"""Profile-scoped opaque credential store for agent-safe secret use.

The model may request and pass credential references, but plaintext is only
resolved inside trusted execution code. Secret values are encrypted on disk and
redacted from model/tool/log boundaries.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from cryptography.fernet import Fernet, InvalidToken

from hermes_constants import get_hermes_home
from utils import atomic_json_write

_REF_PREFIX = "cred_"
_REF_RE = re.compile(r"^cred_[A-Za-z0-9_-]{16,80}$")
_STORE_VERSION = 1
_REDACTION = "«redacted-credential»"
_MIN_REDACT_SECRET_LEN = 4


class CredentialStoreError(RuntimeError):
    """Raised for credential-store failures safe to show after sanitization."""


@dataclass(frozen=True)
class CredentialRecord:
    ref: str
    name: str
    type: str
    status: str
    created_at: float
    updated_at: float
    revoked_at: Optional[float] = None

    def public_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "ref": self.ref,
            "name": self.name,
            "type": self.type,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.revoked_at is not None:
            data["revoked_at"] = self.revoked_at
        return data


def _credentials_dir() -> Path:
    path = get_hermes_home() / "credentials"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path


def _key_path() -> Path:
    return _credentials_dir() / "master.key"


def _store_path() -> Path:
    return _credentials_dir() / "store.json"


def _ensure_file_private(path: Path) -> None:
    if os.name == "posix":
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass


def _load_or_create_key() -> bytes:
    path = _key_path()
    if path.exists():
        key = path.read_bytes().strip()
        if not key:
            raise CredentialStoreError("credential master key file is empty")
        _ensure_file_private(path)
        return key
    key = Fernet.generate_key()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    try:
        os.write(fd, key + b"\n")
    finally:
        os.close(fd)
    _ensure_file_private(path)
    return key


def _fernet() -> Fernet:
    return Fernet(_load_or_create_key())


def _empty_store() -> Dict[str, Any]:
    return {"version": _STORE_VERSION, "credentials": {}}


def _load_store() -> Dict[str, Any]:
    path = _store_path()
    if not path.exists():
        return _empty_store()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CredentialStoreError(f"credential store is unreadable: {type(exc).__name__}") from exc
    if not isinstance(data, dict):
        raise CredentialStoreError("credential store has invalid format")
    creds = data.get("credentials")
    if not isinstance(creds, dict):
        data["credentials"] = {}
    data.setdefault("version", _STORE_VERSION)
    return data


def _save_store(data: Dict[str, Any]) -> None:
    path = _store_path()
    atomic_json_write(path, data, indent=2)
    _ensure_file_private(path)


def _validate_name(name: str) -> str:
    value = (name or "").strip()
    if not value:
        raise CredentialStoreError("credential name is required")
    if len(value) > 160:
        raise CredentialStoreError("credential name is too long")
    return value


def _validate_type(credential_type: str) -> str:
    value = (credential_type or "secret").strip().lower()
    if not re.match(r"^[a-z][a-z0-9_.:-]{0,63}$", value):
        raise CredentialStoreError("credential type must be a short lowercase identifier")
    return value


def _validate_ref(ref: str) -> str:
    value = (ref or "").strip()
    if not _REF_RE.match(value):
        raise CredentialStoreError("invalid credential reference")
    return value


def _new_ref(name: str, credential_type: str) -> str:
    digest = hashlib.sha256(
        f"{credential_type}\0{name}\0{secrets.token_urlsafe(32)}".encode("utf-8")
    ).digest()
    return _REF_PREFIX + base64.urlsafe_b64encode(digest[:24]).decode("ascii").rstrip("=")


def _encrypt(value: str) -> str:
    if not isinstance(value, str) or value == "":
        raise CredentialStoreError("credential value cannot be empty")
    return _fernet().encrypt(value.encode("utf-8")).decode("ascii")


def _decrypt(token: str) -> str:
    try:
        return _fernet().decrypt(token.encode("ascii")).decode("utf-8")
    except (InvalidToken, UnicodeDecodeError, ValueError) as exc:
        raise CredentialStoreError("credential value could not be decrypted") from exc


def request_credential(name: str, credential_type: str = "secret") -> Dict[str, Any]:
    """Create or return a pending opaque reference; never accepts a secret value."""
    name = _validate_name(name)
    credential_type = _validate_type(credential_type)
    data = _load_store()
    now = time.time()
    for ref, record in data["credentials"].items():
        if (
            isinstance(record, dict)
            and record.get("name") == name
            and record.get("type") == credential_type
            and record.get("status") != "deleted"
        ):
            return _public_record(ref, record, pending_secret=("ciphertext" not in record))
    ref = _new_ref(name, credential_type)
    data["credentials"][ref] = {
        "name": name,
        "type": credential_type,
        "status": "pending",
        "created_at": now,
        "updated_at": now,
    }
    _save_store(data)
    result = _public_record(ref, data["credentials"][ref], pending_secret=True)
    result["entry_ui"] = "Run `hermes credentials set %s --type %s` in a terminal; input is masked and never sent to chat." % (name, credential_type)
    return result


def set_credential_value(name: str, credential_type: str, value: str) -> Dict[str, Any]:
    """Store/update a value entered outside chat and return only public metadata."""
    name = _validate_name(name)
    credential_type = _validate_type(credential_type)
    data = _load_store()
    now = time.time()
    ref = None
    record = None
    for candidate_ref, candidate in data["credentials"].items():
        if (
            isinstance(candidate, dict)
            and candidate.get("name") == name
            and candidate.get("type") == credential_type
            and candidate.get("status") != "deleted"
        ):
            ref = candidate_ref
            record = candidate
            break
    if record is None:
        ref = _new_ref(name, credential_type)
        record = {"name": name, "type": credential_type, "created_at": now}
        data["credentials"][ref] = record
    record["ciphertext"] = _encrypt(value)
    record["status"] = "active"
    record["updated_at"] = now
    record.pop("revoked_at", None)
    _save_store(data)
    return _public_record(str(ref), record, pending_secret=False)


def update_credential_value(ref: str, value: str) -> Dict[str, Any]:
    ref = _validate_ref(ref)
    data = _load_store()
    record = data["credentials"].get(ref)
    if not isinstance(record, dict) or record.get("status") == "deleted":
        raise CredentialStoreError("credential reference not found")
    record["ciphertext"] = _encrypt(value)
    record["status"] = "active"
    record["updated_at"] = time.time()
    record.pop("revoked_at", None)
    _save_store(data)
    return _public_record(str(ref), record, pending_secret=False)


def revoke_credential(ref: str) -> Dict[str, Any]:
    ref = _validate_ref(ref)
    data = _load_store()
    record = data["credentials"].get(ref)
    if not isinstance(record, dict) or record.get("status") == "deleted":
        raise CredentialStoreError("credential reference not found")
    now = time.time()
    record["status"] = "revoked"
    record["revoked_at"] = now
    record["updated_at"] = now
    _save_store(data)
    return _public_record(str(ref), record, pending_secret=False)


def delete_credential(ref: str) -> Dict[str, Any]:
    ref = _validate_ref(ref)
    data = _load_store()
    record = data["credentials"].get(ref)
    if not isinstance(record, dict):
        raise CredentialStoreError("credential reference not found")
    public = _public_record(ref, record, pending_secret=False)
    del data["credentials"][ref]
    _save_store(data)
    public["status"] = "deleted"
    return public


def list_credentials() -> list[Dict[str, Any]]:
    data = _load_store()
    rows = []
    for ref, record in sorted(data["credentials"].items(), key=lambda item: str(item[1].get("name", ""))):
        if isinstance(record, dict) and record.get("status") != "deleted":
            rows.append(_public_record(ref, record, pending_secret=("ciphertext" not in record)))
    return rows


def resolve_credential_value(ref: str, *, expected_type: Optional[str] = None) -> str:
    """Resolve plaintext for trusted execution code only.

    Do not expose this as a model tool. Callers must inject the returned value
    directly into an outbound API/client operation and redact any output.
    """
    ref = _validate_ref(ref)
    data = _load_store()
    record = data["credentials"].get(ref)
    if not isinstance(record, dict):
        raise CredentialStoreError("credential reference not found")
    if record.get("status") != "active":
        raise CredentialStoreError(f"credential is not active: {record.get('status', 'unknown')}")
    if expected_type and record.get("type") != _validate_type(expected_type):
        raise CredentialStoreError("credential type mismatch")
    ciphertext = record.get("ciphertext")
    if not isinstance(ciphertext, str) or not ciphertext:
        raise CredentialStoreError("credential has no stored secret value")
    return _decrypt(ciphertext)


def _public_record(ref: str, record: Dict[str, Any], *, pending_secret: bool) -> Dict[str, Any]:
    out = CredentialRecord(
        ref=ref,
        name=str(record.get("name") or ""),
        type=str(record.get("type") or "secret"),
        status=str(record.get("status") or "pending"),
        created_at=float(record.get("created_at") or 0),
        updated_at=float(record.get("updated_at") or 0),
        revoked_at=(float(record["revoked_at"]) if record.get("revoked_at") else None),
    ).public_dict()
    out["has_secret"] = not pending_secret
    return out


def iter_active_secret_values() -> Iterable[str]:
    """Yield active secret plaintexts for forced redaction boundaries only."""
    try:
        data = _load_store()
    except Exception:
        return []
    values = []
    for record in data.get("credentials", {}).values():
        if not isinstance(record, dict) or record.get("status") != "active":
            continue
        ciphertext = record.get("ciphertext")
        if not isinstance(ciphertext, str):
            continue
        try:
            value = _decrypt(ciphertext)
        except Exception:
            continue
        if len(value) >= _MIN_REDACT_SECRET_LEN:
            values.append(value)
    return values


def redact_registered_secrets(text: str) -> str:
    """Redact exact stored secret values from any output string."""
    if text is None:
        return text
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return text
    try:
        for value in iter_active_secret_values():
            if value and value in text:
                text = text.replace(value, _REDACTION)
    except Exception:
        pass
    return text
