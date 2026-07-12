"""Profile-scoped credential storage for the A2A platform.

Inbound bearer tokens are never stored.  Only a per-record salted scrypt
digest is persisted.  Outbound tokens must remain retrievable so the A2A
client can authenticate to a named peer; those values live only in this
owner-readable store, never in config.yaml or a generic dotenv file.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
_STORE_VERSION = 1
_MAX_STORE_BYTES = 256 * 1024
_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SCRYPT_N = 1 << 14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
_SCRYPT_MAXMEM = 64 * 1024 * 1024
_MIN_OUTBOUND_TOKEN_CHARS = 32
_INBOUND_TOKEN_PREFIX = "a2a_"
_PROCESS_CREDENTIAL_LOCK = threading.RLock()


@contextmanager
def _safe_file_lock(path: Path):
    """POSIX no-follow flock anchored to a pinned, owned directory fd."""
    try:
        import fcntl
    except ImportError as exc:
        raise CredentialStoreError("secure A2A file locking is unavailable") from exc
    _secure_store_directory(path)
    required = ("O_DIRECTORY", "O_NOFOLLOW")
    if not all(hasattr(os, name) for name in required) or not hasattr(os, "open"):
        raise CredentialStoreError("secure A2A file locking is unavailable")
    try:
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
    except OSError as exc:
        raise CredentialStoreError("A2A lock directory is unsafe") from exc
    lock_fd = None
    try:
        try:
            directory_info = os.fstat(directory_fd)
            _validate_owned_directory(directory_info, label="A2A lock directory")
            visible = path.parent.stat(follow_symlinks=False)
            if (visible.st_dev, visible.st_ino) != (directory_info.st_dev, directory_info.st_ino):
                raise CredentialStoreError("A2A lock directory changed during access")
            lock_fd = os.open(
                path.name,
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory_fd,
            )
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            actual = os.fstat(lock_fd)
            _validate_store_path_stat(actual)
            pinned = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
            if (actual.st_dev, actual.st_ino) != (pinned.st_dev, pinned.st_ino):
                raise CredentialStoreError("A2A lock changed during access")
            visible = path.parent.stat(follow_symlinks=False)
            if (visible.st_dev, visible.st_ino) != (directory_info.st_dev, directory_info.st_ino):
                raise CredentialStoreError("A2A lock directory changed during access")
            os.fchmod(lock_fd, 0o600)
        except OSError as exc:
            raise CredentialStoreError("A2A file lock is unsafe") from exc
        yield directory_fd
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        os.close(directory_fd)


class CredentialStoreError(RuntimeError):
    """A safe, non-secret-bearing credential store failure."""


class SecretToken(str):
    """A string bearer whose debug representation is always redacted."""

    def __repr__(self) -> str:
        return "<SecretToken redacted>"


def credentials_path() -> Path:
    """Return the credential path for the active profile/HERMES_HOME."""
    return get_hermes_home() / "a2a" / "credentials.json"


def credentials_lock_path() -> Path:
    return get_hermes_home() / "a2a" / "credentials.lock"


@contextmanager
def _locked_credential_mutation():
    """Serialize a fresh read-modify-write across threads and processes."""
    _secure_store_directory(credentials_path())
    with _PROCESS_CREDENTIAL_LOCK, _safe_file_lock(credentials_lock_path()) as directory_fd:
        yield directory_fd


def _empty_store() -> dict[str, Any]:
    return {"version": _STORE_VERSION, "inbound": {}, "outbound": {}}


def _validate_ref(ref: str) -> str:
    value = str(ref or "").strip()
    if not _REF_RE.fullmatch(value):
        raise ValueError("credential reference must use 1-128 safe characters")
    return value


def _secure_store_directory(path: Path) -> None:
    home = get_hermes_home()
    try:
        home_stat = home.lstat()
    except FileNotFoundError:
        home.mkdir(parents=True, mode=0o700)
        home_stat = home.lstat()
    _validate_owned_directory(home_stat, label="Hermes home")

    try:
        parent_stat = path.parent.lstat()
    except FileNotFoundError:
        path.parent.mkdir(mode=0o700)
        parent_stat = path.parent.lstat()
    _validate_owned_directory(parent_stat, label="A2A credential directory")
    try:
        os.chmod(path.parent, 0o700)
    except OSError as exc:
        raise CredentialStoreError("A2A credential directory permissions are unsafe") from exc


def _validate_owned_directory(info: os.stat_result, *, label: str) -> None:
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise CredentialStoreError(f"{label} must be a real directory")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise CredentialStoreError(f"{label} has an unexpected owner")


def _validate_store_path(path: Path) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError:
        raise
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise CredentialStoreError("A2A credential store must be a regular file")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise CredentialStoreError("A2A credential store has an unexpected owner")
    return info


def _load_credentials(directory_fd: int | None = None) -> dict[str, Any]:
    if directory_fd is None:
        with _locked_credential_mutation() as pinned_fd:
            return _load_credentials(pinned_fd)
    path = credentials_path()
    try:
        expected = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return _empty_store()
    if stat.S_ISLNK(expected.st_mode):
        raise CredentialStoreError("A2A credential store must be a regular file")
    _validate_store_path_stat(expected)
    if expected.st_size > _MAX_STORE_BYTES:
        raise CredentialStoreError("A2A credential store is too large")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path.name, flags, dir_fd=directory_fd)
        with os.fdopen(fd, "rb") as stream:
            actual = os.fstat(stream.fileno())
            if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
                raise CredentialStoreError("A2A credential store changed during access")
            _validate_store_path_stat(actual)
            if stat.S_IMODE(actual.st_mode) != 0o600:
                os.fchmod(stream.fileno(), 0o600)
            raw = stream.read(_MAX_STORE_BYTES + 1)
    except CredentialStoreError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise CredentialStoreError("A2A credential store is unreadable or invalid") from exc
    if len(raw) > _MAX_STORE_BYTES:
        raise CredentialStoreError("A2A credential store is too large")
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CredentialStoreError("A2A credential store is unreadable or invalid") from exc
    if not isinstance(data, dict) or set(data) != {"version", "inbound", "outbound"} or data.get("version") != _STORE_VERSION:
        raise CredentialStoreError("A2A credential store has an unsupported format")
    inbound = data.get("inbound")
    outbound = data.get("outbound")
    if not isinstance(inbound, dict) or not isinstance(outbound, dict):
        raise CredentialStoreError("A2A credential store has an invalid structure")
    return data


def _validate_store_path_stat(info: os.stat_result) -> None:
    if not stat.S_ISREG(info.st_mode):
        raise CredentialStoreError("A2A credential store must be a regular file")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise CredentialStoreError("A2A credential store has an unexpected owner")


def _save_credentials(data: dict[str, Any], directory_fd: int) -> None:
    path = credentials_path()
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    if len(encoded) > _MAX_STORE_BYTES:
        raise CredentialStoreError("A2A credential store is too large")
    temp_name = f".{path.name}.{secrets.token_hex(8)}.tmp"
    fd = os.open(
        temp_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory_fd,
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
    finally:
        try:
            os.unlink(temp_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode(value: Any) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError("invalid encoded credential field")
    padded = value + "=" * (-len(value) % 4)
    return base64.b64decode(padded.encode("ascii"), altchars=b"-_", validate=True)


def _derive(token: str, salt: bytes) -> bytes:
    return hashlib.scrypt(
        token.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
        maxmem=_SCRYPT_MAXMEM,
    )


def _new_inbound_token() -> tuple[str, SecretToken, str]:
    credential_id = secrets.token_urlsafe(16)
    secret = secrets.token_urlsafe(32)
    token = SecretToken(f"{_INBOUND_TOKEN_PREFIX}{credential_id}.{secret}")
    return credential_id, token, secret


def generate_token() -> SecretToken:
    """Generate an inbound bearer with a random public lookup identifier."""
    return _new_inbound_token()[1]


def _build_inbound_record(ref: str) -> tuple[str, SecretToken, dict[str, Any]]:
    credential_id, token, secret = _new_inbound_token()
    salt = secrets.token_bytes(16)
    digest = _derive(secret, salt)
    return credential_id, token, {
        "credential_ref": ref,
        "algorithm": "scrypt",
        "n": _SCRYPT_N,
        "r": _SCRYPT_R,
        "p": _SCRYPT_P,
        "salt": _encode(salt),
        "digest": _encode(digest),
        "created_at": _now(),
    }


def create_inbound_credential(ref: str) -> SecretToken:
    """Create or replace an inbound credential and return its token once."""
    ref = _validate_ref(ref)
    credential_id, token, record = _build_inbound_record(ref)
    with _locked_credential_mutation() as directory_fd:
        data = _load_credentials(directory_fd)
        if any(
            isinstance(existing, dict) and existing.get("credential_ref") == ref
            for existing in data["inbound"].values()
        ):
            raise ValueError("inbound credential reference already exists")
        data["inbound"][credential_id] = record
        _save_credentials(data, directory_fd)
    return token


def rotate_inbound_credential(ref: str) -> SecretToken:
    """Rotate an inbound credential, immediately invalidating its old token."""
    ref = _validate_ref(ref)
    credential_id, token, record = _build_inbound_record(ref)
    with _locked_credential_mutation() as directory_fd:
        data = _load_credentials(directory_fd)
        old_id = next(
            (
                existing_id
                for existing_id, existing in data["inbound"].items()
                if isinstance(existing, dict) and existing.get("credential_ref") == ref
            ),
            None,
        )
        if old_id is None:
            raise KeyError("inbound credential reference not found")
        del data["inbound"][old_id]
        data["inbound"][credential_id] = record
        _save_credentials(data, directory_fd)
    return token


def _parse_inbound_token(candidate: str) -> tuple[str, str] | None:
    if not isinstance(candidate, str) or not candidate.startswith(_INBOUND_TOKEN_PREFIX):
        return None
    payload = candidate.removeprefix(_INBOUND_TOKEN_PREFIX)
    try:
        credential_id, secret = payload.split(".", 1)
    except ValueError:
        return None
    if not credential_id or len(secret) < _MIN_OUTBOUND_TOKEN_CHARS:
        return None
    return credential_id, secret


def resolve_inbound_token(candidate: str) -> str | None:
    """Resolve a bearer to its credential ref with at most one scrypt call."""
    parsed = _parse_inbound_token(candidate)
    if parsed is None:
        return None
    credential_id, secret = parsed
    try:
        record = _load_credentials()["inbound"].get(credential_id)
        if not isinstance(record, dict) or record.get("algorithm") != "scrypt":
            return None
        if (
            int(record.get("n")) != _SCRYPT_N
            or int(record.get("r")) != _SCRYPT_R
            or int(record.get("p")) != _SCRYPT_P
        ):
            return None
        salt = _decode(record.get("salt"))
        expected = _decode(record.get("digest"))
        actual = _derive(secret, salt)
        credential_ref = record.get("credential_ref")
        if not isinstance(credential_ref, str):
            return None
        return credential_ref if hmac.compare_digest(actual, expected) else None
    except (CredentialStoreError, KeyError, TypeError, ValueError, OSError):
        return None


def verify_inbound_token(ref: str, candidate: str) -> bool:
    """Constant-time verification that fails closed on every malformed input."""
    try:
        ref = _validate_ref(ref)
        resolved = resolve_inbound_token(candidate)
        return resolved is not None and hmac.compare_digest(resolved, ref)
    except (CredentialStoreError, KeyError, TypeError, ValueError, OSError):
        return False


def store_outbound_credential(ref: str, token: str) -> None:
    """Store the bearer required to call one explicitly configured peer."""
    ref = _validate_ref(ref)
    value = str(token or "").strip()
    if len(value) < _MIN_OUTBOUND_TOKEN_CHARS:
        raise ValueError("outbound credential is too short")
    with _locked_credential_mutation() as directory_fd:
        data = _load_credentials(directory_fd)
        data["outbound"][ref] = {"token": value, "updated_at": _now()}
        _save_credentials(data, directory_fd)


def load_outbound_token(ref: str) -> SecretToken | None:
    """Return an outbound token only to an explicit credential lookup."""
    try:
        with _locked_credential_mutation() as directory_fd:
            return _load_outbound_token_unlocked(ref, directory_fd)
    except CredentialStoreError:
        return None


def _load_outbound_token_unlocked(ref: str, directory_fd: int) -> SecretToken | None:
    """Load one outbound bearer from an already pinned credential transaction."""
    ref = _validate_ref(ref)
    record = _load_credentials(directory_fd)["outbound"].get(ref)
    if not isinstance(record, dict):
        return None
    token = record.get("token")
    return SecretToken(token) if isinstance(token, str) and token else None


def delete_credential(ref: str, *, direction: str) -> bool:
    return _delete_credential_with_snapshot(ref, direction=direction) is not None


def _delete_credential_with_snapshot(
    ref: str, *, direction: str
) -> tuple[str, dict[str, Any]] | None:
    """Delete one record and return an internal rollback snapshot."""
    ref = _validate_ref(ref)
    if direction not in {"inbound", "outbound"}:
        raise ValueError("credential direction must be inbound or outbound")
    with _locked_credential_mutation() as directory_fd:
        data = _load_credentials(directory_fd)
        if direction == "inbound":
            credential_id = next(
                (
                    key
                    for key, record in data["inbound"].items()
                    if isinstance(record, dict) and record.get("credential_ref") == ref
                ),
                None,
            )
            if credential_id is not None:
                snapshot = (credential_id, copy.deepcopy(data["inbound"][credential_id]))
                del data["inbound"][credential_id]
            else:
                snapshot = None
        else:
            record = data[direction].pop(ref, None)
            snapshot = (ref, copy.deepcopy(record)) if isinstance(record, dict) else None
        if snapshot is not None:
            _save_credentials(data, directory_fd)
    return snapshot


def _restore_credential_snapshot(
    snapshot: tuple[str, dict[str, Any]], *, direction: str
) -> None:
    """Restore a record after a setup transaction's config write failed."""
    if direction not in {"inbound", "outbound"}:
        raise ValueError("credential direction must be inbound or outbound")
    key, record = snapshot
    with _locked_credential_mutation() as directory_fd:
        data = _load_credentials(directory_fd)
        if key in data[direction]:
            raise CredentialStoreError("credential rollback conflicts with existing state")
        data[direction][key] = copy.deepcopy(record)
        _save_credentials(data, directory_fd)


def credential_summary() -> dict[str, list[str]]:
    """Return reference names only; secret values and hashes never escape."""
    data = _load_credentials()
    return {
        "inbound": sorted(
            record["credential_ref"]
            for record in data["inbound"].values()
            if isinstance(record, dict) and isinstance(record.get("credential_ref"), str)
        ),
        "outbound": sorted(data["outbound"]),
    }
