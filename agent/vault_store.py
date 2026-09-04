"""Local encrypted vault for browser autofill secrets.

Profile-scoped, model-blind credential store. Metadata (kind, label, origin,
timestamps) lives alongside an encrypted secret payload; the payload is
encrypted at rest with a locally generated Fernet key. The model only ever
sees opaque handles + metadata — secret values are resolved server-side by
the browser fill path and never enter tool results, logs, or the session DB.

Design notes:
- Follows the repo's "default frictionless, 0600 files OK" policy: the key
  file and vault file are created 0600 under ``<HERMES_HOME>/vault/``.
- Ported design (opaque-handle vault fill) from Merit-Systems/OpenInstinct
  (MIT): lib/manager/server/secret-store.ts + vault services.
- Phase 1 supports three item kinds (``login``, ``payment``, ``address``)
  in the store; browser fill support is login-only.
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from hermes_constants import get_hermes_home

VAULT_KINDS = ("login", "payment", "address")

LOGIN_IDENTIFIER_TYPES = ("email", "phone", "username")

_DEFAULT_PORTS = {"http": 80, "https": 443}

_LOCK = threading.Lock()


class VaultError(Exception):
    """Vault failure that is safe to surface (never contains secret values)."""


def normalize_origin(url_or_origin: str) -> str:
    """Normalize a URL or origin to ``scheme://host[:port]``.

    Default ports (80 for http, 443 for https) are stripped so that
    ``https://example.com`` and ``https://example.com:443`` compare equal.
    Raises :class:`VaultError` for values without a scheme + host.
    """
    value = (url_or_origin or "").strip()
    if not value:
        raise VaultError("origin is required")
    if "://" not in value:
        raise VaultError(f"origin must include a scheme (got {value!r})")
    parts = urlsplit(value)
    scheme = (parts.scheme or "").lower()
    host = (parts.hostname or "").lower()
    if not scheme or not host:
        raise VaultError(f"could not parse origin from {value!r}")
    try:
        port = parts.port
    except ValueError as exc:
        raise VaultError(f"invalid port in origin {value!r}") from exc
    if port is None or port == _DEFAULT_PORTS.get(scheme):
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


@dataclass(frozen=True)
class VaultItemMeta:
    """Metadata-only view of a vault item. Never contains secret values.

    For ``kind='login'`` the identifier (email/username/phone) is metadata,
    not a secret: the agent may see it and type it itself. Only the password
    is vault-secret.
    """

    id: str
    kind: str
    label: str
    origin: Optional[str]
    created_at: str
    identifier_type: Optional[str] = None
    identifier: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "origin": self.origin,
            "created_at": self.created_at,
        }
        if self.identifier is not None:
            out["identifier"] = self.identifier
            out["identifier_type"] = self.identifier_type
        return out


class VaultStore:
    """Encrypted, profile-scoped vault under ``<HERMES_HOME>/vault/``."""

    def __init__(self, base_dir: Optional[Path] = None):
        self._base = Path(base_dir) if base_dir is not None else (
            Path(get_hermes_home()) / "vault"
        )
        self._vault_path = self._base / "vault.json.enc"
        self._key_path = self._base / "vault.key"

    # -- key / crypto ------------------------------------------------------

    def _ensure_dir(self) -> None:
        self._base.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Route through the canonical securer (honors managed/NixOS
        # group-share mode and HERMES_UID/GID ownership) rather than a
        # bespoke chmod — same requirement as the browser-profile snapshot
        # dir (f1d05c review).
        try:
            from hermes_cli.config import _secure_dir

            _secure_dir(self._base)
        except Exception:
            try:
                os.chmod(self._base, 0o700)
            except OSError:
                pass

    def _fernet(self):
        from cryptography.fernet import Fernet

        self._ensure_dir()
        if not self._key_path.exists():
            key = Fernet.generate_key()
            fd = os.open(
                self._key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            try:
                os.write(fd, key)
            finally:
                os.close(fd)
        else:
            key = self._key_path.read_bytes().strip()
        try:
            os.chmod(self._key_path, 0o600)
        except OSError:
            pass
        return Fernet(key)

    # -- persistence -------------------------------------------------------

    def _read_all(self) -> List[Dict[str, Any]]:
        if not self._vault_path.exists():
            return []
        blob = self._vault_path.read_bytes()
        if not blob:
            return []
        from cryptography.fernet import InvalidToken

        try:
            raw = self._fernet().decrypt(blob)
        except InvalidToken as exc:
            raise VaultError(
                "vault file could not be decrypted (key mismatch or corruption)"
            ) from exc
        data = json.loads(raw.decode("utf-8"))
        items = data.get("items", [])
        return items if isinstance(items, list) else []

    def _write_all(self, items: List[Dict[str, Any]]) -> None:
        self._ensure_dir()
        payload = json.dumps({"version": 1, "items": items}).encode("utf-8")
        blob = self._fernet().encrypt(payload)
        tmp = self._vault_path.with_suffix(".enc.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, blob)
        finally:
            os.close(fd)
        os.replace(tmp, self._vault_path)
        try:
            os.chmod(self._vault_path, 0o600)
        except OSError:
            pass

    # -- public API ----------------------------------------------------------

    def add_item(
        self,
        kind: str,
        label: str,
        secret: Dict[str, Any],
        origin: Optional[str] = None,
    ) -> VaultItemMeta:
        """Add an item. ``secret`` is the sensitive payload (encrypted at rest).

        For ``kind='login'``, ``origin`` is required and the payload must
        contain ``identifier_type``, ``identifier`` and ``password``. The
        identifier fields are NOT secret — they are moved into item metadata
        (the agent may see and type the identifier itself); only
        ``password`` stays in the encrypted secret payload. ``payment`` and
        ``address`` payloads remain fully secret.
        """
        if kind not in VAULT_KINDS:
            raise VaultError(f"unknown vault kind {kind!r} (expected one of {VAULT_KINDS})")
        label = (label or "").strip()
        if not label:
            raise VaultError("label is required")
        norm_origin: Optional[str] = None
        identifier: Optional[str] = None
        identifier_type: Optional[str] = None
        secret = dict(secret)
        if kind == "login":
            if not origin:
                raise VaultError("origin is required for login items")
            norm_origin = normalize_origin(origin)
            id_type = secret.pop("identifier_type", None)
            if id_type not in LOGIN_IDENTIFIER_TYPES:
                raise VaultError(
                    f"identifier_type must be one of {LOGIN_IDENTIFIER_TYPES}"
                )
            identifier = str(secret.pop("identifier", "") or "").strip()
            if not identifier or not secret.get("password"):
                raise VaultError("login items require identifier and password")
            identifier_type = str(id_type)
            # Login secret payload is password-only; identifier lives in
            # metadata and any stray origin echo is dropped.
            secret = {"password": secret["password"]}
        elif origin:
            norm_origin = normalize_origin(origin)

        item_id = f"vault_{uuid.uuid4().hex[:12]}"
        record = {
            "id": item_id,
            "kind": kind,
            "label": label,
            "origin": norm_origin,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "identifier_type": identifier_type,
            "identifier": identifier,
            "secret": dict(secret),
        }
        with _LOCK:
            items = self._read_all()
            items.append(record)
            self._write_all(items)
        return self._meta(record)

    def list_items(self) -> List[VaultItemMeta]:
        """Metadata-only listing. Secret payloads are never included."""
        with _LOCK:
            return [self._meta(rec) for rec in self._read_all()]

    def has_items(self) -> bool:
        try:
            with _LOCK:
                return bool(self._read_all())
        except Exception:
            return False

    def remove_item(self, item_id: str) -> bool:
        with _LOCK:
            items = self._read_all()
            remaining = [rec for rec in items if rec.get("id") != item_id]
            if len(remaining) == len(items):
                return False
            self._write_all(remaining)
            return True

    def get_meta(self, item_id: str) -> Optional[VaultItemMeta]:
        with _LOCK:
            for rec in self._read_all():
                if rec.get("id") == item_id:
                    return self._meta(rec)
        return None

    def resolve_secret(self, item_id: str) -> Dict[str, Any]:
        """Resolve the decrypted secret payload for server-side use ONLY.

        Callers must never place the returned values into tool results,
        logs, exceptions, or any string that reaches the session DB.
        """
        with _LOCK:
            for rec in self._read_all():
                if rec.get("id") == item_id:
                    return dict(rec.get("secret") or {})
        raise VaultError(f"no vault item with id {item_id!r}")

    @staticmethod
    def _meta(rec: Dict[str, Any]) -> VaultItemMeta:
        identifier = rec.get("identifier")
        return VaultItemMeta(
            id=str(rec.get("id", "")),
            kind=str(rec.get("kind", "")),
            label=str(rec.get("label", "")),
            origin=rec.get("origin"),
            created_at=str(rec.get("created_at", "")),
            identifier_type=rec.get("identifier_type") if identifier else None,
            identifier=identifier or None,
        )


def get_vault_store() -> VaultStore:
    """Default profile-scoped vault store."""
    return VaultStore()


def scrub_secret_from_text(text: str, secret: Dict[str, Any]) -> str:
    """Defensively strip any secret values from a string (e.g. an exception
    message) before it can be surfaced. Case-sensitive exact substring scrub."""
    scrubbed = text
    for value in secret.values():
        if isinstance(value, str) and len(value) >= 3 and value in scrubbed:
            scrubbed = scrubbed.replace(value, "[REDACTED]")
    # Also collapse anything that looks like a leaked password-ish token in
    # common key=value echoes.
    scrubbed = re.sub(r"(password['\"]?\s*[:=]\s*)\S+", r"\1[REDACTED]", scrubbed)
    return scrubbed
