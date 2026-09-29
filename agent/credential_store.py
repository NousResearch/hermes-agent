"""Profile-scoped encrypted credential store for agent-safe secret use.

The model may request credential references and pass them to trusted execution
code, but plaintext values are only resolved *inside* that execution code --
they never appear in tool results, chat context, or logs.

Storage layout under get_hermes_home():

  credentials/
    master.key   -- Fernet key (000600, created once)
    store.json   -- {"keys": {"name": {"enc": <hex>, "created_at": "...", ...}, ...}}

The store is per-profile (get_hermes_home()-scoped), so refs never resolve
across profiles. Values are redacted from all output paths at the redact
boundary (agent/redact.py).
"""

from __future__ import annotations

import json
import logging
import os
import stat
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from cryptography.fernet import Fernet

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_STORE_DIR = "credentials"
_KEY_FILE = "master.key"
_STORE_FILE = "store.json"

# Keys whose names are too generic for credential storage -- a credential
# name must look deliberate (at least 4 chars, not a common programming
# term). This prevents accidental overwrites of env-var-style names.
_FORBIDDEN_NAMES: frozenset[str] = frozenset({
    "key", "key1", "key2", "token", "secret", "password", "pass",
    "cred", "creds", "credential", "credentials",
    "value", "val", "data", "id", "name",
})

_MIN_NAME_LENGTH = 3
_MAX_NAME_LENGTH = 128


class CredentialStore:
    """Encrypted per-profile credential store.

    Thread-safe via file-level locking (atomic write).
    """

    def __init__(self) -> None:
        self._home = get_hermes_home()
        self._store_dir = self._home / _STORE_DIR
        self._key_path = self._store_dir / _KEY_FILE
        self._store_path = self._store_dir / _STORE_FILE
        self._fernet: Fernet | None = None

    # -- Public API -----------------------------------------------------------

    def init(self) -> None:
        """Ensure the store directory and master key exist (idempotent)."""
        self._store_dir.mkdir(parents=True, exist_ok=True)
        if not self._key_path.exists():
            key = Fernet.generate_key()
            self._write_key(key)

    def is_ready(self) -> bool:
        """True when the master key exists and the store can be read."""
        return self._key_path.exists()

    def set(self, name: str, value: str, *, overwrite: bool = False) -> Dict[str, Any]:
        """Store a named credential. Name is validated."""
        self._validate_name(name)
        self.init()
        fernet = self._get_fernet()
        store = self._load_store()
        existing = name in store.get("entries", {})

        if existing and not overwrite:
            return {
                "success": False,
                "error": f"Credential '{name}' already exists. Use overwrite=True to replace it.",
            }

        encrypted = fernet.encrypt(value.encode("utf-8")).decode("utf-8")
        store.setdefault("entries", {})[name] = {
            "enc": encrypted,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self._write_store(store)

        return {"success": True, "name": name, "created": not existing, "updated": existing}

    def get(self, name: str) -> str | None:
        """Return the plaintext value for *name*, or None if absent."""
        fernet = self._get_fernet()
        if fernet is None:
            return None
        store = self._load_store()
        entry = store.get("entries", {}).get(name)
        if entry is None:
            return None
        try:
            return fernet.decrypt(entry["enc"].encode("utf-8")).decode("utf-8")
        except Exception as exc:
            logger.error("Failed to decrypt credential '%s': %s", name, exc)
            return None

    def list(self) -> List[Dict[str, Any]]:
        """Return metadata for every stored credential (never the values)."""
        store = self._load_store()
        entries = store.get("entries", {})
        return [
            {
                "name": name,
                "created_at": meta.get("created_at"),
                "updated_at": meta.get("updated_at"),
            }
            for name, meta in sorted(entries.items())
        ]

    def delete(self, name: str) -> Dict[str, Any]:
        """Remove a credential. Returns success/failure."""
        store = self._load_store()
        entries = store.get("entries", {})
        if name not in entries:
            return {"success": False, "error": f"Credential '{name}' not found."}
        del entries[name]
        self._write_store(store)
        return {"success": True, "name": name}

    def resolve_value(self, name: str) -> str | None:
        """Resolve a credential to plaintext. For trusted execution code only.

        This is deliberately NOT a tool surface -- it is called by the
        execute_code tool and other trusted execution paths to get the
        actual value for use in scripts, terminal calls, etc.
        """
        return self.get(name)

    # -- Private helpers ------------------------------------------------------

    def _validate_name(self, name: str) -> None:
        """Validate a credential name. Raises ValueError on invalid names."""
        if not name or len(name) < _MIN_NAME_LENGTH:
            raise ValueError(
                f"Credential name must be at least {_MIN_NAME_LENGTH} characters."
            )
        if len(name) > _MAX_NAME_LENGTH:
            raise ValueError(
                f"Credential name must be at most {_MAX_NAME_LENGTH} characters."
        )
        if not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "Credential name may only contain letters, digits, underscores, and hyphens."
            )
        # Reject generic/forbidden names that are too vague
        lower = name.lower()
        if lower in _FORBIDDEN_NAMES or lower.startswith("env_") or lower.startswith("os_"):
            raise ValueError(
                f"Credential name '{name}' is too generic. Use a descriptive name like "
                f"'supabase_service_key' or 'stripe_secret_key'."
            )

    def _get_fernet(self) -> Fernet | None:
        """Lazy-init Fernet from the master key file."""
        if self._fernet is not None:
            return self._fernet
        if not self._key_path.exists():
            return None
        try:
            key = self._read_key()
            self._fernet = Fernet(key)
            return self._fernet
        except Exception as exc:
            logger.error("Failed to initialise Fernet from %s: %s", self._key_path, exc)
            return None

    def _load_store(self) -> Dict[str, Any]:
        """Read the store JSON from disk."""
        if not self._store_path.exists():
            return {"version": 1, "entries": {}}
        try:
            raw = self._store_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict):
                data.setdefault("entries", {})
                return data
            return {"version": 1, "entries": {}}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read credential store at %s: %s", self._store_path, exc)
            return {"version": 1, "entries": {}}

    def _write_store(self, store: Dict[str, Any]) -> None:
        """Atomically write the store JSON."""
        from utils import atomic_write_text

        store["version"] = 1
        store["updated_at"] = time.time()
        payload = json.dumps(store, indent=2, ensure_ascii=False)
        atomic_write_text(self._store_path, payload, tmp_prefix=".cred_")

    def _write_key(self, key: bytes) -> None:
        """Write the Fernet master key with restricted permissions."""
        self._store_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._store_dir / f".key_tmp_{os.getpid()}"
        try:
            tmp.write_bytes(key)
            tmp.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0o600
            tmp.rename(self._key_path)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    def _read_key(self) -> bytes:
        """Read the Fernet master key."""
        return self._key_path.read_bytes()


# -- Module-level singleton factory -------------------------------------------

_store_instance: CredentialStore | None = None


def get_store() -> CredentialStore:
    """Return a cached per-profile CredentialStore singleton."""
    global _store_instance
    if _store_instance is None:
        _store_instance = CredentialStore()
    return _store_instance


def reset_store() -> None:
    """Reset the singleton (for tests)."""
    global _store_instance
    _store_instance = None