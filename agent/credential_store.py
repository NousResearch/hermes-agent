"""Profile-scoped encrypted credential store for agent-safe secret use.

Named secrets (Supabase keys, Stripe secrets, DB passwords) the agent needs to
do its work, encrypted at rest under ``<HERMES_HOME>/credentials/``:

  master.key   Fernet key (0600, created once)
  store.enc    Fernet-encrypted JSON ``{name: {value, created_at, updated_at}}`` (0600)

Same shape as ``agent/vault_store.py`` — key and ciphertext sit side by side —
so ``agent/file_safety`` denies the directory and both files to the agent's
generic file tools: reading key + ciphertext together IS the plaintext, and the
model must reach a value only through the paths that register it with the
redactor (``agent.redact.register_secret``), never by reading the store.

Values never appear in tool results or model context on their own: ``get()`` /
``resolve_value()`` exist for trusted execution paths only (execute_code's
``credentials=`` injection, ``/creds show``, ``/creds inject``), and callers
register the value with the redactor before it can be echoed back.

Profile scope: paths resolve ``get_hermes_home()`` on every call (never at
import), so a multiplexed gateway never serves profile A's credentials to
profile B — the per-home singleton below keys on the resolved home.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home
from utils import atomic_write_bytes
import builtins

logger = logging.getLogger(__name__)

STORE_DIR = "credentials"
KEY_FILE = "master.key"
STORE_FILE = "store.enc"
LOCK_FILE = ".store.lock"

# Names too generic to identify a credential: a deliberate name must not be an
# env-var-shaped word that later gets `/creds inject`-ed over PATH or TOKEN.
_FORBIDDEN_NAMES = frozenset({
    "key", "key1", "key2", "token", "secret", "password", "pass", "passwd",
    "cred", "creds", "credential", "credentials", "value", "val", "data",
    "id", "name", "env", "path", "home", "user", "shell", "tmp", "dir",
})
_MIN_NAME_LENGTH = 3
_MAX_NAME_LENGTH = 128

_LOCK = threading.Lock()

# fcntl is POSIX-only; Windows locks a byte range with msvcrt (same shape as
# agent/vault_store.py and tools/skill_usage.py).
msvcrt = None
try:
    import fcntl
except ImportError:  # pragma: no cover - platform-specific fallback
    fcntl = None
    with suppress(ImportError):
        import msvcrt


class CredentialStoreError(RuntimeError):
    """Store could not be read or written safely.

    Never treat this as "empty": an unreadable store that read as empty would
    let the next ``set()`` silently wipe every other credential.
    """


def validate_name(name: str) -> None:
    """Raise ``ValueError`` when *name* is not a usable credential name."""
    if not name or len(name) < _MIN_NAME_LENGTH:
        raise ValueError(f"Credential name must be at least {_MIN_NAME_LENGTH} characters.")
    if len(name) > _MAX_NAME_LENGTH:
        raise ValueError(f"Credential name must be at most {_MAX_NAME_LENGTH} characters.")
    if not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError(
            "Credential name may only contain letters, digits, underscores, and hyphens."
        )
    lower = name.lower()
    if lower in _FORBIDDEN_NAMES or lower.startswith(("env_", "os_")):
        raise ValueError(
            f"Credential name '{name}' is too generic. Use a descriptive name like "
            f"'supabase_service_key' or 'stripe_secret_key'."
        )


class CredentialStore:
    """Encrypted, profile-scoped credential store under ``<HERMES_HOME>/credentials/``."""

    def __init__(self, base_dir: Optional[Path] = None):
        # Resolved per instance (never at import): get_store() keys its cache on
        # this value, so a home switch mid-process cannot leak across profiles.
        self._base = Path(base_dir) if base_dir is not None else Path(get_hermes_home()) / STORE_DIR
        self._store_path = self._base / STORE_FILE
        self._key_path = self._base / KEY_FILE

    # -- key / crypto ------------------------------------------------------

    @property
    def base_dir(self) -> Path:
        return self._base

    def _ensure_dir(self) -> None:
        self._base.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Canonical securer (honors managed/NixOS group-share mode and
        # HERMES_UID/GID ownership), same as the vault dir.
        try:
            from hermes_cli.config import _secure_dir

            _secure_dir(self._base)
        except Exception as exc:
            logger.debug("credential store: _secure_dir failed (%s); chmod fallback", exc, exc_info=True)
            with suppress(OSError):
                os.chmod(self._base, 0o700)

    @contextmanager
    def _locked(self):
        """Serialize read-modify-write across threads AND processes: the gateway,
        a CLI ``/creds set`` and a TUI slash worker all write the same file, and
        two unlocked writers would drop each other's entries."""
        with _LOCK:
            self._ensure_dir()
            lock_path = self._base / LOCK_FILE
            if msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
                lock_path.write_text(" ", encoding="utf-8")  # msvcrt needs a non-empty byte range
            with open(lock_path, "r+" if msvcrt else "a+", encoding="utf-8") as fd:
                if fcntl:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                elif msvcrt:
                    fd.seek(0)
                    msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    with suppress(OSError):
                        if fcntl:
                            fcntl.flock(fd, fcntl.LOCK_UN)
                        elif msvcrt:
                            fd.seek(0)
                            msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)

    def _fernet(self):
        from cryptography.fernet import Fernet

        self._ensure_dir()
        if not self._key_path.exists():
            key = Fernet.generate_key()
            fd = os.open(self._key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.write(fd, key)
            finally:
                os.close(fd)
        else:
            key = self._key_path.read_bytes().strip()
        with suppress(OSError):
            os.chmod(self._key_path, 0o600)
        return Fernet(key)

    # -- persistence -------------------------------------------------------

    def _read_all(self) -> dict[str, dict[str, Any]]:
        """Decrypt the store. Raises CredentialStoreError on anything unreadable —
        an unreadable store must never read as an empty one (that is how a
        corrupt file turns into "next set() wipes every credential")."""
        if not self._store_path.exists():
            return {}
        blob = self._store_path.read_bytes()
        if not blob:
            return {}
        from cryptography.fernet import InvalidToken

        try:
            raw = self._fernet().decrypt(blob)
        except InvalidToken as exc:
            raise CredentialStoreError(
                f"credential store {self._store_path} could not be decrypted "
                "(master.key mismatch or corruption); refusing to read or overwrite it"
            ) from exc
        try:
            data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CredentialStoreError(
                f"credential store {self._store_path} is corrupted (invalid JSON); "
                "refusing to read or overwrite it"
            ) from exc
        entries = data.get("entries") if isinstance(data, dict) else None
        if not isinstance(entries, dict):
            # Unknown or malformed shape must never read as an empty store: the next
            # set() would rewrite the file and delete every credential in it.
            raise CredentialStoreError(
                f"credential store {self._store_path} is corrupted (unexpected shape); "
                "refusing to read or overwrite it"
            )
        return entries

    def _write_all(self, entries: dict[str, dict[str, Any]]) -> None:
        self._ensure_dir()
        payload = json.dumps({"version": 1, "entries": entries}).encode("utf-8")
        blob = self._fernet().encrypt(payload)
        # fsync_dir: the directory entry must be durable too (power loss between
        # rename and the next sync); 0600 keeps the ciphertext private from
        # other local users on a shared host.
        atomic_write_bytes(self._store_path, blob, mode=0o600, fsync_dir=True)

    # -- public API --------------------------------------------------------

    def init(self) -> None:
        """Create the directory and master key (idempotent)."""
        self._fernet()

    def is_ready(self) -> bool:
        """True when the master key exists."""
        return self._key_path.exists()

    def set(self, name: str, value: str, *, overwrite: bool = False) -> dict[str, Any]:
        """Store a named credential. Raises ValueError on a bad name,
        CredentialStoreError when the existing store is unreadable."""
        validate_name(name)
        if not isinstance(value, str) or not value:
            raise ValueError("Credential value must be a non-empty string.")
        with self._locked():
            entries = self._read_all()
            existing = entries.get(name)
            if existing and not overwrite:
                return {
                    "success": False,
                    "error": f"Credential '{name}' already exists. Use overwrite to replace it.",
                }
            now = time.time()
            entries[name] = {
                "value": value,
                "created_at": existing.get("created_at", now) if existing else now,
                "updated_at": now,
            }
            self._write_all(entries)
        return {"success": True, "name": name, "created": existing is None, "updated": existing is not None}

    def get(self, name: str) -> Optional[str]:
        """Plaintext value for *name*, or None when absent. Trusted execution paths only.

        Platform-backed entries (``backend`` + ``ref``) resolve through their
        backend — the value never lives locally for those — and the remote read
        happens outside the lock, since it runs a vendor CLI.
        """
        with self._locked():
            entries = self._read_all()
            entry = entries.get(name)
        if entry is None:
            return None
        backend_name = entry.get("backend")
        if backend_name:
            from agent.credential_backends import read_ref

            return read_ref(str(backend_name), str(entry.get("ref") or ""))
        value = entry.get("value")
        return str(value) if value is not None else None

    def entry(self, name: str) -> Optional[dict[str, Any]]:
        """Metadata for one entry (backend/ref/timestamps) — never the local value."""
        with self._locked():
            entries = self._read_all()
            entry = entries.get(name)
        if entry is None:
            return None
        return {key: val for key, val in entry.items() if key != "value"}

    def list(self) -> builtins.list[dict[str, Any]]:
        """Metadata for every credential (name, backend, timestamps, never values)."""
        with self._locked():
            entries = self._read_all()
        return [
            {"name": name,
             "backend": str(meta.get("backend") or "local"),
             "created_at": meta.get("created_at"),
             "updated_at": meta.get("updated_at")}
            for name, meta in sorted(entries.items())
        ]

    def set_remote(self, name: str, backend: str, ref: str, *,
                   overwrite: bool = False) -> dict[str, Any]:
        """Record an index entry for a credential whose value lives in *backend*.

        This is what makes the store platform-only: the entry carries the
        locator, never the value, so there is no local copy to drift or leak.
        """
        validate_name(name)
        if not backend or not ref:
            raise ValueError("backend and ref are required for a platform entry.")
        with self._locked():
            entries = self._read_all()
            existing = entries.get(name)
            if existing and not overwrite:
                return {
                    "success": False,
                    "error": f"Credential '{name}' already exists. Use overwrite to replace it.",
                }
            now = time.time()
            entries[name] = {
                "backend": backend,
                "ref": ref,
                "created_at": existing.get("created_at", now) if existing else now,
                "updated_at": now,
            }
            self._write_all(entries)
        return {"success": True, "name": name, "backend": backend,
                "created": existing is None, "updated": existing is not None}

    def delete(self, name: str) -> dict[str, Any]:
        """Remove one credential."""
        with self._locked():
            entries = self._read_all()
            if name not in entries:
                return {"success": False, "error": f"Credential '{name}' not found."}
            del entries[name]
            self._write_all(entries)
        return {"success": True, "name": name}

    def clear(self) -> dict[str, Any]:
        """Remove every credential (single atomic write, no per-entry churn)."""
        with self._locked():
            entries = self._read_all()
            count = len(entries)
            if count:
                self._write_all({})
        return {"success": True, "removed": count}

    def resolve_value(self, name: str) -> Optional[str]:
        """Resolve a credential to plaintext for trusted execution code.

        Deliberately not a tool surface: execute_code's ``credentials=`` path
        calls this, and the caller registers the value with the redactor before
        it can reach any output.
        """
        return self.get(name)


# -- per-home singleton ------------------------------------------------------

_store_cache: dict[str, CredentialStore] = {}


def get_store() -> CredentialStore:
    """CredentialStore for the *currently resolved* home (cached per home).

    Keyed on the resolved home rather than a process-lifetime instance: a
    multiplexed gateway changes homes per source, and profile A's store must
    never be handed to profile B.
    """
    home = str(get_hermes_home())
    store = _store_cache.get(home)
    if store is None:
        store = CredentialStore()
        _store_cache[home] = store
    return store


def reset_store() -> None:
    """Drop every cached store (tests, profile teardown)."""
    global _store_cache
    _store_cache = {}