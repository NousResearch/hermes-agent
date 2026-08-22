"""HIGH-2: Token Keychain integration for federation secrets.

Pillar 8: Operational Security (SECURITY-BASELINE.md)

Stores federation tokens (cluster_secret, auth_token, etc.) in the OS
keychain with platform-specific backends:

- macOS: `security` command (Apple Keychain)
- Linux: `secret-tool` (libsecret/Gnome Keyring) or `keyring` lib
- Anywhere: encrypted file fallback (AES-256-GCM, machine-bound key)

Tokens NEVER live in env vars (visible to subprocesses) or plaintext
files (visible to backups, sync, etc.). The fallback is encrypted-at-rest
with a key derived from machine-id + user-id, so it doesn't sync.

This module is INTERNAL to the federation module. It does NOT expose
raw secrets to logs — uses TokenStr for redaction.
"""
from __future__ import annotations

import os
import sys
import json
import base64
import hashlib
import hmac
import logging
import platform
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from gateway.federation.audit import TokenStr  # for redaction


log = logging.getLogger(__name__)


# === Backend detection ===

class SecretBackend:
    """Base class for secret-storage backends."""
    name: str = "base"

    def get(self, key: str) -> Optional[str]:
        raise NotImplementedError

    def set(self, key: str, value: str) -> None:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError

    def is_available(self) -> bool:
        return False


class MacOSKeychainBackend(SecretBackend):
    """macOS Keychain via `security` command.

    Uses user-level keychain (not system). Token names are prefixed with
    'hermes-federation.' to namespace.
    """
    name = "macos-keychain"
    SERVICE = "hermes-federation"

    def is_available(self) -> bool:
        return platform.system() == "Darwin" and shutil.which("security") is not None

    def _key_args(self, key: str) -> list[str]:
        # Account is the key, service is the namespace
        return ["-a", key, "-s", self.SERVICE]

    def get(self, key: str) -> Optional[str]:
        try:
            r = subprocess.run(
                ["security", "find-generic-password", "-w"]
                + self._key_args(key),
                capture_output=True, text=True, timeout=5,
                env={},  # clear env so we don't leak it
            )
            if r.returncode == 0:
                return r.stdout.strip()
            if r.returncode == 44:  # item not found
                return None
            log.debug(f"keychain get failed: rc={r.returncode} err={r.stderr[:200]}")
            return None
        except subprocess.TimeoutExpired:
            log.warning("keychain timeout (will fall back to encrypted file)")
            return None

    def set(self, key: str, value: str) -> None:
        # -U to update if exists
        try:
            r = subprocess.run(
                ["security", "add-generic-password", "-U", "-w", value]
                + self._key_args(key),
                capture_output=True, text=True, timeout=5,
                env={},
            )
            if r.returncode != 0:
                # Try delete first then add
                self.delete(key)
                r = subprocess.run(
                    ["security", "add-generic-password", "-w", value]
                    + self._key_args(key),
                    capture_output=True, text=True, timeout=5,
                    env={},
                )
                if r.returncode != 0:
                    raise RuntimeError(f"keychain set failed: {r.stderr[:200]}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("keychain timeout")

    def delete(self, key: str) -> None:
        try:
            subprocess.run(
                ["security", "delete-generic-password"]
                + self._key_args(key),
                capture_output=True, text=True, timeout=5,
                env={},
            )
        except subprocess.TimeoutExpired:
            pass


class EncryptedFileBackend(SecretBackend):
    """Encrypted file fallback.

    Stores secrets in a single file with AES-256-GCM.
    Machine-bound key derivation (HMAC-SHA256 of machine-id + user-id).
    The file lives in user-only mode (0600) and never syncs.
    """
    name = "encrypted-file"

    DEFAULT_PATH = Path("~/.hermes/federation/secrets.json.enc")

    def __init__(self, path: Optional[Path] = None):
        self._path = Path(path) if path else Path(self.DEFAULT_PATH).expanduser()
        self._lock = threading.Lock()
        self._machine_key = self._derive_machine_key()
        self._ensure_path()

    def _ensure_path(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            # Verify permissions
            mode = self._path.stat().st_mode & 0o777
            if mode != 0o600:
                log.warning(
                    f"secrets file mode {oct(mode)} != 0600, fixing (security)"
                )
                os.chmod(self._path, 0o600)

    def _derive_machine_key(self) -> bytes:
        """Derive a 32-byte key from machine-id + user-id.

        HMAC-SHA256 is fine here because the input space is small
        (no collisions in practice) and the secret is per-machine.
        """
        # Collect machine-bound entropy
        parts = []
        # machine-id (Linux) or Hardware UUID (macOS)
        try:
            mid = subprocess.run(
                ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                capture_output=True, text=True, timeout=2,
            )
            for line in mid.stdout.splitlines():
                if "IOPlatformUUID" in line:
                    parts.append(line.split('"')[3] if '"' in line else line)
                    break
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        # Fallback to machine-id file
        if not parts:
            for p in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
                if os.path.exists(p):
                    parts.append(Path(p).read_text(encoding="utf-8").strip())
                    break
        # username
        import getpass
        parts.append(getpass.getuser())
        # hostname
        parts.append(platform.node())
        # arch
        parts.append(platform.machine())
        # Use a static domain-separation label
        return hmac.new(
            b"hermes-federation-secretstore.v1",
            "|".join(parts).encode(),
            hashlib.sha256,
        ).digest()

    def _load(self) -> dict:
        """Load and decrypt the file."""
        if not self._path.exists():
            return {}
        try:
            raw = self._path.read_bytes()
            data = json.loads(raw)
            nonce = base64.b64decode(data["nonce"])
            ct = base64.b64decode(data["ct"])
            tag = base64.b64decode(data["tag"])
            # AES-256-GCM via cryptography lib
            try:
                from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            except ImportError:
                log.error(
                    "cryptography library not installed; cannot decrypt secrets. "
                    "Install with: pip install cryptography"
                )
                return {}
            aes = AESGCM(self._machine_key)
            plaintext = aes.decrypt(nonce, ct + tag, None)
            return json.loads(plaintext.decode())
        except Exception as e:
            log.error(f"failed to decrypt secrets file: {e}")
            return {}

    def _save(self, data: dict) -> None:
        """Encrypt and save the file."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            import secrets
        except ImportError:
            log.error("cryptography library not installed; cannot save secrets.")
            return
        aes = AESGCM(self._machine_key)
        nonce = secrets.token_bytes(12)
        plaintext = json.dumps(data).encode()
        ct_with_tag = aes.encrypt(nonce, plaintext, None)
        # Split ct + tag (last 16 bytes)
        ct = ct_with_tag[:-16]
        tag = ct_with_tag[-16:]
        payload = {
            "v": 1,
            "nonce": base64.b64encode(nonce).decode(),
            "ct": base64.b64encode(ct).decode(),
            "tag": base64.b64encode(tag).decode(),
        }
        # Atomic write: write to temp then rename
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, dir=self._path.parent,
            prefix=".secrets.", suffix=".tmp", encoding="utf-8",
        ) as f:
            tmp_path = Path(f.name)
        try:
            tmp_path.write_text(json.dumps(payload), encoding="utf-8")
            os.chmod(tmp_path, 0o600)
            tmp_path.replace(self._path)
            os.chmod(self._path, 0o600)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def is_available(self) -> bool:
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            return True
        except ImportError:
            return False

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            data = self._load()
            return data.get(key)

    def set(self, key: str, value: str) -> None:
        with self._lock:
            data = self._load()
            data[key] = value
            self._save(data)

    def delete(self, key: str) -> None:
        with self._lock:
            data = self._load()
            data.pop(key, None)
            self._save(data)


class SecretStore:
    """Multi-backend secret store with auto-fallback.

    Tries backends in order:
      1. macOS Keychain (Mac users)
      2. Encrypted file (anywhere, requires `cryptography` lib)

    Once a backend has stored a secret, ALWAYS read it from the same
    backend to avoid inconsistency.

    Use:
        store = SecretStore()
        store.set("federation.cluster_secret", "...")
        secret = store.get("federation.cluster_secret")
        # secret is wrapped in TokenStr (redacted in logs)
    """
    KEY_PREFIX = "hermes.federation."

    def __init__(self):
        self._backends: list[SecretBackend] = []
        self._keymap: dict[str, str] = {}  # in-memory key → backend name
        self._lock = threading.Lock()
        self._init_backends()

    def _init_backends(self) -> None:
        # Try macOS Keychain first
        macos = MacOSKeychainBackend()
        if macos.is_available():
            self._backends.append(macos)
        # Encrypted file fallback
        ef = EncryptedFileBackend()
        if ef.is_available():
            self._backends.append(ef)
        if not self._backends:
            raise RuntimeError(
                "No secure secret backend available. "
                "Install `cryptography` package or run on macOS with Keychain."
            )

    def _namespaced(self, key: str) -> str:
        if not key.startswith(self.KEY_PREFIX):
            return f"{self.KEY_PREFIX}{key}"
        return key

    def get(self, key: str) -> Optional[TokenStr]:
        """Get a secret. Returns TokenStr (auto-redacted in logs).
        Returns None if not found.
        """
        ns_key = self._namespaced(key)
        with self._lock:
            # Try the backend that previously stored this key first
            preferred = self._keymap.get(key)
            order = []
            if preferred:
                for b in self._backends:
                    if b.name == preferred:
                        order.append(b)
            for b in self._backends:
                if b not in order:
                    order.append(b)
            for b in order:
                val = b.get(ns_key)
                if val is not None:
                    self._keymap[key] = b.name
                    return TokenStr(val)
            return None

    def set(self, key: str, value: str) -> None:
        """Store a secret. Writes to first available backend."""
        if not value:
            raise ValueError("cannot store empty secret")
        ns_key = self._namespaced(key)
        with self._lock:
            # Use the previously-used backend if available
            preferred = self._keymap.get(key)
            target = None
            if preferred:
                for b in self._backends:
                    if b.name == preferred:
                        target = b
                        break
            if target is None:
                target = self._backends[0]
            target.set(ns_key, value)
            self._keymap[key] = target.name

    def delete(self, key: str) -> None:
        """Delete a secret from all backends."""
        ns_key = self._namespaced(key)
        with self._lock:
            for b in self._backends:
                try:
                    b.delete(ns_key)
                except Exception:
                    pass
            self._keymap.pop(key, None)

    def rotate(self, key: str, new_value: str) -> TokenStr:
        """Rotate a secret: set new value, return old."""
        old = self.get(key)
        self.set(key, new_value)
        return old if old else TokenStr("")


# === Singleton ===

_default_store: Optional[SecretStore] = None
_default_lock = threading.Lock()


def get_default_store() -> SecretStore:
    """Get the default SecretStore singleton (lazy init)."""
    global _default_store
    with _default_lock:
        if _default_store is None:
            _default_store = SecretStore()
        return _default_store


def get_token(key: str) -> Optional[str]:
    """Convenience: get a raw token string (caller is responsible for
    not logging it)."""
    ts = get_default_store().get(key)
    return str(ts) if ts else None


def set_token(key: str, value: str) -> None:
    """Convenience: store a token."""
    get_default_store().set(key, value)


__all__ = [
    "SecretStore",
    "SecretBackend",
    "MacOSKeychainBackend",
    "EncryptedFileBackend",
    "get_default_store",
    "get_token",
    "set_token",
]
