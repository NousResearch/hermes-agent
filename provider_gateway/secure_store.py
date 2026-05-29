"""Secure credential store for the opt-in provider gateway.

Encripts and stores API keys securely across various platforms:
1. Native OS keyring (macOS, Windows, Linux Desktop via python-keyring).
2. WSL Windows Credential Manager Interoperability.
3. Machine-bound local AES-256-GCM fallback encryption.
"""

from __future__ import annotations

import getpass
import hashlib
import logging
import os
import platform
import subprocess
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DynamicCredentialStore:
    """Thread-safe and cross-platform secure credentials manager."""

    def __init__(self, secrets_dir: str | Path | None = None) -> None:
        if secrets_dir is not None:
            self.secrets_dir = Path(secrets_dir)
        else:
            from hermes_constants import get_hermes_home
            self.secrets_dir = get_hermes_home()
        self.secrets_dir.mkdir(parents=True, exist_ok=True)
        self.secrets_file = self.secrets_dir / ".hermes_secrets"

    def is_wsl(self) -> bool:
        """Check if running inside Windows Subsystem for Linux (WSL)."""
        if platform.system().lower() != "linux":
            return False
        # Rationale: check proc version or check if wsl interop is mounted
        try:
            with open("/proc/version", "r") as f:
                content = f.read().lower()
                if "microsoft" in content or "wsl" in content:
                    return True
        except Exception:
            pass
        return os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop")

    def has_powershell_interop(self) -> bool:
        """Verify if powershell.exe is available in WSL PATH."""
        if not self.is_wsl():
            return False
        try:
            res = subprocess.run(
                ["which", "powershell.exe"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return res.returncode == 0
        except Exception:
            return False

    def store_credential(self, provider: str, api_key: str) -> bool:
        """Store API key securely using the best available platform backend."""
        provider = str(provider).strip().lower()
        api_key = str(api_key).strip()

        # 1. Try Native Keyring (Desktop OS Host)
        try:
            import keyring
            keyring.set_password("hermes_agent/provider_gateway", provider, api_key)
            logger.debug("Successfully stored credential for %s via native OS keyring.", provider)
            return True
        except Exception as exc:
            logger.debug("Native keyring unavailable or failed for %s: %s. Trying alternatives.", provider, exc)

        # 2. Try WSL Windows Credential Manager Interoperability
        if self.is_wsl() and self.has_powershell_interop():
            try:
                # WinRT PasswordVault powershell payload
                ps_cmd = (
                    f"[void][Windows.Security.Credentials.PasswordVault, Windows.Security.Credentials, ContentType=WindowsRuntime]; "
                    f"$vault = New-Object Windows.Security.Credentials.PasswordVault; "
                    f"$cred = New-Object Windows.Security.Credentials.PasswordCredential('hermes_agent/provider_gateway', '{provider}', '{api_key}'); "
                    f"$vault.Add($cred);"
                )
                res = subprocess.run(
                    ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps_cmd],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if res.returncode == 0:
                    logger.debug("Successfully stored credential for %s via WSL Powershell Windows Vault.", provider)
                    return True
                else:
                    logger.debug("WSL Powershell Windows Vault write failed: %s", res.stderr)
            except Exception as wsl_exc:
                logger.debug("WSL Powershell Vault store failed: %s", wsl_exc)

        # 3. Fallback: Machine-Bound Local AES-256-GCM Encryption
        try:
            return self._store_local_aes(provider, api_key)
        except Exception as aes_exc:
            logger.error("Failed to store credentials using local fallback encryption: %s", aes_exc)
            return False

    def get_credential(self, provider: str) -> str | None:
        """Retrieve API key securely using the best available platform backend."""
        provider = str(provider).strip().lower()

        # 1. Try Native Keyring
        try:
            import keyring
            val = keyring.get_password("hermes_agent/provider_gateway", provider)
            if val is not None:
                return val
        except Exception as exc:
            logger.debug("Native keyring get failed for %s: %s", provider, exc)

        # 2. Try WSL Windows Credential Manager Interop
        if self.is_wsl() and self.has_powershell_interop():
            try:
                ps_cmd = (
                    f"[void][Windows.Security.Credentials.PasswordVault, Windows.Security.Credentials, ContentType=WindowsRuntime]; "
                    f"$vault = New-Object Windows.Security.Credentials.PasswordVault; "
                    f"try {{"
                    f"  $cred = $vault.Retrieve('hermes_agent/provider_gateway', '{provider}');"
                    f"  $cred.RetrievePassword();"
                    f"  Write-Output $cred.Password;"
                    f"}} catch {{"
                    f"  Write-Error 'Not found';"
                    f"}}"
                )
                res = subprocess.run(
                    ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps_cmd],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if res.returncode == 0 and res.stdout.strip():
                    return res.stdout.strip()
            except Exception as wsl_exc:
                logger.debug("WSL Powershell Vault get failed: %s", wsl_exc)

        # 3. Fallback: Local AES-256-GCM
        try:
            return self._get_local_aes(provider)
        except Exception as aes_exc:
            logger.debug("Local AES fallback decryption failed for %s: %s", provider, aes_exc)
            return None

    def delete_credential(self, provider: str) -> bool:
        """Delete API key from all available backends."""
        provider = str(provider).strip().lower()
        deleted = False

        # 1. Native Keyring
        try:
            import keyring
            keyring.delete_password("hermes_agent/provider_gateway", provider)
            deleted = True
        except Exception:
            pass

        # 2. WSL Powershell
        if self.is_wsl() and self.has_powershell_interop():
            try:
                ps_cmd = (
                    f"[void][Windows.Security.Credentials.PasswordVault, Windows.Security.Credentials, ContentType=WindowsRuntime]; "
                    f"$vault = New-Object Windows.Security.Credentials.PasswordVault; "
                    f"try {{"
                    f"  $cred = $vault.Retrieve('hermes_agent/provider_gateway', '{provider}');"
                    f"  $vault.Remove($cred);"
                    f"}} catch {{}}"
                )
                subprocess.run(
                    ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps_cmd],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                deleted = True
            except Exception:
                pass

        # 3. Local AES
        try:
            secrets = self._read_secrets_file()
            if provider in secrets:
                del secrets[provider]
                self._write_secrets_file(secrets)
                deleted = True
        except Exception:
            pass

        return deleted

    # ── Local AES-256-GCM Helper Methods ─────────────────────────────────

    def _get_machine_fingerprint(self) -> bytes:
        """Derive a stable 32-byte hardware machine fingerprint."""
        components = []

        # 1. Machine ID
        for path in ["/etc/machine-id", "/var/lib/dbus/machine-id"]:
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        components.append(f.read().strip())
                        break
                except Exception:
                    pass
        else:
            components.append(platform.node())  # Fallback to hostname

        # 2. MAC Address
        components.append(str(uuid.getnode()))

        # 3. Username
        components.append(getpass.getuser())

        joined = "|".join(components)
        return hashlib.sha256(joined.encode("utf-8")).digest()

    def _derive_aes_key(self, salt: bytes) -> bytes:
        """Derive a secure 32-byte AES key using PBKDF2HMAC with machine fingerprint."""
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        from cryptography.hazmat.primitives import hashes

        fingerprint = self._get_machine_fingerprint()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100_000,
        )
        return kdf.derive(fingerprint)

    def _read_secrets_file(self) -> dict[str, dict[str, str]]:
        """Read and parse the raw secrets JSON mapping."""
        import json
        if not self.secrets_file.exists():
            return {}
        try:
            with open(self.secrets_file, "r") as f:
                content = f.read().strip()
                return json.loads(content) if content else {}
        except Exception:
            return {}

    def _write_secrets_file(self, secrets: dict[str, dict[str, str]]) -> None:
        """Write the secrets JSON mapping to disk with strict permissions (0600)."""
        import json
        temp_file = self.secrets_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w") as f:
                json.dump(secrets, f)
            
            # Restrict permissions before replacing
            if platform.system().lower() != "windows":
                os.chmod(temp_file, 0o600)
            
            os.replace(temp_file, self.secrets_file)
        finally:
            if temp_file.exists():
                try:
                    os.remove(temp_file)
                except Exception:
                    pass

    def _store_local_aes(self, provider: str, api_key: str) -> bool:
        """Encrypt and store credentials locally using AES-256-GCM."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        import base64

        # Generate fresh salt and nonce
        salt = os.urandom(16)
        nonce = os.urandom(12)

        key = self._derive_aes_key(salt)
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, api_key.encode("utf-8"), None)

        # Base64 encode for JSON safety
        payload = {
            "salt": base64.b64encode(salt).decode("utf-8"),
            "nonce": base64.b64encode(nonce).decode("utf-8"),
            "ciphertext": base64.b64encode(ciphertext).decode("utf-8"),
        }

        secrets = self._read_secrets_file()
        secrets[provider] = payload
        self._write_secrets_file(secrets)
        logger.debug("Successfully stored credential for %s via machine-bound local AES.", provider)
        return True

    def _get_local_aes(self, provider: str) -> str | None:
        """Decrypt and retrieve credentials locally using AES-256-GCM."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        import base64

        secrets = self._read_secrets_file()
        payload = secrets.get(provider)
        if payload is None:
            return None

        try:
            salt = base64.b64decode(payload["salt"])
            nonce = base64.b64decode(payload["nonce"])
            ciphertext = base64.b64decode(payload["ciphertext"])

            key = self._derive_aes_key(salt)
            aesgcm = AESGCM(key)
            decrypted = aesgcm.decrypt(nonce, ciphertext, None)
            return decrypted.decode("utf-8")
        except Exception as exc:
            logger.debug("Local AES decryption failed for %s: %s", provider, exc)
            return None
