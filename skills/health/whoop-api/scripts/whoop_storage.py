"""Token storage for Whoop API credentials.

Handles OAuth token persistence across platforms:
  - macOS: Keychain via `security` CLI
  - Linux/Windows: JSON file fallback

Keychain uses a dynamic path (user's login keychain) to handle
profile sandboxes where HOME resolves to a sandbox directory.

Service name: whoop-api
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path

KEYCHAIN_SERVICE = "whoop-api"
KEYCHAIN_ACCOUNT = "whoop-tokens"
KEYCHAIN_CREDENTIAL_ACCOUNT = "whoop-credentials"

TOKEN_FIELDS = ("access_token", "refresh_token", "expires_at")
CREDENTIAL_FIELDS = ("client_id", "client_secret")


def _is_macos() -> bool:
    return platform.system() == "Darwin"


def _keychain_path() -> str | None:
    """Determine the macOS Keychain path dynamically.

    Uses the user's actual login keychain, not a hardcoded path.
    This works across profile sandboxes and different user accounts.
    """
    if not _is_macos():
        return None

    # Try the user's actual home directory login keychain
    home = Path.home()
    keychain = home / "Library" / "Keychains" / "login.keychain-db"
    if keychain.exists():
        return str(keychain)

    # Fallback: try traditional location
    keychain_legacy = home / "Library" / "Keychains" / "login.keychain"
    if keychain_legacy.exists():
        return str(keychain_legacy)

    # Last resort: use default keychain (security CLI will find it)
    return None


def _keychain_available() -> bool:
    """Check if `security` CLI exists and keychain is accessible."""
    if not _is_macos():
        return False
    try:
        # Just check that `security` CLI exists
        result = subprocess.run(
            ["security", "show-keychain-info"],
            capture_output=True, text=True, timeout=5,
        )
        # Exit code 0 or specific keychain errors mean security CLI works
        return result.returncode in (0, 36)  # 36 = keychain not found but CLI works
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _run_security(args: list[str], interactive_password: str | None = None) -> subprocess.CompletedProcess:
    """Run a `security` CLI command with optional keychain path."""
    cmd = ["security"] + args + ["-s", KEYCHAIN_SERVICE]

    # Add keychain path if available (better reliability in profile sandboxes)
    kc_path = _keychain_path()
    if kc_path:
        cmd.append(kc_path)

    return subprocess.run(cmd, capture_output=True, text=True, timeout=10)


def load_tokens() -> dict[str, str | float] | None:
    """Load stored tokens. Returns None if no tokens exist.

    Tries Keychain first (macOS), then JSON file fallback.
    """
    if _keychain_available():
        return _load_from_keychain(KEYCHAIN_ACCOUNT)
    return _load_from_json_file(_token_file_path())


def save_tokens(access_token: str, refresh_token: str, expires_at: float) -> None:
    """Persist tokens. Keychain on macOS, JSON file elsewhere."""
    data = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": expires_at,
    }
    if _is_macos() and _keychain_available():
        _save_to_keychain(KEYCHAIN_ACCOUNT, data)
    else:
        _save_to_json_file(_token_file_path(), data)


def clear_tokens() -> None:
    """Remove stored tokens from Keychain or JSON file."""
    if _is_macos() and _keychain_available():
        _delete_from_keychain(KEYCHAIN_ACCOUNT)
    else:
        path = _token_file_path()
        if path.exists():
            path.unlink()


def load_client_credentials() -> dict[str, str] | None:
    """Load client_id/client_secret from Keychain (macOS) or JSON file."""
    if _keychain_available():
        return _load_from_keychain(KEYCHAIN_CREDENTIAL_ACCOUNT)
    return _load_from_json_file(_credentials_file_path(), CREDENTIAL_FIELDS)


def save_client_credentials(client_id: str, client_secret: str) -> None:
    """Persist client_id/client_secret. Keychain on macOS, JSON file elsewhere."""
    data = {"client_id": client_id, "client_secret": client_secret}
    if _is_macos() and _keychain_available():
        _save_to_keychain(KEYCHAIN_CREDENTIAL_ACCOUNT, data)
    else:
        _save_to_json_file(_credentials_file_path(), data)


# -- Keychain (macOS) --

def _load_from_keychain(account: str) -> dict | None:
    """Read a JSON blob from macOS Keychain."""
    try:
        result = _run_security(["find-generic-password", "-a", account, "-w"])
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout.strip())
        return data
    except (json.JSONDecodeError, subprocess.TimeoutExpired):
        return None


def _save_to_keychain(account: str, data: dict) -> None:
    """Delete old entry (if any) then add new one to Keychain."""
    _delete_from_keychain(account)
    password = json.dumps(data)
    cmd = ["security", "add-generic-password",
           "-s", KEYCHAIN_SERVICE, "-a", account,
           "-w", password]
    kc_path = _keychain_path()
    if kc_path:
        cmd.append(kc_path)
    subprocess.run(cmd, check=True, timeout=10)


def _delete_from_keychain(account: str) -> None:
    """Delete Keychain entry. Ignore errors if it doesn't exist."""
    cmd = ["security", "delete-generic-password",
           "-s", KEYCHAIN_SERVICE, "-a", account]
    kc_path = _keychain_path()
    if kc_path:
        cmd.append(kc_path)
    subprocess.run(cmd, capture_output=True, timeout=10)


# -- JSON file fallback (Linux/Windows) --

def _config_dir() -> Path:
    """Return the config directory for Whoop API files."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "whoop-api"


def _token_file_path() -> Path:
    """Return path to the Whoop tokens JSON file."""
    return _config_dir() / "tokens.json"


def _credentials_file_path() -> Path:
    """Return path to the Whoop credentials JSON file."""
    return _config_dir() / "credentials.json"


def _load_from_json_file(path: Path, required_fields: tuple = TOKEN_FIELDS) -> dict | None:
    """Read JSON from file."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if all(k in data for k in required_fields):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _save_to_json_file(path: Path, data: dict) -> None:
    """Write JSON to file with restricted permissions.

    Uses atomic write (temp + rename) to prevent partial reads
    during concurrent access (e.g. cron refresh vs agent pull).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, indent=2))
    if platform.system() != "Windows":
        tmp_path.chmod(0o600)
    os.replace(tmp_path, path)