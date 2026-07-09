"""hermes_cli.secrets — Windows Credential Manager-backed secret access.

Migrates plaintext tokens out of ~/.hermes/.env into the Windows Credential
Manager (encrypted at rest via DPAPI). Apps can call `secrets.get(KEY)` and
the secret comes from:
  1. Windows Credential Manager (preferred)
  2. os.environ (fallback)
  3. ~/.hermes/.env (fallback)

The .env file is kept as a fallback for one release cycle. After the cycle,
apps can be migrated to read exclusively from keyring by removing the .env
fallback.

Why: plaintext tokens in .env are readable by anything running as bbasketballer75.
Windows Credential Manager encrypts them via DPAPI (per-user, per-machine).
Even if a subprocess reads env vars, it won't see the secret unless it knows
to call keyring.

Usage:
    from hermes_cli.secrets import get, set, migrate_from_env, source

    # Read a secret (returns value + source for logging)
    value, src = get("GITHUB_BUSINESS_TOKEN")
    if value:
        ... use value ...

    # Write a secret
    set("GITHUB_BUSINESS_TOKEN", "ghp_xxxx")

    # Migrate all current .env values into keyring
    migrate_from_env()  # copies all keys, leaves .env intact

Service name is "hermes" so all hermes secrets live under one credential
namespace in Windows Credential Manager.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

SERVICE_NAME = "hermes"
HERMES_HOME = Path(os.environ.get("HERMES_HOME") or Path.home() / "AppData/Local/hermes")
ENV_FILE = HERMES_HOME / ".env"


# --- low-level keyring ops ---

def _keyring_available() -> bool:
    try:
        import keyring
        # Try a no-op set/get round-trip
        return keyring.get_keyring() is not None
    except Exception as e:
        logger.warning(f"keyring unavailable: {e}")
        return False


def set(key: str, value: str, *, service: str = SERVICE_NAME) -> bool:
    """Store a secret in Windows Credential Manager.

    Returns True on success, False on failure (e.g. headless CI without DPAPI).
    """
    if not _keyring_available():
        return False
    try:
        import keyring
        keyring.set_password(service, key, value)
        logger.info(f"secrets.set: stored {service}/{key} in keyring")
        return True
    except Exception as e:
        logger.warning(f"secrets.set: failed to store {service}/{key}: {e}")
        return False


def get(key: str, *, service: str = SERVICE_NAME) -> Tuple[Optional[str], str]:
    """Retrieve a secret.

    Returns (value, source) where source is one of:
      - "keyring" — Windows Credential Manager
      - "env" — os.environ
      - "env_file" — ~/.hermes/.env
      - "" — not found

    The tuple form lets apps log which source served the secret for auditability.
    """
    # 1. Keyring (preferred)
    if _keyring_available():
        try:
            import keyring
            v = keyring.get_password(service, key)
            if v:
                return v, "keyring"
        except Exception as e:
            logger.debug(f"keyring get failed: {e}")

    # 2. Environment variable
    v = os.environ.get(key)
    if v:
        return v, "env"

    # 3. .env file (last-resort fallback)
    v = _read_env_value(key)
    if v:
        return v, "env_file"

    return None, ""


def delete(key: str, *, service: str = SERVICE_NAME) -> bool:
    """Remove a secret from keyring. Returns True on success."""
    if not _keyring_available():
        return False
    try:
        import keyring
        keyring.delete_password(service, key)
        logger.info(f"secrets.delete: removed {service}/{key} from keyring")
        return True
    except Exception as e:
        logger.warning(f"secrets.delete: failed to remove {service}/{key}: {e}")
        return False


def migrate_from_env(path: Optional[Path] = None) -> dict[str, str]:
    """One-shot: read all keys from .env, store them in keyring.

    Leaves the .env file intact (fallback for one cycle). Returns a dict
    of {key: status} where status is one of "stored", "skipped_empty",
    "skipped_keyring_unavailable", "skipped_failed".

    After migration, .env values are still readable via get() (the env_file
    fallback) so apps don't break. The migration just gets the secrets
    into encrypted-at-rest storage for future use.
    """
    path = path or ENV_FILE
    if not path.exists():
        logger.info(f"secrets.migrate_from_env: {path} does not exist, nothing to migrate")
        return {}

    if not _keyring_available():
        logger.warning("secrets.migrate_from_env: keyring unavailable, nothing migrated")
        return {"*": "skipped_keyring_unavailable"}

    results: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if not k or not v:
            results[k] = "skipped_empty"
            continue
        # Use module-level set() (Windows Credential Manager) — not Python builtin.
        # Module-level `set` shadows the builtin within this module.
        if set(key=k, value=v):  # type: ignore[call-arg]
            results[k] = "stored"
        else:
            results[k] = "skipped_failed"

    logger.info(f"secrets.migrate_from_env: migrated {sum(1 for s in results.values() if s == 'stored')}/{len(results)} keys")
    return results


# --- .env reading (fallback) ---

def _read_env_value(key: str, path: Optional[Path] = None) -> Optional[str]:
    path = path or ENV_FILE
    if not path.exists():
        return None
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == key:
                return v.strip().strip('"').strip("'")
    except Exception:
        pass
    return None