"""Runtime config for host-side brain RPC.

All behavioral knobs prefer env vars that company-brain-deploy / operator
wiring already own. Missing optional binds fail closed only when the request
claims a value we *can* check (see auth.py).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Default host cap for vault.read content (contract §4.5).
DEFAULT_READ_MAX_BYTES = 1 * 1024 * 1024  # 1 MiB
HARD_READ_MAX_BYTES = 4 * 1024 * 1024  # 4 MiB
DEFAULT_LIST_LIMIT = 200
HARD_LIST_LIMIT = 500
MAX_IN_FLIGHT = 8
MAX_TIMEOUT_MS = 30_000
DEFAULT_TIMEOUT_MS = 10_000


def is_brain_rpc_enabled() -> bool:
    """Feature gate for Lanyard brain RPC.

    Default ON so an authenticated relay session can serve MVP methods.
    Set ``BRAIN_RPC_ENABLED=0`` (false/no/off) to refuse all requests with
    ``unavailable`` without loading vault handlers — useful for hosts that
    dial the relay for chat only.
    """
    raw = os.environ.get("BRAIN_RPC_ENABLED", "1").strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


def resolve_vault_root() -> Path:
    """Vault root on the customer host.

    Precedence:
      1. ``VAULT_ROOT``
      2. ``COMPANY_BRAIN_VAULT_ROOT``
      3. ``$HERMES_HOME/vault`` (dev/test fallback; production sets VAULT_ROOT)
    """
    for key in ("VAULT_ROOT", "COMPANY_BRAIN_VAULT_ROOT"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
    from hermes_constants import get_hermes_home

    return (get_hermes_home() / "vault").resolve()


def resolve_instance_id() -> Optional[str]:
    """Local instance id this host is bound to (for L1 re-check)."""
    for key in ("GATEWAY_RELAY_INSTANCE_ID", "GATEWAY_RELAY_ID", "BRAIN_INSTANCE_ID"):
        val = os.environ.get(key, "").strip()
        if val:
            return val
    try:
        from gateway.relay import relay_instance_id

        return relay_instance_id()
    except Exception:  # noqa: BLE001
        return None


def resolve_tenant_id() -> Optional[str]:
    """Optional local tenant pin for L1 re-check.

    Deploy may stamp ``BRAIN_TENANT_ID`` or ``GATEWAY_RELAY_TENANT_ID``. When
    absent, host still requires ``auth.tenant_id`` present but cannot verify
    the value against a local pin (relay L1 remains authoritative).
    """
    for key in ("BRAIN_TENANT_ID", "GATEWAY_RELAY_TENANT_ID"):
        val = os.environ.get(key, "").strip()
        if val:
            return val
    return None


def resolve_profiles_dir() -> Path:
    """Directory holding host profile seed JSON files (admin.json, …)."""
    raw = os.environ.get("BRAIN_PROFILES_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "profiles"


@dataclass(frozen=True)
class BrainRpcHostConfig:
    vault_root: Path
    instance_id: Optional[str]
    tenant_id: Optional[str]
    profiles_dir: Path
    read_max_bytes: int = DEFAULT_READ_MAX_BYTES
    hard_read_max_bytes: int = HARD_READ_MAX_BYTES
    list_limit_default: int = DEFAULT_LIST_LIMIT
    list_limit_max: int = HARD_LIST_LIMIT
    max_in_flight: int = MAX_IN_FLIGHT

    @classmethod
    def from_env(cls) -> "BrainRpcHostConfig":
        return cls(
            vault_root=resolve_vault_root(),
            instance_id=resolve_instance_id(),
            tenant_id=resolve_tenant_id(),
            profiles_dir=resolve_profiles_dir(),
        )
