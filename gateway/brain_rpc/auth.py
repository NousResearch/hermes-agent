"""Auth verification for brain RPC (contract §3) — host layers L1/L3/L4/L5.

L0 (channel secret) is enforced at the relay WebSocket upgrade. L2 (portal
session) is BFF-side. Hermes re-checks stamped subject claims fail-closed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from gateway.brain_rpc.config import BrainRpcHostConfig
from gateway.brain_rpc.errors import (
    FORBIDDEN,
    UNAUTHENTICATED,
    BrainRpcError,
)

logger = logging.getLogger(__name__)

# Built-in capability map when host profile seed files are missing.
# Mirrors company-brain-deploy scripts/hermes-profiles/*.json — fail-closed
# if the requested profile is unknown.
_BUILTIN_PROFILE_CAPS: Dict[str, Set[str]] = {
    "admin": {
        "vault_full",
        "vault_read",
        "vault_write",
        "chat",
        "profile_manage",
        "system_ops",
        "shell_exec",
        "gateway_config",
        "beads_admin",
    },
    "contributor": {
        "vault_read",
        "vault_write",
        "chat",
    },
}

# Methods that do not require vault path ACL (still need auth).
NO_PATH_ACL_METHODS = frozenset(
    {
        "brain.ping",
        "brain.health",
        "settings.snapshot",
        "projects.list",
    }
)

# Method → required capabilities (any-of). Empty set = authenticated only.
METHOD_CAPABILITIES: Dict[str, Set[str]] = {
    "brain.ping": set(),
    "brain.health": set(),
    "vault.list": {"vault_read", "vault_full"},
    "vault.stat": {"vault_read", "vault_full"},
    "vault.read": {"vault_read", "vault_full"},
    "settings.snapshot": set(),  # both admin and contributor; payload redacted
    "projects.list": {"vault_read", "vault_full"},
}


@dataclass
class Subject:
    portal_user_id: str
    hermes_profile: str
    roles: List[str] = field(default_factory=list)
    path_prefixes: List[str] = field(default_factory=list)


@dataclass
class AuthContext:
    tenant_id: str
    instance_id: str
    subject: Subject
    session_id: Optional[str] = None
    issued_at: Optional[str] = None
    expires_at: Optional[str] = None
    capabilities: Set[str] = field(default_factory=set)


def _parse_iso8601(value: str) -> datetime:
    """Parse an ISO-8601 timestamp; accept trailing Z."""
    raw = (value or "").strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def load_profile_capabilities(profiles_dir: Path, profile: str) -> Set[str]:
    """Load capabilities for a host profile (seed JSON or builtin fallback)."""
    name = (profile or "").strip()
    if not name:
        return set()
    seed = profiles_dir / f"{name}.json"
    if seed.is_file():
        try:
            data = json.loads(seed.read_text(encoding="utf-8"))
            caps = data.get("capabilities") or []
            if isinstance(caps, list):
                return {str(c).strip() for c in caps if str(c).strip()}
        except Exception as exc:  # noqa: BLE001 - corrupt seed → treat as unknown
            logger.warning("brain_rpc: failed to load profile seed %s: %s", seed, exc)
            return set()
    return set(_BUILTIN_PROFILE_CAPS.get(name, set()))


def verify_auth(
    auth: Any,
    *,
    method: str,
    host: BrainRpcHostConfig,
    now: Optional[datetime] = None,
) -> AuthContext:
    """Validate the request ``auth`` object. Raises BrainRpcError on failure."""
    if not isinstance(auth, dict) or not auth:
        raise BrainRpcError(UNAUTHENTICATED, "missing auth")

    tenant_id = str(auth.get("tenant_id") or "").strip()
    instance_id = str(auth.get("instance_id") or "").strip()
    if not tenant_id or not instance_id:
        raise BrainRpcError(UNAUTHENTICATED, "missing tenant_id or instance_id")

    # L1 — local pins when configured
    if host.instance_id and instance_id != host.instance_id:
        raise BrainRpcError(FORBIDDEN, "instance binding mismatch")
    if host.tenant_id and tenant_id != host.tenant_id:
        raise BrainRpcError(FORBIDDEN, "tenant binding mismatch")

    expires_at = auth.get("expires_at")
    if not expires_at:
        raise BrainRpcError(UNAUTHENTICATED, "missing expires_at")
    try:
        exp = _parse_iso8601(str(expires_at))
    except Exception as exc:
        raise BrainRpcError(UNAUTHENTICATED, "invalid expires_at") from exc
    current = now or datetime.now(timezone.utc)
    if exp <= current:
        raise BrainRpcError(UNAUTHENTICATED, "auth expired")

    subject_raw = auth.get("subject")
    if not isinstance(subject_raw, dict):
        raise BrainRpcError(UNAUTHENTICATED, "missing subject")

    portal_user_id = str(subject_raw.get("portal_user_id") or "").strip()
    hermes_profile = str(subject_raw.get("hermes_profile") or "").strip()
    if not portal_user_id or not hermes_profile:
        raise BrainRpcError(UNAUTHENTICATED, "incomplete subject")

    roles_raw = subject_raw.get("roles") or []
    roles = [str(r) for r in roles_raw] if isinstance(roles_raw, list) else []

    prefixes_raw = subject_raw.get("path_prefixes") or []
    if not isinstance(prefixes_raw, list):
        prefixes_raw = []
    path_prefixes = [_normalize_prefix(p) for p in prefixes_raw if str(p).strip()]

    # L4 — host profile matrix
    capabilities = load_profile_capabilities(host.profiles_dir, hermes_profile)
    if not capabilities and hermes_profile not in _BUILTIN_PROFILE_CAPS:
        # Unknown profile with no seed file — fail closed.
        raise BrainRpcError(FORBIDDEN, "unknown hermes_profile")

    # Empty capability set from a known profile name that failed to load is
    # still fail-closed for methods that require caps; brain.ping is allowed.

    required = METHOD_CAPABILITIES.get(method)
    if required is None:
        # Unknown method handled later; auth still succeeds so dispatcher can
        # return method_not_found (opacity vs auth).
        pass
    elif required:
        if not (capabilities & required):
            raise BrainRpcError(
                FORBIDDEN,
                "profile lacks capability",
                details={"profile": hermes_profile, "method": method},
            )

    # Admin with empty path_prefixes from seed may use full vault via vault_full.
    # Contributor must have path_prefixes for vault path methods (enforced at path check).
    subject = Subject(
        portal_user_id=portal_user_id,
        hermes_profile=hermes_profile,
        roles=roles,
        path_prefixes=path_prefixes,
    )
    return AuthContext(
        tenant_id=tenant_id,
        instance_id=instance_id,
        subject=subject,
        session_id=str(auth.get("session_id") or "") or None,
        issued_at=str(auth.get("issued_at") or "") or None,
        expires_at=str(expires_at),
        capabilities=capabilities,
    )


def _normalize_prefix(prefix: Any) -> str:
    p = str(prefix).strip().replace("\\", "/")
    if not p.startswith("/"):
        p = "/" + p
    # Collapse duplicate slashes; strip trailing slash except root
    while "//" in p:
        p = p.replace("//", "/")
    if len(p) > 1 and p.endswith("/"):
        p = p.rstrip("/")
    return p


def path_allowed(vault_path: str, auth: AuthContext) -> bool:
    """L5 path ACL: vault-root-relative path must fall under path_prefixes.

    ``vault_full`` capability with empty prefixes is treated as full vault
    access (admin seed). Contributor with empty prefixes is denied.
    """
    if "vault_full" in auth.capabilities and not auth.subject.path_prefixes:
        return True
    if not auth.subject.path_prefixes:
        return False
    path = vault_path if vault_path.startswith("/") else f"/{vault_path}"
    # Normalize
    while "//" in path:
        path = path.replace("//", "/")
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    for prefix in auth.subject.path_prefixes:
        if prefix == "/":
            return True
        if path == prefix or path.startswith(prefix + "/"):
            return True
    return False
