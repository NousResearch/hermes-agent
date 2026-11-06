"""Request-bound, opaque credential-scope identity for plugin hooks.

A provider may hold several credentials (credential pools). Downstream plugins
(quota / health observers) need to attribute a request outcome to the credential
that actually served that attempt -- without ever seeing credential material.

The scope id is a host-owned, random, per-pool-entry identifier
(:attr:`agent.credential_pool.PooledCredential.scope_id`). It is:

- generated independently of credential material (never a hash of a secret),
- unusable for authentication and non-reversible,
- stable for one pool entry across ordinary requests and token refreshes
  (refreshes replace the entry's tokens in place, keeping its id/scope),
- distinct for distinct pool entries, and
- replaced together with the entry (removing and re-adding a credential mints
  a fresh scope id).

Resolution is fail-closed: when no pool entry can be attributed to the attempt
(single env/config key, keyless local provider, unsupported backend) the status
is ``unknown`` / ``unavailable`` with a ``None`` id -- an id is never invented.

Attribution is attempt-level: :func:`capture_attempt_credential_scope` snapshots
the scope when an attempt's request is built (before ``pre_api_request``), and
``post_api_request`` / ``api_request_error`` report that snapshot, so a retry
onto another credential is never attributed to the previous attempt's scope.
The snapshot lives on the agent instance (never a global), so concurrent
agents cannot exchange scopes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

CREDENTIAL_SCOPE_CAPABILITY = "llm.credential_scope"

IDENTITY_KNOWN = "known"
IDENTITY_UNKNOWN = "unknown"
IDENTITY_UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class RequestCredentialScope:
    """Opaque identity of the credential selected for one request attempt."""

    provider_id: str
    credential_scope_id: Optional[str]
    identity_status: str  # "known" | "unknown" | "unavailable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "credential_scope_id": self.credential_scope_id,
            "identity_status": self.identity_status,
        }


def _scope_id_for_entry_id(agent: Any, entry_id: str) -> Optional[str]:
    pool = getattr(agent, "_credential_pool", None)
    if pool is None:
        return None
    try:
        entries = pool.entries()
    except Exception:
        return None
    for entry in entries or ():
        if getattr(entry, "id", None) == entry_id:
            scope_id = getattr(entry, "scope_id", None)
            return scope_id if isinstance(scope_id, str) and scope_id else None
    return None


def resolve_credential_scope(agent: Any) -> RequestCredentialScope:
    """The scope of the credential ``agent`` would currently dispatch with.

    Never raises and never touches credential material: only the pool entry's
    host-owned ``scope_id`` is read.
    """
    provider_id = str(getattr(agent, "provider", "") or "")
    if not provider_id:
        return RequestCredentialScope("", None, IDENTITY_UNAVAILABLE)
    entry_id = getattr(agent, "_credential_pool_entry_id", None)
    if isinstance(entry_id, str) and entry_id:
        scope_id = _scope_id_for_entry_id(agent, entry_id)
        if scope_id:
            return RequestCredentialScope(provider_id, scope_id, IDENTITY_KNOWN)
        return RequestCredentialScope(provider_id, None, IDENTITY_UNKNOWN)
    # No pool entry bound: a keyed request is a real credential we cannot
    # attribute; a keyless request (local / no-auth provider) has no scope.
    if getattr(agent, "api_key", None):
        return RequestCredentialScope(provider_id, None, IDENTITY_UNKNOWN)
    return RequestCredentialScope(provider_id, None, IDENTITY_UNAVAILABLE)


def capture_attempt_credential_scope(agent: Any, api_request_id: Any) -> RequestCredentialScope:
    """Snapshot the attempt's scope at request-build time; return the snapshot.

    ``post_api_request`` / ``api_request_error`` read the snapshot (via
    :func:`attempt_credential_scope`) so rotation after a failed attempt cannot
    re-attribute that attempt's outcome.
    """
    scope = resolve_credential_scope(agent)
    try:
        agent._attempt_credential_scope = (api_request_id, scope)
    except Exception:
        pass
    return scope


def attempt_credential_scope(agent: Any, api_request_id: Any = None) -> RequestCredentialScope:
    """The captured scope for this attempt, falling back to a live resolution."""
    captured = getattr(agent, "_attempt_credential_scope", None)
    if isinstance(captured, tuple) and len(captured) == 2:
        captured_id, scope = captured
        if api_request_id is None or captured_id == api_request_id:
            if isinstance(scope, RequestCredentialScope):
                return scope
    return resolve_credential_scope(agent)
