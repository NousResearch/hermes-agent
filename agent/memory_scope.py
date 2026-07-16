"""Memory scope resolver — produces stable hashes from gateway context.

Used by both the built-in MemoryStore (for file paths) and external
memory providers (for container/namespace routing).

The scope key is a short SHA256 hash (16 hex chars) of a stable identifier
derived from gateway context. Raw identifiers are not exposed in paths or
container tags; the resulting value is pseudonymous, not anonymous.

Scope modes
-----------
``identity`` (default)
    No suffix. All conversations sharing the same Hermes profile read and
    write the same memory. This is the pre-scoping behaviour.

``user``
    One memory namespace per platform user. A user's DMs on different
    chats share memory, but different users are isolated.

``conversation``
    One memory namespace per DM, group, channel, or thread. This is stable
    across ``/new`` resets because ``gateway_session_key`` is derived from
    chat identity, not session ID.

``session``
    One memory namespace per Hermes session. Changes on every ``/new``.
    Useful for ephemeral / privacy-sensitive workflows.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

VALID_SCOPES = ("identity", "user", "conversation", "session")
DEFAULT_SCOPE = "identity"
_HASH_DOMAIN = "hermes-memory:v1"


def scope_hash(value: str) -> str:
    """Create a short stable hash suitable for paths and container tags.

    Uses the first 16 hex characters of SHA256 — sufficient collision
    resistance for namespace isolation while keeping paths/tags short.
    This is pseudonymization, not protection against offline enumeration of
    low-entropy identifiers.
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _scoped_hash(scope: str, value: str) -> str:
    """Hash a versioned, domain-separated namespace identifier."""
    return scope_hash(f"{_HASH_DOMAIN}:{scope}:{value}")


def resolve_scope_key(
    scope: str,
    *,
    agent_identity: str = "default",
    platform: str = "",
    user_id: str = "",
    user_id_alt: str = "",
    chat_id: str = "",
    chat_type: str = "",
    thread_id: str = "",
    gateway_session_key: str = "",
    session_id: str = "",
) -> Optional[str]:
    """Resolve a stable scope key from gateway context.

    Returns ``None`` for ``identity`` scope (or on fallback), signalling
    the caller to use the raw identity without a suffix.

    For other scopes, returns a 16-character hex hash. The fallback chain
    ensures we always produce a valid key — worst case degrades to
    ``identity`` with a debug log.
    """
    if scope not in VALID_SCOPES:
        logger.warning("Unknown memory scope '%s'; using identity", scope)
        return None

    if scope == "identity":
        return None

    if scope == "user":
        # Per platform user: stable across different chats from the same user
        raw = user_id_alt or user_id
        if raw:
            return _scoped_hash("user", f"{platform}:{raw}")
        logger.debug(
            "memory scope 'user' has no user_id/user_id_alt; falling back to identity"
        )
        return None

    if scope == "conversation":
        # Per DM/group/channel/thread: stable across /new within the same chat.
        # gateway_session_key is the canonical routing boundary across all
        # gateway platforms — it encodes platform + chat_type + chat_id + thread_id.
        if gateway_session_key:
            return _scoped_hash("conversation:gateway", gateway_session_key)
        # Fallback: reconstruct from available components
        if chat_id:
            components = f"{platform}:{chat_type or 'unknown'}:{chat_id}"
            if thread_id:
                components += f":{thread_id}"
            return _scoped_hash("conversation:components", components)
        # CLI/Desktop fallback: use session_id as the conversation boundary.
        # A resumed session keeps the same id; /new creates a new conversation.
        if session_id:
            return _scoped_hash("conversation:local", session_id)
        logger.debug(
            "memory scope 'conversation' has no gateway_session_key, chat_id, "
            "or session_id; falling back to identity"
        )
        return None

    if scope == "session":
        # Per Hermes session: changes on every /new
        if session_id:
            return _scoped_hash("session", session_id)
        logger.debug(
            "memory scope 'session' has no session_id; falling back to identity"
        )
        return None

    return None


def resolve_scope_suffix(
    scope: str, scope_key: Optional[str], identity: str
) -> str:
    """Produce the string suffix for container tags and directory names.

    For ``identity`` scope (or any ``None`` scope_key): returns the raw
    identity (no hash appended).

    For other scopes with a valid scope_key: returns
    ``'<identity>_<hash>'`` — e.g. ``'default_a4c981e7f2b3d5c9'``.
    """
    if scope_key is None:
        return identity
    return f"{identity}_{scope_key}"


def resolve_active_scope_key() -> Optional[str]:
    """Resolve the current raw scope key from task-local session context.

    Secondary built-in-memory paths use the raw hash because each profile
    already has a separate ``HERMES_HOME``. Provider namespace prefixes are
    handled independently by provider initialization.
    """
    # Read scope from config
    scope = DEFAULT_SCOPE
    try:
        from hermes_cli.config import load_config
        mem_cfg = (load_config() or {}).get("memory", {}) or {}
        scope = str(mem_cfg.get("scope", DEFAULT_SCOPE)).strip().lower()
    except Exception:
        pass

    if scope == "identity":
        return None

    # Read session context (ContextVar-backed, concurrency-safe)
    try:
        from gateway.session_context import get_session_env
        gateway_session_key = get_session_env("HERMES_SESSION_KEY")
        session_id = get_session_env("HERMES_SESSION_ID")
        platform = get_session_env("HERMES_SESSION_PLATFORM")
        user_id = get_session_env("HERMES_SESSION_USER_ID")
        user_id_alt = get_session_env("HERMES_SESSION_USER_ID_ALT")
    except Exception:
        return None

    scope_key = resolve_scope_key(
        scope,
        platform=platform,
        user_id=user_id,
        user_id_alt=user_id_alt,
        gateway_session_key=gateway_session_key,
        session_id=session_id,
    )
    return scope_key


def resolve_active_scope_suffix(
    identity: Optional[str] = None,
) -> Optional[str]:
    """Compatibility helper returning ``<profile>_<hash>`` when scoped."""
    scope_key = resolve_active_scope_key()
    if scope_key is None:
        return None
    if identity is None:
        try:
            from hermes_cli.profiles import get_active_profile_name
            identity = get_active_profile_name() or "default"
        except Exception:
            identity = "default"
    return resolve_scope_suffix("scoped", scope_key, identity)
