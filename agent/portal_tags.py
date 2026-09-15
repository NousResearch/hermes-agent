"""Centralized Nous Portal request tags.

Every Hermes request to the Nous Portal (main loop, auxiliary client, fallback
paths) must carry the same product-attribution tags, sent in OpenAI-compatible
``extra_body['tags']``: ``["product=hermes-agent", "client=hermes-client-v<__version__>"]``.
The version is read live from ``hermes_cli.__version__`` — do NOT pre-compute it
as a module constant in consumers; it can change at runtime (editable installs,
hot reload).
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import List, Optional

# Ambient conversation id (ATTRIBUTION value, sent as ``conversation=<id>``).
# The agent loop publishes it at turn entry; auxiliary call sites funnelling
# through ``auxiliary_client.call_llm`` (no session handle) pick it up via
# ``nous_portal_tags()``. A ContextVar so concurrent agents in one process never
# see each other's id; ``propagate_context_to_thread`` workers inherit it.
_conversation_id: ContextVar[Optional[str]] = ContextVar("nous_portal_conversation_id", default=None)

# Ambient affinity scope (ROUTING value): OpenRouter's sticky ``session_id``, Nous
# Portal's sticky key and xAI's ``x-grok-conv-id`` pin a conversation to one
# backend/prompt cache. Usually equal to the conversation id, but a host that mints
# one physical session per RESPONSE must route on the key it declared for the whole
# chat (``prompt_cache_scope.declared_conversation_scope``). Only that declared value
# is published; unset means consumers fall back to the conversation id, so delegate
# trees keep sharing their parent's sticky key.
_affinity_scope: ContextVar[Optional[str]] = ContextVar("hermes_affinity_scope", default=None)


def _reset_var(var: ContextVar, token) -> None:
    """Reset ``var``; a token from another Context (reset on a different thread)
    falls back to clearing rather than raising in cleanup paths."""
    try:
        var.reset(token)
    except Exception:
        var.set(None)


def set_affinity_scope(scope: Optional[str]):
    """Publish the declared routing/affinity scope; returns the ContextVar token."""
    return _affinity_scope.set(scope or None)


def reset_affinity_scope(token) -> None:
    """Restore the previous affinity scope (pair with ``set_affinity_scope``)."""
    _reset_var(_affinity_scope, token)


def get_affinity_scope() -> Optional[str]:
    return _affinity_scope.get()


def set_conversation_context(conversation_id: Optional[str]):
    """Publish the active conversation id for ambient Portal tagging; returns the token.

    Called by the agent loop at turn entry with the session-lineage ROOT id (so
    the tag survives context-compression rotation). ``None`` clears.
    """
    return _conversation_id.set(conversation_id or None)


def reset_conversation_context(token) -> None:
    """Restore the previous conversation context (pair with ``set_...``)."""
    _reset_var(_conversation_id, token)


def get_conversation_context() -> Optional[str]:
    return _conversation_id.get()


def hermes_client_tag() -> str:
    """``client=hermes-client-v<MAJOR>.<MINOR>.<PATCH>`` ("unknown" if hermes_cli is unimportable)."""
    try:
        from hermes_cli import __version__
    except Exception:
        __version__ = "unknown"
    return f"client=hermes-client-v{__version__}"


def conversation_tag(session_id: str) -> str:
    """``conversation=<session_id>`` — high-cardinality, so only appended when a
    session id is actually available, never in the always-on base set."""
    return f"conversation={session_id}"


def nous_portal_tags(session_id: str | None = None) -> List[str]:
    """Fresh list of the canonical Nous Portal tags.

    The ambient conversation context (lineage ROOT id) wins over the explicit
    ``session_id``, a fallback for callers outside any agent turn.
    """
    tags = ["product=hermes-agent", hermes_client_tag()]
    effective = get_conversation_context() or session_id
    if effective:
        tags.append(conversation_tag(effective))
    return tags
