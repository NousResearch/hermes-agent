"""Ambient conversation / affinity context for outgoing LLM requests.

ContextVars published by the agent loop so auxiliary call sites can pick up the active
conversation id and the routing affinity scope without a session handle."""

from __future__ import annotations

from contextvars import ContextVar
from typing import List, Optional

# Ambient conversation id .
# The agent loop publishes it at turn entry; auxiliary call sites funnelling
# through ``auxiliary_client.call_llm`` (no session handle) pick it up via
# ``get_conversation_context()``. A ContextVar so concurrent agents in one process never
# see each other's id; ``propagate_context_to_thread`` workers inherit it.
_conversation_id: ContextVar[Optional[str]] = ContextVar("rabbit_conversation_id", default=None)

# Ambient affinity scope (ROUTING value): OpenRouter's sticky ``session_id``, and xAI's ``x-grok-conv-id`` pin a conversation to one
# backend/prompt cache. Usually equal to the conversation id, but a host that mints
# one physical session per RESPONSE must route on the key it declared for the whole
# chat (``prompt_cache_scope.declared_conversation_scope``). Only that declared value
# is published; unset means consumers fall back to the conversation id, so delegate
# trees keep sharing their parent's sticky key.
_affinity_scope: ContextVar[Optional[str]] = ContextVar("rabbit_affinity_scope", default=None)


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
    """Publish the active conversation id for ambient request context; returns the token.

    Called by the agent loop at turn entry with the session-lineage ROOT id (so
    the tag survives context-compression rotation). ``None`` clears.
    """
    return _conversation_id.set(conversation_id or None)


def reset_conversation_context(token) -> None:
    """Restore the previous conversation context (pair with ``set_...``)."""
    _reset_var(_conversation_id, token)


def get_conversation_context() -> Optional[str]:
    return _conversation_id.get()
