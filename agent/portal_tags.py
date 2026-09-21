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


# Fixed descriptions identify the operation without copying prompts, tool output,
# or arbitrary task names supplied by plugins into request metadata.
_AUXILIARY_PURPOSES = {
    "compression": "Summarize conversation context so the assistant can continue.",
    "title_generation": "Generate a title for the conversation.",
    "vision": "Interpret an image for the current assistant task.",
    "skills_hub": "Select relevant skills for the assistant task.",
    "approval": "Evaluate whether a requested tool action needs approval.",
    "mcp": "Complete a model sampling request from a connected tool.",
    "memory_query_rewrite": "Rewrite a query to retrieve relevant memories.",
    "tts_audio_tags": "Prepare speech delivery annotations.",
    "triage_specifier": "Expand a task description into a specification.",
    "kanban_decomposer": "Break a task into actionable subtasks.",
    "profile_describer": "Generate a short profile description.",
    "goal_judge": "Define or assess the completion criteria for a goal.",
    "curator": "Review skills and identify useful improvements.",
    "monitor": "Assess the relevance of a monitored item.",
    "background_review": "Review the conversation for memory and skill improvements.",
    "moa_reference": "Produce a candidate response for the current assistant task.",
    "moa_aggregator": "Synthesize candidate responses for the current assistant task.",
}


def nous_request_metadata(
    session_id: str | None = None, *, task: str | None = None, messages: list | None = None,
) -> dict[str, str]:
    """Describe a Nous request and reuse its existing conversation lineage ID."""
    if task is None:
        activity = "assistant_chat"
        purpose = (
            "Continue the assistant response using the requested tool results."
            if messages and messages[-1].get("role") == "tool"
            else "Respond to the user's latest message."
        )
    else:
        activity = task if task in _AUXILIARY_PURPOSES else "auxiliary"
        purpose = _AUXILIARY_PURPOSES.get(task, "Complete a supporting operation for the assistant.")
    metadata = {"hermes_activity": activity, "hermes_purpose": purpose}
    conversation_id = get_conversation_context() or session_id
    if conversation_id and len(conversation_id) <= 512 and conversation_id.strip():
        metadata["hermes_activity_id"] = conversation_id
    return metadata
