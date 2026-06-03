"""Feishu session resolver — pure resolution chain for thread/session binding.

This module implements the v2.10 resolution order:
  1. Explicit card session_id
  2. Explicit alias (reserved for future)
  3. Private current-session binding (p2p)
  4. Thread/root-message binding
  5. Group workspace binding
  6. Ambiguity detection (sends interactive card)
  7. Reject / no global fallback

Does NOT call agents, create tasks, or write ledger events directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from .workspace import DEFAULT_WORKSPACE_ID
from .session import DEFAULT_SESSION_ID
from .entry_event import EntryEvent
from .session_binding import (
    BindingSource,
    SessionBindingValue,
    get_binding_value,
    put_binding,
    resolve_binding,
)

# Feature flag: when False, skip ambiguity detection and fall through to
# default session (current behavior). When True, detect ambiguity and
# return needs_card=True.
FEISHU_AMBIGUITY_CARD_ENABLED = os.getenv("FEISHU_AMBIGUITY_CARD_ENABLED", "false").lower() in ("true", "1", "yes")


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    """Result of resolving a Feishu message to a workspace/session."""

    workspace_id: str
    session_id: str
    source: BindingSource
    ambiguous: bool = False
    needs_card: bool = False
    available_sessions: tuple[str, ...] = ()
    message: str = ""


@dataclass(frozen=True, slots=True)
class AmbiguityInfo:
    """Details when ambiguity is detected."""

    needs_card: bool
    workspace_id: str
    chat_id: str
    thread_id: str | None
    available_sessions: tuple[str, ...] = ()


def resolve_feishu_session(
    event: EntryEvent,
    *,
    active_sessions: tuple[str, ...] | None = None,
) -> ResolutionResult:
    """Resolve a Feishu EntryEvent to workspace/session using the priority chain.

    Args:
        event: The normalized Feishu EntryEvent.
        active_sessions: Tuple of active session IDs for the workspace.
            Used for ambiguity detection. If None or empty, ambiguity
            detection is skipped (falls through to default).

    Returns:
        ResolutionResult with workspace_id, session_id, source, and
        ambiguity info if detected.
    """
    chat_id = event.external_channel_id or ""
    thread_id = event.external_thread_id
    is_group = _is_group_chat(event)

    # 1. Explicit card binding
    card_binding = get_binding_value("feishu", chat_id, thread_id)
    if card_binding and card_binding.source == "card":
        return ResolutionResult(
            workspace_id=card_binding.workspace_id,
            session_id=card_binding.session_id,
            source="card",
            ambiguous=False,
            needs_card=False,
            message="Card binding selected",
        )

    # 2. Explicit alias (reserved for future: "/task abc123" commands)
    # Not yet implemented; skipped.

    # 3. Private current-session binding
    if not is_group:
        p2p_binding = get_binding_value("feishu", chat_id, None)
        if p2p_binding:
            return ResolutionResult(
                workspace_id=p2p_binding.workspace_id,
                session_id=p2p_binding.session_id,
                source=p2p_binding.source,
                ambiguous=False,
                needs_card=False,
                message="Private chat current-session binding",
            )
        # No binding yet for p2p; derive default.
        workspace_id = event.workspace_id or DEFAULT_WORKSPACE_ID
        session_id = event.session_id or DEFAULT_SESSION_ID
        return ResolutionResult(
            workspace_id=workspace_id,
            session_id=session_id,
            source="default",
            ambiguous=False,
            needs_card=False,
            message="Private chat derived session",
        )

    # 4. Thread/root-message binding
    if thread_id:
        thread_binding = get_binding_value("feishu", chat_id, thread_id)
        if thread_binding:
            return ResolutionResult(
                workspace_id=thread_binding.workspace_id,
                session_id=thread_binding.session_id,
                source=thread_binding.source,
                ambiguous=False,
                needs_card=False,
                message="Thread binding",
            )
        # No explicit binding; derive from thread.
        workspace_id = event.workspace_id or DEFAULT_WORKSPACE_ID
        session_id = f"ses-feishu-thread-{thread_id}"
        return ResolutionResult(
            workspace_id=workspace_id,
            session_id=session_id,
            source="thread",
            ambiguous=False,
            needs_card=False,
            message="Thread-derived session",
        )

    # 5. Group workspace binding
    group_binding = get_binding_value("feishu", chat_id, None)
    if group_binding:
        return ResolutionResult(
            workspace_id=group_binding.workspace_id,
            session_id=group_binding.session_id,
            source=group_binding.source,
            ambiguous=False,
            needs_card=False,
            message="Group workspace binding",
        )

    # 6. Ambiguity detection — only when feature flag is enabled
    if FEISHU_AMBIGUITY_CARD_ENABLED and active_sessions and len(active_sessions) > 1:
        return ResolutionResult(
            workspace_id=event.workspace_id or DEFAULT_WORKSPACE_ID,
            session_id=DEFAULT_SESSION_ID,
            source="default",
            ambiguous=True,
            needs_card=True,
            available_sessions=tuple(active_sessions),
            message="Ambiguous: multiple active sessions in group, no thread context",
        )

    # 7. Reject / no global fallback — derive default group session
    workspace_id = event.workspace_id or DEFAULT_WORKSPACE_ID
    session_id = event.session_id or DEFAULT_SESSION_ID
    return ResolutionResult(
        workspace_id=workspace_id,
        session_id=session_id,
        source="default",
        ambiguous=False,
        needs_card=False,
        message="Group chat derived session (no ambiguity)",
    )


def check_ambiguity(
    event: EntryEvent,
    *,
    active_sessions: tuple[str, ...] | None = None,
) -> AmbiguityInfo | None:
    """Check whether an event's context is ambiguous without resolving.

    Returns None if not ambiguous, or AmbiguityInfo if a card should be sent.
    """
    if not FEISHU_AMBIGUITY_CARD_ENABLED:
        return None

    chat_id = event.external_channel_id or ""
    thread_id = event.external_thread_id
    is_group = _is_group_chat(event)

    if not is_group:
        return None

    if thread_id:
        return None

    # Check if there's an explicit binding for this group chat
    existing = get_binding_value("feishu", chat_id, None)
    if existing:
        return None

    # Ambiguous if multiple sessions active
    if active_sessions and len(active_sessions) > 1:
        return AmbiguityInfo(
            needs_card=True,
            workspace_id=event.workspace_id or DEFAULT_WORKSPACE_ID,
            chat_id=chat_id,
            thread_id=thread_id,
            available_sessions=tuple(active_sessions),
        )

    return None


def record_card_session_binding(
    chat_id: str,
    thread_id: str | None,
    workspace_id: str,
    session_id: str,
) -> None:
    """Record a session binding from a card button selection.

    This is called when a user taps a "select_session" card action.
    """
    put_binding(
        entrypoint="feishu",
        external_channel_id=chat_id,
        external_thread_id=thread_id,
        workspace_id=workspace_id,
        session_id=session_id,
        source="card",
    )


def _is_group_chat(event: EntryEvent) -> bool:
    """Determine if a Feishu event came from a group chat.

    Group chat channels start with "oc_" (official Lark group chat ID prefix).
    P2P (direct message) channels start with "ou_" (official Lark user ID prefix).
    """
    channel = event.external_channel_id or ""
    if channel.startswith("oc_"):
        return True
    # If session_id contains group markers, treat as group
    if event.session_id and "group" in event.session_id:
        return True
    return False
