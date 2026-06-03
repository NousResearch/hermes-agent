"""Feishu Session Binding Bridge — sidecar for v2.10 multi-entry session binding.

Called from gateway/platforms/feishu.py after session_key is built but before
message dispatch.  This is an optional sidecar: if any step fails, the main
Feishu message flow continues uninterrupted.

The sidecar:
1. Normalizes the SessionSource into a minimal raw payload.
2. Uses FeishuEntryAdapter to produce a canonical EntryEvent.
3. Writes the workspace/session binding to SessionBinding store.
4. Resolves the session using the priority chain.
5. Optionally detects ambiguity and returns info for interactive card.

Does NOT create tasks, call agents, route, or write ledger events.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gateway.session import SessionSource

from agent.managed_agents.feishu_session_resolver import (
    ResolutionResult,
    AmbiguityInfo,
    resolve_feishu_session,
    check_ambiguity,
    record_card_session_binding,
)
from agent.managed_agents.feishu_session_cards import build_ambiguity_card

logger = logging.getLogger(__name__)


def record_feishu_session_binding(source: "SessionSource", session_key: str) -> None:
    """Record a Feishu session binding for v2.10 without affecting existing flow.

    Called from FeishuAdapter._process_inbound_message() after session_key
    is built but before _dispatch_inbound_event().  Failures are caught and
    logged; the main flow continues.

    Args:
        source: The SessionSource for the incoming message.
        session_key: The existing session key string from build_session_key().
    """
    try:
        from agent.managed_agents.feishu_entry_adapter import FeishuEntryAdapter
        from agent.managed_agents.session_binding import put_binding

        raw = {
            "chat_id": source.chat_id,
            "message_id": getattr(source, "message_id", None) or f"auto-{int(time.time())}",
            "open_id": source.user_id or "unknown",
            "content": "",
            "thread_id": source.thread_id,
            "session_key": session_key,
        }

        adapter = FeishuEntryAdapter()
        event = adapter.normalize_event(raw)

        put_binding(
            entrypoint="feishu",
            external_channel_id=source.chat_id,
            external_thread_id=source.thread_id,
            workspace_id=event.workspace_id,
            session_id=session_key,
        )
        logger.debug("Recorded session binding for Feishu session_key=%s", session_key)
    except Exception:
        logger.debug("Session binding sidecar failed (non-critical): %s", exc_info=True)


def resolve_feishu_session_from_source(
    source: "SessionSource",
    *,
    active_sessions: tuple[str, ...] | None = None,
) -> ResolutionResult | None:
    """Resolve a Feishu session using the full priority chain.

    This is an enhanced sidecar call that goes beyond simple binding
    recording. It resolves the session based on the priority chain:
    card > alias > p2p > thread > group > ambiguity > default.

    Returns None if the sidecar encounters any error (non-critical).
    The main Feishu flow should continue using the existing session_key.

    Args:
        source: The SessionSource for the incoming message.
        active_sessions: Optional tuple of active session IDs for ambiguity detection.

    Returns:
        ResolutionResult or None on error.
    """
    try:
        from agent.managed_agents.feishu_entry_adapter import FeishuEntryAdapter

        raw = {
            "chat_id": source.chat_id,
            "message_id": getattr(source, "message_id", None) or f"auto-{int(time.time())}",
            "open_id": source.user_id or "unknown",
            "content": "",
            "thread_id": source.thread_id,
        }

        adapter = FeishuEntryAdapter()
        event = adapter.normalize_event(raw)
        result = resolve_feishu_session(event, active_sessions=active_sessions)
        return result
    except Exception:
        logger.debug("Session resolution sidecar failed (non-critical): %s", exc_info=True)
        return None


def check_feishu_ambiguity(
    source: "SessionSource",
    *,
    active_sessions: tuple[str, ...] | None = None,
) -> AmbiguityInfo | None:
    """Check if a Feishu message context is ambiguous.

    Returns AmbiguityInfo if an interactive card should be sent,
    or None if the message can proceed normally.

    This is a sidecar call; failures are logged and return None.

    Args:
        source: The SessionSource for the incoming message.
        active_sessions: Optional tuple of active session IDs for the workspace.

    Returns:
        AmbiguityInfo if card should be sent, None otherwise.
    """
    try:
        from agent.managed_agents.feishu_entry_adapter import FeishuEntryAdapter

        raw = {
            "chat_id": source.chat_id,
            "message_id": getattr(source, "message_id", None) or f"auto-{int(time.time())}",
            "open_id": source.user_id or "unknown",
            "content": "",
            "thread_id": source.thread_id,
        }

        adapter = FeishuEntryAdapter()
        event = adapter.normalize_event(raw)
        return check_ambiguity(event, active_sessions=active_sessions)
    except Exception:
        logger.debug("Ambiguity check sidecar failed (non-critical): %s", exc_info=True)
        return None


def build_session_selection_card(
    ambiguity: AmbiguityInfo,
    message_preview: str = "",
) -> dict[str, Any] | None:
    """Build an interactive card for session selection when ambiguity is detected.

    Returns a Feishu card body dict, or None if building fails.

    Args:
        ambiguity: The AmbiguityInfo from check_feishu_ambiguity().
        message_preview: Short preview of the incoming message.

    Returns:
        Card body dict or None.
    """
    try:
        return build_ambiguity_card(ambiguity, message_preview=message_preview)
    except Exception:
        logger.debug("Card building sidecar failed (non-critical): %s", exc_info=True)
        return None


def handle_select_session_card_action(action_value: dict[str, Any]) -> bool:
    """Handle a 'select_session' card action from a Feishu interactive card.

    This is called when a user taps a session selection button in an
    ambiguity card. It writes the binding with source="card".

    Args:
        action_value: The card action value dict with keys:
            session_id, workspace_id, chat_id, thread_id

    Returns:
        True if binding was written, False on error.
    """
    try:
        session_id = action_value.get("session_id", "")
        workspace_id = action_value.get("workspace_id", "")
        chat_id = action_value.get("chat_id", "")
        thread_id = action_value.get("thread_id") or None

        if not session_id or not chat_id:
            logger.warning("select_session card action missing session_id or chat_id")
            return False

        record_card_session_binding(
            chat_id=chat_id,
            thread_id=thread_id,
            workspace_id=workspace_id,
            session_id=session_id,
        )
        logger.info(
            "Recorded card session binding: chat_id=%s session_id=%s",
            chat_id, session_id,
        )
        return True
    except Exception:
        logger.error("Failed to handle select_session card action: %s", exc_info=True)
        return False
