"""Gateway event queue management — extracted from gateway/run.py.

Queue management helpers for the GatewayRunner.  These functions take
a ``runner`` parameter (the GatewayRunner instance) instead of ``self``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def queue_during_drain_enabled(runner: Any) -> bool:
    # Both "queue" and "steer" modes imply the user doesn't want messages
    # to be lost during restart — queue them for the newly-spawned gateway
    # process to pick up.  "interrupt" mode drops them (current behaviour).
    return runner._restart_requested and runner._busy_input_mode in {"queue", "steer"}


def enqueue_fifo(
    runner: Any,
    session_key: str,
    queued_event: Any,
    adapter: Any,
) -> None:
    """Append a /queue event to the FIFO chain for a session."""
    if adapter is None:
        return
    pending_slot = getattr(adapter, "_pending_messages", None)
    if pending_slot is None:
        return
    queued_events = getattr(runner, "_queued_events", None)
    if queued_events is None:
        queued_events = {}
        runner._queued_events = queued_events
    if session_key in pending_slot:
        queued_events.setdefault(session_key, []).append(queued_event)
    else:
        pending_slot[session_key] = queued_event


def promote_queued_event(
    runner: Any,
    session_key: str,
    adapter: Any,
    pending_event: Optional[Any],
) -> Optional[Any]:
    """Promote the next overflow item after the slot was drained.

    Called at the drain site after _dequeue_pending_event consumed
    (or failed to consume) the slot.  If there's an overflow item:
      - When pending_event is None (slot was empty), return the
        overflow head as the new pending_event.
      - When pending_event already exists (slot was populated by an
        interrupt follow-up or similar), stage the overflow head in
        the slot so the NEXT recursion picks it up.
    Returns the (possibly updated) pending_event for drain to use.
    """
    queued_events = getattr(runner, "_queued_events", None)
    if not queued_events:
        return pending_event
    overflow = queued_events.get(session_key)
    if not overflow:
        return pending_event
    next_queued = overflow.pop(0)
    if not overflow:
        queued_events.pop(session_key, None)
    if pending_event is None:
        return next_queued
    if adapter is not None and hasattr(adapter, "_pending_messages"):
        adapter._pending_messages[session_key] = next_queued
    else:
        # No adapter — push back so we don't silently drop the item.
        queued_events.setdefault(session_key, []).insert(0, next_queued)
    return pending_event


def queue_depth(
    runner: Any,
    session_key: str,
    *,
    adapter: Any = None,
) -> int:
    """Total pending /queue items for a session — slot + overflow."""
    queued_events = getattr(runner, "_queued_events", None) or {}
    depth = len(queued_events.get(session_key, []))
    if adapter is not None and session_key in getattr(adapter, "_pending_messages", {}):
        depth += 1
    return depth


def is_goal_continuation_event(event_or_text: Any) -> bool:
    """Return True for synthetic /goal continuation turns.

    Goal continuations are normal queued user-role events, so pause/clear
    must distinguish them from real user /queue messages before removing or
    suppressing them.
    """
    text = getattr(event_or_text, "text", event_or_text) or ""
    return str(text).startswith("[Continuing toward your standing goal]\nGoal:")


def clear_goal_pending_continuations(
    runner: Any,
    session_key: str,
    adapter: Any,
) -> int:
    """Remove queued synthetic /goal continuations for one session.

    User-issued /goal pause/clear can race with a continuation already
    queued by the judge.  Remove only synthetic goal continuations while
    preserving normal /queue and user follow-up events.
    """
    removed = 0
    pending_slot = getattr(adapter, "_pending_messages", None) if adapter is not None else None
    if isinstance(pending_slot, dict):
        pending_event = pending_slot.get(session_key)
        if is_goal_continuation_event(pending_event):
            pending_slot.pop(session_key, None)
            removed += 1

    queued_events = getattr(runner, "_queued_events", None)
    if isinstance(queued_events, dict):
        overflow = queued_events.get(session_key) or []
        if overflow:
            kept = []
            for queued_event in overflow:
                if is_goal_continuation_event(queued_event):
                    removed += 1
                else:
                    kept.append(queued_event)
            if kept:
                queued_events[session_key] = kept
            else:
                queued_events.pop(session_key, None)
    return removed


def goal_still_active_for_session(runner: Any, session_id: str) -> bool:
    """Best-effort fresh DB check before running a queued continuation."""
    if not session_id:
        return False
    try:
        from hermes_cli.goals import GoalManager

        return GoalManager(session_id=session_id).is_active()
    except Exception as exc:
        logger.debug("goal continuation: active-state recheck failed: %s", exc)
        return False


def queue_or_replace_pending_event(runner: Any, session_key: str, event: Any) -> None:
    from gateway.platforms.base import merge_pending_message_event

    adapter = runner.adapters.get(event.source.platform)
    if not adapter:
        return
    merge_pending_message_event(adapter._pending_messages, session_key, event)
