"""Gateway session queue and event management — extracted from gateway/run.py.

Manages the FIFO queue of pending message events per session key,
goal continuation logic, and queue depth tracking.  Stateless helpers
that take adapter references as arguments.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def enqueue_fifo(
    queue: OrderedDict,
    session_key: str,
    queued_event: Any,
    adapter: Any,
    max_queue_depth: int = 20,
) -> None:
    """Add an event to the FIFO queue for a session key.

    Drops the oldest event when the queue exceeds max_queue_depth.
    """
    if session_key not in queue:
        queue[session_key] = []
    queue[session_key].append(queued_event)
    # Capacity limit — drop oldest if exceeded
    while len(queue[session_key]) > max_queue_depth:
        dropped = queue[session_key].pop(0)
        logger.info("Dropped oldest queued event for %s (queue depth %d)", session_key, max_queue_depth)


def promote_queued_event(
    queue: OrderedDict,
    session_key: str,
    *,
    adapter: Any = None,
) -> Optional[Any]:
    """Pop the next queued event for a session key, or None if empty."""
    events = queue.get(session_key)
    if not events:
        return None
    return events.pop(0)


def queue_depth(
    queue: OrderedDict,
    session_key: str,
    *,
    adapter: Any = None,
) -> int:
    """Return the current queue depth for a session key."""
    events = queue.get(session_key)
    return len(events) if events else 0


def queue_or_replace_pending_event(
    queue: OrderedDict,
    pending: Dict[str, Any],
    session_key: str,
    event: Any,
) -> None:
    """Queue a new event or replace an existing pending one for this session."""
    # Add to FIFO
    if session_key not in queue:
        queue[session_key] = []
    queue[session_key].append(event)
    # Track in pending dict
    pending[session_key] = event


def dequeue_pending_event(adapter: Any, session_key: str) -> Any | None:
    """Dequeue the most recent pending event for a session key.

    Uses the adapter's internal queue of pending message events.
    Returns None if nothing is pending.
    """
    try:
        pending = getattr(adapter, "_pending_events", {})
        return pending.pop(session_key, None)
    except Exception:
        return None


def is_goal_continuation_event(event_or_text: Any) -> bool:
    """Check if an event represents a goal continuation (e.g. from /goal)."""
    if isinstance(event_or_text, str):
        text = event_or_text
    elif hasattr(event_or_text, "text"):
        text = event_or_text.text
    elif isinstance(event_or_text, dict):
        text = event_or_text.get("text", "") or ""
    else:
        return False
    return text.strip().startswith("/goal continue") if isinstance(text, str) else False


def clear_goal_pending_continuations(
    queue: OrderedDict,
    session_key: str,
    adapter: Any,
) -> int:
    """Remove all /goal continue events from the queue for this session.

    Returns the number of removed events.
    """
    events = queue.get(session_key, [])
    before = len(events)
    queue[session_key] = [e for e in events if not is_goal_continuation_event(e)]
    return before - len(queue.get(session_key, []))


def drain_queue_for_session(queue: OrderedDict, session_key: str) -> list:
    """Drain and return all queued events for a session key."""
    return queue.pop(session_key, [])
