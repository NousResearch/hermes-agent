"""Durable terminal completion handoff for finite CLI parents."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def persist_oneshot_process_completions(
    session_db: Any,
    session_id: str,
    *,
    registry: Any = None,
) -> list[str]:
    """Append owned terminal completions to a finite parent's transcript.

    One-shot parents have no later turn in which to drain their in-memory
    completion queue. Coalescing all results into one user-role row preserves
    strict role alternation after the parent's final assistant response.
    """
    if session_db is None or not session_id:
        return []
    if registry is None:
        from tools.process_registry import process_registry

        registry = process_registry

    target_session_id = session_db.get_compression_tip(session_id) or session_id

    def _owns_event(event: dict) -> bool:
        event_key = str(event.get("session_key") or "")
        if not event_key:
            return False
        try:
            event_key = session_db.get_compression_tip(event_key) or event_key
        except Exception:
            return False
        return str(event_key) == str(target_session_id)

    drained = registry.drain_notifications(
        session_key=session_id,
        owns_event=_owns_event,
        event_types={"completion"},
    )
    if not drained:
        return []

    process_ids = [str(event.get("session_id") or "unknown") for event, _text in drained]
    content = "\n\n".join(text for _event, text in drained)
    try:
        session_db.append_message(
            target_session_id,
            "user",
            content=content,
            display_kind="internal_notification",
            display_metadata={
                "delivery_kind": "oneshot_process_completion",
                "process_ids": process_ids,
            },
        )
    except Exception:
        for event, _text in drained:
            registry.completion_queue.put(event)
        raise
    logger.info(
        "Persisted %d terminal completion(s) for finite session %s: %s",
        len(process_ids),
        target_session_id,
        ", ".join(process_ids),
    )
    return process_ids