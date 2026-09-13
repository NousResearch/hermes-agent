"""Durable, display-only receipts for completed self-improvement work.

The parent owns persistence. The review fork never writes its replay transcript,
and these receipts are excluded from the model projection on every resume.
"""
from __future__ import annotations

import contextvars
import logging
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# Deferred reviews retain their origin across /new and bounded requeues. The
# idle queue captures this in its Context, not on the reusable parent agent.
REVIEW_SOURCE_SESSION_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "review_source_session_id", default=None,
)


def publish_review_summary(agent: Any, actions: list[str], *, source_session_id: str | None = None) -> None:
    if not actions:
        return
    text = "💾 Self-improvement review: " + " · ".join(dict.fromkeys(actions))
    event: dict[str, Any] = {
        "text": text, "review_id": uuid.uuid4().hex, "timestamp": time.time(),
    }
    session_id = source_session_id or getattr(agent, "session_id", None)
    db = getattr(agent, "_session_db", None)
    if db is not None and session_id and not getattr(agent, "_persist_disabled", False):
        try:
            from hermes_state_errors import CompressionSessionClosedError

            # Follow only compression continuations, not /new or a different profile.
            # A rotation can win between the lookup and append; retry that race, bounded.
            for attempt in range(3):
                target = db.resolve_resume_session_id(session_id) or session_id
                try:
                    row_id = db.append_message(
                        target, "system", text, timestamp=event["timestamp"],
                        display_kind="review_summary", display_metadata={
                            "review_id": event["review_id"], "source_session_id": session_id,
                        },
                    )
                    event.update(row_id=row_id, stored_session_id=target)
                    break
                except CompressionSessionClosedError:
                    if attempt == 2:
                        raise
        except Exception:
            # The skill write already happened: do not hide its confirmation or claim
            # persistence succeeded. Legacy live delivery is still useful on disk failure.
            logger.warning("Could not persist self-improvement receipt for session=%s", session_id, exc_info=True)
    event["persisted"] = "row_id" in event
    agent._safe_print(f"  {text}")
    # Keep the text callback contract for CLI, messaging and third-party consumers.
    # The Desktop gateway opts into metadata so live and stored rows share an identity.
    structured = vars(agent).get("background_review_event_callback")
    callback = structured if callable(structured) else getattr(agent, "background_review_callback", None)
    if callable(callback):
        try:
            callback(event if callable(structured) else text)
        except Exception:
            logger.warning("Could not deliver self-improvement receipt for session=%s", session_id, exc_info=True)
