"""Normalized Dev/subagent event schema helpers."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional


SUBAGENT_EVENT_SCHEMA_VERSION = 1


def normalize_subagent_event(
    payload: Dict[str, Any],
    *,
    session_id: Optional[str] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Return a persisted/SSE-safe normalized subagent event payload.

    Version 1 is intentionally additive: old clients can ignore the field, and
    old replay rows without it remain valid.
    """

    event = dict(payload)
    event.setdefault("schema_version", SUBAGENT_EVENT_SCHEMA_VERSION)
    event.setdefault("event", "subagent.progress")
    if session_id:
        event.setdefault("session_id", session_id)
    event["created_at"] = float(event.get("created_at") or event.get("timestamp") or now or time.time())
    return event

