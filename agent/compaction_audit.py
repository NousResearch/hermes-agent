"""Durable compaction audit events: append-only start/end brackets per attempt.

Issue #104099. Compressing a session is the one sanctioned context mutation, yet it left no
durable trace beyond a content-free log line. This module persists two rows per attempt into
``compaction_events``: ``start`` when the compression lease is acquired, ``end`` when the
attempt settles (committed/aborted/cooldown/etc.). A crash mid-compact leaves an orphaned
``start`` — detectable with one SQL query instead of forensic log archaeology.

Contract (mirrors the activity-heartbeat seam this builds on):
- Observation-only. Writes are best-effort and NEVER raise into the compression path; a failed
  audit write must not abort (or wedges) a compaction that would otherwise succeed.
- Content-free payloads (same policy as ``_emit_compression_attempt_telemetry``): counts,
  statuses, durations, model/provider identifiers. No transcript text.
- No foreign keys onto ``sessions`` — audit rows outlive their session (rotated/ephemeral
  children included); cleanup rides existing session pruning by session_id.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Payloads are metadata; cap guards against a telemetry dict accidentally carrying a summary.
_PAYLOAD_JSON_MAX_CHARS = 8_192


def record_compaction_event(
    session_db: Any,
    session_id: Optional[str],
    attempt_id: str,
    event: str,
    payload: Optional[dict] = None,
    *,
    at: Optional[float] = None,
) -> bool:
    """Append one durable compaction event row. Best-effort; returns True when persisted.

    ``session_db`` is the SessionDB handle (``agent._session_db``); anything without a callable
    ``append_compaction_event`` (older store, test double) is a silent no-op so the recorder can
    never become a compatibility wedge for compression itself.
    """
    if not session_id or not attempt_id:
        return False
    append = getattr(session_db, "append_compaction_event", None)
    if not callable(append):
        return False
    try:
        body = json.dumps(payload or {}, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        body = "{}"
    if len(body) > _PAYLOAD_JSON_MAX_CHARS:
        body = json.dumps({"truncated": True, "chars": len(body)}, separators=(",", ":"))
    try:
        append(session_id, attempt_id, event, float(at if at is not None else time.time()), body)
        return True
    except Exception:
        logger.debug("compaction audit event write failed (ignored)", exc_info=True)
        return False


def find_orphaned_compaction_starts(session_db: Any, session_id: Optional[str] = None) -> list[dict]:
    """``start`` rows with no matching ``end`` — crashed or timed-out compactions.

    Read-side helper for diagnostics (``hermes sessions`` surface or doctor). Grouped per
    (session_id, attempt_id): any attempt whose latest event is ``start`` never settled.
    """
    query = getattr(session_db, "find_orphaned_compaction_starts", None)
    if callable(query):
        try:
            found: Any = query(session_id) if session_id else query()
            return list(found) if found is not None else []
        except Exception:
            logger.debug("orphaned compaction start query failed", exc_info=True)
            return []
    return []
