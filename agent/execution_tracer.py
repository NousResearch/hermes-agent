"""Execution Tracer — lightweight tool-call event collector.

Hooks into handle_function_call (via model_tools enrichment block)
to capture every tool execution event into a bounded ring buffer.
Periodically flushed to the agent_traces table in state.db.

Part of the Agent Observability system (Spike 010).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Ring buffer ────────────────────────────────────────────────────────

_MAX_BUFFER = 1000  # In-memory events before forced flush
_FLUSH_INTERVAL = 60  # Seconds between automatic flushes

_buffer: deque = deque(maxlen=_MAX_BUFFER)
_buffer_lock = threading.Lock()
_last_flush: float = 0.0

# ── Trace event ────────────────────────────────────────────────────────


def trace_event(
    *,
    session_id: str,
    tool_name: str,
    duration_ms: float = 0.0,
    success: bool = True,
    error_class: str = "",
    error_message: str = "",
    confidence: float = 0.0,
    recovery_action: str = "",
    result_summary: str = "",
    task_id: str = "",
) -> None:
    """Record a tool execution event in the ring buffer.

    Called from model_tools.handle_function_call enrichment block.
    Thread-safe, non-blocking, <1µs overhead.
    """
    event = {
        "session_id": session_id,
        "tool": tool_name,
        "duration_ms": round(duration_ms, 2),
        "success": success,
        "error_class": error_class,
        "error_message": error_message[:200],
        "confidence": round(confidence, 4),
        "recovery_action": recovery_action[:100],
        "result_summary": result_summary[:80],
        "task_id": task_id,
        "at": time.time(),
    }
    with _buffer_lock:
        _buffer.append(event)
    _maybe_flush()


# ── Flush to DB ────────────────────────────────────────────────────────


def enrich_last_trace(
    *,
    error_class: str = "",
    error_message: str = "",
    confidence: float = 0.0,
    recovery_action: str = "",
) -> None:
    """Enrich the most recent trace event with error classification data.

    Called after classify_tool_error has determined the error class.
    Mutates the last event in the ring buffer in-place.
    """
    with _buffer_lock:
        if not _buffer:
            return
        last = _buffer[-1]
        last["success"] = False
        last["error_class"] = error_class
        last["error_message"] = error_message
        last["confidence"] = confidence
        last["recovery_action"] = recovery_action


def _maybe_flush() -> None:
    """Flush buffer to state.db if interval elapsed or buffer near full."""
    global _last_flush
    now = time.time()
    if now - _last_flush < _FLUSH_INTERVAL and len(_buffer) < _MAX_BUFFER * 0.8:
        return
    _flush()


def _flush() -> None:
    """Persist buffered events to agent_traces table."""
    global _last_flush
    import json as _json

    with _buffer_lock:
        if not _buffer:
            return
        events = list(_buffer)
        _buffer.clear()
        _last_flush = time.time()

    try:
        from hermes_state import SessionDB

        db = SessionDB()
        now = time.time()

        def _do(conn):
            conn.executemany(
                "INSERT INTO agent_traces "
                "(session_id, tool_name, duration_ms, success, error_class, "
                " error_message, confidence, recovery_action, result_summary, "
                " task_id, at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        e["session_id"],
                        e["tool"],
                        e["duration_ms"],
                        1 if e["success"] else 0,
                        e["error_class"],
                        e["error_message"],
                        e["confidence"],
                        e["recovery_action"],
                        e["result_summary"],
                        e["task_id"],
                        e["at"],
                    )
                    for e in events
                ],
            )

        db._execute_write(_do)
    except Exception:
        logger.debug("ExecutionTracer flush failed (best-effort)", exc_info=True)


# ── Query helpers ──────────────────────────────────────────────────────


def get_recent_events(session_id: str, limit: int = 20) -> list:
    """Return most recent trace events for *session_id*.

    Combines in-memory buffer + DB for complete picture.
    """
    events = []
    now = time.time()

    # In-memory events
    with _buffer_lock:
        for e in _buffer:
            if e["session_id"] == session_id:
                events.append(e)

    # DB events
    try:
        from hermes_state import SessionDB

        db = SessionDB()

        def _do(conn):
            rows = conn.execute(
                "SELECT tool_name, duration_ms, success, error_class, "
                "error_message, recovery_action, result_summary, at "
                "FROM agent_traces "
                "WHERE session_id = ? "
                "ORDER BY at DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
            for row in rows:
                events.append({
                    "tool": row[0],
                    "duration_ms": row[1],
                    "success": bool(row[2]),
                    "error_class": row[3] or "",
                    "error_message": row[4] or "",
                    "recovery_action": row[5] or "",
                    "result_summary": row[6] or "",
                    "at": row[7],
                })
            return None

        db._execute_write(_do)
    except Exception:
        pass

    # Sort by time desc, deduplicate
    events.sort(key=lambda e: e.get("at", 0), reverse=True)
    seen = set()
    deduped = []
    for e in events:
        sig = (e["tool"], e.get("at", 0))
        if sig not in seen:
            seen.add(sig)
            deduped.append(e)
    return deduped[:limit]


def get_session_stats(session_id: str) -> Dict[str, Any]:
    """Aggregate stats for a session: success rate, error breakdown, etc."""
    try:
        from hermes_state import SessionDB

        db = SessionDB()

        def _do(conn):
            total = conn.execute(
                "SELECT COUNT(*) FROM agent_traces WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
            succeeded = conn.execute(
                "SELECT COUNT(*) FROM agent_traces "
                "WHERE session_id = ? AND success = 1",
                (session_id,),
            ).fetchone()[0]
            recovered = conn.execute(
                "SELECT COUNT(*) FROM agent_traces "
                "WHERE session_id = ? AND success = 1 AND error_class != ''",
                (session_id,),
            ).fetchone()[0]
            error_rows = conn.execute(
                "SELECT error_class, COUNT(*) as cnt "
                "FROM agent_traces "
                "WHERE session_id = ? AND success = 0 "
                "GROUP BY error_class ORDER BY cnt DESC",
                (session_id,),
            ).fetchall()

            return {
                "total_tools": total,
                "succeeded": succeeded,
                "failed": total - succeeded,
                "recovered": recovered,
                "success_rate": round(succeeded / max(total, 1), 3),
                "recovery_rate": round(recovered / max(total - succeeded, 1), 3) if total > succeeded else 0.0,
                "error_breakdown": {row[0] or "unknown": row[1] for row in error_rows},
            }

        return db._execute_write(_do)
    except Exception:
        return {}


# ── Force flush (called at session end) ────────────────────────────────


def flush_all() -> None:
    """Force-flush all buffered events. Call at session shutdown."""
    with _buffer_lock:
        if _buffer:
            _flush()