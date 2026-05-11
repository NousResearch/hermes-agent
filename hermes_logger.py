"""Structured JSON logging for Hermes Agent — Milestone A.

Provides correlation-ID-aware structured logging that writes JSON lines to
~/.hermes/logs/structured.jsonl alongside the existing plain-text logs.

Why this exists:
- Existing logs are human-readable but not query-friendly
- Hard to answer "what did task X do?" or "which tool failed on run Y?"
- Structured JSON makes log analysis, dashboards, and benchmarking possible

Usage:
    from hermes_logger import get_logger

    log = get_logger()
    log.event("tool_call", task_id="abc123", tool="terminal", step=3)
    log.event("task_end", task_id="abc123", status="completed", tokens=1200)

    # Or use the context manager to automatically attach IDs to all events:
    with log.context(task_id="abc123", session_id="sess_xyz"):
        log.event("model_call", model="claude-opus-4-6", tokens_in=800)
        log.event("tool_call", tool="web_search", query="python asyncio")

Correlation ID hierarchy:
    request_id  — one per run_conversation() call
    task_id     — matches state.db tasks.task_id
    session_id  — matches state.db sessions.id
    tool_call_id — unique per tool invocation within a request
"""

import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ContextVars hold the active correlation IDs for the current async/thread context.
# These are set at conversation start and cleared on exit.
_ctx_request_id: ContextVar[Optional[str]] = ContextVar("hermes_request_id", default=None)
_ctx_task_id: ContextVar[Optional[str]] = ContextVar("hermes_task_id", default=None)
_ctx_session_id: ContextVar[Optional[str]] = ContextVar("hermes_session_id", default=None)

_LOG_DIR = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes")) / "logs"
_STRUCTURED_LOG = _LOG_DIR / "structured.jsonl"

_write_lock = threading.Lock()
_initialized = False


def _ensure_log_dir() -> bool:
    """Create log directory if needed. Returns True if ready."""
    global _initialized
    if _initialized:
        return True
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        _initialized = True
        return True
    except Exception as e:
        logger.debug("Could not create structured log dir: %s", e)
        return False


def _write_line(record: Dict[str, Any]) -> None:
    """Append a JSON record to the structured log file. No-throw."""
    if not _ensure_log_dir():
        return
    try:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _write_lock:
            with open(_STRUCTURED_LOG, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:
        logger.debug("Structured log write failed (non-fatal): %s", e)


class HermesLogger:
    """Correlation-ID-aware structured logger.

    All events are written as JSON lines to ~/.hermes/logs/structured.jsonl.
    Correlation IDs (request_id, task_id, session_id) are attached from
    contextvars if not explicitly provided, so callers don't need to thread
    them through every call.
    """

    def event(
        self,
        event_type: str,
        *,
        request_id: Optional[str] = None,
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        **fields: Any,
    ) -> None:
        """Write a structured event record.

        Args:
            event_type: Short label for the event (e.g. "task_start", "tool_call",
                "model_call", "task_end", "approval_requested").
            request_id: Override the context request_id.
            task_id: Override the context task_id.
            session_id: Override the context session_id.
            tool_call_id: Per-tool-invocation ID (auto-generated if not provided).
            **fields: Arbitrary additional fields to include in the record.
        """
        record = {
            "ts": time.time(),
            "event": event_type,
            "request_id": request_id or _ctx_request_id.get(),
            "task_id": task_id or _ctx_task_id.get(),
            "session_id": session_id or _ctx_session_id.get(),
        }
        if tool_call_id:
            record["tool_call_id"] = tool_call_id
        record.update(fields)
        # Drop None values to keep records clean
        record = {k: v for k, v in record.items() if v is not None}
        _write_line(record)

    @contextmanager
    def context(
        self,
        *,
        request_id: Optional[str] = None,
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """Context manager that sets correlation IDs for the duration of a block.

        Example:
            with hermes_log.context(task_id="abc", session_id="sess_xyz"):
                hermes_log.event("tool_call", tool="terminal")
                # → record automatically includes task_id and session_id
        """
        rid = request_id or str(uuid.uuid4())
        rid_token = _ctx_request_id.set(rid)
        tid_token = _ctx_task_id.set(task_id) if task_id else None
        sid_token = _ctx_session_id.set(session_id) if session_id else None
        try:
            yield rid
        finally:
            _ctx_request_id.reset(rid_token)
            if tid_token is not None:
                _ctx_task_id.reset(tid_token)
            if sid_token is not None:
                _ctx_session_id.reset(sid_token)

    # ------------------------------------------------------------------
    # Convenience methods for common Hermes event types
    # ------------------------------------------------------------------

    def task_start(
        self,
        task_id: str,
        session_id: str,
        model: str,
        user_message: str,
        *,
        request_id: Optional[str] = None,
    ) -> None:
        self.event(
            "task_start",
            request_id=request_id,
            task_id=task_id,
            session_id=session_id,
            model=model,
            user_message_preview=user_message[:120],
        )

    def task_end(
        self,
        task_id: str,
        status: str,
        *,
        request_id: Optional[str] = None,
        api_calls: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        error: Optional[str] = None,
    ) -> None:
        self.event(
            "task_end",
            request_id=request_id,
            task_id=task_id,
            status=status,
            api_calls=api_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error=error,
        )

    def tool_call(
        self,
        task_id: str,
        tool_name: str,
        *,
        request_id: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        args_preview: Optional[str] = None,
        risk_level: Optional[str] = None,
    ) -> str:
        """Log a tool call start. Returns a generated tool_call_id for correlation."""
        tcid = tool_call_id or f"tc_{uuid.uuid4().hex[:8]}"
        self.event(
            "tool_call",
            request_id=request_id,
            task_id=task_id,
            tool_name=tool_name,
            tool_call_id=tcid,
            args_preview=args_preview,
            risk_level=risk_level,
        )
        return tcid

    def tool_result(
        self,
        task_id: str,
        tool_name: str,
        *,
        request_id: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        self.event(
            "tool_result",
            request_id=request_id,
            task_id=task_id,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            duration_ms=round(duration_ms, 1) if duration_ms is not None else None,
            success=success,
            error=error,
        )

    def model_call(
        self,
        task_id: str,
        model: str,
        *,
        request_id: Optional[str] = None,
        api_call_count: int = 0,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        self.event(
            "model_call",
            request_id=request_id,
            task_id=task_id,
            model=model,
            api_call_number=api_call_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=round(duration_ms, 1) if duration_ms is not None else None,
        )

    def approval_requested(
        self,
        task_id: str,
        command: str,
        description: str,
        *,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.event(
            "approval_requested",
            request_id=request_id,
            task_id=task_id,
            session_id=session_id,
            command_preview=command[:200],
            description=description,
        )

    def approval_resolved(
        self,
        task_id: str,
        decision: str,
        *,
        request_id: Optional[str] = None,
        method: str = "manual",
    ) -> None:
        """Log an approval decision. decision: 'once', 'session', 'always', 'deny'."""
        self.event(
            "approval_resolved",
            request_id=request_id,
            task_id=task_id,
            decision=decision,
            method=method,
        )


# Module-level singleton — import and use directly:
#   from hermes_logger import hermes_log
#   hermes_log.task_start(...)
hermes_log = HermesLogger()


def get_logger() -> HermesLogger:
    """Return the module-level HermesLogger singleton."""
    return hermes_log


# ---------------------------------------------------------------------------
# Log reader utilities — useful for the dashboard and hermes inspect commands
# ---------------------------------------------------------------------------

def read_recent_events(
    n: int = 100,
    event_type: Optional[str] = None,
    task_id: Optional[str] = None,
) -> list:
    """Read the most recent N structured log events.

    Args:
        n: Maximum number of events to return (from the tail of the log).
        event_type: If set, only return events with this type.
        task_id: If set, only return events for this task.

    Returns:
        List of event dicts, newest-last (chronological order).
    """
    if not _STRUCTURED_LOG.exists():
        return []
    try:
        lines = _STRUCTURED_LOG.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    events = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event_type and record.get("event") != event_type:
            continue
        if task_id and record.get("task_id") != task_id:
            continue
        events.append(record)
        if len(events) >= n:
            break

    return list(reversed(events))
