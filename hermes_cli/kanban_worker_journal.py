"""Best-effort Kanban worker activity journaling.

Dispatcher-spawned Kanban workers are ordinary CLI agents with
``HERMES_KANBAN_TASK`` in their environment. This module bridges observable
AIAgent callbacks into durable ``task_events`` rows so Kanban UIs can show a
readable execution story (tool activity + visible worker text) without exposing
private chain-of-thought.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

_LOG = logging.getLogger(__name__)

_MAX_TEXT = 4000
_MAX_RESULT = 2000
_MAX_ARGS_JSON = 4000
_TOOL_PROGRESS_LIFECYCLE_EVENTS = {"tool.started", "tool.completed"}
_PRIVATE_PROGRESS_EVENTS = {"reasoning.available", "_thinking", "thinking"}


def _truncate(value: str, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _redact_text(value: Any, *, limit: int = _MAX_TEXT) -> str:
    try:
        from agent.redact import redact_sensitive_text

        return _truncate(redact_sensitive_text(str(value or ""), force=True), limit)
    except Exception:
        return _truncate(str(value or ""), limit)


def _redact_jsonable(value: Any, *, limit: int = _MAX_ARGS_JSON) -> Any:
    """Redact a JSON-like value while preserving shape when practical."""
    try:
        raw = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return _redact_text(value, limit=limit)
    redacted = _redact_text(raw, limit=limit)
    try:
        return json.loads(redacted)
    except Exception:
        return redacted


def _safe_call(cb: Optional[Callable], *args: Any, **kwargs: Any) -> None:
    if cb is None:
        return
    try:
        cb(*args, **kwargs)
    except Exception:
        _LOG.debug("kanban worker composed callback failed", exc_info=True)


@dataclass
class KanbanWorkerJournal:
    """Write observable worker callback events into the Kanban task journal."""

    task_id: str
    run_id: Optional[int] = None
    _seen_assistant_texts: set[str] = field(default_factory=set)

    @classmethod
    def from_env(cls) -> Optional["KanbanWorkerJournal"]:
        task_id = (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
        if not task_id:
            return None
        run_raw = (os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
        try:
            run_id = int(run_raw) if run_raw else None
        except ValueError:
            run_id = None
        return cls(task_id=task_id, run_id=run_id)

    def _record(self, kind: str, payload: Optional[dict[str, Any]]) -> None:
        try:
            from hermes_cli import kanban_db as kb

            conn = kb.connect()
            try:
                with kb.write_txn(conn):
                    kb._append_event(  # existing internal primitive; best-effort bridge
                        conn,
                        self.task_id,
                        kind,
                        payload or None,
                        run_id=self.run_id,
                    )
            finally:
                conn.close()
        except Exception:
            # Observability must never break the worker loop.
            _LOG.debug("failed to persist kanban worker journal event", exc_info=True)

    def tool_start(self, tool_call_id: str, name: str, args: Any) -> None:
        payload = {
            "tool_call_id": _redact_text(tool_call_id, limit=200),
            "tool": _redact_text(name, limit=200),
            "args": _redact_jsonable(args or {}, limit=_MAX_ARGS_JSON),
            "source": "tool_start_callback",
        }
        self._record("tool_start", payload)

    def tool_complete(self, tool_call_id: str, name: str, args: Any, result: Any) -> None:
        payload = {
            "tool_call_id": _redact_text(tool_call_id, limit=200),
            "tool": _redact_text(name, limit=200),
            "args": _redact_jsonable(args or {}, limit=_MAX_ARGS_JSON),
            "result_preview": _redact_text(result, limit=_MAX_RESULT),
            "source": "tool_complete_callback",
        }
        self._record("tool_end", payload)

    def tool_progress(
        self,
        event_type: str,
        name: str | None = None,
        preview: str | None = None,
        args: Any = None,
        **kwargs: Any,
    ) -> None:
        event = str(event_type or "").strip()
        if event in _TOOL_PROGRESS_LIFECYCLE_EVENTS or event in _PRIVATE_PROGRESS_EVENTS:
            return
        note = preview or kwargs.get("message") or kwargs.get("text") or event
        note = _redact_text(note, limit=1000).strip()
        if not note:
            return
        payload: dict[str, Any] = {"note": note, "source": "tool_progress"}
        if name:
            payload["name"] = _redact_text(name, limit=200)
        if event and event not in {"status", "progress"}:
            payload["event_type"] = _redact_text(event, limit=200)
        self._record("progress_note", payload)

    def assistant_text(self, text: str, *, source: str) -> None:
        visible = _redact_text(text, limit=_MAX_TEXT).strip()
        if not visible:
            return
        seen_key = " ".join(visible.split())
        if seen_key in self._seen_assistant_texts:
            return
        self._seen_assistant_texts.add(seen_key)
        self._record(
            "assistant_text",
            {"text": visible, "source": source},
        )

    def interim_assistant(self, text: str, **kwargs: Any) -> None:
        self.assistant_text(text, source="interim_assistant")


def install_on_agent(agent: Any) -> Optional[KanbanWorkerJournal]:
    """Compose Kanban journaling callbacks onto an AIAgent instance.

    Returns the journal when installed, ``None`` outside dispatcher-spawned
    workers. Existing callbacks are preserved and invoked first.
    """
    journal = KanbanWorkerJournal.from_env()
    if journal is None or agent is None:
        return None
    if getattr(agent, "_kanban_worker_journal_installed", False):
        return getattr(agent, "_kanban_worker_journal", journal)

    old_start = getattr(agent, "tool_start_callback", None)
    old_complete = getattr(agent, "tool_complete_callback", None)
    old_progress = getattr(agent, "tool_progress_callback", None)
    old_interim = getattr(agent, "interim_assistant_callback", None)
    old_stream = getattr(agent, "stream_delta_callback", None)
    stream_buf: list[str] = []

    def flush_stream_assistant_text() -> None:
        if not stream_buf:
            return
        text = "".join(stream_buf)
        stream_buf.clear()
        journal.assistant_text(text, source="stream_delta")

    def tool_start(tool_call_id: str, name: str, args: Any) -> None:
        _safe_call(old_start, tool_call_id, name, args)
        journal.tool_start(tool_call_id, name, args)

    def tool_complete(tool_call_id: str, name: str, args: Any, result: Any) -> None:
        _safe_call(old_complete, tool_call_id, name, args, result)
        journal.tool_complete(tool_call_id, name, args, result)

    def tool_progress(
        event_type: str,
        name: str | None = None,
        preview: str | None = None,
        args: Any = None,
        **kwargs: Any,
    ) -> None:
        _safe_call(old_progress, event_type, name, preview, args, **kwargs)
        journal.tool_progress(event_type, name=name, preview=preview, args=args, **kwargs)

    def interim_assistant(text: str, **kwargs: Any) -> None:
        _safe_call(old_interim, text, **kwargs)
        journal.interim_assistant(text, **kwargs)

    def stream_delta(delta: Any) -> None:
        _safe_call(old_stream, delta)
        if isinstance(delta, str) and delta:
            stream_buf.append(delta)
        elif delta is None:
            flush_stream_assistant_text()

    agent.tool_start_callback = tool_start
    agent.tool_complete_callback = tool_complete
    agent.tool_progress_callback = tool_progress
    agent.interim_assistant_callback = interim_assistant
    if old_stream is not None:
        agent.stream_delta_callback = stream_delta
    agent._kanban_worker_journal_flush = flush_stream_assistant_text
    agent._kanban_worker_journal = journal
    agent._kanban_worker_journal_installed = True
    return journal


def record_final_assistant(agent: Any, text: str) -> None:
    """Flush buffered stream text and persist a final visible response if needed."""
    if agent is None:
        return
    flush = getattr(agent, "_kanban_worker_journal_flush", None)
    if callable(flush):
        try:
            flush()
        except Exception:
            _LOG.debug("kanban worker stream flush failed", exc_info=True)
    journal = getattr(agent, "_kanban_worker_journal", None)
    if isinstance(journal, KanbanWorkerJournal):
        journal.assistant_text(text, source="final_response")
