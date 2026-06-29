"""Structured activity events for dispatcher-spawned Kanban workers.

Kanban workers run as standalone ``hermes chat -q ...`` subprocesses. Their rich
chat/tool stream is not automatically visible to the Kanban WebUI, so this module
bridges existing agent callbacks into durable ``task_events`` rows.

The bridge is intentionally best-effort: it must never break the agent turn when
SQLite is locked, the board disappeared, or the process is not a Kanban worker.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Optional

SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "cookie",
    "credential",
    "password",
    "secret",
    "token",
)


class KanbanWorkerActivityJournal:
    """Persist observable worker progress as structured Kanban task events."""

    EVENT_KINDS = frozenset(
        {
            "tool_start",
            "tool_end",
            "assistant_text",
            "progress_note",
            "heartbeat_note",
            "steer_accepted",
        }
    )

    def __init__(self, *, task_id: str, run_id: Optional[int] = None):
        self.task_id = str(task_id or "").strip()
        self.run_id = int(run_id) if run_id is not None else None
        self._assistant_buffer: list[str] = []
        self._last_stream_emit = 0.0
        self._steer_cursor: Optional[int] = None
        self._steer_stop: Optional[threading.Event] = None

    @classmethod
    def from_environment(cls) -> Optional["KanbanWorkerActivityJournal"]:
        task_id = (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
        if not task_id:
            return None
        run_id = None
        raw_run_id = (os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
        if raw_run_id:
            try:
                run_id = int(raw_run_id)
            except ValueError:
                run_id = None
        return cls(task_id=task_id, run_id=run_id)

    def _record(self, kind: str, payload: Optional[dict[str, Any]] = None) -> None:
        if not self.task_id or kind not in self.EVENT_KINDS:
            return
        safe_payload = self._sanitize_payload(payload or {})
        safe_payload.setdefault("source", "worker_activity")
        safe_payload.setdefault("recorded_at", int(time.time()))
        try:
            from hermes_cli import kanban_db as kb

            conn = kb.connect()
            try:
                with kb.write_txn(conn):
                    kb._append_event(
                        conn,
                        self.task_id,
                        kind,
                        safe_payload,
                        run_id=self.run_id,
                    )
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception:
            # Progress visibility must never interrupt the worker itself.
            return

    def tool_start(self, tool_call_id: str, tool: str, args: Optional[dict[str, Any]] = None) -> None:
        self._record(
            "tool_start",
            {
                "tool_call_id": tool_call_id,
                "tool": tool,
                "args": args or {},
            },
        )

    def tool_end(
        self,
        tool_call_id: str,
        tool: str,
        args: Optional[dict[str, Any]] = None,
        result: Any = None,
        *,
        duration: Optional[float] = None,
        is_error: Optional[bool] = None,
    ) -> None:
        payload: dict[str, Any] = {
            "tool_call_id": tool_call_id,
            "tool": tool,
            "args": args or {},
            "result_preview": self._preview(result),
        }
        if duration is not None:
            try:
                payload["duration"] = round(float(duration), 3)
            except (TypeError, ValueError):
                pass
        if is_error is not None:
            payload["is_error"] = bool(is_error)
        self._record("tool_end", payload)

    def assistant_text(self, text: str) -> None:
        text = str(text or "").strip()
        if not text:
            return
        self._record("assistant_text", {"text": self._truncate(text, 4000)})

    def progress_note(self, note: str, **extra: Any) -> None:
        note = str(note or "").strip()
        if not note:
            return
        payload = {"note": self._truncate(note, 1000)}
        payload.update(extra)
        self._record("progress_note", payload)

    def heartbeat_note(self, note: str = "active", **extra: Any) -> None:
        payload = {"note": self._truncate(str(note or "active"), 1000)}
        payload.update(extra)
        self._record("heartbeat_note", payload)

    def tool_progress(self, event_type: str, name: str = "", preview: Any = None, args: Any = None, **kwargs: Any) -> None:
        if event_type == "tool.started":
            self.progress_note(
                f"tool started: {name}",
                tool=name,
                preview=self._preview(preview),
                args=args or {},
            )
        elif event_type == "tool.completed":
            self.progress_note(
                f"tool completed: {name}",
                tool=name,
                duration=kwargs.get("duration"),
                is_error=kwargs.get("is_error"),
                result_preview=self._preview(kwargs.get("result")),
            )
        elif event_type == "reasoning.available":
            # Do not persist chain-of-thought/reasoning. A coarse note is enough
            # to show the worker is active without exposing private reasoning.
            self.progress_note("reasoning available")
        else:
            self.progress_note(str(event_type or "progress"), tool=name, preview=self._preview(preview))

    def tool_complete(self, tool_call_id: str, tool: str, args: Optional[dict[str, Any]] = None, result: Any = None) -> None:
        self.tool_end(tool_call_id, tool, args or {}, result)

    def stream_delta(self, text: str) -> None:
        text = str(text or "")
        if not text:
            return
        self._assistant_buffer.append(text)
        now = time.monotonic()
        buffered = "".join(self._assistant_buffer)
        if len(buffered) >= 500 or "\n" in text or now - self._last_stream_emit >= 5.0:
            self._assistant_buffer.clear()
            self._last_stream_emit = now
            self.assistant_text(buffered)

    def flush_assistant_text(self) -> None:
        if not self._assistant_buffer:
            return
        buffered = "".join(self._assistant_buffer)
        self._assistant_buffer.clear()
        self.assistant_text(buffered)

    def _latest_event_id(self) -> int:
        try:
            from hermes_cli import kanban_db as kb

            conn = kb.connect()
            try:
                row = conn.execute("SELECT COALESCE(MAX(id), 0) AS latest FROM task_events").fetchone()
                return int(row["latest"] or 0)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception:
            return 0

    def consume_steer_once(self, agent: Any) -> int:
        """Consume pending ``steer_note`` events and inject them via ``agent.steer``.

        This is the Kanban equivalent of the chat `/steer` control: it does not
        interrupt the running tool, but the next model iteration receives the
        user note through Hermes Agent's existing pending-steer mechanism.
        """
        if not self.task_id or agent is None or not hasattr(agent, "steer"):
            return 0
        cursor = int(self._steer_cursor or 0)
        rows = []
        try:
            from hermes_cli import kanban_db as kb

            conn = kb.connect()
            try:
                rows = conn.execute(
                    "SELECT id, run_id, payload FROM task_events "
                    "WHERE task_id = ? AND kind = 'steer_note' AND id > ? "
                    "ORDER BY id ASC LIMIT 20",
                    (self.task_id, cursor),
                ).fetchall()
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception:
            return 0

        accepted = 0
        for row in rows:
            try:
                event_id = int(row["id"])
                cursor = max(cursor, event_id)
                row_run_id = row["run_id"]
                if self.run_id is not None and row_run_id is not None and int(row_run_id) != self.run_id:
                    continue
                payload = json.loads(row["payload"]) if row["payload"] else {}
                if not isinstance(payload, dict):
                    continue
                message = str(payload.get("message") or payload.get("text") or payload.get("note") or "").strip()
                if not message:
                    continue
                if agent.steer(message):
                    accepted += 1
                    self._record(
                        "steer_accepted",
                        {
                            "message": self._truncate(message, 1000),
                            "steer_event_id": event_id,
                        },
                    )
            except Exception:
                continue
        self._steer_cursor = cursor
        return accepted

    def start_steer_polling(self, agent: Any, *, interval: float = 2.0) -> Optional[threading.Event]:
        """Start a daemon poller that injects WebUI steer notes into the worker."""
        if not self.task_id or agent is None or not hasattr(agent, "steer"):
            return None
        if getattr(self, "_steer_stop", None) is not None:
            return self._steer_stop
        self._steer_cursor = self._latest_event_id()
        stop = threading.Event()
        self._steer_stop = stop

        def _loop() -> None:
            while not stop.is_set():
                try:
                    self.consume_steer_once(agent)
                except Exception:
                    pass
                stop.wait(max(0.5, float(interval or 2.0)))

        thread = threading.Thread(target=_loop, name="kanban-worker-steer", daemon=True)
        thread.start()
        return stop

    @classmethod
    def _sanitize_payload(cls, value: Any) -> Any:
        if isinstance(value, dict):
            out = {}
            for key, item in value.items():
                key_str = str(key)
                if any(part in key_str.lower() for part in SENSITIVE_KEY_PARTS):
                    out[key_str] = "[REDACTED]"
                else:
                    out[key_str] = cls._sanitize_payload(item)
            return out
        if isinstance(value, (list, tuple)):
            return [cls._sanitize_payload(item) for item in value]
        if isinstance(value, str):
            return cls._truncate(value, 4000)
        return value

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        text = str(text)
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)] + "…"

    @classmethod
    def _preview(cls, value: Any, limit: int = 2000) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return cls._truncate(value.strip(), limit)
        try:
            import json

            return cls._truncate(json.dumps(cls._sanitize_payload(value), ensure_ascii=False), limit)
        except Exception:
            return cls._truncate(str(value), limit)
