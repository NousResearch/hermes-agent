"""Buffered Matrix thinking/acting panes for agent introspection."""

from __future__ import annotations

import asyncio
import html
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from gateway.platforms.matrix import MatrixAdapter

logger = logging.getLogger(__name__)

_MIN_EDIT_INTERVAL = 3.0
_MAX_BODY_SIZE = 60_000


@dataclass
class ThinkingSession:
    room_id: str
    event_id: str
    task_id: str
    started_at: float
    last_update: float
    step_count: int = 0
    content_lines: list[str] = field(default_factory=list)
    finalized: bool = False
    field_kind: str = "thinking"
    title: str = "Agent Thinking: Hermes"
    summary: str = ""
    model_label: str = ""
    dirty: bool = False
    flush_task: Optional[asyncio.Task] = None
    thread_id: Optional[str] = None


class ThinkingManager:
    def __init__(self, adapter: "MatrixAdapter"):
        self._adapter = adapter
        self._sessions: Dict[str, ThinkingSession] = {}
        self._lock = asyncio.Lock()

    async def start(
        self,
        room_id: str,
        task_id: str,
        initial_summary: str = "Processing request...",
        *,
        field_kind: str = "thinking",
        model_label: str = "",
        initial_content_md: str = "",
        thread_id: Optional[str] = None,
    ) -> Optional[str]:
        key = self._session_key(task_id, field_kind)
        async with self._lock:
            _, existing = self._lookup_session_locked(task_id, field_kind)
            if existing and not existing.finalized:
                if model_label:
                    existing.model_label = model_label
                if initial_content_md:
                    existing.content_lines.append(initial_content_md)
                    existing.dirty = True
                    self._ensure_flush_locked(key, existing, 0.0)
                return existing.event_id

        now = time.time()
        title = self._field_title(field_kind, model_label)
        initial_lines = [initial_content_md] if initial_content_md else []
        html_body = self._build_html(
            title=title,
            summary=initial_summary,
            step=0,
            ts=now,
            content_html=self._lines_to_html(initial_lines),
            open_tag=True,
        )
        plaintext = self._plaintext_summary(title, initial_summary)
        content = self._msg_content(html_body, plaintext, thread_id=thread_id)

        try:
            raw_event_id = await asyncio.wait_for(
                self._adapter._client.send_message_event(room_id, "m.room.message", content),
                timeout=30,
            )
        except Exception as exc:
            logger.error("Matrix thinking: failed to start %s in %s: %s", field_kind, room_id, exc)
            return None

        event_id = str(getattr(raw_event_id, "event_id", raw_event_id) or "")
        if not event_id:
            return None

        session = ThinkingSession(
            room_id=room_id,
            event_id=event_id,
            task_id=task_id,
            started_at=now,
            last_update=now,
            field_kind=field_kind,
            title=title,
            summary=initial_summary,
            model_label=model_label,
            content_lines=initial_lines,
            thread_id=thread_id,
        )
        async with self._lock:
            self._sessions[key] = session
        return event_id

    async def update(
        self,
        task_id: str,
        step_info: str,
        content_md: str = "",
        *,
        field_kind: str = "thinking",
        model_label: Optional[str] = None,
        append_line: bool = True,
    ) -> None:
        snapshot = None
        key = self._session_key(task_id, field_kind)

        async with self._lock:
            key, session = self._lookup_session_locked(task_id, field_kind)
            if not session or session.finalized:
                return
            if step_info:
                session.summary = step_info
            if model_label:
                session.model_label = model_label
                session.title = self._field_title(field_kind, model_label)
            if content_md and append_line:
                session.content_lines.append(content_md)
            session.step_count += 1
            session.dirty = True

            now = time.time()
            elapsed = now - session.last_update
            if elapsed >= _MIN_EDIT_INTERVAL and (session.flush_task is None or session.flush_task.done()):
                snapshot = self._snapshot_locked(session)
                session.last_update = now
                session.dirty = False
            else:
                self._ensure_flush_locked(key, session, max(0.0, _MIN_EDIT_INTERVAL - elapsed))

        if snapshot:
            await self._send_edit_snapshot(snapshot)

    async def finalize(
        self,
        task_id: str,
        final_summary: str = "Task complete",
        collapse: bool = True,
        *,
        field_kind: str = "thinking",
        model_label: Optional[str] = None,
    ) -> None:
        key = self._session_key(task_id, field_kind)
        snapshot = None
        flush_task = None

        async with self._lock:
            key, session = self._lookup_session_locked(task_id, field_kind)
            if not session:
                return
            session.finalized = True
            if field_kind == "thinking" and final_summary and not final_summary.startswith(("✅", "⚠️")):
                session.summary = f"✅ {final_summary}"
            else:
                session.summary = final_summary
            if model_label:
                session.model_label = model_label
                session.title = self._field_title(field_kind, model_label)
            flush_task = session.flush_task
            session.flush_task = None
            snapshot = self._snapshot_locked(session)

        if flush_task and not flush_task.done():
            flush_task.cancel()
            try:
                await flush_task
            except asyncio.CancelledError:
                pass

        await self._send_edit_snapshot(snapshot, final=True, collapse=collapse)
        async with self._lock:
            self._sessions.pop(key, None)

    async def abort(
        self,
        task_id: str,
        reason: str = "Aborted",
        *,
        field_kind: str = "thinking",
        model_label: Optional[str] = None,
    ) -> None:
        await self.finalize(
            task_id,
            f"⚠️ {reason}",
            collapse=True,
            field_kind=field_kind,
            model_label=model_label,
        )

    async def abort_all(self, reason: str = "Gateway restarting") -> None:
        async with self._lock:
            pending = [(session.task_id, session.field_kind) for session in self._sessions.values()]
        for task_id, field_kind in pending:
            try:
                await self.abort(task_id, reason, field_kind=field_kind)
            except Exception as exc:
                logger.debug("Matrix thinking: abort_all failed for %s/%s: %s", task_id, field_kind, exc)

    def has_session(self, task_id: str, field_kind: str = "thinking") -> bool:
        return self._session_key(task_id, field_kind) in self._sessions

    @staticmethod
    def _field_title(field_kind: str, model_label: str = "") -> str:
        if field_kind == "tools":
            return "Agent Acting:"
        suffix = f" via {model_label}" if model_label else ""
        return f"Agent Thinking: Hermes{suffix}"

    @staticmethod
    def _plaintext_summary(title: str, summary: str) -> str:
        return f"{title}\n{summary}".strip()

    @staticmethod
    def _elapsed_str(started_at: float) -> str:
        elapsed = max(0, int(time.time() - started_at))
        minutes, seconds = divmod(elapsed, 60)
        if minutes:
            return f"{minutes}m{seconds}s"
        return f"{seconds}s"

    @staticmethod
    def _msg_content(formatted_body: str, body: str, *, thread_id: Optional[str] = None) -> Dict[str, Any]:
        content: Dict[str, Any] = {
            "msgtype": "m.text",
            "body": body,
            "format": "org.matrix.custom.html",
            "formatted_body": formatted_body,
        }
        if thread_id:
            content["m.relates_to"] = {
                "rel_type": "m.thread",
                "event_id": thread_id,
                "is_falling_back": True,
            }
        return content

    @staticmethod
    def _edit_content(event_id: str, formatted_body: str, body: str) -> Dict[str, Any]:
        return {
            "msgtype": "m.text",
            "body": f"* {body}",
            "format": "org.matrix.custom.html",
            "formatted_body": f"* {formatted_body}",
            "m.new_content": {
                "msgtype": "m.text",
                "body": body,
                "format": "org.matrix.custom.html",
                "formatted_body": formatted_body,
            },
            "m.relates_to": {
                "rel_type": "m.replace",
                "event_id": event_id,
            },
        }

    @staticmethod
    def _lines_to_html(lines: list[str]) -> str:
        if not lines:
            return ""
        escaped = [html.escape(line) for line in lines]
        return "<br/>".join(escaped)

    def _build_html(
        self,
        *,
        title: str,
        summary: str,
        step: int,
        ts: float,
        content_html: str,
        open_tag: bool = True,
    ) -> str:
        _ = ts
        summary_html = html.escape(summary or "")
        title_html = html.escape(title)
        details_open = " open" if open_tag else ""
        step_label = "Starting" if step == 0 else f"Update {step}"
        body = content_html or html.escape(summary or "")
        return (
            f"<details{details_open}><summary>{title_html}</summary>"
            f"<p><strong>{summary_html}</strong> · {step_label}</p>"
            f"<div>{body}</div>"
            f"</details>"
        )[:_MAX_BODY_SIZE]

    def _session_key(self, task_id: str, field_kind: str) -> str:
        return f"{task_id}:{field_kind}"

    def _lookup_session_locked(self, task_id: str, field_kind: str):
        key = self._session_key(task_id, field_kind)
        return key, self._sessions.get(key)

    def _snapshot_locked(self, session: ThinkingSession) -> Dict[str, Any]:
        return {
            "room_id": session.room_id,
            "event_id": session.event_id,
            "task_id": session.task_id,
            "field_kind": session.field_kind,
            "title": session.title,
            "summary": session.summary,
            "step_count": session.step_count,
            "content_html": self._lines_to_html(session.content_lines),
            "started_at": session.started_at,
            "thread_id": session.thread_id,
        }

    def _ensure_flush_locked(self, key: str, session: ThinkingSession, delay: float) -> None:
        if session.flush_task and not session.flush_task.done():
            return

        async def _flush_later() -> None:
            try:
                await asyncio.sleep(delay)
                async with self._lock:
                    current = self._sessions.get(key)
                    if not current or current.finalized or not current.dirty:
                        return
                    snapshot = self._snapshot_locked(current)
                    current.last_update = time.time()
                    current.dirty = False
                await self._send_edit_snapshot(snapshot)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Matrix thinking: deferred flush failed: %s", exc)

        session.flush_task = asyncio.create_task(_flush_later())

    async def _send_edit_snapshot(self, snapshot: Dict[str, Any], *, final: bool = False, collapse: bool = True) -> None:
        if not snapshot:
            return
        title = snapshot["title"]
        body = self._plaintext_summary(title, snapshot["summary"])
        formatted = self._build_html(
            title=title,
            summary=snapshot["summary"],
            step=snapshot["step_count"],
            ts=time.time(),
            content_html=snapshot["content_html"],
            open_tag=not (final and collapse),
        )
        content = self._edit_content(snapshot["event_id"], formatted, body)
        try:
            await self._adapter._client.send_message_event(snapshot["room_id"], "m.room.message", content)
        except Exception as exc:
            logger.debug("Matrix thinking: edit failed for %s: %s", snapshot["event_id"], exc)
