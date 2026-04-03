"""Matrix collapsible introspection fields for agent reasoning and tool activity.

Lossless, reviewable, rate-limited live introspection for Matrix agent runs.
Uses stable Matrix primitives only:
  - m.room.message with org.matrix.custom.html + <details><summary>
  - Live message edits via m.replace relation + m.new_content
"""

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
    """Tracks one active introspection field per task/kind."""

    room_id: str
    event_id: str
    task_id: str
    started_at: float
    last_update: float
    step_count: int = 0
    content_lines: list = field(default_factory=list)
    finalized: bool = False
    field_kind: str = "thinking"
    title: str = "Hermes Agent"
    summary: str = ""
    model_label: str = ""
    dirty: bool = False
    flush_task: Optional[asyncio.Task] = None


class ThinkingManager:
    """Manages buffered Matrix introspection fields.

    Supports two lossless buffered field types:
      - thinking
      - tools
    """

    def __init__(self, adapter: "MatrixAdapter"):
        self._adapter = adapter
        self._sessions: Dict[str, ThinkingSession] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(
        self,
        room_id: str,
        task_id: str,
        initial_summary: str = "Processing request...",
        *,
        field_kind: str = "thinking",
        model_label: str = "",
        initial_content_md: str = "",
    ) -> Optional[str]:
        """Send initial field and return event_id, or None on failure."""
        import nio

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
        title = self._field_title(field_kind)
        initial_lines = [initial_content_md] if initial_content_md else []
        html_body = self._build_html(
            summary=initial_summary,
            step=0,
            ts=now,
            content_html=self._lines_to_html(initial_lines),
            open_tag=True,
            field_kind=field_kind,
            model_label=model_label,
        )
        plaintext = self._plaintext_summary(title, initial_summary, model_label=model_label)
        content = self._msg_content(html_body, plaintext)

        try:
            resp = await asyncio.wait_for(
                self._adapter._client.room_send(
                    room_id,
                    "m.room.message",
                    content,
                    ignore_unverified_devices=True,
                ),
                timeout=30,
            )
        except Exception as exc:
            logger.error("Matrix introspection: failed to start %s in %s: %s", field_kind, room_id, exc)
            return None

        if not isinstance(resp, nio.RoomSendResponse):
            logger.error(
                "Matrix introspection: unexpected response %s",
                getattr(resp, "message", resp),
            )
            return None

        session = ThinkingSession(
            room_id=room_id,
            event_id=resp.event_id,
            task_id=task_id,
            started_at=now,
            last_update=now,
            field_kind=field_kind,
            title=title,
            summary=initial_summary,
            model_label=model_label,
            content_lines=initial_lines,
        )
        async with self._lock:
            self._sessions[key] = session

        logger.info(
            "Matrix introspection started in %s (task=%s kind=%s event=%s)",
            room_id,
            task_id,
            field_kind,
            resp.event_id,
        )
        return resp.event_id

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
        """Append data losslessly and flush on a rate-limited schedule."""
        key = self._session_key(task_id, field_kind)
        snapshot = None

        async with self._lock:
            key, session = self._lookup_session_locked(task_id, field_kind)
            if not session or session.finalized:
                return

            if step_info:
                session.summary = step_info
            if model_label:
                session.model_label = model_label
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
                delay = max(0.0, _MIN_EDIT_INTERVAL - elapsed)
                self._ensure_flush_locked(key, session, delay)

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
        """Flush all buffered data, then collapse the field."""
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
        """Abort a field while preserving all buffered content."""
        await self.finalize(
            task_id,
            f"⚠️ {reason}",
            collapse=True,
            field_kind=field_kind,
            model_label=model_label,
        )

    def has_session(self, task_id: str, field_kind: str = "thinking") -> bool:
        key = self._session_key(task_id, field_kind)
        return key in self._sessions or (field_kind == "thinking" and task_id in self._sessions)

    async def cleanup_stale(self, max_age: float = 1800) -> None:
        now = time.time()
        async with self._lock:
            stale = [
                key
                for key, session in self._sessions.items()
                if now - session.started_at > max_age
            ]
            for key in stale:
                session = self._sessions.pop(key, None)
                if session and session.flush_task and not session.flush_task.done():
                    session.flush_task.cancel()
                logger.warning("Matrix introspection: cleaned up stale session %s", key)

    async def abort_all(self, reason: str = "Gateway restarting") -> None:
        """Abort all active fields before shutdown/restart to avoid dangling blocks."""
        async with self._lock:
            snapshots = [
                (key, session.task_id, session.field_kind, session.flush_task)
                for key, session in self._sessions.items()
            ]

        for _key, task_id, field_kind, flush_task in snapshots:
            if flush_task and not flush_task.done():
                flush_task.cancel()
                try:
                    await flush_task
                except asyncio.CancelledError:
                    pass
            await self.abort(task_id, reason, field_kind=field_kind)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _session_key(task_id: str, field_kind: str) -> str:
        return f"{task_id}:{field_kind}"

    def _lookup_session_locked(
        self, task_id: str, field_kind: str
    ) -> tuple[str, Optional[ThinkingSession]]:
        key = self._session_key(task_id, field_kind)
        session = self._sessions.get(key)
        if session is None and field_kind == "thinking":
            session = self._sessions.get(task_id)
            if session is not None:
                key = task_id
        return key, session

    @staticmethod
    def _field_title(field_kind: str) -> str:
        return "Tool Activity" if field_kind == "tools" else "Hermes Agent"

    @staticmethod
    def _field_icon(field_kind: str) -> str:
        return "🛠️" if field_kind == "tools" else "🤔"

    def _snapshot_locked(self, session: ThinkingSession) -> Dict[str, Any]:
        return {
            "room_id": session.room_id,
            "event_id": session.event_id,
            "task_id": session.task_id,
            "started_at": session.started_at,
            "step": session.step_count,
            "summary": session.summary,
            "content_lines": list(session.content_lines),
            "field_kind": session.field_kind,
            "title": session.title,
            "model_label": session.model_label,
        }

    def _ensure_flush_locked(self, key: str, session: ThinkingSession, delay: float) -> None:
        if session.flush_task is None or session.flush_task.done():
            session.flush_task = asyncio.create_task(self._delayed_flush(key, delay))

    async def _delayed_flush(self, key: str, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
            await self._flush(key)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Matrix introspection: delayed flush failed for %s: %s", key, exc)

    async def _flush(self, key: str) -> None:
        snapshot = None
        async with self._lock:
            session = self._sessions.get(key)
            if not session or session.finalized:
                return
            session.flush_task = None
            if not session.dirty:
                return
            snapshot = self._snapshot_locked(session)
            session.last_update = time.time()
            session.dirty = False

        await self._send_edit_snapshot(snapshot)

    async def _send_edit_snapshot(
        self,
        snapshot: Dict[str, Any],
        *,
        final: bool = False,
        collapse: bool = True,
    ) -> None:
        elapsed = self._elapsed_str(snapshot["started_at"])
        content_html = self._lines_to_html(snapshot["content_lines"])
        summary = snapshot["summary"] or ("Complete" if final else "Working")
        open_tag = not final or not collapse

        html_body = self._build_html(
            summary=summary,
            step=snapshot["step"],
            ts=time.time(),
            content_html=content_html,
            open_tag=open_tag,
            elapsed=elapsed,
            final=final,
            field_kind=snapshot["field_kind"],
            model_label=snapshot.get("model_label") or "",
        )
        plaintext = self._plaintext_summary(
            snapshot["title"],
            summary,
            elapsed=elapsed,
            model_label=snapshot.get("model_label") or "",
        )
        edit_content = self._edit_content(snapshot["event_id"], html_body, plaintext)

        try:
            await asyncio.wait_for(
                self._adapter._client.room_send(
                    snapshot["room_id"],
                    "m.room.message",
                    edit_content,
                    ignore_unverified_devices=True,
                ),
                timeout=15,
            )
        except Exception as exc:
            logger.debug(
                "Matrix introspection: edit failed for task %s kind %s: %s",
                snapshot["task_id"],
                snapshot["field_kind"],
                exc,
            )
            # Mark dirty again for non-final sessions so later updates/finalization retry.
            if not final:
                key = self._session_key(snapshot["task_id"], snapshot["field_kind"])
                async with self._lock:
                    session = self._sessions.get(key)
                    if session and not session.finalized:
                        session.dirty = True
                        self._ensure_flush_locked(key, session, _MIN_EDIT_INTERVAL)

    # ------------------------------------------------------------------
    # HTML generation
    # ------------------------------------------------------------------

    def _build_html(
        self,
        summary: str,
        step: int,
        ts: float,
        content_html: str,
        open_tag: bool = True,
        elapsed: str = "",
        final: bool = False,
        field_kind: str = "thinking",
        model_label: str = "",
    ) -> str:
        open_attr = " open" if open_tag else ""
        timestamp = time.strftime("%H:%M:%S", time.localtime(ts))
        step_info = f"Step {step}" if step > 0 else "Starting"
        elapsed_info = f" • {elapsed}" if elapsed else ""
        icon = self._field_icon(field_kind)
        title = self._field_title(field_kind)

        if len(content_html.encode("utf-8")) > _MAX_BODY_SIZE:
            content_html = content_html[:_MAX_BODY_SIZE] + "\n… (truncated)"

        meta_html = (
            f"<p><em>Model: {html.escape(model_label)}</em></p>" if model_label else ""
        )
        details_body = meta_html
        if content_html:
            details_body += f"<pre><code>{content_html}</code></pre>"

        return (
            f"<details{open_attr}>"
            f"<summary>{icon} <strong>{html.escape(title)}</strong> "
            f"({step_info}{elapsed_info} • {timestamp}) — "
            f"{html.escape(summary)}</summary>"
            f"{details_body}"
            f"</details>"
        )

    def _lines_to_html(self, lines: list) -> str:
        if not lines:
            return ""
        return "\n".join(html.escape(line) for line in lines)

    @staticmethod
    def _elapsed_str(started_at: float) -> str:
        elapsed = time.time() - started_at
        if elapsed < 60:
            return f"{elapsed:.0f}s"
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        return f"{minutes}m{seconds}s"

    def _plaintext_summary(
        self,
        title: str,
        summary: str,
        *,
        elapsed: str = "",
        model_label: str = "",
    ) -> str:
        icon = "🛠️" if title == "Tool Activity" else "🤔"
        parts = [f"{icon} {title} — {summary}"]
        if elapsed:
            parts.append(f"({elapsed})")
        if model_label:
            parts.append(f"[{model_label}]")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Message content builders
    # ------------------------------------------------------------------

    @staticmethod
    def _msg_content(html_body: str, plaintext: str) -> Dict[str, Any]:
        return {
            "msgtype": "m.text",
            "body": plaintext,
            "format": "org.matrix.custom.html",
            "formatted_body": html_body,
        }

    @staticmethod
    def _edit_content(original_event_id: str, html_body: str, plaintext: str) -> Dict[str, Any]:
        new_content = {
            "msgtype": "m.text",
            "body": plaintext,
            "format": "org.matrix.custom.html",
            "formatted_body": html_body,
        }
        return {
            **new_content,
            "body": f"* {plaintext}",
            "formatted_body": f"* {html_body}",
            "m.new_content": new_content,
            "m.relates_to": {
                "rel_type": "m.replace",
                "event_id": original_event_id,
            },
        }
