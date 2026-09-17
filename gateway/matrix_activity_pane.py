"""Per-turn coordinator for Matrix's sticky activity pane."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Optional

from gateway.matrix_tool_activity import matrix_tool_activity_bodies


logger = logging.getLogger(__name__)


class MatrixActivityPane:
    """Serialize one turn's Matrix activity into a single editable root."""

    def __init__(
        self,
        *,
        adapter: Any,
        chat_id: str,
        reply_to: Optional[str] = None,
        metadata: Optional[dict] = None,
        coalescing_enabled: bool = True,
    ) -> None:
        self.adapter = adapter
        self.chat_id = chat_id
        self.reply_to = reply_to
        self.metadata = dict(metadata or {})
        self.coalescing_enabled = bool(coalescing_enabled)
        self.root_event_id: Optional[str] = None
        self.activity_lines: list[str] = []
        self.footer: Optional[str] = None
        self.lock = asyncio.Lock()
        self.closed = False
        self.closing = False  # Desired terminal state; closed means acknowledged.
        self.publish_interval = 0.0
        self._last_publish = float("-inf")
        self._delivered_snapshot = None

    SEAL_ATTEMPTS = 3
    TRANSPORT_TIMEOUT = 2.0
    SEAL_RETRY_DELAY = 0.1

    async def append_activity(self, line: str, *, publish: bool = True) -> Any:
        """Append one status/tool label, optionally publishing the snapshot."""

        text = str(line or "").strip()
        if not text:
            return None
        return await self._coordinate(lambda: self.activity_lines.append(text), publish=publish)

    async def replace_activity(
        self, previous: str, replacement: str, *, publish: bool = True,
    ) -> Any:
        """Replace the latest matching activity line, or append if absent."""

        old = str(previous or "").strip()
        new = str(replacement or "").strip()
        if not new:
            return None

        def _mutate() -> None:
            for index in range(len(self.activity_lines) - 1, -1, -1):
                current = self.activity_lines[index]
                if current == old or current.startswith(f"{old} (\u00d7"):
                    self.activity_lines[index] = new
                    return
            self.activity_lines.append(new)

        return await self._coordinate(_mutate, publish=publish)

    async def set_footer(self, footer: str | None) -> Any:
        """Replace the heartbeat footer and publish the complete snapshot."""

        value = str(footer or "").strip() or None
        return await self._coordinate(lambda: setattr(self, "footer", value))

    async def close(self) -> None:
        """Stop mutations, then acknowledge a bounded, cancellation-safe seal.

        Failed delivery remains retryable by a later close(), never a new root.
        """
        self.closing = True
        task = asyncio.create_task(self._seal())
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        task.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _seal(self) -> None:
        async with self.lock:
            if self.closed:
                return
            target = (tuple(self.activity_lines), None)
            if self._delivered_snapshot == target or (
                self.root_event_id is None and not self.activity_lines
                and self.footer is None
            ):
                self.footer = None
                self.closed = True
                return
            for attempt in range(self.SEAL_ATTEMPTS):
                result = await self._transport_locked(seal=True)
                if getattr(result, "success", False) and self.root_event_id:
                    self.footer = None
                    self.closed = True
                    return
                if attempt + 1 < self.SEAL_ATTEMPTS:
                    await asyncio.sleep(self.SEAL_RETRY_DELAY)
            logger.warning("Matrix activity pane seal delivery exhausted; remains pending")

    async def flush(self) -> Any:
        """Publish a pending snapshot when the shared edit interval permits."""
        async with self.lock:
            if self.closing or not self.coalescing_enabled:
                return None
            return await self._flush_locked()

    async def _flush_locked(self) -> Any:
        if self._delivered_snapshot == (tuple(self.activity_lines), self.footer):
            return None
        if not self.activity_lines and self.footer is None:
            return None
        if time.monotonic() - self._last_publish < self.publish_interval:
            return None
        self._last_publish = time.monotonic()
        return await self._transport_locked()

    async def _coordinate(self, mutate: Callable[[], None], *, publish: bool = True) -> Any:
        """Mutate and optionally publish under one turn-local lock."""

        async with self.lock:
            if self.closing or not self.coalescing_enabled:
                return None
            mutate()
            # Queue consumers absorb without I/O, so cancellation cannot leave
            # them unsure whether a dequeued label was applied before an await.
            if publish:
                return await self._flush_locked()
            return None

    async def _transport_locked(self, *, seal: bool = False) -> Any:
        """Send or edit the current snapshot. Caller MUST hold ``self.lock``."""

        lines = tuple(self.activity_lines)
        footer = None if seal else self.footer
        body, formatted_body = matrix_tool_activity_bodies(lines, footer)
        metadata = dict(self.metadata)
        metadata["matrix_formatted_body"] = formatted_body
        metadata["matrix_formatted_body_unprefixed"] = True
        metadata["_interim_send"] = True

        try:
            if self.root_event_id is None:
                result = await self._await_transport(
                    self.adapter.send(
                        chat_id=self.chat_id,
                        content=body,
                        reply_to=self.reply_to,
                        metadata=metadata,
                    )
                )
                if getattr(result, "success", False) and getattr(
                    result, "message_id", None
                ):
                    self.root_event_id = str(result.message_id)
                    self._delivered_snapshot = (lines, footer)
                return result

            kwargs = {
                "chat_id": self.chat_id,
                "message_id": self.root_event_id,
                "content": body,
                "metadata": metadata,
            }
            if getattr(self.adapter, "REQUIRES_EDIT_FINALIZE", False):
                kwargs["finalize"] = True
            result = await self._await_transport(self.adapter.edit_message(**kwargs))
            if getattr(result, "success", False):
                self._delivered_snapshot = (lines, footer)
            return result
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("Matrix activity pane transport failed", exc_info=True)
            return None

    async def _await_transport(self, awaitable: Any) -> Any:
        """Finish the Matrix call even if the consumer task is cancelled."""

        task = asyncio.ensure_future(asyncio.wait_for(awaitable, self.TRANSPORT_TIMEOUT))
        cancelled = False
        try:
            while True:
                try:
                    result = await asyncio.shield(task)
                    break
                except asyncio.CancelledError:
                    if task.cancelled():
                        raise
                    cancelled = True
            if (
                cancelled and self.root_event_id is None
                and getattr(result, "success", False)
                and getattr(result, "message_id", None)
            ):
                self.root_event_id = str(result.message_id)
                self._delivered_snapshot = (tuple(self.activity_lines), self.footer)
            return result
        finally:
            # A timeout/error after cancellation must not replace CancelledError:
            # the fail-soft transport handler would swallow it and revive the
            # progress loop that turn cleanup is waiting to finish.
            if cancelled:
                raise asyncio.CancelledError
