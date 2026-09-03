"""Per-turn coordinator for Matrix's sticky activity pane."""

from __future__ import annotations

import asyncio
import logging
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

    async def append_activity(self, line: str) -> Any:
        """Append one status/tool label and publish the complete snapshot."""

        text = str(line or "").strip()
        if not text:
            return None
        return await self._coordinate(lambda: self.activity_lines.append(text))

    async def replace_activity(self, previous: str, replacement: str) -> Any:
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

        return await self._coordinate(_mutate)

    async def set_footer(self, footer: str | None) -> Any:
        """Replace the heartbeat footer and publish the complete snapshot."""

        value = str(footer or "").strip() or None
        return await self._coordinate(lambda: setattr(self, "footer", value))

    async def close(self) -> None:
        """Seal the pane after any in-flight publish; drop the live footer."""

        async with self.lock:
            if self.closed:
                return
            if self.footer is not None:
                self.footer = None
                if self.root_event_id:
                    await self._transport_locked()
            self.closed = True

    async def _coordinate(self, mutate: Callable[[], None]) -> Any:
        """Mutate, snapshot, render, and transport under one turn-local lock."""

        async with self.lock:
            if self.closed or not self.coalescing_enabled:
                return None
            mutate()
            return await self._transport_locked()

    async def _transport_locked(self) -> Any:
        """Send or edit the current snapshot. Caller MUST hold ``self.lock``."""

        lines = tuple(self.activity_lines)
        footer = self.footer
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
                return result

            kwargs = {
                "chat_id": self.chat_id,
                "message_id": self.root_event_id,
                "content": body,
                "metadata": metadata,
            }
            if getattr(self.adapter, "REQUIRES_EDIT_FINALIZE", False):
                kwargs["finalize"] = True
            return await self._await_transport(self.adapter.edit_message(**kwargs))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("Matrix activity pane transport failed", exc_info=True)
            return None

    async def _await_transport(self, awaitable: Any) -> Any:
        """Finish the Matrix call even if the consumer task is cancelled."""

        task = asyncio.ensure_future(awaitable)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            result = await task
            if (
                self.root_event_id is None
                and getattr(result, "success", False)
                and getattr(result, "message_id", None)
            ):
                self.root_event_id = str(result.message_id)
            raise
