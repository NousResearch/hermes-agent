"""Per-turn coordinator for Matrix's sticky interim-commentary pane."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from gateway.matrix_commentary import matrix_commentary_bodies


logger = logging.getLogger(__name__)


class MatrixCommentaryPane:
    """Serialize one turn's completed commentary into a single editable root."""

    SEAL_ATTEMPTS = 3
    TRANSPORT_TIMEOUT = 2.0
    SEAL_RETRY_DELAY = 0.1

    def __init__(
        self,
        *,
        adapter: Any,
        chat_id: str,
        reply_to: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        self.adapter = adapter
        self.chat_id = chat_id
        self.reply_to = reply_to
        self.metadata = dict(metadata or {})
        self.root_event_id: Optional[str] = None
        self.items: list[str] = []
        self.lock = asyncio.Lock()
        self.closed = False
        self._pending: Optional[tuple[str, str, dict]] = None

    def _snapshot_payload(self) -> Optional[tuple[str, str, dict]]:
        if not self.items:
            return None
        body, html = matrix_commentary_bodies(self.items)
        metadata = dict(self.metadata)
        metadata["matrix_formatted_body"] = html
        metadata["matrix_formatted_body_unprefixed"] = True
        metadata["_interim_send"] = True
        return body, html, metadata

    async def append(self, text: str) -> Any:
        """Append one completed commentary item and publish the full snapshot."""
        value = str(text or "").strip()
        if not value:
            return None
        async with self.lock:
            if self.closed:
                return None
            self.items.append(value)
            return await self._publish_locked()

    async def close(self) -> None:
        """Leave the pane in the room; retry a pending snapshot on the same root."""
        async with self.lock:
            self.closed = True
            if self._pending is None:
                return
            for attempt in range(self.SEAL_ATTEMPTS):
                result = await self._transport_locked()
                if getattr(result, "success", False):
                    self._pending = None
                    return
                if attempt + 1 < self.SEAL_ATTEMPTS:
                    await asyncio.sleep(self.SEAL_RETRY_DELAY)
            logger.warning("Matrix commentary pane seal exhausted; remains pending")

    async def _publish_locked(self) -> Any:
        payload = self._snapshot_payload()
        if payload is None:
            return None
        self._pending = payload
        result = await self._transport_locked()
        if getattr(result, "success", False):
            self._pending = None
        return result

    async def _transport_locked(self) -> Any:
        payload = self._pending or self._snapshot_payload()
        if payload is None:
            return None
        body, _html, metadata = payload
        try:
            if self.root_event_id is None:
                result = await asyncio.wait_for(
                    self.adapter.send(
                        chat_id=self.chat_id,
                        content=body,
                        reply_to=self.reply_to,
                        metadata=metadata,
                    ),
                    self.TRANSPORT_TIMEOUT,
                )
                if getattr(result, "success", False) and getattr(result, "message_id", None):
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
            return await asyncio.wait_for(self.adapter.edit_message(**kwargs), self.TRANSPORT_TIMEOUT)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("Matrix commentary pane transport failed", exc_info=True)
            return None
