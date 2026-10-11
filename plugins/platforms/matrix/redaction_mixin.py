"""Effective event invalidation for Matrix redactions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from urllib.parse import quote
from plugins.platforms.matrix.client_events import Method
import logging
from typing import TYPE_CHECKING, Any

from plugins.platforms.matrix.sync_transport import DurableSyncStore

if TYPE_CHECKING:
    from gateway.platforms.base import BasePlatformAdapter
    from gateway.platforms.event import MessageEvent
    from plugins.platforms.matrix.followup_mixin import _MatrixFollowupChoice
    from plugins.platforms.matrix.reply_context import MatrixEventContextCache
    from plugins.platforms.matrix.voice_mention import ParkedVoices

logger = logging.getLogger(__name__)


def _redacted_event_id(event: Any) -> str:
    """The event ID that an ``m.room.redaction`` event redacts, or an empty string."""
    content = event.get("content") if isinstance(event, dict) else getattr(event, "content", None)
    if isinstance(event, dict) and content is not None and not isinstance(content, dict):
        return ""
    target = event.get("redacts") if isinstance(event, dict) else getattr(event, "redacts", None)
    # Room version 11 moved ``redacts`` into the content.
    return str(target or (content.get("redacts") if content else None) or "")


@dataclass(frozen=True)
class _PendingRedaction:
    room_id: str
    target: str
    actor: str
    event_id: str

    def confirmed_author(self, current: Any) -> str | None:
        if not isinstance(current, dict) or current.get("event_id") != self.target or current.get("room_id") != self.room_id:
            return None
        unsigned = current.get("unsigned")
        redaction = unsigned.get("redacted_because") if isinstance(unsigned, dict) else None
        if (not isinstance(redaction, dict) or redaction.get("type") != "m.room.redaction"
                or redaction.get("room_id", self.room_id) != self.room_id
                or redaction.get("event_id") != self.event_id or redaction.get("sender") != self.actor
                or _redacted_event_id(redaction) != self.target):
            return None
        author = current.get("sender")
        return author if isinstance(author, str) and author else None



class MatrixRedactionMixin:
    """Redaction handling for the Matrix adapter."""

    if TYPE_CHECKING:
        _client: Any
        _pending_text_batches: dict[str, MessageEvent]
        _text_batch_intakes: dict[int, list[tuple[str, asyncio.Future[bool]]]]
        _event_context_cache: MatrixEventContextCache
        _reaction_followup_actions: dict[str, _MatrixFollowupChoice]
        _parked_voices: ParkedVoices
        withdraw_pending_message = BasePlatformAdapter.withdraw_pending_message

    async def _on_redaction(self, event: Any) -> None:
        room_id = str(getattr(event, "room_id", "") or "")
        target = _redacted_event_id(event)
        if room_id and target:
            self._event_context_cache.redact(room_id, target)
            for action in self._reaction_followup_actions.values():
                if action.room_id == room_id:
                    action.pending.discard(target)

        if room_id and target:
            sender = str(getattr(event, "sender", "") or "")
            if await self._withdraw_redacted_message(room_id, sender, target):
                return
            await self._withdraw_confirmed_redaction(_PendingRedaction(
                room_id, target, sender, str(getattr(event, "event_id", "") or "")))

    async def _withdraw_redacted_message(
        self, room_id: str, sender: str, target: str
    ) -> bool:
        """Drop input that its sender redacted before its turn started."""
        batches = tuple(self._pending_text_batches.items())
        withdrawn = self._parked_voices.discard(room_id, sender, target)
        withdrawn = (
            self.withdraw_pending_message(target, chat_id=room_id, sender_id=sender)
            or withdrawn
        )
        removed = []
        for key, previous in batches:
            remaining = self._pending_text_batches.get(key)
            if remaining is previous:
                continue
            receipts = self._text_batch_intakes.pop(id(previous), [])
            retained_ids = (
                {remaining.message_id, *remaining.merged_message_ids}
                if remaining is not None
                else set()
            )
            retained = []
            for event_id, receipt in receipts:
                if event_id in retained_ids:
                    retained.append((event_id, receipt))
                    continue
                removed.append((event_id, receipt))
            if retained and remaining is not None:
                self._text_batch_intakes.setdefault(id(remaining), []).extend(retained)
        if removed:
            store = getattr(self._client, "sync_store", None)
            try:
                if isinstance(store, DurableSyncStore):
                    await store.accept_intakes(
                        tuple(event_id for event_id, _ in removed)
                    )
            except BaseException as exc:
                for _event_id, receipt in removed:
                    if receipt.done():
                        continue
                    if isinstance(exc, asyncio.CancelledError):
                        receipt.cancel()
                        continue
                    receipt.set_exception(exc)
                raise
            for _event_id, receipt in removed:
                if not receipt.done():
                    receipt.set_result(True)
        if withdrawn:
            logger.info(
                "Matrix: dropped queued message %s in %s; its sender redacted it",
                target,
                room_id,
            )
        return withdrawn

    async def _withdraw_confirmed_redaction(self, redaction: _PendingRedaction) -> None:
        if self._client is None or not redaction.actor or not redaction.event_id:
            return
        cached = self._event_context_cache.history_entry(redaction.room_id, redaction.target)
        if cached is not None and cached.sender == redaction.actor:
            return
        path = (f"/_matrix/client/v3/rooms/{quote(redaction.room_id, safe='')}"
                f"/event/{quote(redaction.target, safe='')}")
        try:
            current = await asyncio.wait_for(self._client.api.request(Method.GET, path),
                                             self._event_context_cache.timeout_seconds)
        except Exception:
            logger.debug("Matrix: could not verify redaction %s", redaction.event_id, exc_info=True)
            return
        author = redaction.confirmed_author(current)
        if author is not None:
            await self._withdraw_redacted_message(redaction.room_id, author, redaction.target)
