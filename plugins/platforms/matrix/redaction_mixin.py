"""Effective event invalidation for Matrix redactions."""

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from plugins.platforms.matrix.sync_transport import DurableSyncStore

if TYPE_CHECKING:
    from gateway.platforms.event import MessageEvent

logger = logging.getLogger(__name__)


class MatrixRedactionMixin:
    """Redaction handling for the Matrix adapter."""

    if TYPE_CHECKING:
        _client: Any
        _pending_text_batches: dict[str, MessageEvent]
        _text_batch_intakes: dict[int, list[tuple[str, asyncio.Future[bool]]]]

    async def _on_redaction(self, event: Any) -> None:
        room_id = str(getattr(event, "room_id", "") or "")
        target = str(getattr(event, "redacts", "") or "")
        if not target:
            content = getattr(event, "content", None)
            target = str(content.get("redacts") or "") if isinstance(content, dict) else ""
        if room_id and target:
            self._event_context_cache.redact(room_id, target)
            for action in self._reaction_followup_actions.values():
                if action.room_id == room_id:
                    action.pending.discard(target)

        sender = str(getattr(event, "sender", "") or "")
        if not (room_id and target and sender):
            return
        batches = tuple(self._pending_text_batches.items())
        withdrawn = self._parked_voices.discard(room_id, sender, target)
        withdrawn = self.withdraw_pending_message(target, chat_id=room_id, sender_id=sender) or withdrawn
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
            logger.info("Matrix: dropped queued message %s in %s; its sender redacted it", target, room_id)
