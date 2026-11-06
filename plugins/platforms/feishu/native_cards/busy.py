"""仅在官方 busy handler 的动态作用域内合卡，不根据回执文字猜测。"""

from contextvars import ContextVar
from functools import wraps

import logging

from gateway.platforms.base import SendResult


logger = logging.getLogger(__name__)

class BusyCardMixin:
    def set_busy_session_handler(self, handler):
        if handler is None:
            return super().set_busy_session_handler(None)
        scope = ContextVar("feishu_cards_busy_delivery", default=None)
        self._busy_card_scope = scope

        @wraps(handler)
        async def scoped(event, session_key):
            token = scope.set((event, session_key))
            try:
                return await handler(event, session_key)
            finally:
                scope.reset(token)

        return super().set_busy_session_handler(scoped)

    def _busy_card_target(self, chat_id, reply_to, metadata):
        scope = getattr(self, "_busy_card_scope", None)
        bound = scope.get() if scope is not None else None
        if bound is None:
            return None
        event, session_key = bound
        source = event.source
        if (
            getattr(event, "internal", False)
            or str(event.text or "").lstrip().startswith("/")
            or str(chat_id) != str(source.chat_id)
            or str(reply_to or "") != str(event.message_id)
            or str((metadata or {}).get("thread_id") or "") != str(source.thread_id or "")
        ):
            return None
        owners = {str(v) for v in (source.user_id, getattr(source, "user_id_alt", None)) if v}
        candidates = [
            turn
            for turn in self._card_turns.values()
            if (
                turn.session_key == session_key
                and turn.chat_id == str(chat_id)
                and turn.thread_id == str(source.thread_id or "")
                and turn.reply_to != str(event.message_id)
                and not turn.final
                and not turn.cancelled
                and owners.intersection(turn.owner_ids)
            )
        ]
        return candidates[0] if len(candidates) == 1 else None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        turn = self._busy_card_target(chat_id, reply_to, metadata)
        if turn is None:
            return await super().send(chat_id, content, reply_to=reply_to, metadata=metadata)
        result = await self._send_busy_card(turn, content)
        if result is not None:
            return result
        return await super().send(chat_id, content, reply_to=reply_to, metadata=metadata)

    async def _send_busy_card(self, turn, content):
        # 使用原卡传输及取消屏障；busy 提示不替换回答，也不提前封卡。
        async with turn.io_lock:
            if turn.final or turn.cancelled:
                return None
            from .cardkit_mutations import await_settled

            try:
                await await_settled(self._write_busy_card(turn, content))
                return SendResult(success=True, message_id=turn.message_id)
            except Exception as exc:
                logger.warning("Feishu card delivery failed", exc_info=True)
                return SendResult(success=False, message_id=turn.message_id or None, error=str(exc))

    async def _write_busy_card(self, turn, content):
        await self._ensure_card(turn)
        text = "\n".join(
            self._split_streaming_card_content(self._stream_payload(turn) + "\n\n" + content)
        )
        if turn.fallback:
            import json

            await self._patch_card_message(
                turn.message_id, json.loads(self._build_streaming_card_payload(text))
            )
        else:
            await self._stream_cardkit_content_with_recovery(turn.card_id, text)
