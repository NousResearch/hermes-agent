"""命令及系统文字回执统一使用静态卡片，媒体与交互提示仍走各自接口。"""

import json

import logging

from gateway.platforms.base import SendResult

from .cardkit_mutations import await_settled


logger = logging.getLogger(__name__)

class NoticeCardMixin:
    async def send(self, chat_id, content, reply_to=None, metadata=None):
        if not self._client:
            return SendResult(success=False, error="Not connected")

        async def publish():
            response = await self._feishu_send_with_retry(
                chat_id=chat_id,
                msg_type="interactive",
                payload=self._build_streaming_card_payload(content, finalize=True),
                reply_to=reply_to,
                metadata=metadata,
            )
            result = self._finalize_send_result(response, "回执卡片发送失败")
            if result.success and result.message_id:
                self._notice_cards[(str(chat_id), str(result.message_id))] = None
                while len(self._notice_cards) > 512:
                    self._notice_cards.popitem(last=False)
            return result

        try:
            return await await_settled(publish())
        except Exception as exc:
            logger.warning("Feishu receipt delivery failed", exc_info=True)
            return SendResult(success=False, error=str(exc))

    async def edit_message(self, chat_id, message_id, content, *, finalize=False, metadata=None):
        if (str(chat_id), str(message_id)) not in self._notice_cards:
            return await super().edit_message(chat_id, message_id, content, finalize=finalize)
        try:
            payload = json.loads(self._build_streaming_card_payload(content, finalize=True))
            await await_settled(self._patch_card_message(message_id, payload))
            return SendResult(success=True, message_id=message_id)
        except Exception as exc:
            logger.warning("Feishu receipt delivery failed", exc_info=True)
            return SendResult(success=False, message_id=message_id, error=str(exc))
