"""同一逻辑发送复用 UUID，避免响应丢失后重试创建重复卡片。"""

from __future__ import annotations
import asyncio
import logging
from typing import Any, Dict, Optional
import uuid
from lark_oapi.api.im.v1.model import (
    ReplyMessageRequest,
    ReplyMessageRequestBody,
    CreateMessageRequest,
    CreateMessageRequestBody,
)
from plugins.platforms.feishu.adapter import (
    _FEISHU_SEND_ATTEMPTS,
    _FEISHU_REPLY_FALLBACK_CODES,
    _POST_CONTENT_INVALID_RE,
)

logger = logging.getLogger(__name__)


def _sdk_build(request_cls, **fields):
    builder = request_cls.builder()
    for name, value in fields.items():
        builder = getattr(builder, name)(value)
    return builder.build()


class MessageSendMixin:
    async def _send_raw_message(
        self,
        *,
        chat_id: str,
        msg_type: str,
        payload: str,
        reply_to: Optional[str],
        metadata: Optional[dict[str, Any]],
    ) -> Any:
        thread_id = (metadata or {}).get("thread_id")
        effective_reply_to = reply_to or (
            (metadata or {}).get("reply_to_message_id") if thread_id else None
        )
        if effective_reply_to:
            body = self._build_reply_message_body(
                content=payload,
                msg_type=msg_type,
                reply_in_thread=bool(thread_id),
                uuid_value=(metadata or {}).get("__hermes_send_uuid") or str(uuid.uuid4()),
            )
            request = self._build_reply_message_request(effective_reply_to, body)
            return await self._run_blocking(self._client.im.v1.message.reply, request)
        if thread_id:
            # reply→create fallback inside a topic: thread_id as receive_id keeps it in the topic.
            receive_id, receive_id_type = thread_id, "thread_id"
        elif chat_id.startswith("feishu_user_id:"):
            receive_id, receive_id_type = chat_id.split(":", 1)[1], "user_id"
        else:
            receive_id, receive_id_type = (
                chat_id,
                "open_id" if chat_id.startswith("ou_") else "chat_id",
            )
        body = self._build_create_message_body(
            receive_id=receive_id,
            msg_type=msg_type,
            content=payload,
            uuid_value=(metadata or {}).get("__hermes_send_uuid") or str(uuid.uuid4()),
        )
        request = self._build_create_message_request(receive_id_type, body)
        return await self._run_blocking(self._client.im.v1.message.create, request)

    async def _feishu_send_with_retry(
        self,
        *,
        chat_id: str,
        msg_type: str,
        payload: str,
        reply_to: Optional[str],
        metadata: Optional[dict[str, Any]],
    ) -> Any:
        last_error: Optional[Exception] = None
        active_reply_to = reply_to
        retry_metadata = dict(metadata or {})
        retry_metadata["__hermes_send_uuid"] = str(uuid.uuid4())

        async def _raw(reply_target: Optional[str]) -> Any:
            return await self._send_raw_message(
                chat_id=chat_id,
                msg_type=msg_type,
                payload=payload,
                reply_to=reply_target,
                metadata=retry_metadata,
            )

        from .cardkit_mutations import cancellation_requested

        for attempt in range(_FEISHU_SEND_ATTEMPTS):
            if attempt and cancellation_requested():
                raise last_error or RuntimeError("Feishu send cancelled during retry backoff")
            try:
                response = await _raw(active_reply_to)
                # Reply target withdrawn/missing → post a new message to the chat instead.
                if active_reply_to and not self._response_succeeded(response):
                    code = getattr(response, "code", None)
                    if code in _FEISHU_REPLY_FALLBACK_CODES:
                        if (metadata or {}).get("thread_id"):
                            logger.warning(
                                "[Feishu] Reply to %s failed in thread %s (code %s — message withdrawn/missing); "
                                "skipping top-level fallback to avoid creating a new topic",
                                active_reply_to,
                                (metadata or {}).get("thread_id"),
                                code,
                            )
                            return response
                        logger.warning(
                            "[Feishu] Reply to %s failed (code %s — message withdrawn/missing); "
                            "falling back to new message in chat %s",
                            active_reply_to,
                            code,
                            chat_id,
                        )
                        active_reply_to = None
                        response = await _raw(None)
                return response
            except Exception as exc:
                last_error = exc
                if msg_type == "post" and _POST_CONTENT_INVALID_RE.search(str(exc)):
                    raise
                if attempt >= _FEISHU_SEND_ATTEMPTS - 1 or cancellation_requested():
                    raise
                wait_seconds = 2**attempt
                logger.warning(
                    "[Feishu] Send attempt %d/%d failed for chat %s; retrying in %ds: %s",
                    attempt + 1,
                    _FEISHU_SEND_ATTEMPTS,
                    chat_id,
                    wait_seconds,
                    exc,
                )
                await asyncio.sleep(wait_seconds)
        raise last_error or RuntimeError("Feishu send failed")

    @staticmethod
    def _build_reply_message_body(
        *, content: str, msg_type: str, reply_in_thread: bool, uuid_value: str
    ) -> Any:
        return _sdk_build(
            ReplyMessageRequestBody,
            content=content,
            msg_type=msg_type,
            reply_in_thread=reply_in_thread,
            uuid=uuid_value,
        )

    @staticmethod
    def _build_reply_message_request(message_id: str, request_body: Any) -> Any:
        return _sdk_build(ReplyMessageRequest, message_id=message_id, request_body=request_body)

    @staticmethod
    def _build_create_message_body(
        *, receive_id: str, msg_type: str, content: str, uuid_value: str
    ) -> Any:
        return _sdk_build(
            CreateMessageRequestBody,
            receive_id=receive_id,
            msg_type=msg_type,
            content=content,
            uuid=uuid_value,
        )

    @staticmethod
    def _build_create_message_request(receive_id_type: str, request_body: Any) -> Any:
        return _sdk_build(
            CreateMessageRequest, receive_id_type=receive_id_type, request_body=request_body
        )
