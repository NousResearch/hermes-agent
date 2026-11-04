"""Raw text/post delivery shared by public sends and explicit standalone delivery."""

import json
from typing import Any

from gateway.platforms.base import SendResult


async def send_text_direct(
    self, chat_id: str, content: str, reply_to: str | None = None, metadata: dict[str, Any] | None = None,
) -> SendResult:
    from .adapter import _MARKDOWN_HINT_RE, _POST_CONTENT_INVALID_RE, _strip_markdown_to_plain_text, logger

    if not self._client:
        return SendResult(success=False, error="Not connected")
    formatted = self.format_message(content)
    chunks = self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH)
    # A split markdown message keeps one rendering decision for all its chunks (#26841).
    prefer_post = bool(_MARKDOWN_HINT_RE.search(formatted))
    last_response = None

    async def _send_plain(chunk: str) -> Any:
        return await self._feishu_send_with_retry(
            chat_id=chat_id,
            msg_type="text",
            payload=json.dumps({"text": _strip_markdown_to_plain_text(chunk)}, ensure_ascii=False),
            reply_to=reply_to,
            metadata=metadata,
        )

    try:
        for chunk in chunks:
            msg_type, payload = self._build_outbound_payload(chunk, prefer_post=prefer_post)
            try:
                response = await self._feishu_send_with_retry(
                    chat_id=chat_id, msg_type=msg_type, payload=payload, reply_to=reply_to, metadata=metadata,
                )
            except Exception as exc:
                if msg_type != "post" or not _POST_CONTENT_INVALID_RE.search(str(exc)):
                    raise
                logger.warning("[Feishu] Invalid post payload rejected by API; falling back to plain text")
                response = await _send_plain(chunk)
            if (
                msg_type == "post"
                and not self._response_succeeded(response)
                and _POST_CONTENT_INVALID_RE.search(str(getattr(response, "msg", "") or ""))
            ):
                logger.warning("[Feishu] Post payload rejected by API response; falling back to plain text")
                response = await _send_plain(chunk)
            last_response = response

        return self._finalize_send_result(last_response, "send failed")
    except Exception as exc:
        logger.error("[Feishu] Send error: %s", exc, exc_info=True)
        return SendResult(success=False, error=str(exc))
