"""Text-only REST enrichment for WebSocket merge_forward placeholders."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_MAX_TRANSCRIPT_CHARS = 12000
_TRUNCATION_MARKER = "\n[转发内容已截断]"


async def fetch_merge_forward_text(adapter: Any, message_id: str) -> str | None:
    """REST returns the parent followed by children linked by upper_message_id."""
    if not adapter._client or not message_id:
        return None
    try:
        request = adapter._build_get_message_request(message_id)
        response = await adapter._run_blocking(adapter._client.im.v1.message.get, request)
        if not response or not response.success():
            logger.warning("[Feishu] Failed to fetch merge_forward children for %s", message_id)
            return None
        items = getattr(getattr(response, "data", None), "items", None) or []
        transcript = ""
        for item in items:
            if getattr(item, "upper_message_id", None) != message_id:
                continue
            kind = getattr(item, "msg_type", "") or ""
            if kind in {"image", "file"}:
                text = "[图片]" if kind == "image" else "[文件]"
            else:
                text = adapter._extract_text_from_raw_content(
                    msg_type=kind,
                    raw_content=getattr(getattr(item, "body", None), "content", "") or "",
                    mentions=getattr(item, "mentions", None),
                )
            if not text:
                continue
            sender = getattr(item, "sender", None)
            name = getattr(sender, "name", None) or getattr(sender, "id", None)
            line = f"[{name}] {text}" if name else text
            transcript += ("\n" if transcript else "") + line
            if len(transcript) > _MAX_TRANSCRIPT_CHARS:
                transcript = transcript[:_MAX_TRANSCRIPT_CHARS - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER
                break
        if not transcript:
            logger.debug("[Feishu] No usable children found for merge_forward %s", message_id)
        return transcript or None
    except Exception:
        logger.warning("[Feishu] Failed to fetch merge_forward children for %s", message_id, exc_info=True)
        return None
