"""Turn-input translation for the SDK: image/content blocks and the streaming
user-message shape. Extracted from ``claude_agent_sdk_session.py``;
``agent.claude_sdk_aux_client`` reaches these through the facade.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("agent.transports.claude_agent_sdk_session")


def _sdk_image_content_block(item: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Translate one Hermes/OpenAI image part to an SDK-native image block.

    The Agent SDK accepts structured user messages in streaming-input mode.
    Preserve base64 data URIs and http(s) image URLs instead of pretending a
    text-only query still carries the attachment. Return ``None`` for an
    unsupported/malformed source; callers add an explicit user-visible marker.
    """
    import base64 as _base64
    import re as _re
    from urllib.parse import urlsplit as _urlsplit

    source = item.get("source")
    if isinstance(source, dict):
        source_type = source.get("type")
        if source_type == "base64":
            media_type = source.get("media_type")
            data = source.get("data")
            if (
                isinstance(media_type, str)
                and media_type.startswith("image/")
                and isinstance(data, str)
                and data
            ):
                try:
                    _base64.b64decode(data, validate=True)
                except Exception:
                    return None
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data,
                    },
                }
        elif source_type == "url":
            raw_url = source.get("url")
            if isinstance(raw_url, str):
                parsed = _urlsplit(raw_url)
                if parsed.scheme in {"http", "https"} and parsed.netloc:
                    return {
                        "type": "image",
                        "source": {"type": "url", "url": raw_url},
                    }

    raw_url: Any = item.get("image_url")
    if isinstance(raw_url, dict):
        raw_url = raw_url.get("url")
    if not isinstance(raw_url, str):
        raw_url = item.get("url")
    if not isinstance(raw_url, str) or not raw_url:
        return None

    data_match = _re.fullmatch(
        r"data:(image/[A-Za-z0-9.+-]+);base64,(.+)",
        raw_url,
        flags=_re.DOTALL,
    )
    if data_match:
        media_type, data = data_match.groups()
        try:
            _base64.b64decode(data, validate=True)
        except Exception:
            return None
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": data,
            },
        }

    parsed = _urlsplit(raw_url)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return {
            "type": "image",
            "source": {"type": "url", "url": raw_url},
        }
    return None


def _coerce_turn_input(user_input: Any) -> Any:
    """Preserve Hermes/OpenAI rich images while keeping text-only turns plain.

    ClaudeSDKClient.query accepts either a string or an async stream of SDK
    message dictionaries. A content list with at least one valid image becomes
    SDK-native blocks; a text-only list keeps the historical joined-string
    behavior. Invalid image sources become truthful text, never a fabricated
    claim that an image is attached.
    """
    if isinstance(user_input, str):
        return user_input
    if isinstance(user_input, list):
        blocks: list[dict[str, Any]] = []
        has_valid_image = False
        for item in user_input:
            if isinstance(item, str):
                if item.strip():
                    blocks.append({"type": "text", "text": item})
                continue
            if not isinstance(item, dict):
                if item is not None:
                    blocks.append({"type": "text", "text": str(item)})
                continue
            item_type = item.get("type")
            if item_type in {"text", "input_text"}:
                text = item.get("text") or item.get("content") or ""
                if text:
                    blocks.append({"type": "text", "text": str(text)})
            elif item_type in {"image", "image_url", "input_image"}:
                image = _sdk_image_content_block(item)
                if image is not None:
                    blocks.append(image)
                    has_valid_image = True
                else:
                    logger.warning(
                        "claude-agent-sdk: image attachment has an unsupported "
                        "or malformed source; sending an explicit unavailable marker"
                    )
                    blocks.append({
                        "type": "text",
                        "text": (
                            "[image attachment unavailable: unsupported or "
                            "malformed source]"
                        ),
                    })
        if has_valid_image:
            return blocks
        return "\n\n".join(
            str(block.get("text") or "")
            for block in blocks
            if block.get("type") == "text"
        ).strip()
    return "" if user_input is None else str(user_input)


async def _sdk_user_message_stream(content: list[dict[str, Any]]):
    """One-message async stream accepted by ``ClaudeSDKClient.query``."""
    yield {
        "type": "user",
        "message": {"role": "user", "content": content},
        "parent_tool_use_id": None,
    }
