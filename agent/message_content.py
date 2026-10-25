from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_NON_TEXT_PART_TYPES = {"image", "image_url", "input_image", "audio", "input_audio"}
_TEXT_KEYS = ("text", "content", "input_text", "output_text", "summary_text")


def _field(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, Mapping) else getattr(value, key, None)


def _text_from_part(part: Any) -> str:
    if part is None:
        return ""
    if isinstance(part, str):
        return part
    if str(_field(part, "type") or "").strip().lower() in _NON_TEXT_PART_TYPES:
        return ""
    for key in _TEXT_KEYS:
        text = _field(part, key)
        if isinstance(text, str):
            return text
    return ""


def flatten_message_text(content: Any, *, sep: str = "\n") -> str:
    """Return the visible text from common chat/Responses message content shapes."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return sep.join(chunk for chunk in map(_text_from_part, content) if chunk)
    text = _text_from_part(content)
    if text:
        return text
    try:
        return str(content)
    except Exception:
        return ""


def delta_reasoning_text(delta: Any) -> Any:
    """Reasoning fragment of one streamed chat delta (SDK object or dict); ``None`` when absent.

    Order: ``reasoning_content`` / ``reasoning``, the same keys in pydantic ``model_extra`` (a
    reasoning-only stream otherwise trips the empty-stream guard, #56516), then typed ``thinking``
    parts inside ``content``. One field per chunk, so mirrored gateways never double the text.
    """
    text = _field(delta, "reasoning_content") or _field(delta, "reasoning")
    extra = getattr(delta, "model_extra", None)
    if text is None and isinstance(extra, dict):
        text = extra.get("reasoning_content") or extra.get("reasoning")
    return text if text is not None else (thinking_text_from_content(_field(delta, "content")) or None)


def thinking_text_from_content(content: Any) -> str:
    """Reasoning carried as typed ``thinking`` parts inside ``content``; ``""`` for any other shape.

    Mistral (Large 4, Medium 3.5, GLM via Mistral) streams reasoning this way, and its ``thinking``
    value is itself a list of text parts. ``flatten_message_text`` skips these parts, so without
    this the reasoning is silently dropped.
    """
    if not isinstance(content, list):
        return ""
    return "".join(
        flatten_message_text(value, sep="")
        for part in content
        if str(_field(part, "type") or "").strip().lower() == "thinking"
        and isinstance(value := _field(part, "thinking"), (str, list))
    )
