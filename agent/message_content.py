from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_NON_TEXT_PART_TYPES = {"image", "image_url", "input_image", "audio", "input_audio"}
_TEXT_KEYS = ("text", "content", "input_text", "output_text", "summary_text")


def get_message_field(value: Any, key: str, default: Any = None) -> Any:
    """Read a field from mapping-backed or attribute-backed SDK values."""
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _field(value: Any, key: str) -> Any:
    return get_message_field(value, key)


def to_plain_data(
    value: Any,
    *,
    _depth: int = 0,
    _path: set[int] | None = None,
) -> Any:
    """Recursively convert provider SDK objects to cycle-safe plain data."""
    if _depth > 20:
        return str(value)
    if _path is None:
        _path = set()

    obj_id = id(value)
    if obj_id in _path:
        return str(value)

    if hasattr(value, "model_dump"):
        _path.add(obj_id)
        result = to_plain_data(
            value.model_dump(),
            _depth=_depth + 1,
            _path=_path,
        )
        _path.discard(obj_id)
        return result
    if isinstance(value, Mapping):
        _path.add(obj_id)
        result = {
            str(key): to_plain_data(item, _depth=_depth + 1, _path=_path)
            for key, item in value.items()
        }
        _path.discard(obj_id)
        return result
    if isinstance(value, (list, tuple)):
        _path.add(obj_id)
        result = [
            to_plain_data(item, _depth=_depth + 1, _path=_path)
            for item in value
        ]
        _path.discard(obj_id)
        return result
    if hasattr(value, "__dict__"):
        _path.add(obj_id)
        result = {
            key: to_plain_data(item, _depth=_depth + 1, _path=_path)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
        _path.discard(obj_id)
        return result
    return value


def _text_from_part(part: Any) -> str:
    if part is None:
        return ""
    if isinstance(part, str):
        return part

    part_type = str(_field(part, "type") or "").strip().lower()
    if part_type in _NON_TEXT_PART_TYPES:
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
        chunks = [_text_from_part(part) for part in content]
        return sep.join(chunk for chunk in chunks if chunk)

    text = _text_from_part(content)
    if text:
        return text
    try:
        return str(content)
    except Exception:
        return ""
