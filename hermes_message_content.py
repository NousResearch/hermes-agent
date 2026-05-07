"""Helpers for storing, replaying, and displaying structured message content."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from agent.memory_manager import sanitize_context

STRUCTURED_CONTENT_FORMAT = "json"

_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def sanitize_storage_text(text: str) -> str:
    """Replace invalid UTF-8 surrogate code points before SQLite binding."""
    if _SURROGATE_RE.search(text):
        return _SURROGATE_RE.sub("\ufffd", text)
    return text


def sanitize_content_for_storage(content: Any) -> Any:
    if isinstance(content, str):
        return sanitize_storage_text(content)
    if isinstance(content, list):
        return [sanitize_content_for_storage(item) for item in content]
    if isinstance(content, dict):
        return {
            (sanitize_storage_text(key) if isinstance(key, str) else key): sanitize_content_for_storage(value)
            for key, value in content.items()
        }
    return content


def content_to_text(
    content: Any,
    *,
    separator: str = "\n",
    attachment_style: str = "detailed",
) -> str:
    """Render structured content as text safe for previews, display, and FTS."""
    if content is None:
        return ""
    if isinstance(content, str):
        return sanitize_storage_text(content)
    if isinstance(content, list):
        parts = []
        for part in content:
            text = content_part_to_text(part, separator=separator, attachment_style=attachment_style)
            if text:
                parts.append(text)
        return separator.join(parts)
    if isinstance(content, dict):
        return (
            content_part_to_text(content, separator=separator, attachment_style=attachment_style)
            or "[structured content]"
        )
    return str(content)


def content_part_to_text(
    part: Any,
    *,
    separator: str = "\n",
    attachment_style: str = "detailed",
) -> str:
    if isinstance(part, str):
        return sanitize_storage_text(part)
    if not isinstance(part, dict):
        return str(part) if part is not None else ""

    ptype = str(part.get("type") or "").strip().lower()
    if ptype in {"text", "input_text", "output_text"}:
        return sanitize_storage_text(str(part.get("text") or ""))

    if ptype in {"image_url", "input_image", "image"}:
        if attachment_style == "compact":
            return "[image]"
        source = part.get("source")
        if isinstance(source, dict) and source.get("data") is not None:
            return "[image attachment]"
        image_ref = part.get("image_url") or part.get("url") or part.get("image")
        if not image_ref and isinstance(source, dict):
            image_ref = source.get("url") or source.get("uri")
        if isinstance(image_ref, dict):
            image_ref = image_ref.get("url") or image_ref.get("uri") or ""
        image_url = sanitize_storage_text(str(image_ref or "").strip())
        if image_url.lower().startswith("data:"):
            return "[image attachment]"
        return f"[image: {image_url}]" if image_url else "[image attachment]"

    if ptype in {"file", "input_file"}:
        if attachment_style == "compact":
            return "[file]"
        file_data = part.get("file_data") or part.get("data")
        file_ref = part.get("file_id") or part.get("filename")
        file_obj = part.get("file")
        if isinstance(file_obj, dict):
            file_data = file_data or file_obj.get("file_data") or file_obj.get("data")
            file_ref = (
                file_ref
                or file_obj.get("file_id")
                or file_obj.get("id")
                or file_obj.get("filename")
                or file_obj.get("name")
            )
        elif file_obj:
            file_ref = file_ref or file_obj
        if file_data:
            return "[file attachment]"
        file_text = sanitize_storage_text(str(file_ref or "").strip())
        return f"[file: {file_text}]" if file_text else "[file attachment]"

    if ptype in {"input_audio", "audio"}:
        return "[audio]" if attachment_style == "compact" else "[audio attachment]"

    source = part.get("source")
    if (
        part.get("data") is not None
        or part.get("file_data") is not None
        or (
            isinstance(source, dict)
            and (
                source.get("data") is not None
                or source.get("file_data") is not None
            )
        )
    ):
        return "[attachment]"

    fallback = part.get("text") or part.get("content")
    if isinstance(fallback, (list, dict)):
        return content_to_text(
            fallback,
            separator=separator,
            attachment_style=attachment_style,
        )
    return sanitize_storage_text(str(fallback)) if fallback is not None else ""


def serialize_message_content(content: Any) -> tuple[Any, Optional[str], Optional[str]]:
    """Convert message content into indexed text plus optional replay payload."""
    content = sanitize_content_for_storage(content)
    if isinstance(content, (list, dict)):
        return (
            content_to_text(content),
            STRUCTURED_CONTENT_FORMAT,
            json.dumps(content, ensure_ascii=False),
        )
    return content, None, None


def deserialize_message_content(
    content: Any,
    content_format: Optional[str] = None,
    content_payload: Optional[str] = None,
) -> Any:
    """Restore content only when Hermes marked it as structured JSON."""
    if content_format != STRUCTURED_CONTENT_FORMAT or not isinstance(content, str):
        return content
    try:
        return json.loads(content_payload if content_payload is not None else content)
    except (json.JSONDecodeError, TypeError):
        return content


def sanitize_replay_content(content: Any) -> Any:
    """Strip transient memory context from text before replaying to models."""
    if isinstance(content, str):
        return sanitize_context(content).strip()
    if isinstance(content, list):
        return [sanitize_replay_content(item) for item in content]
    if isinstance(content, dict):
        return {
            key: sanitize_replay_content(value)
            for key, value in content.items()
        }
    return content


def normalize_replay_content(content: Any) -> Any:
    """Return provider-acceptable replay content."""
    content = sanitize_replay_content(content)
    if isinstance(content, dict):
        if content.get("type"):
            return [content]
        return content_to_text(content)
    return content


def content_identity(content: Any) -> Optional[str]:
    """Stable content key for replay duplicate detection."""
    if content is None or content == "":
        return None
    if isinstance(content, (list, dict)):
        try:
            return json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            return str(content)
    return str(content)
