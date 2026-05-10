"""Helpers for keeping persisted transcripts compact and model-safe."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from typing import Any, Dict

_DATA_IMAGE_URL_RE = re.compile(
    r"^data:(image/[a-zA-Z0-9.+-]+);base64,(?P<data>[A-Za-z0-9+/=\s]+)$"
)
_DATA_IMAGE_URL_ANY_RE = re.compile(
    r"data:(image/[a-zA-Z0-9.+-]+);base64,[A-Za-z0-9+/=]+"
)


def _image_persistence_placeholder(data_url: str) -> str:
    """Return metadata for an inline image without persisting raw image bytes."""
    match = _DATA_IMAGE_URL_RE.match(data_url)
    if not match:
        digest = hashlib.sha256(data_url.encode("utf-8", errors="ignore")).hexdigest()
        return (
            "[Inline data URL omitted from persistent transcript: "
            f"sha256={digest[:16]}. Raw base64 was not stored.]"
        )

    mime_type = match.group(1)
    encoded = "".join(match.group("data").split())
    try:
        raw = base64.b64decode(encoded, validate=True)
        byte_count = len(raw)
        digest = hashlib.sha256(raw).hexdigest()
    except (binascii.Error, ValueError):
        byte_count = int(len(encoded) * 0.75)
        digest = hashlib.sha256(encoded.encode("ascii", errors="ignore")).hexdigest()

    return (
        "[Image omitted from persistent transcript: "
        f"{mime_type}, {byte_count} bytes, sha256={digest[:16]}. "
        "Raw base64 was not stored.]"
    )


def sanitize_for_persistence(value: Any) -> Any:
    """Remove inline image bytes before writing session history.

    Runtime model requests may need data:image URLs, but persisted JSON/JSONL
    and SQLite rows should store only metadata. Otherwise one screenshot can
    add tens of megabytes to resume context, FTS, and downstream memory sync.
    """
    if isinstance(value, str):
        if "data:image/" in value:
            return _DATA_IMAGE_URL_ANY_RE.sub(
                lambda match: _image_persistence_placeholder(match.group(0)),
                value,
            )
        return value

    if isinstance(value, list):
        return [sanitize_for_persistence(item) for item in value]

    if isinstance(value, dict):
        # OpenAI chat-completions multimodal shape:
        # {"type": "image_url", "image_url": {"url": "data:image/..."}}
        if value.get("type") == "image_url":
            image_url = value.get("image_url")
            if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                url = image_url["url"]
                if url.startswith("data:image/"):
                    return {"type": "text", "text": _image_persistence_placeholder(url)}

        # Responses-style shapes commonly use input_image/image_url strings.
        if value.get("type") in {"input_image", "image"}:
            image_url = value.get("image_url") or value.get("url")
            if isinstance(image_url, str) and image_url.startswith("data:image/"):
                return {"type": "input_text", "text": _image_persistence_placeholder(image_url)}

        return {key: sanitize_for_persistence(item) for key, item in value.items()}

    return value


def sanitize_message_for_persistence(message: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize all structured message fields before persistence."""
    sanitized = dict(message)
    for key in (
        "content",
        "tool_calls",
        "reasoning_details",
        "codex_reasoning_items",
        "codex_message_items",
    ):
        if key in sanitized:
            sanitized[key] = sanitize_for_persistence(sanitized[key])
    return sanitized


def summarize_for_log(value: Any, *, limit: int = 160) -> str:
    """Return a compact one-line preview that never includes inline image bytes."""
    sanitized = sanitize_for_persistence(value)
    if isinstance(sanitized, str):
        preview = sanitized
    else:
        try:
            preview = json.dumps(sanitized, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            preview = str(sanitized)
    preview = " ".join(preview.split())
    if len(preview) > limit:
        return f"{preview[: max(0, limit - 1)]}…"
    return preview
