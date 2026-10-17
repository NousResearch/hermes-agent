"""Lossless /retry text extraction for a user-originated turn's live content."""

from __future__ import annotations

from typing import Any


def retryable_user_text(content: Any) -> str:
    """Lossless retry text, or raise before destructive mutation (media/unknown parts fail closed: no replay protocol)."""
    if not isinstance(content, (str, list)):
        raise ValueError("retry does not support non-text content")
    chunks: list[str] = []
    for part in [content] if isinstance(content, str) else content:
        if isinstance(part, str):
            chunks.append(part)
            continue
        if not isinstance(part, dict):
            raise ValueError("retry does not support non-text content")
        if part.get("type") not in {"text", "input_text", "output_text"}:
            raise ValueError("retry does not support media or unknown content parts")
        if set(part) - {"type", "text"}:
            raise ValueError("retry cannot losslessly flatten annotated text parts")
        if not isinstance(part.get("text"), str):
            raise ValueError("retry text parts must contain text")
        chunks.append(part["text"])
    text = "".join(chunks)
    if not text.strip():
        raise ValueError("retry found no text to send")
    return text
