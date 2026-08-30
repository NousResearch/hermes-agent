"""Safe display-only projection of persisted Codex Responses sidecars."""

from __future__ import annotations

import json
from typing import Any


def _decode_list(value: Any) -> list[Any] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
    return value if isinstance(value, list) else None


def _safe_content(value: Any) -> list[dict[str, str]] | None:
    if not isinstance(value, list) or not value:
        return None
    result: list[dict[str, str]] = []
    for part in value:
        if not isinstance(part, dict) or part.get("type") != "output_text":
            return None
        text = part.get("text")
        if not isinstance(text, str) or not text:
            return None
        result.append({"type": "output_text", "text": text})
    return result


def project_codex_display_items(message: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Return only safe reasoning summaries and assistant display text, or fail closed."""
    projected: list[dict[str, Any]] = []
    raw_reasoning = message.get("codex_reasoning_items")
    if raw_reasoning is not None:
        reasoning_items = _decode_list(raw_reasoning)
        if reasoning_items is None:
            return None
        for item in reasoning_items:
            if not isinstance(item, dict) or item.get("type") != "reasoning":
                return None
            item_id = item.get("id")
            summary = item.get("summary")
            if not isinstance(item_id, str) or not item_id or not isinstance(summary, list) or not summary:
                return None
            safe_summary: list[dict[str, str]] = []
            for part in summary:
                if not isinstance(part, dict) or part.get("type") != "summary_text":
                    return None
                text = part.get("text")
                if not isinstance(text, str) or not text:
                    return None
                safe_summary.append({"type": "summary_text", "text": text})
            projected.append({"type": "reasoning", "id": item_id, "summary": safe_summary})

    raw_messages = message.get("codex_message_items")
    if raw_messages is not None:
        message_items = _decode_list(raw_messages)
        if message_items is None:
            return None
        for item in message_items:
            if not isinstance(item, dict) or item.get("type") != "message" or item.get("role") != "assistant":
                return None
            item_id = item.get("id")
            phase = item.get("phase")
            content = _safe_content(item.get("content"))
            if content is None or phase not in {"analysis", "commentary", "final", "final_answer"}:
                return None
            if phase == "analysis":
                continue
            if phase == "commentary" and (not isinstance(item_id, str) or not item_id):
                return None
            safe_item: dict[str, Any] = {
                "type": "message", "role": "assistant", "phase": phase, "content": content,
            }
            if isinstance(item_id, str) and item_id:
                safe_item["id"] = item_id
            projected.append(safe_item)

    return projected or None
