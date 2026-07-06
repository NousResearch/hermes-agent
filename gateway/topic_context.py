from __future__ import annotations

import json
from typing import Any

MAX_FIELD_CHARS = 800
MAX_ITEM_CHARS = 800
MAX_ITEMS = 20


def _clean(value: object, *, limit: int = MAX_FIELD_CHARS) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    text = text.replace("\x00", "")
    text = "".join(ch if ch >= " " or ch in "\n\t" else " " for ch in text)
    if limit and len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _clean_items(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned: list[str] = []
    for value in values[:MAX_ITEMS]:
        item = _clean(value, limit=MAX_ITEM_CHARS)
        if item:
            cleaned.append(item)
    return cleaned


def format_topic_context_prompt(ctx: dict[str, Any] | None) -> str:
    """Render compact persistent topic metadata for the gateway prompt."""
    if not ctx:
        return ""

    lines = [
        "Telegram group topic context (persistent, compact metadata; not chat history):",
    ]
    chat_name = _clean(ctx.get("chat_name"))
    topic_name = _clean(ctx.get("topic_name"))
    thread_id = _clean(ctx.get("thread_id"), limit=80)
    purpose = _clean(ctx.get("purpose"))
    workdir = _clean(ctx.get("workdir"), limit=500)

    if chat_name:
        # The group title comes from Telegram (any member can rename the
        # group), unlike the other fields, which are set by authorized users
        # via /topic. Render it as an inert quoted string, matching how
        # gateway/session.py treats untrusted platform metadata.
        lines.append(f"Group: {json.dumps(chat_name, ensure_ascii=False)}")
    if topic_name:
        lines.append(f"Topic: {topic_name}")
    if thread_id:
        lines.append(f"Topic ID: {thread_id}")
    if purpose:
        lines.append(f"Purpose: {purpose}")
    if workdir:
        lines.append(
            f"Workdir: {workdir} (session working directory; project rules are loaded from here)"
        )

    skills = _clean_items(ctx.get("skills"))
    if skills:
        lines.append("Topic-bound skills: " + ", ".join(skills))

    return "\n".join(lines)
