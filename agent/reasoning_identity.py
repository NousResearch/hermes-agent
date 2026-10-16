"""Native summary coordinates, accepted only when they describe the existing visible text.

Adapted from royalaid's #79727. This does not classify commentary, decode encrypted
reasoning or choose display policy; history_commentary owns the display boundary.
"""

import json
from typing import Any


def native_reasoning_items(value: Any, expected_text: str) -> list[dict]:
    """Return only IDs and summaries covering *all* expected text, otherwise no identities.

    Unknown/malformed siblings or extra analysis fall back to flat reasoning. Native
    compaction checkpoints have no display text and can coexist with summary items.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return []
    if (
        not isinstance(value, list)
        or not isinstance(expected_text, str)
        or not expected_text
    ):
        return []
    result, groups, seen = [], [], set()
    for item in value:
        if not isinstance(item, dict):
            return []
        if item.get("type") == "compaction":
            continue
        item_id, summary = item.get("id"), item.get("summary")
        if (
            item.get("type") != "reasoning"
            or not isinstance(item_id, str)
            or not item_id
            or item_id in seen
            or not isinstance(summary, list)
            or not summary
        ):
            return []
        parts = []
        for part in summary:
            if (
                not isinstance(part, dict)
                or part.get("type") != "summary_text"
                or not isinstance(part.get("text"), str)
                or not part["text"]
            ):
                return []
            parts.append({"type": "summary_text", "text": part["text"]})
        seen.add(item_id)
        groups.append("\n".join(part["text"] for part in parts))
        result.append({"type": "reasoning", "id": item_id, "summary": parts})
    return result if "\n\n".join(groups) == expected_text else []


def project_reasoning_identity(message: dict) -> dict:
    """Annotate the already-selected reasoning view without adding or replacing its text."""
    if message.get("role") != "assistant" or message.get("display_kind") == "hidden":
        return message
    reasoning = message.get(
        "display_reasoning",
        (
            message.get("reasoning")
            or message.get("reasoning_content")
            or message.get("reasoning_details")
            or ""
        ),
    )
    items = native_reasoning_items(message.get("codex_reasoning_items"), reasoning)
    if not items:
        return message
    return {**message, "display_reasoning_items": items}
