"""Helpers for API-call-time prefill messages."""

from __future__ import annotations

from typing import Any, Dict, List


def fold_system_prefill_messages(
    api_messages: List[Dict[str, Any]],
    prefill_messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Fold system prefill messages into the single leading system prompt."""
    if not prefill_messages:
        return api_messages
    messages = [msg.copy() for msg in api_messages]

    system_parts: List[str] = []
    non_system_prefill: List[Dict[str, Any]] = []
    for prefill in prefill_messages:
        copied = prefill.copy()
        if copied.get("role") == "system":
            content = copied.get("content", "")
            if content:
                system_parts.append(content if isinstance(content, str) else str(content))
        else:
            non_system_prefill.append(copied)

    if system_parts:
        folded_system = "\n\n".join(part.strip() for part in system_parts if part.strip())
        if messages and messages[0].get("role") == "system":
            existing = messages[0].get("content", "")
            existing_text = existing if isinstance(existing, str) else str(existing)
            combined = "\n\n".join(part for part in (existing_text.strip(), folded_system) if part)
            messages[0] = {**messages[0], "content": combined}
        else:
            messages.insert(0, {"role": "system", "content": folded_system})

    sys_offset = 1 if messages and messages[0].get("role") == "system" else 0
    for idx, prefill in enumerate(non_system_prefill):
        messages.insert(sys_offset + idx, prefill)
    return messages
