"""Recovery helpers for tool-call turns that end in empty model output."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def empty_tool_result_recovery_response(messages: List[Dict[str, Any]]) -> Optional[str]:
    """Build a useful final response when a model goes empty after tool calls."""
    recent_tools: list[Dict[str, str]] = []
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "tool":
            name = str(message.get("name") or "tool")
            content = message.get("content")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            recent_tools.append({
                "name": name,
                "content": content.strip(),
            })
            if len(recent_tools) >= 5:
                break
            continue
        if recent_tools and role == "assistant" and message.get("tool_calls"):
            break
        if recent_tools and role not in {"tool"}:
            break

    if not recent_tools:
        return None

    recent_tools.reverse()
    lines = [
        "The model returned an empty message after running tools, so Hermes is returning the latest tool results instead of `(empty)`.",
        "",
        "Latest tool results:",
    ]
    for tool in recent_tools:
        content = tool["content"] or "(no tool output)"
        parsed = None
        try:
            parsed = json.loads(content)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            compact = json.dumps(parsed, ensure_ascii=False, indent=2)
            preview = compact[:4000] + "\n..." if len(compact) > 4000 else compact
        else:
            preview = content[:4000] + "\n..." if len(content) > 4000 else content
        lines.extend([
            f"- `{tool['name']}`:",
            "```text",
            preview,
            "```",
        ])
    return "\n".join(lines).strip()
