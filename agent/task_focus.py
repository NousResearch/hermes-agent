"""Task focus state — preserves goal/artifacts/verified_state across compaction.
Extracted from OpenHarness task focus pattern."""

import re
from typing import Any, Dict, List, Optional


GOAL_PATTERNS = [
    r"(?:goal|objective|task|TODO|目标|任务)[:\s]+(.+)",
    r"(?:I need to|I want to|请|帮我|需要)[:\s]+(.+)",
]


def extract_task_focus(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Extract task focus from conversation messages."""
    goal = ""
    active_artifacts: List[str] = []
    verified_state: List[str] = []

    for msg in messages:
        if msg.get("role") == "user":
            content = _extract_text(msg.get("content", ""))
            for pattern in GOAL_PATTERNS:
                match = re.search(pattern, content, re.IGNORECASE)
                if match and not goal:
                    goal = match.group(1).strip()[:200]

        if msg.get("role") == "tool":
            content = _extract_text(msg.get("content", ""))
            for path_match in re.finditer(r"(?:/[\w.-]+){2,}", content):
                path = path_match.group(0)
                if path not in active_artifacts and len(active_artifacts) < 5:
                    active_artifacts.append(path)

        if msg.get("role") == "assistant":
            content = _extract_text(msg.get("content", ""))
            lowered = content.lower()
            if any(kw in lowered for kw in ["passed", "success", "完成", "通过", "fixed"]):
                summary = content[:150].strip()
                if summary and summary not in verified_state and len(verified_state) < 4:
                    verified_state.append(summary)

    if not goal and not active_artifacts and not verified_state:
        return None

    return {
        "goal": goal,
        "active_artifacts": active_artifacts,
        "verified_state": verified_state,
    }


def format_task_focus_for_summary(focus: Dict[str, Any]) -> str:
    """Format task focus as markdown section for compaction summary."""
    lines = ["## Active Task"]
    if focus.get("goal"):
        lines.append(f"Goal: {focus['goal']}")
    if focus.get("active_artifacts"):
        lines.append("Key files:")
        for artifact in focus["active_artifacts"][:3]:
            lines.append(f"- {artifact}")
    if focus.get("verified_state"):
        lines.append("Verified:")
        for verified in focus["verified_state"][:2]:
            lines.append(f"- {verified}")
    return "\n".join(lines)


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return " ".join(parts)
    return ""
