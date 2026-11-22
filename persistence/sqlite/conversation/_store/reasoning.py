"""Stored-once reasoning text for reasoning-content providers (#125273)."""

from typing import Any


def _reasoning_is_shared(role: Any, reasoning: Any, reasoning_content: Any) -> bool:
    """Blank thinking-mode pads never count as shared reasoning text."""
    return (role == "assistant" and isinstance(reasoning_content, str) and bool(reasoning_content.strip())
            and reasoning == reasoning_content)


def _restore_shared_reasoning(msg: dict[str, Any]) -> None:
    """Refill ``reasoning`` stored only as ``reasoning_content``."""
    if msg.get("reasoning") is None and _reasoning_is_shared(
            msg.get("role"), msg.get("reasoning_content"), msg.get("reasoning_content")):
        msg["reasoning"] = msg["reasoning_content"]
