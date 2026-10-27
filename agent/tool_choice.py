"""Wire-level tool enforcement (``agent.tool_choice: required``).

Small local models often answer "I'll install it now" and stop, or explain why they can't act,
instead of calling a tool; ``tool_use_enforcement`` only nudges via the system prompt. With
``required`` the request carries ``tool_choice="required"`` so a text-only reply is impossible,
plus a ``respond`` tool that is the model's one way to talk to the user and end the turn. A
respond-only reply is unwrapped back into a plain text message at intake, so final-response
delivery, persistence and verification run unchanged (the gap that left #125444 inert).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

RESPOND_TOOL_NAME = "respond"
# chat_completions only: it is where local/small models run, and Anthropic rejects forced tool
# use ({"type": "any"}) whenever extended thinking is on, so enabling it there would 400.
_SUPPORTED_API_MODES = ("chat_completions",)

RESPOND_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": RESPOND_TOOL_NAME,
        "description": (
            "Send your reply to the user and end your turn. This is the only way to talk to the "
            "user: plain text replies are disabled. Do the work with your other tools first; call "
            "this when the task is done, when you need the user's input, or to report a real "
            "blocker. Never use it to announce work you have not done yet."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "The full reply shown to the user (markdown ok)."},
            },
            "required": ["message"],
        },
    },
}


def tool_choice_required(agent: Any) -> bool:
    """True when ``agent.tool_choice`` is ``required`` for this agent's platform and its api_mode
    can enforce it. The value is a string, or a ``{platform: value, "default": value}`` mapping:
    local servers cap the forced-call grammar size (Rapid-MLX: 64 KiB), so a lean chat platform
    can force calls while the CLI's larger tool surface stays on ``auto``."""
    value = getattr(agent, "_tool_choice", "auto") or "auto"
    if isinstance(value, dict):
        value = value.get(getattr(agent, "platform", None) or "cli", value.get("default", "auto"))
    value = str(value or "auto").strip().lower()
    return value == "required" and getattr(agent, "api_mode", None) in _SUPPORTED_API_MODES


def apply_tool_choice(agent: Any, tools_for_api: Optional[list], request_overrides: dict) -> tuple:
    """``(tools, request_overrides)`` for one request: the respond tool plus
    ``tool_choice="required"`` when enabled. Explicit overrides win; tool-less requests pass."""
    if not tools_for_api or "tool_choice" in request_overrides or not tool_choice_required(agent):
        return tools_for_api, request_overrides
    return with_respond_tool(tools_for_api), {**request_overrides, "tool_choice": "required"}


def with_respond_tool(tools: Optional[list]) -> Optional[list]:
    """``tools`` plus the respond schema (appended once, so the tool block stays cache-stable)."""
    if not tools:
        return tools
    if any((t.get("function") or {}).get("name") == RESPOND_TOOL_NAME for t in tools):
        return tools
    return [*tools, RESPOND_TOOL_SCHEMA]


def _respond_message(arguments: str) -> str:
    try:
        parsed = json.loads(arguments or "{}")
    except (TypeError, ValueError):
        return str(arguments or "")
    message = parsed.get("message") if isinstance(parsed, dict) else parsed
    if message is None:
        return ""
    return message if isinstance(message, str) else json.dumps(message)


def unwrap_respond_call(assistant_message: Any) -> bool:
    """Turn a ``respond`` call into the plain-text reply it carries; True when it did.

    respond-only -> content = message, no tool calls, finish_reason "stop". respond mixed with
    real tool calls -> the respond call is dropped so the real tools run first; the model replies
    on a later iteration with their results in hand.
    """
    tool_calls = getattr(assistant_message, "tool_calls", None) or []
    responds = [tc for tc in tool_calls if getattr(tc, "name", None) == RESPOND_TOOL_NAME]
    if not responds:
        return False
    others = [tc for tc in tool_calls if getattr(tc, "name", None) != RESPOND_TOOL_NAME]
    if others:
        logger.info("tool_choice=required: dropping respond call issued alongside %d tool call(s)", len(others))
        assistant_message.tool_calls = others
        return False
    # Text emitted alongside the call is usually a narration of the same reply; the respond
    # message wins and that text is only the fallback for an empty message.
    text = "\n\n".join(m for m in (_respond_message(tc.arguments) for tc in responds) if m)
    assistant_message.content = text or assistant_message.content
    assistant_message.tool_calls = None
    assistant_message.finish_reason = "stop"
    return True
