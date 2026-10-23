"""Messaging-session-only terminal response control; deliberately not a registry tool."""

from __future__ import annotations

import json


TOOL_NAME = "skip_response"


def available(agent) -> bool:
    """Use the session's own chat identity, never process environment or profile config."""
    from gateway.config import Platform

    try:
        platform = Platform(getattr(agent, "platform", None))
    except ValueError:
        return False
    return bool(getattr(agent, "_chat_id", None)) and platform not in {
        Platform.LOCAL, Platform.API_SERVER, Platform.WEBHOOK, Platform.MSGRAPH_WEBHOOK,
    }


def tool_schema() -> dict:
    return {"type": "function", "function": {
        "name": TOOL_NAME,
        "description": (
            "End this messaging turn successfully without sending a reply. Use when the user "
            "asks you to stop replying, or the conversation is addressed to someone else and "
            "you should stay out. Call this tool alone, without acknowledgement, commentary, "
            "or other tools. Do not use it instead of answering an ordinary request or reporting "
            "an error. This skips only this turn; it does not mute the channel or future turns. "
            "The reason is recorded in the transcript, not sent to chat; do not include secrets."
        ),
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string", "minLength": 1, "maxLength": 500,
                       "description": "Brief non-sensitive reason for not replying."},
        }, "required": ["reason"], "additionalProperties": False},
    }}


def inject(agent, tools_list: list, name_set: set) -> None:
    """Keep dynamic rebuilds and surface switches session-scoped and idempotent."""
    if not available(agent):
        return
    if TOOL_NAME not in name_set:
        tools_list.append(tool_schema())
        name_set.add(TOOL_NAME)


def execute(agent, args: dict, ctx) -> str:
    """An explicit successful outcome, not an interrupt or a magic model-text marker."""
    reason = args.get("reason")
    tail = (ctx.messages or [{}])[-1]
    calls = tail.get("tool_calls") or []
    if not available(agent) or len(calls) != 1 or calls[0].get("function", {}).get("name") != TOOL_NAME:
        return json.dumps({"error": "skip_response must be called alone in a messaging turn."})
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 500 or set(args) != {"reason"}:
        return json.dumps({"error": "A non-empty reason of at most 500 characters is required."})
    agent._response_skip_reason = reason.strip()
    return json.dumps({"success": True, "response_disposition": TOOL_NAME, "reason": reason.strip()})


def receipt(reason: str) -> dict:
    """Close the tool sequence with a durable, hidden, provider-safe assistant row."""
    return {"role": "assistant", "content": "", "api_content": "[Response intentionally skipped.]",
            "display_kind": "hidden", "display_metadata": {
                "response_disposition": TOOL_NAME, "reason": reason,
            }}
