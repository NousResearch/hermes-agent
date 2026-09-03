"""Current-message-only Slack tools.

The model never supplies Slack identifiers.  Each call is bound to the inbound
message that started this turn through task-local gateway session context.
"""
from __future__ import annotations

import json
import re

_CURRENT_MESSAGE_SCHEMA = {
    "name": "slack_current_message",
    "description": "React to the Slack message that triggered this turn, or return that exact message's canonical Slack permalink. This tool cannot target any other message, thread, channel, or workspace.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["react", "permalink"],
                "description": "'react' adds an emoji to the current inbound Slack message; 'permalink' returns its canonical Slack URL.",
            },
            "emoji": {
                "type": "string",
                "description": "Slack emoji name for action='react', optionally colon-wrapped (for example ':beer:' or 'eyes').",
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}
_EMOJI_NAME_RE = re.compile(r"^[a-z0-9_+\-]{1,128}$")


def _bound_current_message() -> tuple[str, str, str] | tuple[None, None, str]:
    from gateway.session_context import get_session_env

    if get_session_env("HERMES_SESSION_PLATFORM", "").strip().lower() != "slack":
        return None, None, "This tool is available only while handling a Slack message."
    channel = get_session_env("HERMES_SESSION_CHAT_ID", "").strip()
    message_id = get_session_env("HERMES_SESSION_MESSAGE_ID", "").strip()
    team_id = get_session_env("HERMES_SESSION_SCOPE_ID", "").strip()
    if not channel or not message_id or not team_id:
        return None, None, "Current Slack message context is unavailable; refusing to target a message."
    return channel, message_id, team_id


def _current_slack_adapter():
    try:
        from gateway.config import Platform
        from gateway.run import _gateway_runner_ref

        runner = _gateway_runner_ref()
        adapter = runner.adapters.get(Platform.SLACK) if runner is not None else None
    except Exception:
        adapter = None
    if adapter is None or not callable(getattr(adapter, "react_to_current_message", None)):
        return None
    return adapter


def _handle_current_message(args: dict, **_: object) -> str:
    action = str(args.get("action") or "").strip().lower()
    channel, message_id, team_or_error = _bound_current_message()
    if channel is None:
        return json.dumps({"success": False, "error": team_or_error})
    team_id = team_or_error
    adapter = _current_slack_adapter()
    if adapter is None:
        return json.dumps({"success": False, "error": "A live Slack gateway adapter is required."})

    from model_tools import _run_async

    if action == "react":
        emoji = str(args.get("emoji") or "").strip().strip(":").lower()
        if not _EMOJI_NAME_RE.fullmatch(emoji):
            return json.dumps({"success": False, "error": "emoji must be a Slack emoji name such as 'beer' or ':beer:'."})
        try:
            success = bool(_run_async(adapter.react_to_current_message(
                channel=channel, timestamp=message_id, emoji=emoji, team_id=team_id,
            )))
        except Exception:
            success = False
        return json.dumps({"success": success, "action": "react", "emoji": emoji})

    if action == "permalink":
        try:
            permalink = _run_async(adapter.current_message_permalink(
                channel=channel, timestamp=message_id, team_id=team_id,
            ))
        except Exception:
            permalink = None
        if not isinstance(permalink, str) or not permalink:
            return json.dumps({"success": False, "error": "Slack could not resolve the current message permalink."})
        return json.dumps({"success": True, "action": "permalink", "permalink": permalink})

    return json.dumps({"success": False, "error": "action must be 'react' or 'permalink'."})


def _check_slack_tool_available() -> bool:
    try:
        from agent.secret_scope import get_secret
        return bool((get_secret("SLACK_BOT_TOKEN", "") or "").strip())
    except Exception:
        return False


def register_tools(ctx) -> None:
    ctx.register_tool(
        name="slack_current_message",
        toolset="slack",
        schema=_CURRENT_MESSAGE_SCHEMA,
        handler=_handle_current_message,
        check_fn=_check_slack_tool_available,
        emoji="💬",
    )
