"""ClawTeam agent-facing tools.

Five verbs an LLM agent inside Hermes can call to drive a ClawTeam:
discover, status, spawn, inbox-send, inbox-peek. Each handler shells
to `clawteam --json …` via the shared driver and wraps the result in
the standard tool_result / tool_error envelope Hermes' registry uses.
"""

from __future__ import annotations

from typing import Any

from tools.registry import tool_error, tool_result  # type: ignore[import-not-found]

from ._clawteam_cli import (
    CliError,
    check_clawteam_available,
    require_str,
    run_clawteam_json,
    validate_name,
)


def _err(exc: Exception) -> str:
    if isinstance(exc, CliError):
        return tool_error(str(exc), status_code=exc.status_code)
    return tool_error(f"clawteam tool failed: {type(exc).__name__}: {exc}")


def _handle_team_discover(args: dict, **_kw: Any) -> str:
    try:
        teams = run_clawteam_json("team", "discover") or []
    except Exception as exc:
        return _err(exc)
    return tool_result({"teams": teams})


def _handle_team_status(args: dict, **_kw: Any) -> str:
    try:
        name = validate_name(require_str(args.get("team"), field="team"), field="team")
        status = run_clawteam_json("team", "status", "--", name)
    except Exception as exc:
        return _err(exc)
    return tool_result({"team": status})


def _handle_team_spawn(args: dict, **_kw: Any) -> str:
    try:
        name = validate_name(require_str(args.get("name"), field="name"), field="team name")
        cmd = ["team", "spawn-team", name]
        description = args.get("description")
        if isinstance(description, str) and description.strip():
            cmd += ["-d", description.strip()]
        leader_name = args.get("leader_name")
        if isinstance(leader_name, str) and leader_name.strip():
            leader_name = validate_name(leader_name.strip(), field="leader_name")
            cmd += ["-n", leader_name]
        result = run_clawteam_json(*cmd)
    except Exception as exc:
        return _err(exc)
    return tool_result({"spawned": result, "name": name})


def _handle_inbox_send(args: dict, **_kw: Any) -> str:
    try:
        team = validate_name(require_str(args.get("team"), field="team"), field="team")
        to = validate_name(require_str(args.get("to"), field="to"), field="to")
        content = require_str(args.get("content"), field="content")
        cmd = ["inbox", "send", team, to, content]
        msg_type = args.get("type")
        if isinstance(msg_type, str) and msg_type.strip():
            cmd += ["--type", msg_type.strip()]
        sender_raw = args.get("from")
        sender = sender_raw.strip() if isinstance(sender_raw, str) and sender_raw.strip() else "hermes-agent"
        cmd += ["--from", sender]
        result = run_clawteam_json(*cmd)
    except Exception as exc:
        return _err(exc)
    return tool_result({"sent": result})


def _handle_inbox_peek(args: dict, **_kw: Any) -> str:
    try:
        team = validate_name(require_str(args.get("team"), field="team"), field="team")
        cmd = ["inbox", "peek", team]
        agent = args.get("agent")
        if isinstance(agent, str) and agent.strip():
            agent = validate_name(agent.strip(), field="agent")
            cmd += ["--agent", agent]
        result = run_clawteam_json(*cmd)
    except Exception as exc:
        return _err(exc)
    return tool_result(result or {"count": 0, "messages": []})


# OpenAI-format schemas. Names use the clawteam_ prefix so Hermes' tool
# list groups them visibly.
_TEAM_DISCOVER_SCHEMA = {
    "name": "clawteam_team_discover",
    "description": "List all discoverable ClawTeam teams. Returns the team list with name, description, lead agent id, and member count.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

_TEAM_STATUS_SCHEMA = {
    "name": "clawteam_team_status",
    "description": "Get full status for one ClawTeam team — members, lead agent, creation time, recent activity.",
    "parameters": {
        "type": "object",
        "properties": {
            "team": {"type": "string", "description": "Team name (as returned by clawteam_team_discover)"},
        },
        "required": ["team"],
    },
}

_TEAM_SPAWN_SCHEMA = {
    "name": "clawteam_team_spawn",
    "description": "Spawn a new ClawTeam team and register its leader agent. Returns the spawned team metadata.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Team name (kebab-case recommended)"},
            "description": {"type": "string", "description": "Optional human-readable team description"},
            "leader_name": {"type": "string", "description": "Optional leader agent name (default: 'leader')"},
        },
        "required": ["name"],
    },
}

_INBOX_SEND_SCHEMA = {
    "name": "clawteam_inbox_send",
    "description": "Send a point-to-point message to one agent inside a ClawTeam team. The message is written to that agent's inbox; the recipient can read it via clawteam_inbox_peek or its own inbox watcher.",
    "parameters": {
        "type": "object",
        "properties": {
            "team": {"type": "string", "description": "Team name"},
            "to": {"type": "string", "description": "Recipient agent name (use clawteam_team_status to list members)"},
            "content": {"type": "string", "description": "Message body"},
            "type": {"type": "string", "description": "Message type (default: 'message')"},
            "from": {"type": "string", "description": "Sender label shown in the recipient's inbox (default: 'hermes-agent')"},
        },
        "required": ["team", "to", "content"],
    },
}

_INBOX_PEEK_SCHEMA = {
    "name": "clawteam_inbox_peek",
    "description": "Read messages from a team agent's inbox WITHOUT consuming them. Returns {count, messages[]}. Non-destructive — safe to call repeatedly.",
    "parameters": {
        "type": "object",
        "properties": {
            "team": {"type": "string", "description": "Team name"},
            "agent": {"type": "string", "description": "Agent name to peek for (default: from clawteam env identity)"},
        },
        "required": ["team"],
    },
}


# (name, schema, handler) tuples consumed by __init__.py's register(ctx).
TOOLS = [
    (_TEAM_DISCOVER_SCHEMA["name"], _TEAM_DISCOVER_SCHEMA, _handle_team_discover),
    (_TEAM_STATUS_SCHEMA["name"], _TEAM_STATUS_SCHEMA, _handle_team_status),
    (_TEAM_SPAWN_SCHEMA["name"], _TEAM_SPAWN_SCHEMA, _handle_team_spawn),
    (_INBOX_SEND_SCHEMA["name"], _INBOX_SEND_SCHEMA, _handle_inbox_send),
    (_INBOX_PEEK_SCHEMA["name"], _INBOX_PEEK_SCHEMA, _handle_inbox_peek),
]
