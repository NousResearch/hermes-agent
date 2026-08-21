"""Model-facing Buzz channel operations for the bundled Buzz platform."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from .adapter import (
    _cli_error_message,
    _exec_buzz,
    _resolve_cli_path,
    _resolve_private_key,
)

_CHANNEL_TYPES = ("stream", "forum")
_VISIBILITIES = ("open", "private")


def _result(**payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _check_requirements() -> bool:
    """Return whether this profile can launch authenticated Buzz CLI calls."""
    return bool(
        os.getenv("BUZZ_RELAY_URL", "").strip()
        and _resolve_private_key()
        and _resolve_cli_path(os.getenv("BUZZ_CLI_PATH", "").strip())
    )


async def buzz_channels(args: dict | None = None, **_: Any) -> str:
    """Create or list channels with the active Hermes profile's Buzz identity."""
    values = args or {}
    action = str(values.get("action") or "").strip().lower()
    relay_url = os.getenv("BUZZ_RELAY_URL", "").strip()
    private_key = _resolve_private_key()
    cli_path = _resolve_cli_path(os.getenv("BUZZ_CLI_PATH", "").strip())

    if not relay_url:
        return _result(ok=False, error="BUZZ_RELAY_URL is not configured")
    if not private_key:
        return _result(
            ok=False, error="BUZZ_PRIVATE_KEY is not configured for this profile"
        )
    if not cli_path:
        return _result(ok=False, error="buzz CLI binary was not found")

    if action == "create":
        name = str(values.get("name") or "").strip()
        if not name:
            return _result(ok=False, error="name is required for create")
        channel_type = str(values.get("channel_type") or "stream").strip().lower()
        if channel_type not in _CHANNEL_TYPES:
            return _result(ok=False, error="channel_type must be stream or forum")
        visibility = str(values.get("visibility") or "private").strip().lower()
        if visibility not in _VISIBILITIES:
            return _result(ok=False, error="visibility must be open or private")
        cli_args = [
            "channels",
            "create",
            "--name",
            name,
            "--type",
            channel_type,
            "--visibility",
            visibility,
        ]
        description = str(values.get("description") or "").strip()
        if description:
            cli_args.extend(["--description", description])
    elif action == "list":
        limit = values.get("limit", 100)
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 500
        ):
            return _result(ok=False, error="limit must be an integer from 1 to 500")
        cli_args = ["channels", "list", "--limit", str(limit)]
        visibility = str(values.get("visibility") or "").strip().lower()
        if visibility:
            if visibility not in _VISIBILITIES:
                return _result(ok=False, error="visibility must be open or private")
            cli_args.extend(["--visibility", visibility])
        if values.get("member_only") is True:
            cli_args.append("--member")
    else:
        return _result(ok=False, error="action must be create or list")

    try:
        code, stdout, stderr = await _exec_buzz(
            cli_path,
            cli_args,
            relay_url=relay_url,
            private_key=private_key,
        )
    except asyncio.CancelledError:
        raise
    except OSError as exc:
        return _result(ok=False, error=f"failed to launch buzz CLI: {exc}")

    if code != 0:
        return _result(ok=False, error=_cli_error_message(stderr, code))
    try:
        data = json.loads(stdout)
    except ValueError:
        return _result(ok=False, error="buzz CLI returned invalid JSON")
    return _result(ok=True, action=action, result=data)


_SCHEMA = {
    "type": "function",
    "function": {
        "name": "buzz_channels",
        "description": (
            "Create or list Buzz channels using this Hermes profile's Buzz identity. "
            "Creating a channel is a durable workspace change."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list"],
                    "description": "Channel operation to perform.",
                },
                "name": {
                    "type": "string",
                    "description": "Channel name. Required for create.",
                },
                "channel_type": {
                    "type": "string",
                    "enum": list(_CHANNEL_TYPES),
                    "description": "Channel type for create. Defaults to stream.",
                },
                "visibility": {
                    "type": "string",
                    "enum": list(_VISIBILITIES),
                    "description": "Visibility for create or list filtering. Create defaults to private.",
                },
                "description": {
                    "type": "string",
                    "description": "Optional channel description for create.",
                },
                "member_only": {
                    "type": "boolean",
                    "description": "For list, return only channels where this identity is a member.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 500,
                    "description": "Maximum channels returned by list. Defaults to 100.",
                },
            },
            "required": ["action"],
        },
    },
}


def register_tools(ctx) -> None:
    """Register the opt-in Buzz channel-management toolset."""
    ctx.register_tool(
        name="buzz_channels",
        toolset="buzz_admin",
        schema=_SCHEMA,
        handler=buzz_channels,
        check_fn=_check_requirements,
        is_async=True,
        description=_SCHEMA["function"]["description"],
        emoji="🐝",
    )
