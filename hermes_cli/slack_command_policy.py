"""Slack-native slash naming and finite manifest policy."""

import re


_SLACK_MAX_SLASH_COMMANDS = 50
_SLACK_NAME_LIMIT = 32
_SLACK_INVALID_CHARS = re.compile(r"[^a-z0-9_\-]")
_SLACK_RESERVED_COMMANDS = frozenset({
    # Built-in Slack slash commands that cannot be registered by apps.
    "me",
    "status",
    "away",
    "dnd",
    "shrug",
    "remind",
    "msg",
    "feed",
    "who",
    "collapse",
    "expand",
    "leave",
    "join",
    "open",
    "search",
    "topic",
    "mute",
    "pro",
    "shortcuts",
})

# Pinned aliases claim slots before canonical names. This stays deliberately
# empty now that /bg and /btw are canonical commands.
_SLACK_PRIORITY_ALIASES: tuple[str, ...] = ()

# Slack caps an app at 50 slash commands. These low-frequency commands stay
# reachable through `/hermes <command>` instead of silently displacing another
# native command when the shared registry grows.
_SLACK_VIA_HERMES_ONLY = frozenset({
    "topup",
    "moa",
    "debug",
    "egress",
    "init",
    "version",
    "diff",
    "update",
    "heartbeat",
    "refine",
    "review",
    "pause",
    "whoami",
    "platform",
    "insights",
    "group",
})


def _sanitize_slack_name(raw: str) -> str:
    """Convert a command name to Slack's native slash-name shape."""
    name = raw.lower()
    name = _SLACK_INVALID_CHARS.sub("", name)
    name = name.strip("-_")
    return name[:_SLACK_NAME_LIMIT]
