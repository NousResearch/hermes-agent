"""Slack slash-command payload decoding: the gateway text and the thread anchor of a slash."""

from __future__ import annotations

from typing import Optional


def slash_command_text(command: dict) -> str:
    """Gateway message text for a slash payload. Native slashes keep Slack's raw argument
    payload verbatim (internal/trailing spacing). ``/hermes`` (or a missing ``command``) maps
    ``<subcommand> [args]`` via the registry, else free-form text is a regular question."""
    slash_name = (command.get("command") or "").lstrip("/").strip()
    raw_text = str(command.get("text") or "")
    if slash_name not in {"hermes", ""}:
        return f"/{slash_name}" if not raw_text else f"/{slash_name} {raw_text}"
    legacy_text = raw_text.strip()
    from hermes_cli.commands_platforms import slack_subcommand_map
    subcommand_map = slack_subcommand_map()
    subcommand_map["compact"] = "/compress"
    first_word = legacy_text.split()[0] if legacy_text.split() else ""
    if first_word in subcommand_map:
        rest = legacy_text[len(first_word) :].strip()
        mapped = subcommand_map[first_word]
        return f"{mapped} {rest}".strip() if rest else mapped
    return legacy_text or "/help"


def slash_thread_id(command: dict) -> Optional[str]:
    """Thread anchor for a slash payload so session-scoped commands (``/model``)
    hit the same thread session. Shape varies by surface: top-level or nested
    ``message``/``container``; ``thread_ts`` preferred over ``message_ts``."""
    nested = (command.get(k) for k in ("message", "container"))
    candidates = [command] + [n for n in nested if isinstance(n, dict)]
    for ts_key in ("thread_ts", "message_ts"):
        for payload in candidates:
            value = payload.get(ts_key)
            if value:
                return str(value)
    return None
