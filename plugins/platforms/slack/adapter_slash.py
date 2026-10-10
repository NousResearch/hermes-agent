"""Slash- and bang-command text parsing for the Slack adapter.

Stateless helpers that turn a Slack slash payload or a typed ``!cmd`` into the gateway's
``/command`` text (consulting the command registry), including the optional namespace
prefix shared by both surfaces.
"""

from __future__ import annotations

from typing import Optional


def _rewrite_known_bang_command(text: str, prefix: str = "") -> str:
    """Rewrite a known leading ``!cmd`` to the gateway ``/cmd`` form.

    Mirrors the native-slash surface: with a namespace prefix configured
    (e.g. ``myorg-``), bang commands carry the same prefix (``!myorg-stop``)
    and it is stripped before dispatch. Unprefixed bangs stay plain text so
    two namespaced apps sharing a channel do not both execute ``!stop``.
    """
    if not text.startswith("!"):
        return text
    tokens = text[1:].split(maxsplit=1)
    if not tokens:
        return text
    cmd_name = tokens[0].split("@", 1)[0].lower()
    rest = text[1:]
    if prefix:
        if not cmd_name.startswith(prefix):
            return text
        cmd_name = cmd_name[len(prefix) :]
        rest = rest[len(prefix) :]
    if not cmd_name or "/" in cmd_name:
        return text
    from hermes_cli.commands import is_gateway_known_command
    return "/" + rest if is_gateway_known_command(cmd_name) else text


def _slash_command_text(command: dict, prefix: str = "") -> str:
    """Gateway message text for a slash payload. Native slashes keep Slack's raw argument
    payload verbatim (internal/trailing spacing). ``/hermes`` (or a missing ``command``) maps
    ``<subcommand> [args]`` via the registry, else free-form text is a regular question.

    ``prefix`` is the configured namespace: ``/myorg-model`` resolves to ``model`` and
    ``/myorg-hermes`` to the legacy catch-all, so the gateway never sees the namespace."""
    slash_name = (command.get("command") or "").lstrip("/").strip()
    if prefix and slash_name.startswith(prefix):
        slash_name = slash_name[len(prefix):]
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


def _slash_thread_id(command: dict) -> Optional[str]:
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
