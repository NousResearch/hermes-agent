"""Shared validation contracts for Gemini delegation routing."""

from __future__ import annotations

from typing import Any


_ANTIGRAVITY_CONTROLLED_ARG_PREFIXES = (
    "--dangerously-skip-permissions",
    "--add-dir",
    "--continue",
    "--conversation",
    "--print",
    "--model",
    "--effort",
    "--mode",
    "--sandbox",
    "--disable-slash-commands",
    "--output-format",
    "--print-timeout",
    "--json-schema",
)


def extra_arg_overrides_contract(arg: object) -> bool:
    """Return whether an extra CLI argument can alter a pinned worker boundary."""
    if not isinstance(arg, str):
        return True
    return any(
        arg == prefix
        or arg.startswith(prefix + "=")
        or arg == "--no-" + prefix.removeprefix("--")
        or arg.startswith("--no-" + prefix.removeprefix("--") + "=")
        for prefix in _ANTIGRAVITY_CONTROLLED_ARG_PREFIXES
    )


def validate_antigravity_extra_args(value: Any) -> list[str]:
    """Validate and copy a configured Antigravity extra-argument list."""
    if type(value) is not list:
        raise ValueError("extra_args must be a list of strings")
    if any(extra_arg_overrides_contract(arg) for arg in value):
        raise ValueError("extra_args cannot override the Antigravity execution contract")
    return list(value)


def parse_exact_slack_target(value: Any) -> str:
    """Return the channel ID from exactly one ``slack:<channel-id>`` target."""
    if not isinstance(value, str) or value != value.strip() or not value.startswith("slack:"):
        raise ValueError("review.alert_target must be an exact slack:<channel-id> target")
    channel_id = value.removeprefix("slack:")
    if not channel_id or ":" in channel_id or any(char.isspace() for char in channel_id):
        raise ValueError("review.alert_target must be an exact slack:<channel-id> target")
    return channel_id
