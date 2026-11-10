"""Opt-in presentation for public Slack commentary, not model reasoning."""

import re

_STARTS_WITH_BALLOON = re.compile(r"^\s*(?:💬\ufe0f?|:speech_balloon:)")
_TRUTHY = {"1", "true", "yes", "on"}


def _enabled(extra: dict) -> bool:
    value = extra.get("interim_speech_balloon")
    return value.strip().lower() in _TRUTHY if isinstance(value, str) else bool(value)


def style_public_message(content: str, metadata: dict | None, extra: dict) -> str:
    """Prefix assistant commentary with 💬 when enabled.

    Additive-only and interim-only: prefixes a balloon when ``_assistant_commentary``
    is true and the message does not already start with a speech balloon marker.
    Never strips, mutates, or rewrites existing content or final responses.
    """
    if not _enabled(extra):
        return content
    if not (metadata or {}).get("_assistant_commentary"):
        return content
    if not content.strip() or _STARTS_WITH_BALLOON.match(content):
        return content
    return f"💬 {content.lstrip()}"
