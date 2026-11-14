"""Whether an inbound QQ group message should get a reply.

Mention mode (the default) replies only when the bot is @-ed. Always mode
replies to every group message the platform delivers. ``GROUP_AT_MESSAGE_CREATE``
is itself proof of an @. Full-push ``GROUP_MESSAGE_CREATE`` is checked against
``mentions[].is_you`` and an ``<@!app_id>`` tag for this bot only.
"""

from __future__ import annotations

import re
from typing import Any, Optional

GROUP_AT_MESSAGE_CREATE = "GROUP_AT_MESSAGE_CREATE"
GROUP_MESSAGE_CREATE = "GROUP_MESSAGE_CREATE"

# An @ of some other member must not count. The captured id has to be our app id.
_MENTION_TAG_RE = re.compile(r"<@!?(\d+)>")


def config_flag(value: Any, default: bool) -> bool:
    """YAML bool, or the strings true/false. Missing stays at *default*."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def per_group_require_mention(groups: Any) -> dict[str, bool]:
    """``groups.{openid}.require_mention`` overrides. Other keys are ignored."""
    if not isinstance(groups, dict):
        return {}
    overrides: dict[str, bool] = {}
    for group_id, cfg in groups.items():
        if isinstance(cfg, dict) and "require_mention" in cfg:
            overrides[str(group_id)] = config_flag(cfg["require_mention"], True)
    return overrides


def history_limit(value: Any, default: int = 20) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def detect_mentioned(event_type: str, payload: dict[str, Any], content: str, app_id: str = "") -> bool:
    """True when this message addresses the bot.

    ``GROUP_AT_MESSAGE_CREATE`` always does. A full-push event does when a
    mention entry has ``is_you is True``, or the raw text tags our app id.
    """
    if event_type == GROUP_AT_MESSAGE_CREATE:
        return True
    mentions = payload.get("mentions")
    if isinstance(mentions, list):
        for entry in mentions:
            if isinstance(entry, dict) and entry.get("is_you") is True:
                return True
    if app_id and content:
        target = str(app_id)
        for match in _MENTION_TAG_RE.finditer(content):
            if match.group(1) == target:
                return True
    return False


def resolve_require_mention(
    group_openid: str, *, global_default: bool,
    per_group: Optional[dict[str, bool]] = None,
    runtime_overrides: Optional[dict[str, bool]] = None,
) -> bool:
    """Effective mention gate. Runtime override, then per-group, then global.

    True means reply only when @-ed. False means every delivered group message
    may start a turn.
    """
    if runtime_overrides and group_openid in runtime_overrides:
        return bool(runtime_overrides[group_openid])
    if per_group and group_openid in per_group:
        return bool(per_group[group_openid])
    return bool(global_default)
