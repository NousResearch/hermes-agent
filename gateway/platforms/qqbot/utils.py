"""QQBot shared utilities — User-Agent, HTTP helpers, config coercion."""

from __future__ import annotations

import platform
import re
import sys
from typing import Any, Dict, List

from .constants import QQBOT_VERSION


def _get_hermes_version() -> str:
    """Return the hermes-agent package version, or 'dev' if unavailable."""
    try:
        from importlib.metadata import version
        return version("hermes-agent")
    except Exception:
        return "dev"


def build_user_agent() -> str:
    """``QQBotAdapter/<qqbot_version> (Python/<py_version>; <os>; Hermes/<hermes_version>)``."""
    v = sys.version_info
    return (f"QQBotAdapter/{QQBOT_VERSION} (Python/{v.major}.{v.minor}.{v.micro}; "
            f"{platform.system().lower()}; Hermes/{_get_hermes_version()})")


def get_api_headers() -> Dict[str, str]:
    """Standard QQBot API headers. ``q.qq.com`` requires ``Accept: application/json``
    — without it the server returns a JavaScript anti-bot challenge page."""
    return {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": build_user_agent()}


def coerce_list(value: Any) -> List[str]:
    """Coerce a comma-separated string / list / tuple / set / scalar into a trimmed string list."""
    if value is None:
        return []
    items = value.split(",") if isinstance(value, str) else value if isinstance(value, (list, tuple, set)) else [value]
    return [s for s in (str(item).strip() for item in items) if s]


def format_mentions(value: Any) -> str:
    """Expose QQ's separate mention list without inventing positions in content.

    Official event schema: /wiki/develop/api-v2/autogen/event/group_at_message_create.html.
    Missing lists convey no identity; bot mentions are not member mentions.
    """
    if not isinstance(value, list):
        return ""
    members: List[str] = []
    seen: set[str] = set()
    for user in value:
        if not isinstance(user, dict) or user.get("bot") is True:
            continue
        member_id = str(user.get("member_openid") or user.get("id") or "").strip()
        name = str(user.get("username") or "").strip()
        identity = member_id or name
        if not identity or identity in seen:
            continue
        seen.add(identity)
        label = f"@{name}" if name else "@member"
        members.append(f"{label} (member_openid={member_id})" if member_id else label)
    return "[QQ mentioned users]: " + "; ".join(members) if members else ""


def render_group_mentions(content: str, mentions: Any, bot_id: str = "") -> str:
    """Use full-mode <@OpenID> positions; remove only this bot's invocation."""
    names: Dict[str, str] = {}
    if isinstance(mentions, list):
        for user in mentions:
            if isinstance(user, dict):
                member_id = str(user.get("member_openid") or user.get("id") or "").strip()
                name = str(user.get("username") or "").strip()
                if member_id and name:
                    names[member_id] = name

    def replace(match: re.Match[str]) -> str:
        member_id = match.group(1)
        if bot_id and member_id == bot_id:
            return ""
        return "@" + names[member_id] if member_id in names else match.group(0)

    return re.sub(r"<@!?([^<>\s]+)>", replace, content).strip()


def mentions_this_bot(content: str, mentions: Any, bot_id: str) -> bool:
    """All-message delivery must not change which messages invoke the bot."""
    if not bot_id:
        return False
    if re.search(r"<@!?" + re.escape(bot_id) + r">", content):
        return True
    return isinstance(mentions, list) and any(
        isinstance(user, dict) and str(user.get("member_openid") or user.get("id") or "") == bot_id
        for user in mentions
    )
