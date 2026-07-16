"""Fork-owned pure helpers for hermes_state.py.

This module intentionally imports nothing from hermes_state. Keep SQL execution,
connection handling, and SessionDB methods in hermes_state.py.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Tuple

# Keep the ORIGINAL logger name: these records logged as "hermes_state" for
# their whole life pre-extraction; renaming the logger would silently move them
# out of any log-filter/handler configured on "hermes_state" (Greptile P2 —
# logger name is an observable output the golden did not capture).
logger = logging.getLogger("hermes_state")

def _sql_placeholders(values) -> str:
    return ",".join("?" for _ in values)


def _session_list_denorm_enabled() -> bool:
    """Lazy config.yaml-only gate for the dormant session.list denorm path."""
    try:
        from hermes_cli.config import cfg_get, read_raw_config

        value = cfg_get(
            read_raw_config(),
            "dashboard",
            "session_list_denorm",
            default=False,
        )
    except Exception as exc:
        logger.debug("dashboard.session_list_denorm read failed: %s", exc)
        return False
    return value is True


# Platform names + common aliases -> sessions.source values, so cross-surface
# search understands "discord" / "tg" / "imessage" as platform intent. Kept in
# sync with the desktop sidebar's SOURCE_LABELS/SOURCE_ALIASES
# (apps/desktop/src/lib/session-source.ts).
_PLATFORM_SEARCH_ALIASES: Dict[str, str] = {
    "bluebubbles": "bluebubbles",
    "imessage": "bluebubbles",
    "cli": "cli",
    "codex": "codex",
    "desktop": "desktop",
    "discord": "discord",
    "email": "email",
    "matrix": "matrix",
    "mattermost": "mattermost",
    "qq": "qqbot",
    "qqbot": "qqbot",
    "signal": "signal",
    "slack": "slack",
    "sms": "sms",
    "telegram": "telegram",
    "tg": "telegram",
    "tui": "tui",
    "webhook": "webhook",
    "wechat": "weixin",
    "weixin": "weixin",
    "whatsapp": "whatsapp",
    "wa": "whatsapp",
    "yuanbao": "yuanbao",
}

def _session_title_search_like(term: str) -> str:
    return (
        "%"
        + term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        + "%"
    )


def _session_title_search_platform_sources(tokens: Iterable[str]) -> List[str]:
    return sorted({src for tok in tokens if (src := _PLATFORM_SEARCH_ALIASES.get(tok))})


def _session_title_search_score(row: Any, tokens: List[str], needle: str) -> Tuple[int, int]:
    title = (row["title"] or "").strip().lower()
    display = (row["display_name"] or "").strip().lower()
    source = (row["source"] or "").strip().lower()
    matched = 0
    best_tier = 6
    for tok in tokens:
        tiers = []
        if tok in title:
            tiers.append(3)
        if tok in display:
            tiers.append(4)
        if _PLATFORM_SEARCH_ALIASES.get(tok) == source:
            tiers.append(5)
        if tiers:
            matched += 1
            best_tier = min(best_tier, *tiers)
    # Whole-phrase title tiers preserve the historical contract that
    # an exact/prefix title hit beats everything else.
    if title == needle:
        best_tier = 0
    elif title.startswith(needle):
        best_tier = 1
    elif needle in title:
        best_tier = 2
    return (-matched, best_tier)
