"""live-time plugin — inject live current time into every LLM call.

Fixes the stale "Conversation started" timestamp problem: Hermes stamps the
session-start time once and never refreshes it, so long or cross-day
conversations leave the model with no sense of "now". This plugin injects the
real current time at every LLM call — ephemeral, on the user-message side,
never touching the cached system prompt (prompt caching stays intact).

Timezone resolution:
    1. ``HERMES_TIMEZONE`` environment variable
    2. top-level ``timezone`` key in ``<HERMES_HOME | ~/.hermes>/config.yaml``
       (parsed with ``yaml.safe_load``; note this is the default-profile
       config — see README for the profile caveat)
    3. local system timezone

See https://github.com/NousResearch/hermes-agent/issues/10421
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _resolve_timezone_name() -> str:
    """Return a configured IANA timezone name, or "" for system-local."""
    tz = os.environ.get("HERMES_TIMEZONE", "").strip()
    if tz:
        return tz
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    cfg_path = Path(home) / "config.yaml"
    try:
        if cfg_path.exists():
            import yaml  # PyYAML ships with the Hermes runtime

            loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                tz_cfg = loaded.get("timezone")
                if isinstance(tz_cfg, str) and tz_cfg.strip():
                    return tz_cfg.strip()
    except (OSError, ImportError, ValueError):
        pass  # unreadable/unparsable config — fall through to system local
    return ""


def _current_time() -> tuple[datetime, str]:
    """Now in the configured timezone; fall back to system-local time.

    Returns ``(now, tz_name)`` where ``tz_name`` is "" when system-local.
    An unknown/invalid configured timezone falls back to system-local time
    rather than raising.
    """
    tz_name = _resolve_timezone_name()
    if tz_name:
        try:
            from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

            return datetime.now(ZoneInfo(tz_name)), tz_name
        except (ZoneInfoNotFoundError, ValueError):
            pass  # unknown/invalid tz — fall through to system local
    return datetime.now().astimezone(), ""


def _on_pre_llm_call(**_: Any) -> Optional[dict]:
    """Return live-time context injected before every LLM call.

    The returned ``{"context": ...}`` block is appended on the user-message
    side of the request, so the agent always sees the current local time
    without invalidating prompt-cache prefixes.
    """
    now, tz_name = _current_time()
    offset_h = now.utcoffset().total_seconds() / 3600
    weekday = _WEEKDAYS[now.weekday()]
    tz_label = tz_name or f"UTC{offset_h:+.0f}"

    context = (
        f"[LIVE-TIME] Now: {now.strftime('%Y-%m-%d %H:%M:%S')} "
        f"(Weekday {now.weekday() + 1}/7, {weekday}), "
        f"TZ {tz_label}. "
        "Injected by live-time plugin at THIS LLM call's moment. "
        "Use THIS as the authoritative current time for any today/now/elapsed/"
        "date judgment. Ignore the 'Conversation started' timestamp in the "
        "system prompt for anything except session-creation facts."
    )
    return {"context": context}


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
