"""``origin_url`` on session rows: a link that opens a messaging session's conversation in its
platform's own client ("Open in Slack"), built from the row's stored ``origin_json``.

The platform plugin owns the URL shape through ``PlatformEntry.thread_url_fn`` (its adapter's
``thread_url``); core never knows a platform's link format. Rows are read from one profile's store
at a time, and each profile loads its own plugins, so the lookup runs under that profile's home:
one ``serve`` lists every profile's sessions, and the launch profile's registry would answer for
plugins another profile disabled or replaced.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from agent.runtime_scope import bind_profile_scope

logger = logging.getLogger(__name__)


def _origin(row: dict[str, Any]) -> Optional[dict[str, Any]]:
    raw = row.get("origin_json")
    if not raw:
        return None
    try:
        origin = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return origin if isinstance(origin, dict) and origin.get("platform") and origin.get("chat_id") else None


def _link_builder(platform: str) -> Optional[Callable[[Any], Optional[str]]]:
    from gateway.platform_registry import platform_registry
    entry = platform_registry.get(platform)
    return getattr(entry, "thread_url_fn", None)


def _thread_url(build: Callable[[Any], Optional[str]], origin: dict[str, Any]) -> Optional[str]:
    from gateway.session import SessionSource
    try:
        url = build(SessionSource.from_dict(origin))
    except Exception:  # health: allow BLE001 -- plugin code: one bad origin must not fail the list
        logger.debug("thread_url failed for a %s session", origin.get("platform"), exc_info=True)
        return None
    return url if isinstance(url, str) and url.startswith("https://") else None


def stamp_origin_urls(rows: Iterable[dict[str, Any]], home: str | Path) -> None:
    """Set ``row["origin_url"]`` (None when unknown) on every row read from *home*'s store."""
    rows = list(rows)
    pending = [(row, origin) for row in rows if (origin := _origin(row)) is not None]
    for row in rows:
        row["origin_url"] = None
    if not pending:
        return
    from hermes_cli.plugins import discover_plugins

    with bind_profile_scope(home, hydrate_secrets=False):
        discover_plugins()
        builders: dict[str, Optional[Callable[[Any], Optional[str]]]] = {}
        for row, origin in pending:
            platform = str(origin["platform"])
            if platform not in builders:
                builders[platform] = _link_builder(platform)
            if (build := builders[platform]) is not None:
                row["origin_url"] = _thread_url(build, origin)
