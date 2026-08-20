"""Host chrome hook for Walkie-Talkie peer presence (G7).

The peer plugin (``hermes_peer``) is loaded in-process by the plugin system.
This module is the DEFENSIVE host-side seam: it queries the plugin's
``get_manager().summary()`` for the CLI ●N prompt pill and the TUI status
bar, and degrades to an empty result when the plugin is absent/inactive so
the host chrome never breaks because of it.

Contract: ``peer_presence_summary()`` returns a small dict with keys
``active``, ``total``, ``offline``, ``you`` (name or None), ``live``
(age string or "") — or None when the plugin is unavailable. The caller
renders nothing when the result is None (plugin not installed) and renders
an ambient pill when present.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("hermes_cli.peer_presence")

# Ambient presence chrome renders ONLY when at least two sessions are known
# (total >= 2). With a single session the only peer is you — the indicator
# would be pure noise ("● 1 live" about yourself). All ambient surfaces
# (CLI pill, TUI status bar, desktop pill) share this one threshold so the
# behaviour cannot drift between them.
_MIN_VISIBLE_TOTAL = 2

# Cache the summary for a short window so polling the status bar (which
# repaints often) does not hammer the registry probe on every frame.
_CACHE_SECONDS = 2.0
_cache: Dict[str, Any] = {"at": 0.0, "value": None}


def clear_peer_presence_cache() -> None:
    """Drop the cached summary so the NEXT query re-reads live state.

    Called at turn boundaries so the CLI prompt reflects sessions that
    opened/closed during the previous turn instead of a stale count.
    """
    _cache["at"] = 0.0
    _cache["value"] = None


def _query_peer_summary() -> Optional[Dict[str, Any]]:
    """Query the peer plugin manager summary; None when unavailable."""
    try:
        from hermes_peer.plugin import get_manager
    except Exception:
        return None
    try:
        mgr = get_manager()
        if mgr is None:
            return None
        return mgr.summary()
    except Exception:
        logger.debug("hermes_cli: peer presence summary unavailable", exc_info=True)
        return None


def peer_presence_summary(force: bool = False) -> Optional[Dict[str, Any]]:
    """Return the cached peer-presence summary (or None when inactive).

    ``force=True`` bypasses the short cache (used on the CLI status line,
    which prints once per prompt, not per repaint).
    """
    import time

    now = time.monotonic()
    if not force and now - _cache["at"] < _CACHE_SECONDS:
        return _cache["value"]
    value = _query_peer_summary()
    _cache["at"] = now
    _cache["value"] = value
    return value


def peer_presence_pill(force: bool = False) -> str:
    """Compact ambient pill for the CLI prompt: ``●N`` / ``""``.

    Returns ``●N`` (N = live OPEN interactive sessions) when ``total >= 2``,
    and ``""`` otherwise (single session = you only, or plugin absent). The
    pill is ambient chrome — it exists to signal OTHER live sessions, so
    it hides entirely when there is nothing to signal.
    """
    summary = peer_presence_summary(force=force)
    if summary is None:
        return ""
    live = int(summary.get("live_count") or 0)
    if live < _MIN_VISIBLE_TOTAL:
        return ""
    return f"●{live}"


def peer_presence_status_line(force: bool = True) -> str:
    """Single-line status for the CLI status line (G7).

    Copy uses distinct dot glyphs per state so the renderer can colour them
    independently:
      ``● N Live · ○ M Idle · × K Offline``
    Live = OPEN interactive sessions (probe-live, cli/tui/desktop surface),
    regardless of mid-turn state. Idle = live but not working. Offline = PID
    alive but socket-dead, outside the registration grace period.

    Empty string when the plugin is absent OR fewer than two live sessions
    (single session = you only = no signal to convey).
    """
    summary = peer_presence_summary(force=force)
    if summary is None:
        return ""
    live = int(summary.get("live_count") or 0)
    if live < _MIN_VISIBLE_TOTAL:
        return ""
    active = int(summary.get("active_count") or 0)
    idle = int(summary.get("idle_count") or 0)
    offline = int(summary.get("offline_count") or 0)
    parts = []
    if live:
        parts.append(f"● {live} Live")
    if active and active < live:
        parts.append(f"{active} working")
    if idle:
        parts.append(f"○ {idle} Idle")
    if offline:
        parts.append(f"× {offline} Offline")
    you_id = summary.get("you_peer_id")
    if you_id:
        you_name = None
        for row in (summary.get("peers") or []):
            if row.get("peer_id") == you_id:
                you_name = row.get("name")
                break
        if you_name:
            parts.append(f"you: {you_name}")
    return " · ".join(parts)
