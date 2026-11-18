"""Which cache tier a chat a person types into asks for under ``cache_ttl: auto``: 5m until its first pause.

A 1h write costs 2x input against 1.25x for 5m, and only pays when the person pauses for more than five
minutes. A chat that never pauses (back-to-back tool loops) would pay the premium on every write for
nothing, so an ``auto`` human chat starts on 5m and asks for 1h from the first request after an idle gap
longer than five minutes. That request finds its 5m entry already expired, so changing the markers' TTL
breaks no warm prefix: the upgrade only happens once the clock says the 5m window has passed. Once
upgraded a session stays on 1h: the choice is kept per session id (an agent rebuilt for the same session
by gateway eviction, the child session a compaction continues in) and read back from the session's stored
clock after a restart (its last request asked for 1h).

Before the upgrade no idle keep-alive is sent: the chat has not shown a pause yet, and the first long
one pays a single 5m rewrite and then upgrades. Keep-alives while a tool runs are unchanged. Explicit
pins (``cache_ttl: 5m`` / ``1h``) and machine-paced sessions never take this path.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Optional

#: What the adaptive start asks for before and after the first pause.
START_TIER, UPGRADED_TIER = "5m", "1h"
#: The gap (seconds since the cache was last touched) that upgrades: the 5m window has passed.
UPGRADE_AFTER_SECONDS = 300.0

#: Session ids that upgraded (bounded LRU): an agent rebuilt for the session, or the compaction child, keeps 1h.
_UPGRADED: OrderedDict[str, bool] = OrderedDict()
_UPGRADED_MAX = 1000
_lock = threading.Lock()


def adaptive_start_tier(agent: Any, resolved: str) -> str:
    """The tier an ``auto`` agent starts on: ``resolved`` (the source's choice) except that a 1h choice
    starts on 5m and is marked for the upgrade."""
    agent._cache_ttl_adaptive = resolved == UPGRADED_TIER
    return START_TIER if agent._cache_ttl_adaptive else resolved


def _remember(session_id: Optional[str]) -> None:
    if not session_id:
        return
    with _lock:
        _UPGRADED[session_id] = True
        _UPGRADED.move_to_end(session_id)
        while len(_UPGRADED) > _UPGRADED_MAX:
            _UPGRADED.popitem(last=False)


def _upgraded(session_id: Optional[str]) -> bool:
    with _lock:
        return bool(session_id) and session_id in _UPGRADED


def _session_upgraded(agent: Any) -> bool:
    """The session already asks for 1h: remembered in this process, or its stored clock (restored on a
    resumed or rebuilt agent) was last written by a 1h request, so 1h markers match what is cached."""
    from agent.prompt_cache_clock import CacheEvent, cache_window

    if _upgraded(getattr(agent, "session_id", None)):
        return True
    cache_window(agent)  # restores the stored clock once per agent
    event = getattr(agent, "_prompt_cache_clock", None)
    return isinstance(event, CacheEvent) and event.requested_ttl == UPGRADED_TIER


def carry_cache_tier(old_session_id: Optional[str], new_session_id: Optional[str]) -> None:
    """A compaction continued the conversation in a child session: its upgrade goes with it."""
    if _upgraded(old_session_id):
        _remember(new_session_id)


def _paused_past_start_window(agent: Any, now: float) -> bool:
    """The clock says the 5m entry is gone: no guarantee is left and the cache was last touched more
    than :data:`UPGRADE_AFTER_SECONDS` ago. A chat that never touched a cache has not paused."""
    from agent.prompt_cache_clock import cache_certainly_warm, cache_window

    window = cache_window(agent)
    refreshed = getattr(window, "refreshed_at", None)
    if window is None or refreshed is None or cache_certainly_warm(window, now=now):
        return False
    return now - float(refreshed) >= UPGRADE_AFTER_SECONDS


def maybe_upgrade_cache_tier(agent: Any, *, now: Optional[float] = None) -> bool:
    """Turn start of a new user message: an adaptive chat still on 5m asks for 1h from now on when the
    session already upgraded (a rebuilt agent, a compaction child) or this message ends its first pause
    longer than five minutes. Returns whether the tier is 1h after the call."""
    from agent.prompt_cache_lifecycle.edge import human_paced

    if not getattr(agent, "_cache_ttl_adaptive", False) or not getattr(agent, "_cache_ttl_auto", False):
        return False
    if getattr(agent, "_cache_ttl", None) == UPGRADED_TIER:
        return True
    if not human_paced(agent):
        return False
    if not _session_upgraded(agent) and not _paused_past_start_window(agent, time.time() if now is None else now):
        return False
    agent._cache_ttl = UPGRADED_TIER
    _remember(getattr(agent, "session_id", None))
    return True


def idle_warming_allowed(agent: Any) -> bool:
    """Idle keep-alives (after a turn ends) only once an adaptive chat has upgraded to 1h."""
    return not getattr(agent, "_cache_ttl_adaptive", False) or getattr(agent, "_cache_ttl", None) == UPGRADED_TIER


def tier_pending(agent: Any) -> bool:
    """An adaptive chat still on 5m (``/context`` says it moves to 1h after the first pause)."""
    return bool(getattr(agent, "_cache_ttl_adaptive", False)) and getattr(agent, "_cache_ttl", None) == START_TIER


def reset_cache_tier(agent: Any) -> None:
    """A fresh session (``/new``): an adaptive chat starts on 5m again."""
    if getattr(agent, "_cache_ttl_adaptive", False):
        agent._cache_ttl = START_TIER
