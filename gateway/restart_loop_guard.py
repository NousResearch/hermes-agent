"""Auto-resume restart-loop breaker (#30719, defense-3).

Defenses 1 and 2 (the ``_HERMES_GATEWAY`` guard on ``hermes gateway
stop|restart`` + ``terminal_tool``, and the cron-creation lifecycle
filter) stop the agent from scheduling its own restart via the cron and
CLI paths.  They do NOT cover every SIGTERM source: an agent running a
raw ``terminal("launchctl kickstart -k gui/<uid>/ai.hermes.gateway")``,
an external monitor with a bad trigger, or any other repeated crash can
still drive the supervisor (launchd ``KeepAlive`` / systemd ``Restart=``)
into a tight respawn loop.  On each boot the gateway auto-resumes the
restart-interrupted session, whose next turn re-runs the offending
logic — SIGTERM every ~10 seconds until manually broken.

This module is the last-resort circuit breaker: it records a timestamp
each time the gateway boots with restart-interrupted sessions pending,
keeps the current chain of such boots persisted across processes (each
boot is a fresh process, so in-memory state is useless), and reports the
loop as "tripped" once too many of them chain together.  Boots chain
while consecutive gaps stay within ``max_gap_seconds``, so the breaker
sees slow crash cycles (a wedged event loop killed by the liveness
watchdog every ~150s, #81642) exactly as well as the fast ~10s respawn
loop it was originally written for.
When tripped, the caller SKIPS auto-resume for that boot — the gateway
still starts and serves real inbound messages, it just stops replaying
the session that keeps killing it, which breaks the cycle and puts a
human back in the loop.

State lives in ``<HERMES_HOME>/gateway/restart_loop.json`` so it is
profile-scoped and survives process death.  It is intentionally tiny and
best-effort: any read/write failure fails OPEN (no false trip) because a
broken breaker must never wedge a healthy gateway.
"""

from __future__ import annotations

import json
import logging
import time
from typing import List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger("gateway.run")

# Defaults chosen so a legitimate operator restart (or two) never trips the
# breaker, but the documented ~10s respawn loop does within a few cycles.
DEFAULT_MAX_RESTARTS = 3
DEFAULT_WINDOW_SECONDS = 60

# Longest gap between two consecutive restart-interrupted boots that still
# counts them as the SAME loop (#81642).  A fixed ``window_seconds`` prune can
# only see crash cycles faster than the window: a loop whose period exceeds it
# drops its own history on every boot, so the counter never leaves 1 and the
# breaker never trips no matter how long the loop runs.  The reported cycle was
# ~150s (wedged event loop -> ~90s liveness watchdog hard-exit -> respawn ->
# auto-resume replays the same session), i.e. structurally invisible to the 60s
# window.  Chaining on the inter-boot GAP instead makes the breaker period-
# agnostic: any repeating cycle trips once ``max_restarts`` links accumulate,
# and a single boot followed by real quiet resets the chain.
DEFAULT_MAX_GAP_SECONDS = 300

# Cap the persisted chain so a long-running loop cannot grow the state file
# without bound.  Only the newest ``max_restarts`` entries can change a
# verdict; the rest are kept for forensics.
_MAX_STORED_BOOTS = 50


def _state_path():
    return get_hermes_home() / "gateway" / "restart_loop.json"


def _load_boots() -> List[float]:
    try:
        raw = _state_path().read_text(encoding="utf-8")
        data = json.loads(raw)
        boots = data.get("boots", [])
        return [float(t) for t in boots if isinstance(t, (int, float))]
    except (OSError, ValueError, TypeError):
        return []


def _save_boots(boots: List[float]) -> None:
    try:
        path = _state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"boots": boots}), encoding="utf-8")
    except OSError:
        pass


def _chain_gap(window_seconds: int, max_gap_seconds: int) -> float:
    """Effective inter-boot gap that still links two boots into one loop.

    Floored by ``window_seconds`` so an operator who widens the window never
    ends up with a breaker that is *less* sensitive than they asked for.
    """
    return float(max(1, window_seconds, max_gap_seconds))


def _chain_ending_at(boots: List[float], ts: float, gap: float) -> List[float]:
    """Return the unbroken chain of boots leading up to ``ts``.

    Walks backwards from ``ts`` and keeps boots while each successive gap stays
    within ``gap``.  The first gap that exceeds it ends the chain: everything
    older belongs to a previous, already-resolved episode.  A chain broken at
    the head (nothing recent enough) yields an empty list, which is how a
    healthy gateway forgets an old loop.
    """
    chain: List[float] = []
    prev = ts
    for t in sorted(boots, reverse=True):
        if t > ts:
            # Clock moved backwards (NTP step, restored state file). Treat the
            # future entry as adjacent rather than dropping the whole chain.
            chain.append(t)
            continue
        if prev - t > gap:
            break
        chain.append(t)
        prev = t
    chain.reverse()
    return chain


def record_restart_interrupted_boot(
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    *,
    now: Optional[float] = None,
    max_gap_seconds: int = DEFAULT_MAX_GAP_SECONDS,
) -> List[float]:
    """Record that the gateway just booted with restart-interrupted sessions.

    Drops boots that belong to an earlier, already-broken chain (any gap wider
    than ``max_gap_seconds``) and appends the current time.  Returns the
    pruned+appended list (most recent last).  Best-effort — a persistence
    failure returns the in-memory list without raising.
    """
    ts = time.time() if now is None else now
    gap = _chain_gap(window_seconds, max_gap_seconds)
    boots = _chain_ending_at(_load_boots(), ts, gap)
    boots.append(ts)
    _save_boots(boots[-_MAX_STORED_BOOTS:])
    return boots


def is_restart_loop_tripped(
    max_restarts: int = DEFAULT_MAX_RESTARTS,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    *,
    now: Optional[float] = None,
    max_gap_seconds: int = DEFAULT_MAX_GAP_SECONDS,
) -> bool:
    """Return True if the gateway has restarted ``>= max_restarts`` times with
    restart-interrupted sessions in one unbroken chain ending at ``now``.

    Reads the persisted boot log written by
    ``record_restart_interrupted_boot`` and counts the boots that still chain
    together (consecutive gaps within ``max_gap_seconds``), so the verdict does
    not depend on how fast the crash cycle happens to be.
    Fails OPEN (returns False) on any error — a broken breaker must never
    wedge a healthy gateway.
    """
    if max_restarts <= 0:
        return False
    ts = time.time() if now is None else now
    gap = _chain_gap(window_seconds, max_gap_seconds)
    try:
        recent = _chain_ending_at(_load_boots(), ts, gap)
    except Exception:  # pragma: no cover — _load_boots already guards
        return False
    return len(recent) >= max_restarts


def clear() -> None:
    """Remove the persisted boot log (used on clean shutdown / by tests)."""
    try:
        _state_path().unlink(missing_ok=True)
    except OSError:
        pass


def check_and_record(
    max_restarts: int = DEFAULT_MAX_RESTARTS,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    *,
    now: Optional[float] = None,
    max_gap_seconds: int = DEFAULT_MAX_GAP_SECONDS,
) -> bool:
    """Record this restart-interrupted boot and report whether the loop is now
    tripped.

    This is the single entry point the gateway calls: it appends the current
    boot, then checks whether the (now-updated) chain has reached the
    threshold.  Returns True when auto-resume should be SKIPPED to break the
    loop.
    """
    boots = record_restart_interrupted_boot(
        window_seconds, now=now, max_gap_seconds=max_gap_seconds
    )
    tripped = len(boots) >= max_restarts if max_restarts > 0 else False
    if tripped:
        logger.warning(
            "Restart-loop breaker TRIPPED: %d chained restart-interrupted "
            "gateway boots (no gap wider than %ds; threshold %d). Skipping "
            "auto-resume to break a suspected SIGTERM-respawn loop (#30719, "
            "#81642). Restart-interrupted sessions stay resume-pending and "
            "will continue on the next real user message. If this is a false "
            "positive, delete %s.",
            len(boots),
            int(_chain_gap(window_seconds, max_gap_seconds)),
            max_restarts,
            _state_path(),
        )
    return tripped
