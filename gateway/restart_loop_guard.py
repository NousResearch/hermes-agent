"""Auto-resume restart-loop breaker (defense-3).

Defenses 1-2 (``_HERMES_GATEWAY`` guard on ``hermes gateway stop|restart`` /
``terminal_tool``, cron lifecycle filter) stop the agent scheduling its own restart
but not every SIGTERM source: the supervisor respawns, the gateway auto-resumes the
restart-interrupted session, whose next turn re-runs the offending logic.  Boots are
persisted to ``<HERMES_HOME>/gateway/restart_loop.json`` and CHAIN while gaps stay
within ``max_gap_seconds`` (a ~150s watchdog-kill cycle trips like a ~10s loop).
Tripped → caller SKIPS auto-resume.  Any I/O failure fails OPEN, never wedging.

The same chain, kept in its own file, mutes the home-channel shutdown broadcast
(``notice_loop_tripped``): a loop with nothing to resume still posts one notice
per cycle otherwise.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from typing import List, Optional

from hermes_constants import get_hermes_home, get_process_hermes_home

logger = logging.getLogger("gateway.run")

# A legitimate operator restart (or two) never trips; a ~10s respawn loop does
# within a few cycles.
DEFAULT_MAX_RESTARTS = 3
DEFAULT_WINDOW_SECONDS = 60
# Longest gap between consecutive restart-interrupted boots still counted as the
# SAME loop.  A fixed-window prune only sees cycles faster than the window (a slower
# loop drops its history every boot and never trips); chaining on the inter-boot
# gap is period-agnostic, and real quiet resets it.
DEFAULT_MAX_GAP_SECONDS = 300
# Only the newest ``max_restarts`` entries can change a verdict; the rest are forensics.
_MAX_STORED_BOOTS = 50


def _state_path():
    return get_hermes_home() / "gateway" / "restart_loop.json"


def _notice_state_path():
    # The host process's home, like the planned-restart marker it pairs with: one process, one loop.
    return get_process_hermes_home() / "gateway" / "lifecycle_notice_loop.json"


def _load_boots(path=None) -> list[float]:
    try:
        # utf-8-sig: our BOM-tolerant read fix for the persisted boot log.
        raw = (path or _state_path()).read_text(encoding="utf-8-sig")
        data = json.loads(raw)
        boots = data.get("boots", [])
        return [float(t) for t in boots if isinstance(t, (int, float))]
    except (OSError, ValueError, TypeError):
        return []


def _save_boots(boots: list[float], path=None) -> None:
    with contextlib.suppress(OSError):
        path = path or _state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"boots": boots}), encoding="utf-8")


def _chain_gap(window_seconds: int, max_gap_seconds: int) -> float:
    """Inter-boot gap that still links two boots; floored by ``window_seconds`` so
    widening the window never makes the breaker *less* sensitive."""
    return float(max(1, window_seconds, max_gap_seconds))


def _chain_ending_at(boots: list[float], ts: float, gap: float) -> list[float]:
    """Unbroken chain of boots leading up to ``ts`` (oldest first): walks backwards
    while each gap stays within ``gap``; the first wider gap ends the chain (older
    boots are a resolved episode).  Empty when nothing is recent — how a healthy
    gateway forgets a loop."""
    chain: list[float] = []
    prev = ts
    for t in sorted(boots, reverse=True):
        if t > ts:
            # Clock moved backwards (NTP step, restored RTC/file): a *small* future entry is
            # adjacent to `ts`, not a break. An arbitrary distant one is not — admitting it
            # would chain boots from an unrelated episode weeks ago into "now" and trip the
            # breaker, violating this module's fail-open contract (#93427). Bound the
            # adjacency by the same `gap` every other link uses, and `continue` rather than
            # `break`: the walk is sorted descending, so a distant future entry is visited
            # FIRST and breaking there would silently drop an otherwise valid chain.
            if t - ts > gap:
                continue
            chain.append(t)
            continue
        if prev - t > gap:
            break
        chain.append(t)
        prev = t
    return chain[::-1]


def record_restart_interrupted_boot(
    window_seconds: int = DEFAULT_WINDOW_SECONDS, *, now: Optional[float] = None,
    max_gap_seconds: int = DEFAULT_MAX_GAP_SECONDS,
) -> list[float]:
    """Record a restart-interrupted boot; return the pruned chain + now (most recent
    last).  A persistence failure returns the in-memory list without raising."""
    ts = time.time() if now is None else now
    boots = _chain_ending_at(_load_boots(), ts, _chain_gap(window_seconds, max_gap_seconds)) + [ts]
    _save_boots(boots[-_MAX_STORED_BOOTS:])
    return boots


def clear() -> None:
    """Remove the persisted boot log (used on clean shutdown / by tests)."""
    with contextlib.suppress(OSError):
        _state_path().unlink(missing_ok=True)


def check_and_record(
    max_restarts: int = DEFAULT_MAX_RESTARTS, window_seconds: int = DEFAULT_WINDOW_SECONDS, *,
    now: Optional[float] = None, max_gap_seconds: int = DEFAULT_MAX_GAP_SECONDS,
) -> bool:
    """Gateway entry point: record this boot; True when the chain reached
    ``max_restarts`` and auto-resume should be SKIPPED."""
    boots = record_restart_interrupted_boot(window_seconds, now=now, max_gap_seconds=max_gap_seconds)
    tripped = max_restarts > 0 and len(boots) >= max_restarts
    if tripped:
        logger.warning(
            "Restart-loop breaker TRIPPED: %d chained restart-interrupted gateway boots (no gap wider than %ds; "
            "threshold %d). Skipping auto-resume to break a suspected SIGTERM-respawn loop (#30719, #81642). "
            "Restart-interrupted sessions stay resume-pending and will continue on the next real user message. "
            "If this is a false positive, delete %s.",
            len(boots), int(_chain_gap(window_seconds, max_gap_seconds)), max_restarts, _state_path(),
        )
    return tripped


def _load_sent(path) -> list[float]:
    try:
        sent = json.loads(path.read_text(encoding="utf-8-sig")).get("sent", [])
        return [float(t) for t in sent if isinstance(t, (int, float))]
    except (OSError, ValueError, TypeError, AttributeError):
        return []


def _load_chain_start(path) -> Optional[float]:
    try:
        start = json.loads(path.read_text(encoding="utf-8-sig")).get("start")
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    return float(start) if isinstance(start, (int, float)) else None


def _notice_chain_start(chain: list[float], stored: list[float], recorded: Optional[float], ts: float) -> float:
    """First stop of the chain ``ts`` extends. The stored boots are capped, so a long loop pushes its
    real first stop out of them: when the chain still reaches the oldest stored boot it began at the
    recorded start, which outlives the cap (or the delivered notices before ``chain[0]`` are
    forgotten and a continuous loop posts again)."""
    if not chain:
        return ts
    if recorded is not None and chain[0] == min(stored):
        return min(recorded, chain[0])
    return chain[0]


def notice_loop_tripped(
    max_restarts: int = DEFAULT_MAX_RESTARTS, window_seconds: int = DEFAULT_WINDOW_SECONDS, *,
    now: Optional[float] = None, max_gap_seconds: int = DEFAULT_MAX_GAP_SECONDS,
) -> bool:
    """Record one stop in the home-notice chain; True when ``max_restarts`` notices were already
    DELIVERED in this chain, so this one stays quiet. Every checked stop links the chain (a muted or
    undeliverable one too: the loop stays muted until the gateway has been up longer than the chain
    gap), but only :func:`record_notice_delivered` spends a slot, so a cycle with no connected
    adapter does not use up a notice the user never saw."""
    ts = time.time() if now is None else now
    path = _notice_state_path()
    stored = _load_boots(path)
    chain = _chain_ending_at(stored, ts, _chain_gap(window_seconds, max_gap_seconds))
    start = _notice_chain_start(chain, stored, _load_chain_start(path), ts)
    sent = [t for t in _load_sent(path) if chain and t >= start]
    _save_notice_state(chain + [ts], sent, path, start)
    tripped = max_restarts > 0 and len(sent) >= max_restarts
    if tripped:
        logger.warning(
            "Restart loop: %d home-channel shutdown notices already sent with no gap wider than %ds; this one "
            "stays quiet until the gateway stays up. If this is a false positive, delete %s.",
            len(sent), int(_chain_gap(window_seconds, max_gap_seconds)), path,
        )
    return tripped


def record_notice_delivered(*, now: Optional[float] = None) -> None:
    """Spend one slot of the current chain: this stop's goodbye reached a home chat."""
    path = _notice_state_path()
    _save_notice_state(_load_boots(path), _load_sent(path) + [time.time() if now is None else now], path,
                       _load_chain_start(path))


def _save_notice_state(boots: list[float], sent: list[float], path, start: Optional[float] = None) -> None:
    state: dict[str, object] = {"boots": boots[-_MAX_STORED_BOOTS:], "sent": sent[-_MAX_STORED_BOOTS:]}
    if start is not None:
        state["start"] = start
    with contextlib.suppress(OSError):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state), encoding="utf-8")
