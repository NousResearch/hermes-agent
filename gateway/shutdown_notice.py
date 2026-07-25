"""Durable cooldown for the home-channel "gateway shutting down" broadcast.

``GatewayRunner._notify_active_sessions_of_shutdown`` deduplicates its sends
through a ``notified`` set. That set is **per-process**, so it only ever
suppresses a duplicate *within a single shutdown*. Every fresh gateway process
starts with an empty set and is free to re-broadcast the identical message.

That is fine when a restart is deliberate and rare. It is not fine when
something outside Hermes is cycling the host: each activation starts a gateway,
the gateway is told to stop seconds later, and the home channel gets another
copy of

    ⚠️ Gateway shutting down — Your current task will be interrupted.

Observed on a WSL host where Windows Modern Standby repeatedly tore down and
re-activated the distro on a ~33 s cycle: **240** home-channel shutdown
notifications, 19 of them inside a single 10-minute window, against **1**
active-session notification. Nothing was wrong with the gateway — it was
started and stopped 240 times, and each process correctly believed it was
announcing its first shutdown.

This module makes that dedup survive process death by recording the last
broadcast time per destination under ``HERMES_HOME``, so the *next* gateway
process can see that the same channel was already told moments ago.

Scope is deliberately narrow — the **home-channel broadcast only**. The
per-active-session pings in the same function stay ungated, matching the
existing ``suppress_notification`` drain flag: those carry the genuinely useful
"your task was cut off, message me to resume" hint, and are empty by
construction on the idle shutdowns that produce this flood.

Fail-open everywhere: an unreadable, malformed, or unwritable state file must
never suppress a notification, and must never raise into the shutdown path.
Setting ``gateway.shutdown_notification_cooldown_seconds: 0`` restores the
original always-send behaviour.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home
from utils import atomic_json_write

_log = logging.getLogger(__name__)

_STATE_FILENAME = ".shutdown_notice_state.json"

# Entries older than this are dropped on write so the file cannot grow
# unbounded across a long-lived install with many home channels. Generous
# relative to any sane cooldown: an entry only matters while it is younger
# than the configured window.
_MAX_ENTRY_AGE_SECONDS = 7 * 24 * 60 * 60


def notice_state_path(home: Optional[Path] = None) -> Path:
    """Absolute path to the shutdown-notice state file, respecting HERMES_HOME."""
    base = home if home is not None else get_hermes_home()
    return Path(base) / _STATE_FILENAME


def destination_key(platform: str, chat_id: Any, thread_id: Any = None) -> str:
    """Stable string key for one delivery destination.

    Mirrors the in-process ``dedup_key`` tuple so the durable record and the
    per-process set agree on what "the same destination" means.
    """
    thread = str(thread_id) if thread_id else ""
    return f"{platform}:{chat_id}:{thread}"


def _read_state(path: Path) -> dict[str, float]:
    """Return ``{destination_key: last_sent_epoch}``; ``{}`` on any problem."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:  # malformed / unreadable / permission
        _log.debug("Ignoring unreadable shutdown-notice state %s: %s", path, e)
        return {}

    if not isinstance(raw, dict):
        return {}
    entries = raw.get("home_notices")
    if not isinstance(entries, dict):
        return {}

    out: dict[str, float] = {}
    for key, value in entries.items():
        if not isinstance(key, str):
            continue
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def should_send_home_notice(
    key: str,
    *,
    cooldown_seconds: float,
    now: Optional[float] = None,
    home: Optional[Path] = None,
) -> bool:
    """Return True when the home-channel broadcast for *key* may be sent.

    Suppresses only when a previous send is recorded **and** it happened within
    ``cooldown_seconds`` of *now*. A non-positive cooldown disables the check
    entirely (always send), which is the documented opt-out.

    A recorded timestamp in the future is treated as unusable rather than as an
    infinitely-long cooldown: a backwards wall-clock step (NTP correction, VM
    resume, the very host suspend/resume cycle this guard exists for) must not
    be able to silence the channel indefinitely.
    """
    try:
        if cooldown_seconds is None or cooldown_seconds <= 0:
            return True

        now_ts = time.time() if now is None else float(now)
        last = _read_state(notice_state_path(home)).get(key)
        if last is None:
            return True
        if last > now_ts:
            # Clock moved backwards — distrust the record, don't extend silence.
            return True
        return (now_ts - last) >= cooldown_seconds
    except Exception as e:
        # Never let bookkeeping block a notification.
        _log.debug("shutdown-notice cooldown check failed for %s: %s", key, e)
        return True


def record_home_notice(
    key: str,
    *,
    now: Optional[float] = None,
    home: Optional[Path] = None,
) -> None:
    """Record that the home-channel broadcast for *key* was just sent.

    Best-effort and atomic: a failure to persist means the next process simply
    sends again (the pre-fix behaviour), which is the safe direction.
    """
    try:
        now_ts = time.time() if now is None else float(now)
        path = notice_state_path(home)
        entries = _read_state(path)
        entries[key] = now_ts
        # Keep only entries within one window of now, in EITHER direction: old
        # ones are irrelevant (any cooldown has long lapsed) and wildly-future
        # ones are clock-skew garbage. Bounds the file without extra state.
        fresh = {
            k: v for k, v in entries.items()
            if abs(v - now_ts) <= _MAX_ENTRY_AGE_SECONDS
        }
        atomic_json_write(path, {"home_notices": fresh})
    except Exception as e:
        _log.debug("Failed recording shutdown notice for %s: %s", key, e)
