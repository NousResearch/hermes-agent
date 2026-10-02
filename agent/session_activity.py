"""Shared session activity observation contract: timestamp + bounded description/provenance, observation
only (notification, timeout, kill and retry policy live elsewhere). Provenance is a small closed enum of
*noun* sources; the default agent clock stamps ``unknown`` unless a writer passes ``provenance=``."""

from __future__ import annotations

import sys
import time
from contextlib import suppress
from enum import Enum
from typing import Any, Mapping, Optional

from agent.i18n import t

ACTIVITY_DESCRIPTION_MAX = 120

# Durable SessionDB heartbeat cadence. Contract: MUST stay >= 30s — the SessionDB write path is contended and
# this observation-only projection never justifies extra write pressure. A code constant on purpose (no config
# can make it a high-frequency writer); matches the kanban auto-heartbeat. force_persist and terminal
# compression stamps (see TERMINAL_COMPRESSION_PROVENANCES) are the only bypasses.
SESSION_ACTIVITY_HEARTBEAT_MIN_INTERVAL_SECONDS = 60.0


class ActivityProvenance(str, Enum):
    """Where a durable/in-memory activity stamp came from."""

    UNKNOWN = "unknown"
    # Compression writers: heartbeat, host timeout, cooldown, turn hold.
    # See #72424.
    AGENT_COMPRESSION = "agent.compression"
    AGENT_COMPRESSION_TIMEOUT = "agent.compression_timeout"
    AGENT_COMPRESSION_COOLDOWN = "agent.compression_cooldown"
    AGENT_COMPRESSION_TURNHOLD = "agent.compression_turnhold"


# Provenances that END the user-visible "compressing" phase: the host's progress timeout, a cooldown/
# backoff block, and the gateway's hygiene turn-hold. Each is written once, at a terminal edge, and must
# reach the durable projection IMMEDIATELY — the heartbeat above wrote "context compression in progress"
# moments earlier, so a rate-limited terminal stamp leaves `sessions.last_activity_description` advertising
# a compression that has already stopped, with no later writer to correct it (the agent may be idle, the
# process gone). That is the permanently "stuck / compressing" chat users see.
#
# Distinct from ``conversation_compression._TERMINAL_COMPRESSION_PROVENANCES``, which answers a different
# question — "may a detached heartbeat still overwrite this stamp?" — and deliberately excludes TURNHOLD,
# because after a turn-hold the worker may still be alive and adoptable.
TERMINAL_COMPRESSION_PROVENANCES = frozenset(
    {
        ActivityProvenance.AGENT_COMPRESSION_TIMEOUT,
        ActivityProvenance.AGENT_COMPRESSION_COOLDOWN,
        ActivityProvenance.AGENT_COMPRESSION_TURNHOLD,
    }
)


def is_terminal_compression_provenance(provenance: Optional[ActivityProvenance | str]) -> bool:
    """True when this stamp ends a compression phase and must bypass the persist rate limit."""
    return normalize_activity_provenance(provenance) in TERMINAL_COMPRESSION_PROVENANCES


def bound_activity_description(description: Optional[str]) -> str:
    """Clamp free-form activity text to the shared description budget."""
    text = (description or "").strip()
    return text if len(text) <= ACTIVITY_DESCRIPTION_MAX else text[: ACTIVITY_DESCRIPTION_MAX - 1] + "…"


def normalize_activity_provenance(provenance: Optional[ActivityProvenance | str]) -> ActivityProvenance:
    """Return a known provenance, or ``UNKNOWN`` when unset/unrecognized."""
    if isinstance(provenance, ActivityProvenance):
        return provenance
    try:
        return ActivityProvenance((provenance or "").strip())
    except ValueError:
        return ActivityProvenance.UNKNOWN


def format_iteration_progress(api_call_count: Any, max_iterations: Any) -> str:
    """``iteration N/M`` for user-facing status lines, or ``iteration N`` when the cap is unbounded.

    ``AIAgent.max_iterations`` defaults to ``sys.maxsize`` (unlimited), so printing the pair verbatim
    shows ``iteration 3/9223372036854775807`` in busy acks, heartbeats and timeout diagnostics (#102806).
    """
    try:
        cap = int(max_iterations)
    except (TypeError, ValueError):
        cap = sys.maxsize
    if cap >= sys.maxsize:
        return t("display.iteration_progress.unbounded", n=api_call_count)
    return t("display.iteration_progress.bounded", n=api_call_count, max=cap)


def reset_session_activity_persist_window(agent: Any) -> None:
    """Clear the persist rate-limit so the next stamp writes through (terminal compression labels must not stick on mid-compress text)."""
    with suppress(Exception):
        agent._session_activity_last_persist_mono = 0.0


def build_activity_snapshot(
    *,
    last_activity_at: Optional[float],
    last_activity_description: Optional[str],
    last_activity_provenance: Optional[ActivityProvenance | str] = None,
    now: Optional[float] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build the shared activity snapshot (plus optional caller extras)."""
    when = float(last_activity_at) if last_activity_at is not None else None
    clock = float(now if now is not None else time.time())
    desc = bound_activity_description(last_activity_description)
    prov = normalize_activity_provenance(last_activity_provenance).value
    return {
        "last_activity_at": when,
        "last_activity_description": desc,
        "last_activity_provenance": prov,
        "seconds_since_activity": round(clock - when, 1) if when is not None else None,
        # Short aliases used by existing gateway/delegate readers.
        "last_activity_ts": when, "last_activity_desc": desc, "description": desc, "provenance": prov,
        **(extra or {}),
    }


class AwakeIdleMeter:
    """Removes host sleep from a polling watchdog's idle readings.

    Idle time is measured on the wall clock, which keeps running while the host sleeps, so a laptop
    that sleeps 15 minutes mid-turn wakes up "idle" for 15 minutes and the first poll after resume
    kills a turn that never had a chance to run. ``time.monotonic()`` pauses during sleep on macOS
    (``mach_absolute_time``) and Linux (``CLOCK_MONOTONIC``), so the two clocks drift apart by the
    time spent asleep. Call :meth:`measure` once per poll with the wall-clock idle seconds.
    """

    def __init__(self) -> None:
        self._wall, self._mono = time.time(), time.monotonic()
        self._asleep_s = 0.0

    def measure(self, idle_s: float) -> float:
        """Return *idle_s* minus the sleep observed since the agent's last activity."""
        wall, mono = time.time(), time.monotonic()
        self._asleep_s += max(0.0, (wall - self._wall) - (mono - self._mono))
        self._wall, self._mono = wall, mono
        # New activity pulls idle below the credit, which then drops with it (to within one poll).
        self._asleep_s = max(0.0, min(self._asleep_s, idle_s))
        return idle_s - self._asleep_s
