"""Opt-in wall budget for a request that has produced no first byte, spanning its retries.

Every streaming attempt has its own stale timer (180 s by default, raised for large contexts and
reasoning models), and a silent attempt is retried in place before the fallback chain is
considered. A provider that accepts the connection and never sends a byte therefore holds the turn
for several stale windows before ``fallback_providers`` is reached, even though a fallback is
configured exactly for this case.

``fallback.no_first_byte_budget_seconds`` (0 = off) puts one clock over that wait. It is armed when
the request is built, keeps running across in-place retries and fallback restarts, and is disarmed
by the first real stream event or a completed response. Time the agent spends in its own retry
backoff is credited back (``note_excused_wait``): the clock measures provider silence, not our
pauses. Once the budget is spent and a fallback entry is still pending, the streaming monitor cuts
the attempt once and the error path moves to the next entry in chain order. With no fallback left
nothing is cut and the per-attempt timers decide as before.

State lives on the agent as ``_first_byte_budget`` (reset per turn).
"""

from __future__ import annotations

import time
from typing import Any, Optional


class FirstByteBudgetExceeded(TimeoutError):
    """The attempt was cut because the request's no-first-byte budget was spent.

    A ``TimeoutError``, so the classifier gives it the transport-timeout verdict. ``provider`` is
    the route that stayed silent and ``elapsed_s`` the budget clock at the cut."""

    def __init__(self, message: str = "", *, provider: Optional[str] = None,
                 elapsed_s: Optional[float] = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.elapsed_s = elapsed_s


def budget_seconds() -> float:
    """Configured ``fallback.no_first_byte_budget_seconds``; 0 (off) when unset or invalid."""
    try:
        from hermes_cli.config import load_config_readonly
        cfg = (load_config_readonly() or {}).get("fallback") or {}
        value = float(cfg.get("no_first_byte_budget_seconds") or 0) if isinstance(cfg, dict) else 0.0
    except Exception:
        return 0.0
    return value if value > 0 else 0.0


def _state(agent: Any) -> Optional[dict]:
    st = getattr(agent, "_first_byte_budget", None)
    return st if isinstance(st, dict) else None


def arm(agent: Any) -> None:
    """A request is about to wait. An already-armed clock is kept: a retry or a fallback restart is
    the same wait continuing."""
    if _state(agent) is not None:
        return
    budget = budget_seconds()
    agent._first_byte_budget = {"started": time.monotonic(), "budget_s": budget, "fired": False} if budget else None


def disarm(agent: Any) -> None:
    """A real byte or a full response arrived: the wait is over."""
    if getattr(agent, "_first_byte_budget", None) is not None:
        agent._first_byte_budget = None


def note_excused_wait(agent: Any, seconds: float) -> None:
    """Credit time the agent itself spent waiting (retry backoff), so only silence on the wire
    counts against the budget. No-op when nothing is armed."""
    st = _state(agent)
    if st is not None and seconds > 0:
        st["started"] += seconds


def elapsed(agent: Any) -> Optional[float]:
    st = _state(agent)
    return None if st is None else time.monotonic() - st["started"]


def should_cut(agent: Any) -> bool:
    """True once per wait: the budget is spent and a fallback entry is still pending. The entry the
    cut moves to keeps its own per-attempt timers instead of being cut on arrival."""
    st = _state(agent)
    if st is None or st["fired"] or time.monotonic() - st["started"] < st["budget_s"]:
        return False
    if not agent._has_pending_fallback():
        return False
    st["fired"] = True
    return True
