"""One-shot session wake deadlines - the orchestrator's own alarm (#122444).

Inspired by ChatGPT Work's dots ("your dot can decide when to pause and wake up to continue
work; you don't need a fixed schedule for every follow-up"): the AGENT arms the deadline with
the ``schedule_wake`` tool, the session's owning driver fires it once ``fires_at`` passes, and
the session re-enters its loop with zero external input instead of sleeping forever.

Session-scoped and in-process (the owning CLI / TUI session-owner / gateway must be running;
durable cross-process scheduling stays ``hermes cron``). State lives in SessionDB ``state_meta``
under ``wake:<session_id>`` like ``heartbeat:``/``loop:``. Drivers:

* classic CLI - ``CLILoopsMixin._maybe_fire_wake`` (idle hook);
* TUI / Desktop / dashboard - ``tui_gateway/session_notifications.py::_maybe_fire_tui_wake``;
* messaging gateway - ``gateway/run_goals.py::_wake_watcher`` for wakes armed inside a routed chat
  (the tool captures the chat route at arm time, exactly like ``/loop``).

Invariants (mirrors heartbeat.py): injection is a plain user message rendered through
``WAKE_PROMPT_TEMPLATE`` - the rendered text starts with ``[``, so a model-authored prompt can
never be dispatched as a slash/``!`` command; no system-prompt mutation or toolset swap, so prompt
caching stays intact; the wake is one-shot (consumed on fire, re-armed only by the agent); a real
user message always wins (fires only into an idle session with an empty input queue); and a
per-session fire budget (``wake.max_fires``, default 100, 0 = unlimited) is the deterministic
backstop against a model that re-arms itself forever.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_WAKE_PREFIX = "wake:"

DEFAULT_MAX_FIRES = 100  # per session, carried across re-arms; 0 = unlimited (config: wake.max_fires)

WAKE_PROMPT_TEMPLATE = (
    "[Wake - one-shot instruction you scheduled for yourself{budget}]\n{prompt}\n\n"
    "If nothing needs doing yet, say so briefly and stop. Re-arm schedule_wake only while "
    "work is genuinely outstanding."
)


def max_fires() -> int:
    """``wake.max_fires`` from config (clamped >= 0); ``DEFAULT_MAX_FIRES`` on any bad value."""
    try:
        from hermes_cli.config import load_config

        section = (load_config() or {}).get("wake") or {}
        return max(0, int(section.get("max_fires", DEFAULT_MAX_FIRES)))
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        logger.debug("wake: config read failed (%s); using default budget", exc)
        return DEFAULT_MAX_FIRES


@dataclass
class WakeState:
    """Serializable per-session one-shot wake. ``armed`` is False once the wake has fired (the row
    is kept so ``fire_count`` survives re-arms); ``route`` is the gateway chat the wake was armed
    from (empty for CLI/TUI-owned sessions)."""

    prompt: str
    fires_at: float
    armed: bool = True
    fire_count: int = 0
    created_at: float = 0.0
    route: Dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "WakeState":
        d = json.loads(raw)
        route = d.get("route")
        return cls(
            prompt=str(d.get("prompt", "")),
            fires_at=float(d.get("fires_at", 0.0)),
            armed=bool(d.get("armed", True)),
            fire_count=int(d.get("fire_count") or 0),
            created_at=float(d.get("created_at") or 0.0),
            route=dict(route) if isinstance(route, dict) else {},
        )

    def is_due(self, now: Optional[float] = None) -> bool:
        return self.armed and bool(self.prompt) and (time.time() if now is None else now) >= self.fires_at

    def render_prompt(self, limit: int) -> str:
        budget = f" ({self.fire_count}/{limit})" if limit else ""
        return WAKE_PROMPT_TEMPLATE.format(budget=budget, prompt=self.prompt)


def route_is_gateway_chat(route: Optional[Dict[str, str]]) -> bool:
    """A wake armed inside a messaging chat carries ``platform`` + ``chat_id``; the gateway's
    ``_wake_watcher`` owns those, CLI/TUI drivers skip them (and vice versa)."""
    route = route or {}
    return bool(route.get("platform") and route.get("chat_id"))


def _get_session_db() -> Optional[Any]:
    """Persistence goes through the goals module's per-HERMES_HOME cached SessionDB (one shared connection)."""
    try:
        from hermes_cli.goals import _get_session_db as _goals_db

        return _goals_db()
    except Exception:  # health: allow BLE001 -- SessionDB bootstrap boundary; the heartbeat/loop twins degrade to no-op the same way
        logger.debug("wake: SessionDB bootstrap failed", exc_info=True)
        return None


def _parse(raw: Any, session_id: str) -> Optional[WakeState]:
    try:
        return WakeState.from_json(raw) if raw else None
    except (ValueError, TypeError, KeyError) as exc:
        logger.warning("wake: could not parse stored wake for %s: %s", session_id, exc)
        return None


def load_wake(session_id: str) -> Optional[WakeState]:
    db = _get_session_db() if session_id else None
    if db is None:
        return None
    try:
        raw = db.get_meta(_WAKE_PREFIX + session_id)
    except Exception:  # health: allow BLE001 -- sqlite read boundary (OperationalError, closed db); a wake must never crash the idle loop
        logger.debug("wake: get_meta failed", exc_info=True)
        return None
    return _parse(raw, session_id)


def save_wake(session_id: str, state: WakeState) -> None:
    if not session_id:
        return
    db = _get_session_db()
    if db is None:
        from hermes_cli.goals import _warn_dropped_write

        _warn_dropped_write("wake", "wake", session_id)
        return
    try:
        db.set_meta(_WAKE_PREFIX + session_id, state.to_json())
    except Exception:  # health: allow BLE001 -- sqlite write boundary; same contract as save_heartbeat
        logger.debug("wake: set_meta failed", exc_info=True)


def clear_wake(session_id: str) -> None:
    if not session_id:
        return
    db = _get_session_db()
    if db is None:
        return
    try:
        db.set_meta(_WAKE_PREFIX + session_id, "")
    except Exception:  # health: allow BLE001 -- sqlite write boundary; same contract as save_heartbeat
        logger.debug("wake: clear failed", exc_info=True)


def store_has_armed_wake(db: Any) -> bool:
    """True when *db* holds an ARMED ``wake:*`` row - or one that cannot be parsed (unknown, so the
    caller keeps its full scan). Read errors propagate: "unavailable" is the caller's call."""
    for key, raw in db.list_meta_prefix(_WAKE_PREFIX):
        if not raw:
            continue
        state = _parse(raw, key[len(_WAKE_PREFIX):])
        if state is None or state.armed:
            return True
    return False


def list_armed_wakes() -> List[Tuple[str, WakeState]]:
    """``[(session_id, WakeState), ...]`` for every ARMED wake in the current store; ``[]`` on DB error."""
    db = _get_session_db()
    if db is None:
        return []
    out: List[Tuple[str, WakeState]] = []
    try:
        rows = db.list_meta_prefix(_WAKE_PREFIX)
    except Exception:  # health: allow BLE001 -- sqlite read boundary on the gateway watcher's scan; empty list = retry next tick
        logger.debug("wake: list_meta_prefix failed", exc_info=True)
        return []
    for key, raw in rows:
        sid = key[len(_WAKE_PREFIX):]
        state = _parse(raw, sid) if sid and raw else None
        if state is not None and state.armed:
            out.append((sid, state))
    return out


class WakeBudgetExhausted(ValueError):
    """The session has used its ``wake.max_fires`` budget; re-arming needs a human decision."""


def schedule_wake(
    session_id: str, prompt: str, fires_at: float, *, route: Optional[Dict[str, str]] = None,
) -> WakeState:
    """Arm a one-shot wake for ``session_id``; replaces any wake already armed (latest wins). The
    fire budget is per session and survives re-arms: once ``fire_count`` reaches ``wake.max_fires``
    the call raises :class:`WakeBudgetExhausted` instead of arming the (N+1)th wake."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("wake prompt is empty")
    previous = load_wake(session_id)
    fired = previous.fire_count if previous is not None else 0
    limit = max_fires()
    if limit and fired >= limit:
        raise WakeBudgetExhausted(
            f"this session has fired {fired} self-scheduled wakes, the wake.max_fires budget ({limit}); "
            "ask the user to raise it or to set a /heartbeat before arming another"
        )
    state = WakeState(
        prompt=prompt, fires_at=float(fires_at), armed=True, fire_count=fired, created_at=time.time(),
        route={k: str(v) for k, v in (route or {}).items() if v},
    )
    save_wake(session_id, state)
    return state


def due_wake_prompt(session_id: str, now: Optional[float] = None) -> Optional[str]:
    """Return the rendered prompt if the wake deadline passed, consuming the wake (one-shot), else None.

    The fire is recorded BEFORE the caller queues the prompt: like heartbeat's claim-first
    ``due_prompt``, a consumed wake may be lost only on a crash, never double-fired into two
    turns. A driver whose dispatch never started a turn calls :func:`abandon_wake_fire`.
    """
    state = load_wake(session_id)
    if state is None or not state.is_due(now):
        return None
    state.armed = False
    state.fire_count += 1
    save_wake(session_id, state)
    return state.render_prompt(max_fires())


def abandon_wake_fire(session_id: str) -> bool:
    """Rewind the fire recorded by the last :func:`due_wake_prompt` whose turn never started, so the
    wake stays armed for the next poll instead of being silently consumed. Skipped (False) when the
    agent re-armed in between (the row is armed again) - that newer deadline wins."""
    state = load_wake(session_id)
    if state is None or state.armed or state.fire_count <= 0:
        return False
    state.armed = True
    state.fire_count -= 1
    save_wake(session_id, state)
    return True


def migrate_wake_to_session(old_session_id: str, new_session_id: str) -> bool:
    """Carry a wake across a compression-driven session rotation (copy to child, clear parent,
    never raise). Same shape as ``heartbeat.migrate_heartbeat_to_session``: without this the
    deadline is stranded on the dead session id exactly when the issue's compaction case would
    need it. The fired-but-unarmed row migrates too so the fire budget follows the session."""
    if not old_session_id or not new_session_id or old_session_id == new_session_id:
        return False
    try:
        state = load_wake(old_session_id)
        if state is None or load_wake(new_session_id) is not None:
            return False
        save_wake(new_session_id, state)
        clear_wake(old_session_id)
        return True
    except Exception:  # health: allow BLE001 -- compression must never fail on wake bookkeeping; same contract as migrate_heartbeat_to_session
        logger.debug("wake: migrate failed", exc_info=True)
        return False


__all__ = [
    "DEFAULT_MAX_FIRES",
    "WAKE_PROMPT_TEMPLATE",
    "WakeBudgetExhausted",
    "WakeState",
    "abandon_wake_fire",
    "clear_wake",
    "due_wake_prompt",
    "list_armed_wakes",
    "load_wake",
    "max_fires",
    "migrate_wake_to_session",
    "route_is_gateway_chat",
    "save_wake",
    "schedule_wake",
    "store_has_armed_wake",
]
