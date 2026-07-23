# SPDX-License-Identifier: Apache-2.0
"""In-memory reference bindings for ``ConsciousnessLoop``.

The heartbeat module is intentionally dependency-injected — every
callable is passed in via :class:`HeartbeatDeps`. This file provides a
small, self-contained set of bindings that plug into that interface
without touching Hermes' main agent loop, so:

* :mod:`agent.heartbeat` can be unit-tested in isolation.
* Downstream consumers (gateway, TUI, dashboard) get a clear template
  to wire their own queues/reminders/quota into the loop later.

Nothing here is imported from ``run_agent.py``. Turning the feature on
still requires an explicit call in a follow-up integration PR.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from .heartbeat import HeartbeatDeps, Priorities
from .heartbeat_policy import TickerStatus, build_autonomous_tick_directions


# ── Minimal in-memory state ────────────────────────────────────────────────


@dataclass
class InMemoryHeartbeatState:
    """Tiny state container the reference bindings mutate.

    A production wiring replaces each field with a real service (Hermes
    session queue, cron reminders table, credential-pool quota view,
    etc.). Kept intentionally boring so tests can drive the loop
    deterministically.
    """

    messages: list[Any] = field(default_factory=list)
    reminders: list[dict[str, Any]] = field(default_factory=list)
    running: bool = True
    processing_execution: Optional[dict[str, Any]] = None
    abort_controller: Optional[Any] = None
    ticker: dict[str, Any] = field(
        default_factory=lambda: {"active": False, "ttl": 0, "revision": 0}
    )
    awakening_ticks: int = 0
    task_active: bool = False
    rate_limited: bool = False
    base_tick_interval_seconds: float = 60.0
    startup_self_check_active: bool = False
    startup_self_check_version: str = "0"
    scheduler: Optional[Callable[[], None]] = None
    interrupt_callback: Optional[Callable[[Any], None]] = None
    events: list[tuple[str, Mapping[str, Any]]] = field(default_factory=list)
    sticky_events: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    priorities: Priorities = field(default_factory=Priorities)


def build_in_memory_deps(
    state: InMemoryHeartbeatState,
    *,
    run_turn: Callable[[Any, str, Optional[Any]], Any],
    run_turn_watchdog_s: float = 300.0,
    format_tick: Optional[Callable[[], str]] = None,
) -> HeartbeatDeps:
    """Build a :class:`HeartbeatDeps` backed by ``state``.

    ``run_turn`` is the only argument every caller must supply; it
    represents the actual agent-turn coroutine. ``format_tick`` defaults
    to :func:`build_autonomous_tick_directions` with no extras, which is
    exactly the prompt BaiLongma emits on an idle L2 tick.
    """
    if format_tick is None:

        def _default_format_tick() -> str:
            ticker_arg: Optional[TickerStatus] = None
            if state.ticker.get("active"):
                ticker_arg = TickerStatus(
                    active=bool(state.ticker.get("active", False)),
                    seconds=float(state.ticker.get("seconds", 60.0)),
                    ttl=int(state.ticker.get("ttl", 0)),
                    reason=state.ticker.get("reason"),
                    revision=int(state.ticker.get("revision", 0)),
                )
            return build_autonomous_tick_directions(
                awakening_ticks=state.awakening_ticks,
                ticker_status=ticker_arg,
            )

        format_tick = _default_format_tick

    def emit_event(name: str, payload: Mapping[str, Any]) -> None:
        state.events.append((name, dict(payload)))

    def set_sticky_event(name: str, payload: Mapping[str, Any]) -> None:
        state.sticky_events[name] = dict(payload)

    def enqueue_due_reminders() -> None:
        now = time.time()
        due = [r for r in state.reminders if r.get("due_at_ms", 0) / 1000 <= now]
        for reminder in due:
            state.messages.append(
                type(
                    "Msg",
                    (),
                    {
                        "raw": reminder.get("message"),
                        "fromId": reminder.get("from_id", "reminder"),
                        "queueName": "background",
                        "priority": state.priorities.background,
                    },
                )()
            )
            state.reminders.remove(reminder)

    def has_messages() -> bool:
        return bool(state.messages)

    def pop_message() -> Any:
        return state.messages.pop(0)

    def has_user_messages() -> bool:
        return any(
            getattr(m, "queueName", None) != "background" for m in state.messages
        )

    def get_queue_snapshot() -> Mapping[str, int]:
        user = sum(
            1
            for m in state.messages
            if getattr(m, "queueName", None) != "background"
        )
        background = len(state.messages) - user
        return {"user": user, "background": background}

    def consume_ticker_tick() -> None:
        if state.ticker.get("active"):
            state.ticker["ttl"] = max(0, state.ticker.get("ttl", 0) - 1)
            if state.ticker["ttl"] == 0:
                state.ticker["active"] = False

    def decrement_awakening_tick() -> None:
        state.awakening_ticks = max(0, state.awakening_ticks - 1)

    def get_tick_interval(base: float) -> float:
        # In the reference wiring, rate-limited fallback just reuses the
        # base interval. Real wiring reads the credential pool's cooldown.
        return base

    def get_custom_interval_ms() -> Optional[float]:
        if state.ticker.get("active"):
            return float(state.ticker.get("seconds", 60.0)) * 1000.0
        return None

    def get_next_pending_reminder() -> Optional[Mapping[str, Any]]:
        if not state.reminders:
            return None
        return min(state.reminders, key=lambda r: r.get("due_at_ms", float("inf")))

    def get_quota_status() -> Mapping[str, Any]:
        return {"rpm_used": 0, "tpm_used": 0, "ratio": 0.0}

    def set_scheduler(fn: Callable[[], None]) -> None:
        state.scheduler = fn

    def set_interrupt_callback(fn: Callable[[Any], None]) -> None:
        state.interrupt_callback = fn

    return HeartbeatDeps(
        run_turn=run_turn,
        run_turn_watchdog_s=run_turn_watchdog_s,
        get_current_execution=lambda: state.processing_execution,
        get_current_abort_controller=lambda: state.abort_controller,
        clear_current_execution=lambda: setattr(state, "processing_execution", None),
        emit_event=emit_event,
        enqueue_due_reminders=enqueue_due_reminders,
        has_messages=has_messages,
        pop_message=pop_message,
        has_user_messages=has_user_messages,
        get_queue_snapshot=get_queue_snapshot,
        format_tick=format_tick,
        consume_ticker_tick=consume_ticker_tick,
        decrement_awakening_tick=decrement_awakening_tick,
        is_startup_self_check_active=lambda: state.startup_self_check_active,
        is_running=lambda: state.running,
        set_scheduler=set_scheduler,
        set_interrupt_callback=set_interrupt_callback,
        is_rate_limited=lambda: state.rate_limited,
        get_tick_interval=get_tick_interval,
        get_base_tick_interval=lambda: state.base_tick_interval_seconds,
        get_custom_interval_ms=get_custom_interval_ms,
        get_ticker_status=lambda: state.ticker,
        get_awakening_ticks=lambda: state.awakening_ticks,
        is_task_active=lambda: state.task_active,
        get_next_pending_reminder=get_next_pending_reminder,
        get_quota_status=get_quota_status,
        start_consolidation_loop=lambda: None,
        ensure_startup_self_check_state=lambda: None,
        set_sticky_event=set_sticky_event,
        startup_self_check_version=state.startup_self_check_version,
        priorities=state.priorities,
    )


def load_heartbeat_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Resolve the ``agent.heartbeat`` block from a merged Hermes config.

    Applied bounds mirror the docstring in ``config.py``:

    * ``base_tick_interval_seconds`` clamped to [10, 3600].
    * ``run_turn_watchdog_seconds`` floored at 5 (anything shorter is
      almost certainly a config typo).
    * ``max_daily_ticks`` floored at 0 (0 disables the cap).
    """
    raw = dict((config.get("agent") or {}).get("heartbeat") or {})

    base_raw = raw.get("base_tick_interval_seconds", 60)
    base = float(base_raw) if base_raw is not None else 60.0
    raw["base_tick_interval_seconds"] = max(10.0, min(3600.0, base))

    watchdog_raw = raw.get("run_turn_watchdog_seconds", 300)
    watchdog = float(watchdog_raw) if watchdog_raw is not None else 300.0
    raw["run_turn_watchdog_seconds"] = max(5.0, watchdog)

    max_daily_raw = raw.get("max_daily_ticks", 288)
    max_daily = int(max_daily_raw) if max_daily_raw is not None else 288
    raw["max_daily_ticks"] = max(0, max_daily)

    raw.setdefault("enabled", False)
    raw.setdefault("run_immediate_tick_on_start", False)
    raw.setdefault("quiet_hours_start", "")
    raw.setdefault("quiet_hours_end", "")
    return raw


__all__ = [
    "InMemoryHeartbeatState",
    "build_in_memory_deps",
    "load_heartbeat_config",
]
