# SPDX-License-Identifier: Apache-2.0
"""Tests for the autonomous heartbeat loop.

These exercise :class:`agent.heartbeat.ConsciousnessLoop` through the
in-memory reference bindings so the loop's public contract is enforced
without requiring a real ``AIAgent``.
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

import pytest

from agent.heartbeat import (
    ConsciousnessLoop,
    Priorities,
    WatchdogTimeoutError,
)
from agent.heartbeat_bindings import (
    InMemoryHeartbeatState,
    build_in_memory_deps,
    load_heartbeat_config,
)
from agent.heartbeat_policy import (
    TickerStatus,
    build_autonomous_tick_directions,
)


# ── Policy prompt ──────────────────────────────────────────────────────────


def test_policy_always_includes_core_directives() -> None:
    """Every autonomous tick must carry the six non-negotiable directives."""
    text = build_autonomous_tick_directions()
    assert "autonomous L2 heartbeat" in text
    assert "private working text" in text
    assert "send_message" in text
    assert "runtime guardrails" in text.lower()
    assert "find_tool" in text
    # Silence-when-unanswered must be present so a heartbeat never
    # justifies pinging a user who hasn't replied.
    assert "unanswered" in text.lower()


def test_policy_appends_awakening_and_ticker_only_when_relevant() -> None:
    """Awakening / ticker blocks are opt-in, not always emitted."""
    base = build_autonomous_tick_directions()
    with_awakening = build_autonomous_tick_directions(awakening_ticks=3)
    with_ticker = build_autonomous_tick_directions(
        ticker_status=TickerStatus(
            active=True, seconds=15.0, ttl=4, reason="user asked", revision=1
        )
    )
    assert "awakening period" not in base
    assert "awakening period" in with_awakening
    assert "3 heartbeat" in with_awakening
    assert "Custom heartbeat cadence" not in base
    assert "Custom heartbeat cadence" in with_ticker
    assert "15.0s interval" in with_ticker
    assert "Reason: user asked." in with_ticker


def test_policy_ignores_inactive_ticker() -> None:
    text = build_autonomous_tick_directions(
        ticker_status=TickerStatus(active=False, seconds=15.0, ttl=0, revision=0)
    )
    assert "Custom heartbeat cadence" not in text


# ── Config resolution ─────────────────────────────────────────────────────


def test_load_heartbeat_config_defaults_when_missing() -> None:
    """Missing / empty config falls back to safe defaults."""
    hb = load_heartbeat_config({})
    assert hb["enabled"] is False
    assert hb["base_tick_interval_seconds"] == 60.0
    assert hb["run_turn_watchdog_seconds"] == 300.0
    assert hb["max_daily_ticks"] == 288
    assert hb["quiet_hours_start"] == ""
    assert hb["quiet_hours_end"] == ""


def test_load_heartbeat_config_clamps_absurd_values() -> None:
    """A user who sets base=1s or watchdog=0 shouldn't be able to
    lock the runtime up; clamping happens in resolution."""
    hb = load_heartbeat_config(
        {
            "agent": {
                "heartbeat": {
                    "enabled": True,
                    "base_tick_interval_seconds": 1,
                    "run_turn_watchdog_seconds": 0,
                    "max_daily_ticks": -5,
                }
            }
        }
    )
    assert hb["enabled"] is True
    assert hb["base_tick_interval_seconds"] == 10.0  # floored
    assert hb["run_turn_watchdog_seconds"] == 5.0  # floored
    assert hb["max_daily_ticks"] == 0  # floored


def test_load_heartbeat_config_caps_base_interval_at_hour() -> None:
    hb = load_heartbeat_config(
        {"agent": {"heartbeat": {"base_tick_interval_seconds": 999999}}}
    )
    assert hb["base_tick_interval_seconds"] == 3600.0


# ── Loop behaviour ─────────────────────────────────────────────────────────


def _make_loop_and_state() -> tuple[ConsciousnessLoop, InMemoryHeartbeatState, list]:
    state = InMemoryHeartbeatState()
    calls: list[tuple[Any, str, Optional[Any]]] = []

    async def run_turn(inp: Any, label: str, msg: Optional[Any]) -> None:
        calls.append((inp, label, msg))

    deps = build_in_memory_deps(state, run_turn=run_turn)
    return ConsciousnessLoop(deps), state, calls


@pytest.mark.asyncio
async def test_on_tick_runs_autonomous_l2_when_idle() -> None:
    """No messages → the loop fires an L2 TICK and consumes state."""
    loop, state, calls = _make_loop_and_state()
    state.awakening_ticks = 2
    await loop.on_tick()
    assert len(calls) == 1
    inp, label, msg = calls[0]
    assert label == "L2 TICK"
    assert msg is None
    assert "autonomous L2 heartbeat" in inp
    # A completed tick spends one awakening tick.
    assert state.awakening_ticks == 1
    # No pending message → loop is idle again after the tick.
    assert loop.is_processing() is False


@pytest.mark.asyncio
async def test_on_tick_pumps_pending_message_instead_of_ticking() -> None:
    """When a message is pending, it must be dispatched before any L2 tick."""
    loop, state, calls = _make_loop_and_state()

    class _Msg:
        raw = "hello"
        fromId = "user-1"
        queueName = "user"
        priority = 3

    state.messages.append(_Msg())
    state.awakening_ticks = 1

    await loop.on_tick()
    assert len(calls) == 1
    inp, label, msg = calls[0]
    assert inp == "hello"
    assert "L1 message from user-1" in label
    assert msg is not None
    # Awakening state is preserved because this wasn't an autonomous tick.
    assert state.awakening_ticks == 1


@pytest.mark.asyncio
async def test_on_tick_reentry_guard() -> None:
    """A second on_tick while one is running must be a no-op."""
    loop, state, calls = _make_loop_and_state()

    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_turn(inp: Any, label: str, msg: Optional[Any]) -> None:
        started.set()
        await release.wait()
        calls.append((inp, label, msg))

    # Rebind run_turn to a slow variant.
    deps = build_in_memory_deps(state, run_turn=slow_turn)
    loop = ConsciousnessLoop(deps)

    first = asyncio.create_task(loop.on_tick())
    await started.wait()
    assert loop.is_processing() is True

    # This must return immediately without adding another call.
    await loop.on_tick()
    assert len(calls) == 0

    release.set()
    await first
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_watchdog_aborts_stuck_turn_and_marks_tick_aborted() -> None:
    """A run_turn that never returns must be aborted by the watchdog."""
    state = InMemoryHeartbeatState()
    aborted = asyncio.Event()

    async def stuck_turn(inp: Any, label: str, msg: Optional[Any]) -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            aborted.set()
            raise

    class _AbortController:
        def abort(self, reason: str = "") -> None:
            state.abort_controller = None

    state.processing_execution = {
        "priority": 1,
        "started_at": 0.0,
        "label": "L2 TICK",
    }
    state.abort_controller = _AbortController()

    deps = build_in_memory_deps(
        state, run_turn=stuck_turn, run_turn_watchdog_s=0.05
    )
    loop = ConsciousnessLoop(deps)

    with pytest.raises(WatchdogTimeoutError):
        await loop._run_turn_with_watchdog("prompt", "L2 TICK", None)

    # The watchdog emitted an error event.
    assert any(name == "error" for name, _ in state.events)
    # And it cleared the current execution reference.
    assert state.processing_execution is None


def test_should_preempt_when_no_current_execution() -> None:
    """No running turn → any incoming entry preempts (starts immediately)."""
    loop, state, _ = _make_loop_and_state()
    # processing = False, execution = None ⇒ preempt.
    assert loop._should_preempt_for({"priority": 1}) is True


@pytest.mark.asyncio
async def test_should_preempt_concurrent_user_messages() -> None:
    """A second user message must preempt an already-running user turn
    (per BaiLongma's rule: user-vs-user can barge)."""
    loop, state, _ = _make_loop_and_state()

    # Simulate: one turn currently running at user priority.
    state.processing_execution = {"priority": Priorities().user}
    loop._processing = True

    incoming = {"priority": Priorities().user, "queueName": "user"}
    assert loop._should_preempt_for(incoming) is True

    # A lower-priority background message does NOT preempt.
    background = {
        "priority": Priorities().background,
        "queueName": "background",
    }
    assert loop._should_preempt_for(background) is False


@pytest.mark.asyncio
async def test_ticker_reconfigured_mid_tick_is_not_consumed() -> None:
    """If a tick installs a new custom cadence, the fresh TTL must
    survive the same tick (mirrors BaiLongma's ``tickerWasReconfigured``
    guard so the very tick that requested a new cadence doesn't spend
    one of its own TTL slots)."""
    state = InMemoryHeartbeatState()

    async def turn_that_installs_cadence(
        inp: Any, label: str, msg: Optional[Any]
    ) -> None:
        # Simulate the model calling set_tick_interval mid-turn.
        state.ticker = {
            "active": True,
            "seconds": 15.0,
            "ttl": 4,
            "revision": (state.ticker.get("revision", 0) or 0) + 1,
        }

    deps = build_in_memory_deps(state, run_turn=turn_that_installs_cadence)
    loop = ConsciousnessLoop(deps)

    await loop.on_tick()
    assert state.ticker["ttl"] == 4  # untouched
    assert state.ticker["active"] is True
