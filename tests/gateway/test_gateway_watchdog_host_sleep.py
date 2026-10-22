"""Gateway inactivity watchdogs must not charge host sleep (laptop lid closed) as turn idleness.

The agent's ``seconds_since_activity`` is wall-clock; after resume the first watchdog poll reads
the whole sleep as idle and killed / warned / evicted a turn that never had a chance to run.
Sleep is simulated by advancing the wall clock while ``time.monotonic()`` stays put, which is
what ``CLOCK_MONOTONIC`` / ``mach_absolute_time`` do across suspend.
"""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from agent import session_activity
import gateway.run as gateway_run
from gateway.run import GatewayRunner, _watch_gateway_turn_inactivity
from gateway.session_state import TurnState
from gateway.turn_context import TurnContext

TIMEOUT_S = 1800.0
SLEEP_S = 2000.0


class _Clock:
    def __init__(self):
        self.wall = 1_000_000.0
        self.mono = 500.0

    def time(self):
        return self.wall

    def monotonic(self):
        return self.mono

    def sleep(self, seconds):
        self.wall += seconds

    def awake(self, seconds):
        self.wall += seconds
        self.mono += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(session_activity, "time", fake)
    return fake


class _ScriptedAgent:
    """Each activity read applies the next clock step, then reports wall-clock idle since the
    agent's last activity (stamped at clock creation)."""

    def __init__(self, clock, steps):
        self._clock = clock
        self._origin = clock.wall
        self._steps = list(steps)
        self.reads = 0

    def get_activity_summary(self):
        if self._steps:
            self._steps.pop(0)(self._clock)
        self.reads += 1
        return {"seconds_since_activity": self._clock.wall - self._origin}


def _resume_then_idle_steps():
    return [lambda c: c.awake(1.0), lambda c: c.sleep(SLEEP_S), lambda c: c.awake(TIMEOUT_S)]


def test_turn_watchdog_thread_ignores_host_sleep(clock, monkeypatch):
    agent = _ScriptedAgent(clock, _resume_then_idle_steps())
    fired_at_read = []
    monkeypatch.setattr(
        gateway_run, "_abandon_timed_out_gateway_turn", lambda **_kw: fired_at_read.append(agent.reads))

    _watch_gateway_turn_inactivity(
        agent_holder=[agent], task_id="t", process_baseline=None, timeout=TIMEOUT_S,
        worker_done=threading.Event(), timeout_fired=threading.Event(),
        cleanup_lock=threading.Lock(), poll_interval=0.01)

    # Read 2 is the first poll after resume (2001 s wall idle, 1 s awake): it must not fire.
    assert fired_at_read == [3]


@pytest.mark.asyncio
async def test_turn_poll_loop_ignores_host_sleep(clock, monkeypatch):
    turn_ctx = TurnContext()
    agent = _ScriptedAgent(clock, [lambda c: c.sleep(SLEEP_S), lambda c: c.awake(TIMEOUT_S)])
    turn_ctx.agent_holder[0] = agent
    runner = object.__new__(GatewayRunner)
    runner._agent_activity_summary = lambda a: a.get_activity_summary()
    warned_at_read = []
    fired_at_read = []

    async def _warn(*_args):
        warned_at_read.append(agent.reads)

    async def _no_backup_check(*_args):
        return None

    async def _poll_without_waiting(_fs, timeout=None):
        await asyncio.sleep(0)
        return set(), set()

    runner._run_agent_inactivity_warning = _warn
    runner._run_agent_backup_interrupt_check = _no_backup_check
    runner._run_agent_timeout_result = lambda _worker, _ctx: "timed out"
    monkeypatch.setattr(asyncio, "wait", _poll_without_waiting)
    monkeypatch.setattr(
        gateway_run, "_abandon_timed_out_gateway_turn", lambda **_kw: fired_at_read.append(agent.reads))
    worker = GatewayRunner._RunAgentWorker(
        agent_timeout=TIMEOUT_S, agent_warning=900.0, task_id="t", process_baseline=None,
        worker_done=threading.Event(), timeout_fired=threading.Event(),
        cleanup_lock=threading.Lock(), is_current=lambda: True)
    worker.executor_task = object()

    result = await runner._run_agent_await_turn_worker(worker, turn_ctx, asyncio.Event(), None)

    assert result == "timed out"
    assert warned_at_read == [2]
    assert agent.reads == 2


def test_stale_agent_eviction_ignores_host_sleep(clock, monkeypatch):
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", str(TIMEOUT_S))
    turn = TurnState(ctx=TurnContext(), started_ts=clock.wall)
    agent = _ScriptedAgent(clock, [lambda c: c.sleep(SLEEP_S), lambda c: c.awake(TIMEOUT_S)])
    turn.agent = agent
    runner = object.__new__(GatewayRunner)
    runner._peek_session_state = lambda _key: SimpleNamespace(turn=turn)
    evicted_at_read = []
    runner._hm_evict_running_agent = lambda _key, _reason: evicted_at_read.append(agent.reads)

    runner._hm_evict_idle_stale_agent("k")
    assert evicted_at_read == []

    runner._hm_evict_idle_stale_agent("k")
    assert evicted_at_read == [2]


def test_stale_agent_eviction_drops_sleep_credit_once_the_agent_was_active_again(clock, monkeypatch):
    """Eviction samples the meter only when a message arrives. Sleep before the agent's
    post-resume activity must not keep discounting the idle time that followed that activity."""
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", str(TIMEOUT_S))
    turn = TurnState(ctx=TurnContext(), started_ts=clock.wall)

    def work_after_resume_then_idle(c):
        c.awake(10.0)
        agent._origin = c.wall  # the agent's last activity
        c.awake(TIMEOUT_S)

    agent = _ScriptedAgent(clock, [lambda c: c.sleep(SLEEP_S), work_after_resume_then_idle])
    turn.agent = agent
    runner = object.__new__(GatewayRunner)
    runner._peek_session_state = lambda _key: SimpleNamespace(turn=turn)
    evicted_at_read = []
    runner._hm_evict_running_agent = lambda _key, _reason: evicted_at_read.append(agent.reads)

    runner._hm_evict_idle_stale_agent("k")
    assert evicted_at_read == []

    runner._hm_evict_idle_stale_agent("k")
    assert evicted_at_read == [2]
