"""Profile-scope regressions for gateway timeout watchdog/reaper thread hops."""

from __future__ import annotations

import asyncio
import dataclasses
import threading
from types import SimpleNamespace
from typing import Any, Optional

import gateway.run as gateway_run
import gateway.run_turn as gateway_run_turn
from gateway.run_turn import GatewayTurnMixin
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from tools.process_registry import process_registry


@dataclasses.dataclass
class _Worker:
    executor_task: Any = None
    agent_timeout: Optional[float] = None
    agent_warning: Optional[float] = None
    task_id: str = ""
    process_baseline: frozenset = frozenset()
    worker_done: threading.Event = dataclasses.field(default_factory=threading.Event)
    timeout_fired: threading.Event = dataclasses.field(default_factory=threading.Event)
    cleanup_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    is_current: Any = None


class _Runner(GatewayTurnMixin):
    _RunAgentWorker = _Worker

    def _is_session_run_current(self, _session_key, _generation):
        return True

    async def _run_in_executor_with_context(self, fn):
        return fn()


class _IdleAgent:
    def get_activity_summary(self):
        return {
            "seconds_since_activity": 10.0,
            "last_activity_desc": "waiting",
            "api_call_count": 1,
            "max_iterations": 10,
        }


def _homes(tmp_path, monkeypatch):
    launch_home = tmp_path / "launch"
    served_home = tmp_path / "served"
    launch_home.mkdir()
    served_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    return launch_home, served_home


def test_inactivity_watchdog_keeps_turn_profile_context(tmp_path, monkeypatch):
    """The event-loop-independent watchdog must retain the routed turn's profile scope."""
    _launch_home, served_home = _homes(tmp_path, monkeypatch)
    seen_homes = []
    watchdog_ran = threading.Event()

    def _record_watchdog(**_kwargs):
        seen_homes.append(get_hermes_home())
        watchdog_ran.set()

    monkeypatch.setattr(gateway_run, "_watch_gateway_turn_inactivity", _record_watchdog)
    monkeypatch.setattr(
        gateway_run,
        "_float_env",
        lambda name, _default: 1.0 if name == "HERMES_AGENT_TIMEOUT" else 0.0,
    )
    monkeypatch.setattr(process_registry, "snapshot_running_ids", lambda _task_id: frozenset())

    async def _exercise():
        turn_ctx = SimpleNamespace(
            agent_holder=[None], session_key="served-session", run_generation=1,
            session_id="served-session", process_task_id="", process_baseline=frozenset(),
        )
        worker = _Runner()._run_agent_start_turn_worker(turn_ctx, lambda: None)
        assert watchdog_ran.wait(timeout=1.0), "watchdog thread did not run"
        await worker.executor_task

    token = set_hermes_home_override(served_home)
    try:
        asyncio.run(_exercise())
    finally:
        reset_hermes_home_override(token)

    assert seen_homes == [served_home]


def test_asyncio_timeout_reaper_keeps_turn_profile_context(tmp_path, monkeypatch):
    """The loop-side timeout fallback must reap under the same routed profile as its turn."""
    _launch_home, served_home = _homes(tmp_path, monkeypatch)
    seen_homes = []
    reaper_ran = threading.Event()

    def _record_reaper(**_kwargs):
        seen_homes.append(get_hermes_home())
        reaper_ran.set()
        return True

    async def _not_done(_futures, timeout=None):
        return set(), set()

    monkeypatch.setattr(gateway_run, "_abandon_timed_out_gateway_turn", _record_reaper)
    monkeypatch.setattr(gateway_run, "request_hard_interrupt", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(gateway_run_turn.asyncio, "wait", _not_done)

    async def _exercise():
        loop = asyncio.get_running_loop()
        pending = loop.create_future()
        worker = _Worker(
            executor_task=pending,
            agent_timeout=1.0,
            agent_warning=None,
            task_id="served-session",
            process_baseline=frozenset(),
            worker_done=threading.Event(),
            timeout_fired=threading.Event(),
            cleanup_lock=threading.Lock(),
            is_current=lambda: True,
        )
        turn_ctx = SimpleNamespace(
            session_key="served-session",
            agent_holder=[_IdleAgent()],
            result_holder=[None],
            tools_holder=[None],
        )
        try:
            await _Runner()._run_agent_await_turn_worker(
                worker, turn_ctx, asyncio.Event(), None,
            )
            assert reaper_ran.wait(timeout=1.0), "timeout reaper thread did not run"
        finally:
            pending.cancel()

    token = set_hermes_home_override(served_home)
    try:
        asyncio.run(_exercise())
    finally:
        reset_hermes_home_override(token)

    assert seen_homes == [served_home]
