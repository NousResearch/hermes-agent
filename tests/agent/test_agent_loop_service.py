"""The plugin-facing ``agent_loop`` pool: one thread per run, bounded queue, stall release."""

from __future__ import annotations

import contextvars
import threading
import time

import pytest
from tests.waiting import wait_until

from agent import agent_loop_service
from agent.agent_loop_service import (
    AgentLoopQueueFull, AgentLoopService, PluginAgentLoop, reset_service,
)


@pytest.fixture(autouse=True)
def _fresh_pool():
    reset_service()
    yield
    reset_service()


def test_simultaneous_runs_each_hold_their_own_thread():
    service = AgentLoopService(max_workers=3, max_run_seconds=0)
    rendezvous = threading.Barrier(3, timeout=10)

    def task() -> int:
        ident = threading.get_ident()
        rendezvous.wait()  # only three live threads can reach this together
        return ident

    jobs = [service.submit(task) for _ in range(3)]
    idents = [job.future.result(timeout=10) for job in jobs]

    assert len(set(idents)) == 3, "three concurrent runs must use three threads"
    assert threading.get_ident() not in idents, "no run executes on the caller's thread"


def test_the_caller_context_follows_the_run():
    service = AgentLoopService(max_workers=1, max_run_seconds=0)
    lane = contextvars.ContextVar("test_agent_loop_lane", default="none")
    seen: list[str] = []

    token = lane.set("plugin-profile")
    try:
        assert service.run(lambda: seen.append(lane.get()) or "ok") == "ok"
    finally:
        lane.reset(token)

    assert seen == ["plugin-profile"], "a pool thread must inherit the submitter's contextvars"


def test_a_queued_job_reports_its_place_and_runs_in_order():
    service = AgentLoopService(max_workers=1, max_queued=2, max_run_seconds=0)
    release = threading.Event()
    first = service.submit(lambda: release.wait(10))
    wait_until(lambda: first.started_at, what="the first job started")

    second = service.submit(lambda: "second")
    third = service.submit(lambda: "third")
    assert (second.queue_position, third.queue_position) == (1, 2)
    with pytest.raises(AgentLoopQueueFull):
        service.submit(lambda: "fourth")

    release.set()
    assert second.future.result(timeout=10) == "second"
    assert third.future.result(timeout=10) == "third"


def test_an_exception_reaches_the_future_and_frees_the_slot():
    service = AgentLoopService(max_workers=1, max_run_seconds=0)

    def boom() -> None:
        raise ValueError("no route")

    failed = service.submit(boom)
    with pytest.raises(ValueError, match="no route"):
        failed.future.result(timeout=10)
    assert service.run(lambda: "next") == "next"


def test_a_stalled_run_is_settled_once_and_its_slot_released():
    service = AgentLoopService(max_workers=1, max_run_seconds=0.05, stall_check_seconds=1.0)
    release = threading.Event()
    settled: list[tuple[str, BaseException | None]] = []
    stuck = service.submit(
        lambda: release.wait(10),
        on_settle=lambda job, _value, error: settled.append((job.job_id, error)))
    wait_until(lambda: stuck.started_at, what="the job took its slot")
    stuck.started_at = time.time() - 60  # pretend it has held the slot for a minute

    assert service._scan_stalls() == [stuck.job_id]
    with pytest.raises(TimeoutError, match="slot was released"):
        stuck.future.result(timeout=1)
    assert [job_id for job_id, _error in settled] == [stuck.job_id], (
        "a stalled run must still deliver the caller's completion callback")
    assert isinstance(settled[0][1], TimeoutError)
    assert service.run(lambda: "next") == "next", "the released slot must serve the next run"
    assert service._scan_stalls() == [], "a settled job cannot be reported twice"
    release.set()
    assert len(settled) == 1, "the late finish must not report the same job again"


def test_settle_callback_runs_once_per_job():
    service = AgentLoopService(max_workers=1, max_run_seconds=0)
    seen: list[tuple[str, object]] = []

    job = service.submit(lambda: "value",
                         on_settle=lambda settled, value, error: seen.append(
                             (settled.job_id, value)))
    assert job.future.result(timeout=10) == "value"
    wait_until(lambda: seen, what="the settle callback ran")
    assert seen == [(job.job_id, "value")]


def test_plugin_facade_forwards_the_run_parameters_and_its_identity(monkeypatch):
    from agent import agent_run

    calls: list[tuple[str, dict]] = []
    result = agent_run.AgentRunResult(answer="ok", api_calls=1, model="m", provider="p",
                                      toolsets=("file",))

    def fake_run(message, **kwargs):
        calls.append((message, kwargs))
        return result

    monkeypatch.setattr(agent_run, "run", fake_run)
    loop = PluginAgentLoop("minir")

    assert loop.run("hello", toolsets=["file"], history_session="s1") is result
    assert calls == [("hello", {"layer": "minir", "toolsets": ["file"],
                                "history_session": "s1"})]
    assert loop.capability_version == agent_loop_service.CAPABILITY_VERSION

    job = loop.submit("later", model="m")
    assert job.future.result(timeout=10) is result
    assert calls[-1] == ("later", {"layer": "minir", "model": "m"})


def test_the_plugin_context_exposes_agent_loop():
    from hermes_cli.plugins import PluginContext
    from hermes_cli.plugins_manifest import PluginManifest

    ctx = PluginContext(PluginManifest(name="demo"), object())

    assert isinstance(ctx.agent_loop, PluginAgentLoop)
    assert ctx.agent_loop.plugin_id == "demo"
    assert ctx.agent_loop is ctx.agent_loop, "the facade is created once per context"
