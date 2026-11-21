"""A gateway stop sends its in-process Stop during the wait, not only after it.

``settle_gateway_runtime`` waits TURN_SETTLE_SECONDS for the authority's tasks to settle. The Stop
reached an executing in-process turn, a turn worker, a side agent or a cron job only in
``join_authority_work``, after the whole wait, so a turn whose agent was built a few seconds into the
stop rode the wait out. The wait now sends the in-process half first and again every
STOP_RESEND_SECONDS, interrupting each agent once.
"""
import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from gateway import run_runtime


def _runner(authority, monkeypatch):
    runner = SimpleNamespace(session_authority=authority, _running_agents={},
                             session_ticket_store=SimpleNamespace(revoke=lambda: None),
                             session_api=SimpleNamespace())
    authority.runner = runner
    authority.persistence = SimpleNamespace(
        bindings=SimpleNamespace(local_session_ids=lambda: []),
    )

    async def stop_api(_handle):
        return None

    monkeypatch.setattr("gateway.run_api.stop_gateway_api", stop_api)
    return runner


def _live(task):
    return SimpleNamespace(task=task, route="r", event_stream=SimpleNamespace(execution={}))


@pytest.mark.asyncio
async def test_a_turn_worker_built_after_the_first_sweep_is_stopped_within_the_wait(monkeypatch):
    """The card's scenario: the agent exists only after the first sweep, and the wait must reach it."""
    interrupts = []
    done = threading.Event()

    class _Agent:
        def hard_interrupt(self, message=None):
            interrupts.append((time.monotonic(), message))

    holder = [_Agent()]
    hold = asyncio.Event()
    task = asyncio.create_task(hold.wait())
    authority = SimpleNamespace(sessions={"s": _live(task)}, _managed_workers={}, pending_stops={},
                                adopted={}, profile_id="p",
                                _turn_workers={1: (SimpleNamespace(worker_done=done), holder)})
    runner = _runner(authority, monkeypatch)
    monkeypatch.setattr(run_runtime, "TURN_SETTLE_SECONDS", 1.6)
    began = time.monotonic()
    try:
        await asyncio.wait_for(run_runtime.settle_gateway_runtime(runner), 30)
    finally:
        hold.set()
        await asyncio.gather(task, return_exceptions=True)
    assert interrupts, "the in-process stop never reached the turn worker"
    assert interrupts[0][0] - began < 1.0, "the interrupt arrived only after the wait"


class _RecordingEvent(threading.Event):
    def __init__(self):
        super().__init__()
        self.at = None

    def set(self):
        if self.at is None:
            self.at = time.monotonic()
        super().set()


@pytest.mark.asyncio
async def test_a_cron_job_is_cancelled_at_the_start_of_the_wait(monkeypatch):
    cancel = _RecordingEvent()
    hold = asyncio.Event()
    task = asyncio.create_task(hold.wait())
    authority = SimpleNamespace(sessions={"s": _live(task)}, _managed_workers={}, pending_stops={},
                                adopted={}, profile_id="p", _cron_cancellations={"job": cancel})
    runner = _runner(authority, monkeypatch)
    monkeypatch.setattr(run_runtime, "TURN_SETTLE_SECONDS", 1.6)
    began = time.monotonic()
    try:
        await asyncio.wait_for(run_runtime.settle_gateway_runtime(runner), 30)
    finally:
        hold.set()
        await asyncio.gather(task, return_exceptions=True)
    assert cancel.is_set(), "the cron cancellation was never set"
    assert cancel.at - began < 1.0, "the cron cancellation arrived only after the wait"


@pytest.mark.asyncio
async def test_the_stuck_agent_is_interrupted_once_per_send_not_twice_a_second(monkeypatch):
    """Each hard interrupt aborts the request and logs a line: a stuck agent must not be hit 0.5 s apart."""
    interrupts = []
    done = threading.Event()

    class _Agent:
        def hard_interrupt(self, message=None):
            interrupts.append(message)

    stuck = [_Agent()]
    hold = asyncio.Event()
    task = asyncio.create_task(hold.wait())
    authority = SimpleNamespace(sessions={"s": _live(task)}, _managed_workers={}, pending_stops={},
                                adopted={}, profile_id="p",
                                _turn_workers={1: (SimpleNamespace(worker_done=done), stuck)})
    runner = _runner(authority, monkeypatch)
    monkeypatch.setattr(run_runtime, "TURN_SETTLE_SECONDS", 1.6)
    try:
        await asyncio.wait_for(run_runtime.settle_gateway_runtime(runner), 30)
    finally:
        hold.set()
        await asyncio.gather(task, return_exceptions=True)
    # The wait's sends plus the join's own send: the same agent is not interrupted on every resend.
    assert len(interrupts) <= 2, interrupts
