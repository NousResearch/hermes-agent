"""t_b9436ad2: managed worker turns yield COOPERATIVELY at drain start, not at the timeout edge.

The stop drain waits on agents PLUS managed worker turns (``uncounted_runtime_work``), while the
timeout WARNING printed only ``_running_agents`` — so one held kanban/polling worker pinned every
restart to the full launchd-capped drain ("drain timed out after 50.1s with 0 active agent(s)",
five times on 2026-11-26), and the overrun left zero settle budget for the authority tasks.
``_stop_drain_active_work`` now sends the same cooperative Stop a user's Stop sends BEFORE the
wait loop, so a yielding worker settles in seconds and the drain exits clean (no timeout path,
``.clean_shutdown`` marker intact). In-process chat agents are untouched.
"""
import asyncio
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from gateway.run import GatewayRunner
from gateway.run_shutdown import GatewayShutdownMixin
from tests.gateway.restart_test_helpers import make_restart_runner


@pytest.mark.asyncio
async def test_cooperative_stop_is_sent_once_before_the_drain_loop(monkeypatch):
    runner, _adapter = make_restart_runner()
    runner.session_authority = SimpleNamespace(sessions={}, _managed_workers={})
    order = []

    def fake_stop(r, reason=None):
        order.append(("stop", reason))
        return 0

    async def fake_drain(timeout, cron_timeout=None):
        order.append(("drain", timeout))
        return {}, False

    monkeypatch.setattr("gateway.run_shutdown.stop_managed_turns", fake_stop)
    monkeypatch.setattr(runner, "_drain_active_agents", fake_drain)
    ctx = GatewayShutdownMixin._StopContext(deferred_count=lambda: 0)
    ctx.started_at = time.monotonic()

    await GatewayRunner._stop_drain_active_work(runner, 5.0, ctx)

    assert [kind for kind, _payload in order] == ["stop", "drain"]
    # The restart/shutdown reason rides the same Stop a user's Stop sends, so the turn settles
    # with the right cause and the card/session resumes instead of counting a failure.
    assert order[0][1] == runner._shutdown_interrupt_reason()


@pytest.mark.asyncio
async def test_a_yielding_managed_worker_lets_the_drain_finish_early(tmp_path, monkeypatch):
    """Real child, real control pipe: the child exits at its first stdin frame — the Stop sent at
    drain start reaches it, the count drops, and the wait returns clean in seconds. Before the
    fix the same wait held the FULL budget (the stop only flew after the timeout)."""
    from gateway.session_managed_worker import ManagedWorker

    seen = tmp_path / "controls"
    child = subprocess.Popen(  # noqa: ASYNC220 - Popen returns immediately; the drain drives the child
        [sys.executable, "-c",
         'import sys; open(sys.argv[1], "wb").write(sys.stdin.buffer.readline()); sys.exit(0)',
         str(seen)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    worker = ManagedWorker(child)
    worker.writer.start()
    runner, _adapter = make_restart_runner()
    runner.session_authority = SimpleNamespace(sessions={}, _managed_workers={"s": worker})
    # The owner loop normally closes a worker whose child exited; this double models that with the
    # child's own liveness — exactly the count the drain loop polls.
    monkeypatch.setattr("gateway.session_runtime_workers.uncounted_runtime_work",
                        lambda runner: 1 if child.poll() is None else 0)
    try:
        ctx = GatewayShutdownMixin._StopContext(deferred_count=lambda: 0)
        ctx.started_at = time.monotonic()
        started = time.monotonic()
        await GatewayRunner._stop_drain_active_work(runner, 30.0, ctx)
        waited = time.monotonic() - started
        assert ctx.timed_out is False
        assert waited < 10.0, f"held-managed-turn regression: drain pinned {waited:.1f}s of 30s"
        assert seen.read_bytes() == b'{"type":"stop"}\n'
    finally:
        await asyncio.to_thread(child.wait, 10)
        await asyncio.to_thread(worker.close)


@pytest.mark.asyncio
async def test_yielding_managed_worker_does_not_pin_the_restart_when_idle_wait(tmp_path, monkeypatch):
    """The restart-when-idle wait (#77184 redesign) yields durable managed turns at WAIT START:
    a real control-pipe child exits at its first frame, the wait reaches idle in seconds, and a
    busy fleet no longer holds the restart open indefinitely. Chat turns, delegations and
    notices keep their untouched "never cut on a timer" semantics."""
    from gateway.restart_when_idle import await_idle
    from gateway.session_managed_worker import ManagedWorker

    seen = tmp_path / "controls"
    child = subprocess.Popen(  # noqa: ASYNC220 - Popen returns immediately; the wait drives the child
        [sys.executable, "-c",
         'import sys; open(sys.argv[1], "wb").write(sys.stdin.buffer.readline()); sys.exit(0)',
         str(seen)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    worker = ManagedWorker(child)
    worker.writer.start()
    runner, _adapter = make_restart_runner()
    runner._restart_requested = True
    runner._restart_after_turn_timeout = 30.0
    runner.session_authority = SimpleNamespace(sessions={}, _managed_workers={"s": worker})
    monkeypatch.setattr("gateway.session_runtime_workers.uncounted_runtime_work",
                        lambda runner: 1 if child.poll() is None else 0)
    monkeypatch.setattr(runner, "_scale_to_zero_status", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_describe_active_work", lambda: [])
    try:
        started = time.monotonic()
        idle = await await_idle(runner)
        waited = time.monotonic() - started
        assert idle is True
        assert waited < 10.0, f"held-managed-turn regression: wait pinned {waited:.1f}s of 30s"
        assert seen.read_bytes() == b'{"type":"stop"}\n'
    finally:
        await asyncio.to_thread(child.wait, 10)
        await asyncio.to_thread(worker.close)
