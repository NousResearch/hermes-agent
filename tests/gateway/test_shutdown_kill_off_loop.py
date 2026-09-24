"""Gateway shutdown must not run process teardown on the event loop (#116327).

``ProcessRegistry.kill_all()`` is synchronous and blocking (per-target
``kill_process`` does disk I/O via ``_write_checkpoint`` and may spawn
``subprocess.run`` up to 15 s via ``_stop_systemd_unit``). Driving it inline
from ``_stop_interrupt_remaining_work`` monopolizes the asyncio loop. These
tests pin the invariant: the kill sweep runs off-loop, in phase order.
"""

import asyncio
import threading
import time

import pytest

from gateway.run_shutdown import GatewayShutdownMixin
from tests.gateway.restart_test_helpers import make_restart_runner


def _make_phase_runner(monkeypatch, events):
    runner, _adapter = make_restart_runner()
    runner._restart_drain_timeout = 0.01

    loop_thread = threading.current_thread()

    def _fake_kill_all(task_id=None):
        events.append(("kill_all", threading.current_thread()))
        return 2

    import tools.process_registry as _pr

    monkeypatch.setattr(_pr.process_registry, "kill_all", _fake_kill_all)
    monkeypatch.setattr(
        "cron.scheduler.mark_running_jobs_interrupted", lambda *a, **k: []
    )
    monkeypatch.setattr("tools.async_delegation.interrupt_all", lambda *a, **k: 0)
    monkeypatch.setattr(
        "tools.terminal_tool_lifecycle.cleanup_all_environments",
        lambda: events.append(("cleanup_envs", threading.current_thread())),
    )
    monkeypatch.setattr(
        "tools.browser_tool_lifecycle.cleanup_all_browsers",
        lambda: events.append(("cleanup_browsers", threading.current_thread())),
    )
    return runner, loop_thread


def _make_ctx():
    ctx = GatewayShutdownMixin._StopContext(deferred_count=lambda: 0)
    ctx.started_at = time.monotonic()
    return ctx


@pytest.mark.asyncio
async def test_post_interrupt_kill_runs_off_event_loop(monkeypatch):
    """kill_all must execute on a worker thread, not the loop thread (#116327)."""
    events: list = []
    runner, loop_thread = _make_phase_runner(monkeypatch, events)

    await runner._stop_interrupt_remaining_work(_make_ctx())

    kill_threads = [t for name, t in events if name == "kill_all"]
    assert kill_threads, f"expected kill_all to run, got events: {events}"
    for t in kill_threads:
        assert t is not loop_thread, (
            "kill_all ran on the event-loop thread; it must be offloaded "
            "via asyncio.to_thread (#116327)"
        )


@pytest.mark.asyncio
async def test_post_interrupt_kill_preserves_phase_order(monkeypatch):
    """Offloading must not reorder the teardown sequence within the phase."""
    events: list = []
    runner, _loop_thread = _make_phase_runner(monkeypatch, events)

    await runner._stop_interrupt_remaining_work(_make_ctx())

    names = [name for name, _t in events]
    assert "kill_all" in names
    assert names.index("kill_all") < names.index("cleanup_envs")
    assert names.index("cleanup_envs") < names.index("cleanup_browsers")
    # The loop must still be responsive while the sweep runs: a callback
    # scheduled during teardown fires without waiting for the phase.
    probe = asyncio.Event()
    asyncio.get_running_loop().call_soon(probe.set)
    await asyncio.wait_for(probe.wait(), timeout=5)
