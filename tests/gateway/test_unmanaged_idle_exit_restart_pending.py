"""An auto-started gateway that installed a dependency it cannot load retires once its turn ends.

Without this the CLI's next attempt (or a fresh ``hermes chat``) reattached to the same process and
failed the same way until the idle window ran out. Real asyncio and a real ticket store; the runner
is a stand-in exposing only what the idle predicate reads.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from gateway import run_idle_exit
from gateway.config import Platform
from gateway.runtime_bootstrap import TicketStore
from pm import environments_adopt
from tests.waiting import async_wait_until


def _runner(tmp_path, monkeypatch, *, work):
    """``work[0]``: agent turns in flight; ``work[1]``: how often the watcher has probed for them."""
    home = tmp_path / "home"
    home.mkdir()
    stops = []

    def active_work_count():
        work[1] += 1
        return work[0]

    async def stop():
        stops.append(True)
        runner._running = False
    runner = SimpleNamespace(
        _running=True, _draining=False, adapters={Platform.LOCAL: object()},
        _profile_adapters={}, _profile_failed_platforms={}, _failed_platforms={},
        config=SimpleNamespace(platforms={}), session_ticket_store=TicketStore("i", frozenset({str(home)})),
        session_runtime_descriptor={"state": "ready", "capabilities": ["x"]}, _exit_reason=None,
        _active_work_count=active_work_count, _scale_to_zero_has_live_background_work=lambda: False, stop=stop)
    # The default 600 s window: only a pending restart may end it within this test.
    monkeypatch.setattr(run_idle_exit, "idle_exit_seconds", lambda r: 600.0)
    for probe in ("_ledger_busy", "_cron_busy", "_store_gates_busy"):
        monkeypatch.setattr(run_idle_exit, probe, lambda *_a: None)
    monkeypatch.setattr(run_idle_exit, "_kanban_busy", lambda: None)
    monkeypatch.setattr(run_idle_exit, "_served_homes", lambda r: [home])
    monkeypatch.setattr(environments_adopt, "_restart_pending", None)
    return runner, stops


@pytest.mark.asyncio
async def test_pending_restart_retires_once_the_turn_ends_even_with_a_client_attached(tmp_path, monkeypatch):
    work = [1, 0]
    runner, stops = _runner(tmp_path, monkeypatch, work=work)
    with run_idle_exit.attached_client(runner):
        watcher = asyncio.ensure_future(run_idle_exit.unmanaged_idle_exit_watcher(runner))
        environments_adopt.note_restart_pending("anthropic installed (dependency generation g2 was published)")
        await async_wait_until(lambda: work[1] >= 4, timeout=10, what="two restart probes")
        assert not watcher.done() and runner.session_runtime_descriptor["state"] == "ready"  # turn in flight

        # A client mid-attach (minted, not redeemed) holds it: its refused first connect never rediscovers.
        store = runner.session_ticket_store
        store.mint(profile_id=next(iter(store.profile_ids)), subject="cli", purpose="interactive")
        checks = [0]
        real_outstanding = store.outstanding

        def outstanding():
            checks[0] += 1
            return real_outstanding()
        monkeypatch.setattr(store, "outstanding", outstanding)
        work[0] = 0
        await async_wait_until(lambda: checks[0] >= 4, timeout=10, what="probes while attaching")
        assert not watcher.done() and stops == []

        store.revoke()
        await asyncio.wait_for(watcher, 5)
        await runner._idle_exit_stop_task
    assert stops == [True]
    # The drain clients already wait out before starting a fresh gateway.
    assert runner.session_runtime_descriptor["drain_reason"] == run_idle_exit.IDLE_EXIT_REASON
    assert runner._exit_reason.startswith(run_idle_exit.IDLE_EXIT_STOP_PREFIX)
    assert "anthropic installed" in runner._exit_reason


@pytest.mark.asyncio
async def test_pending_restart_never_drops_a_duty_nothing_would_restart(tmp_path, monkeypatch):
    """Clients come back and start a fresh gateway; a messaging adapter would just go dark."""
    work = [0, 0]
    runner, stops = _runner(tmp_path, monkeypatch, work=work)
    runner.config.platforms[Platform.TELEGRAM] = SimpleNamespace(enabled=True)
    watcher = asyncio.ensure_future(run_idle_exit.unmanaged_idle_exit_watcher(runner))
    environments_adopt.note_restart_pending("anthropic installed")
    await async_wait_until(lambda: work[1] >= 4, timeout=10, what="two restart probes")
    assert not watcher.done() and stops == []
    assert runner.session_runtime_descriptor["state"] == "ready"
    watcher.cancel()
