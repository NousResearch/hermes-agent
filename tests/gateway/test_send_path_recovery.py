"""Deterministic recovery coalescing and stale-transport regressions for PR #123929."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from gateway import delivery_ledger as dl
from gateway import run_startup
from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
from hermes_constants import get_hermes_home


class _ManualDebounce:
    """Keep real Event wakes; only the quiet-window deadline is manually expired."""

    def __init__(self):
        self.windows = asyncio.Queue()

    async def wait_for(self, awaitable, *, timeout):
        assert timeout == run_startup._SEND_PATH_RECOVERY_DEBOUNCE_SECONDS
        waiter = asyncio.create_task(awaitable)
        expired = asyncio.get_running_loop().create_future()
        self.windows.put_nowait(expired)
        try:
            done, _ = await asyncio.wait(
                (waiter, expired), return_when=asyncio.FIRST_COMPLETED
            )
            if waiter in done:
                return waiter.result()
            raise asyncio.TimeoutError
        finally:
            waiter.cancel()
            expired.cancel()
            await asyncio.gather(waiter, return_exceptions=True)

    async def next_window(self):
        # A watchdog bounds broken tests; it never advances the debounce clock.
        return await asyncio.wait_for(self.windows.get(), timeout=5)


def _adapter(profile="default"):
    return SimpleNamespace(
        platform=Platform.TELEGRAM,
        _owner_profile=profile,
        is_connected=True,
        has_fatal_error=False,
        send_path_degraded=False,
        send=AsyncMock(return_value=SendResult(success=True, message_id="recovered")),
    )


@pytest_asyncio.fixture
async def recovery_runner(monkeypatch):
    # Match the bare-runner / real SQLite pattern in test_delivery_ledger.py.
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._draining = False
    runner.adapters = {}
    runner._profile_adapters = {}
    runner._primary_profile_name = "default"
    runner._background_tasks = set()
    runner.session_store = None
    runner._async_session_store = SimpleNamespace(
        clear_resume_pending=AsyncMock(), _store=None
    )
    clock = _ManualDebounce()
    # Patch the sibling's asyncio reference, leaving pytest's watchdogs and
    # every other module's asyncio.wait_for untouched.
    local_asyncio = SimpleNamespace(**{name: getattr(asyncio, name) for name in dir(asyncio)})
    local_asyncio.wait_for = clock.wait_for
    monkeypatch.setattr(run_startup, "asyncio", local_asyncio)
    yield runner, clock
    tasks = list(runner._background_tasks)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_recovery_bursts_debounce_and_coalesce_one_trailing_sweep(recovery_runner):
    runner, clock = recovery_runner
    adapter = _adapter()
    runner.adapters[Platform.TELEGRAM] = adapter
    sweep = runner._redeliver_failed_obligations_for_platform = AsyncMock(return_value=0)
    key = ("telegram", "default")

    runner._schedule_send_path_recovery(adapter, reason="recovered")
    worker = runner._send_path_recovery_jobs[key]["task"]
    await clock.next_window()
    # Each burst interrupts a live window. Seeing the next window is the
    # handshake that the worker consumed the wake without sweeping.
    for _ in range(3):
        for _ in range(20):
            runner._schedule_send_path_recovery(adapter, reason="failed-finalized")
        quiet = await clock.next_window()
        assert runner._send_path_recovery_jobs[key]["task"] is worker
        sweep.assert_not_awaited()
    quiet.set_result(None)
    await asyncio.wait_for(worker, timeout=5)
    sweep.assert_awaited_once_with(Platform.TELEGRAM, profile="default")
    assert runner._send_path_recovery_jobs == {}

    started, release = asyncio.Event(), asyncio.Event()

    async def blocked_first_sweep(platform, *, profile):
        if sweep.await_count == 1:
            started.set()
            await release.wait()
        return 0

    sweep.reset_mock()
    sweep.side_effect = blocked_first_sweep
    runner._schedule_send_path_recovery(adapter, reason="recovered")
    worker = runner._send_path_recovery_jobs[key]["task"]
    quiet = await clock.next_window()
    quiet.set_result(None)
    await asyncio.wait_for(started.wait(), timeout=5)
    for _ in range(20):
        runner._schedule_send_path_recovery(adapter, reason="failed-finalized")
    assert runner._send_path_recovery_jobs[key]["task"] is worker
    assert sweep.await_count == 1
    release.set()
    trailing = await clock.next_window()
    assert sweep.await_count == 1
    trailing.set_result(None)
    await asyncio.wait_for(worker, timeout=5)
    assert sweep.await_args_list == [
        ((Platform.TELEGRAM,), {"profile": "default"}),
        ((Platform.TELEGRAM,), {"profile": "default"}),
    ]
    assert runner._send_path_recovery_jobs == {}


@pytest.mark.asyncio
async def test_failed_finalized_resolves_current_row_adapter_and_rejects_stale_instance(
    recovery_runner, tmp_path, monkeypatch
):
    runner, clock = recovery_runner
    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "state.db")
    profile_home = tmp_path / "reviewer"
    profile_home.mkdir()
    monkeypatch.setattr(runner, "_routed_profile_home", lambda profile: profile_home)
    old, replacement, default = _adapter("reviewer"), _adapter("reviewer"), _adapter()
    runner.adapters[Platform.TELEGRAM] = default
    runner._profile_adapters["reviewer"] = {Platform.TELEGRAM: old}
    started, release = asyncio.Event(), asyncio.Event()

    async def fail_after_replacement(**kwargs):
        started.set()
        await release.wait()
        return SendResult(success=False, error="send_path_degraded", retryable=True)

    async def recovered_send(**kwargs):
        assert get_hermes_home() == profile_home
        return SendResult(success=True, message_id="recovered")

    old.send.side_effect = fail_after_replacement
    replacement.send.side_effect = recovered_send
    notify = Mock(wraps=runner._schedule_send_path_recovery)
    monkeypatch.setattr(runner, "_schedule_send_path_recovery", notify)
    dl.record_obligation(
        obligation_id="late-failure", session_key="agent:reviewer:telegram:dm:123",
        platform="telegram", chat_id="123", thread_id="77", content="final answer",
        adapter_profile="reviewer",
    )
    dl.mark_failed("late-failure", "send_path_degraded")
    claimed = dl.sweep_failed_for_runtime("telegram", profile="reviewer")
    assert len(claimed) == 1
    sending = asyncio.create_task(runner._redeliver_claimed_obligations(claimed))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        runner._profile_adapters["reviewer"][Platform.TELEGRAM] = replacement
        release.set()
        assert await asyncio.wait_for(sending, timeout=5) == 0
    finally:
        sending.cancel()
        await asyncio.gather(sending, return_exceptions=True)

    notify.assert_called_once_with(replacement, reason="failed-finalized")
    with dl._connect() as conn:
        assert conn.execute(
            "SELECT state, last_error FROM delivery_obligations WHERE obligation_id='late-failure'"
        ).fetchone() == ("failed", "send_path_degraded")
    quiet = await clock.next_window()
    key = ("telegram", "reviewer")
    entry = runner._send_path_recovery_jobs[key]
    # A direct notification from the retired transport cannot replace or wake
    # the current transport's worker (and cannot create a second worker).
    runner._schedule_send_path_recovery(old, reason="failed-finalized")
    assert list(runner._send_path_recovery_jobs) == [key]
    assert runner._send_path_recovery_jobs[key] is entry
    assert entry["adapter"] is replacement
    assert not entry["wake"].is_set()
    quiet.set_result(None)
    await asyncio.wait_for(entry["task"], timeout=5)
    old.send.assert_awaited_once()
    replacement.send.assert_awaited_once()
    assert replacement.send.await_args.kwargs["chat_id"] == "123"
    assert replacement.send.await_args.kwargs["metadata"] == {"thread_id": "77"}
    default.send.assert_not_awaited()
    with dl._connect() as conn:
        assert conn.execute(
            "SELECT state FROM delivery_obligations WHERE obligation_id='late-failure'"
        ).fetchone() == ("delivered",)
    assert runner._send_path_recovery_jobs == {}
