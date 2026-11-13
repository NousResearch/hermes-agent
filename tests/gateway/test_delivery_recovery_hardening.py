"""A reconnect or drain must preserve a reply's durable delivery outcome."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway import delivery_ledger as ledger
from gateway.config import Platform
from gateway.delivery_work import messaging_work_count
from gateway.run_reconnect_recovery import recover_reconnected_sessions
from gateway.run_startup import GatewayStartupMixin
from tests.gateway.test_delivery_ledger import _orphan, _record, _row
from tests.gateway.test_recovery_continuation import continuation_runner, settle
from tests.waiting import async_wait_until


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    runner, adapter, _, _ = continuation_runner(tmp_path, monkeypatch)
    return runner, adapter


def fail_once(monkeypatch, name):
    original = getattr(ledger, name)
    calls = []

    def failing(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise OSError(f"{name} unavailable")
        return original(*args, **kwargs)

    monkeypatch.setattr(ledger, name, failing)
    return calls


def hold_first(monkeypatch, name, *, fails):
    original = getattr(ledger, name)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def held(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            entered.set()
            assert release.wait(10)
            if fails:
                raise OSError(f"{name} unavailable")
        return original(*args, **kwargs)

    monkeypatch.setattr(ledger, name, held)
    return entered, release, calls


def claim_reply():
    _record(adapter_profile="default")
    ledger.mark_failed("ob-1", "send_path_degraded")
    [row] = ledger.sweep_failed_for_runtime("slack")
    return row


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [True, False])
async def test_cancelling_a_settlement_keeps_the_write_outcome(
    gateway, monkeypatch, fails
):
    from gateway.run_delivery_settlement import settle_claim

    runner, adapter = gateway
    row = claim_reply()
    name = "mark_delivered" if fails else "release_runtime_claim"
    state = "delivered" if fails else "unsent"
    entered, release, calls = hold_first(monkeypatch, name, fails=fails)
    runner._running = False
    task = asyncio.create_task(settle_claim(runner, row, state, "send_path_degraded"))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert messaging_work_count(runner) == 1
    finally:
        release.set()
    await async_wait_until(lambda: not runner._delivery_settlement_writes)
    assert bool(runner._delivery_settlements) == fails
    runner._running = True
    await runner._redeliver_failed_obligations_for_platform(Platform.SLACK)
    await settle(runner)
    assert adapter.send.await_count == (0 if fails else 1)
    assert len(calls) == (2 if fails else 1)
    assert (_row("ob-1")["state"], _row("ob-1")["attempts"]) == ("delivered", 1)


@pytest.mark.asyncio
async def test_cancelling_replay_refunds_only_undispatched_claims(gateway, monkeypatch):
    runner, adapter = gateway
    for chat in ("CA", "CB", "CC"):
        _record(
            chat,
            f"agent:main:slack:channel:{chat}",
            chat_id=chat,
            adapter_profile="default",
        )
        ledger.mark_failed(chat, "send_path_degraded")
    entered, blocked = asyncio.Event(), asyncio.Event()

    async def send(*, chat_id, content, metadata=None):
        if chat_id == "CA":
            return SimpleNamespace(success=False, error="temporary API error")
        assert chat_id == "CB"
        entered.set()
        await blocked.wait()

    adapter.send.side_effect = send
    fail_once(monkeypatch, "mark_failed")
    runner._running = False  # keep the retry worker out of the cancellation window
    task = asyncio.create_task(
        runner._redeliver_failed_obligations_for_platform(Platform.SLACK)
    )
    try:
        await asyncio.wait_for(entered.wait(), 10)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert {
        chat: (_row(chat)["state"], _row(chat)["attempts"])
        for chat in ("CA", "CB", "CC")
    } == {
        "CA": ("failed", 1),
        "CB": ("failed", 1),
        "CC": ("failed", 0),
    }
    assert _row("CA")["last_error"] == "temporary API error"


@pytest.mark.asyncio
async def test_partial_claim_refund_is_retried_after_correlated_store_fault(
    gateway, monkeypatch
):
    runner, adapter = gateway
    _record(adapter_profile="default")
    _orphan("ob-1")
    fail_once(monkeypatch, "sweep_failed_for_runtime")
    refunds = fail_once(monkeypatch, "release_runtime_claim")
    await recover_reconnected_sessions(runner, Platform.SLACK)
    await settle(runner)
    assert len(refunds) == 2
    adapter.send.assert_awaited_once()
    assert (_row("ob-1")["state"], _row("ob-1")["attempts"]) == ("delivered", 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_partial_claim_arms_the_adopted_flood_timer(gateway, monkeypatch, cancel):
    runner, adapter = gateway
    _record(adapter_profile="default")
    ledger.mark_failed("ob-1", "flood_control:30")
    _orphan("ob-1")
    runner._schedule_flood_redelivery = Mock()
    entered, release, _ = hold_first(
        monkeypatch, "sweep_failed_for_runtime", fails=True
    )
    task = asyncio.create_task(
        runner._redeliver_failed_obligations_for_platform(Platform.SLACK)
    )
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        if cancel:
            task.cancel()
    finally:
        release.set()
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        assert await task is None
    runner._schedule_flood_redelivery.assert_called_once_with(
        Platform.SLACK, profile="default"
    )
    adapter.send.assert_not_awaited()
    assert (_row("ob-1")["state"], _row("ob-1")["attempts"]) == ("failed", 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("relay", [False, True])
@pytest.mark.parametrize("stop", [False, True])
async def test_drain_counts_and_retries_an_inflight_receipt(
    tmp_path, monkeypatch, relay, stop
):
    from gateway.run_delivery_settlement import settle_claim

    runner, adapter, connector, _ = continuation_runner(
        tmp_path, monkeypatch, relay=relay
    )
    row = claim_reply()
    entered, release, calls = hold_first(monkeypatch, "mark_delivered", fails=True)
    task = asyncio.create_task(settle_claim(runner, row, "delivered"))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        runner._draining = True
        if stop:
            runner._running = False
            runner._stop_draining_delivery = True
        assert runner._drain_work_counts()[0] == 1
    finally:
        release.set()
    assert not await task
    await settle(runner)
    assert runner._drain_work_counts()[0] == 0
    assert len(calls) == 2
    assert _row("ob-1")["state"] == "delivered"
    if relay:
        assert not connector.sent
    else:
        adapter.send.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["sweep", "clear"])
async def test_boot_wiring_defers_resume_until_recovery_finishes(
    gateway, monkeypatch, fault
):
    runner, adapter = gateway
    _record(adapter_profile="default")
    _orphan("ob-1")
    if fault == "sweep":
        fail_once(monkeypatch, "sweep_recoverable")
    else:
        clear = runner._async_session_store.clear_resume_pending
        original = clear.side_effect

        async def clear_once(key):
            clear.side_effect = original
            raise OSError("session store busy")

        clear.side_effect = clear_once
    monkeypatch.setattr("gateway.shutdown_flush.recover_gateway_pending", Mock())
    monkeypatch.setattr("gateway.run._restart_notification_pending", lambda: False)
    monkeypatch.setattr(
        "gateway.run._planned_restart_notification_pending", lambda: False
    )
    monkeypatch.setattr("gateway.run._startup_restore_drain_timeout_secs", lambda: 0)
    monkeypatch.setattr(
        "gateway.run_runtime.recover_gateway_native_sessions", AsyncMock()
    )
    monkeypatch.setattr("gateway.cron_store_notices.install_cron_store_notices", Mock())
    runner._start_post_connect_services = AsyncMock()
    runner._send_restart_notification = AsyncMock()
    runner._finish_startup_restore = AsyncMock()
    runner._send_session_db_warning_notifications = AsyncMock()
    runner._spawn_supervised = Mock()
    resume = runner._schedule_resume_pending_sessions = Mock(return_value=0)
    await GatewayStartupMixin._start_finish_wiring(runner, 0)
    await settle(runner)
    resume.assert_called_once_with(platform=Platform.SLACK, profile="default")
    adapter.send.assert_awaited_once()
    assert _row("ob-1")["state"] == "delivered"
