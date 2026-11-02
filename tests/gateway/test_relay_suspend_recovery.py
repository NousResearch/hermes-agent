"""A disconnected final reply must survive the gap before reconnect recovery starts."""

import asyncio
import sqlite3
from contextlib import closing
from unittest.mock import AsyncMock

import pytest

from gateway import delivery_ledger as ledger
from gateway.config import Platform
from gateway.run_delivery_recovery import schedule_relay_recovery
from tests.gateway.restart_test_helpers import make_restart_runner
from tests.gateway.test_relay_ledger_recovery import record, setup_runner


def idle_relay(tmp_path, monkeypatch):
    runner, adapter, connector = setup_runner(tmp_path, monkeypatch)
    runner._background_tasks = set()
    runner._last_inbound_at = 0
    runner._draining = False
    runner._scale_to_zero_cooldown_until = 0
    runner._scale_to_zero_direct_platform_logged = False
    monkeypatch.setattr(runner, "_running_cron_job_count", lambda: 0)
    monkeypatch.setattr(runner, "_active_cron_job_count", lambda: 0)
    monkeypatch.setattr(runner, "_api_server_hook", lambda *args: 0)
    monkeypatch.setattr(runner, "_scale_to_zero_idle_timeout_seconds", lambda: 120)
    monkeypatch.setattr(
        "gateway.scale_to_zero.dashboard_client_last_seen", lambda: None
    )
    monkeypatch.setattr("gateway.scale_to_zero.suspend_available", lambda: True)
    adapter.hold_redial = lambda: True
    return runner, adapter, connector


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["default", "secondary"])
async def test_disconnect_blocks_suspend_until_reconnect_reply_settles(
    tmp_path, monkeypatch, profile
):
    runner, adapter, connector = idle_relay(tmp_path, monkeypatch)
    adapter._owner_profile = profile
    if profile != "default":
        runner._profile_adapters = {profile: runner.adapters}
        runner._profile_configs = {profile: runner.config}
        runner.adapters = {}
    record(profile=profile)
    ledger.mark_failed("owed", "relay transport not connected")
    connector.delivery_ready = False
    assert ledger.pending_retries() == []
    assert (
        runner._active_work_count() == 0
    )  # waiting on reconnect must not block restart

    dormant = AsyncMock(return_value=True)
    monkeypatch.setattr(runner, "_relay_adapter_for_dormancy", lambda: adapter)
    monkeypatch.setattr(adapter, "go_dormant", dormant)
    monkeypatch.setattr(runner, "_scale_to_zero_status", lambda *args: None)
    suspends = []

    async def suspend():
        suspends.append(True)
        runner._running = False

    monkeypatch.setattr(runner, "_scale_to_zero_self_suspend", suspend)
    real_idle = runner._scale_to_zero_is_idle

    def one_check():
        idle = real_idle()
        runner._running = False
        return idle

    monkeypatch.setattr(runner, "_scale_to_zero_is_idle", one_check)
    await asyncio.wait_for(runner._scale_to_zero_watcher(interval=0), 10)
    dormant.assert_not_awaited()
    assert not suspends

    # Handshake starts recovery without any new inbound message or model turn.
    runner._running = True
    monkeypatch.setattr(runner, "_scale_to_zero_is_idle", real_idle)
    connector.delivery_ready = True
    entered, release = asyncio.Event(), asyncio.Event()
    original_send = connector.send_outbound

    async def held_send(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original_send(*args, **kwargs)

    monkeypatch.setattr(connector, "send_outbound", held_send)
    schedule_relay_recovery(adapter)
    tasks = list(runner._background_tasks)
    try:
        await asyncio.wait_for(entered.wait(), 10)
        assert real_idle() is False
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), 10)
    assert len(connector.sent) == 1
    with ledger._transaction() as db:
        assert db.execute(
            "SELECT state, attempts FROM delivery_obligations"
        ).fetchone() == ("delivered", 1)
    assert real_idle() is True
    runner._scale_to_zero_cooldown_until = 0
    await asyncio.wait_for(runner._scale_to_zero_watcher(interval=0), 10)
    dormant.assert_awaited_once()
    assert suspends == [True]


@pytest.mark.parametrize(
    "case,blocks",
    [
        ("waiting", True),
        ("refunded", True),
        ("expired", False),
        ("exhausted", False),
        ("other_process", False),
        ("reused_pid", False),
        ("unserved_profile", False),
        ("unadvertised_platform", False),
        ("native_priority", False),
        ("disabled", False),
        ("delivered", False),
        ("ledger_disabled", False),
        ("unknown_owner", True),
        ("unreadable", True),
    ],
)
def test_idle_reconnect_wait_respects_recovery_ownership_and_limits(
    tmp_path, monkeypatch, case, blocks
):
    runner, _, _ = idle_relay(tmp_path, monkeypatch)
    record(
        profile="unserved" if case == "unserved_profile" else "default",
        logical="telegram" if case == "unadvertised_platform" else "slack",
    )
    ledger.mark_failed(
        "owed", "send_path_degraded" if case == "refunded" else "relay transport closed"
    )
    updates = {
        "expired": (
            "created_at=?",
            (ledger.time.time() - ledger.STALE_AFTER_SECONDS - 1,),
        ),
        "exhausted": ("attempts=?", (ledger.MAX_ATTEMPTS,)),
        "other_process": ("owner_pid=?", (424243,)),
        "reused_pid": ("owner_started_at=?", (1235.5,)),
    }
    if case in updates:
        assignment, values = updates[case]
        with ledger._transaction() as db:
            db.execute(f"UPDATE delivery_obligations SET {assignment}", values)

    def cannot_read(*args, **kwargs):
        raise sqlite3.OperationalError("database unavailable")

    actions = {
        "native_priority": lambda: monkeypatch.setitem(
            runner.adapters, Platform.SLACK, object()
        ),
        "disabled": lambda: monkeypatch.setattr(
            runner.config.platforms[Platform.RELAY], "enabled", False
        ),
        "delivered": lambda: ledger.mark_delivered("owed"),
        "ledger_disabled": lambda: monkeypatch.setattr(
            ledger, "ledger_enabled", lambda: False
        ),
        "unknown_owner": lambda: monkeypatch.setattr(
            ledger, "_owner_stamp", lambda: (424242, None)
        ),
        "unreadable": lambda: monkeypatch.setattr(
            ledger.sqlite3, "connect", cannot_read
        ),
    }
    action = actions.get(case)
    if action:
        action()
    assert runner._scale_to_zero_is_idle() is (not blocks)


def test_idle_probe_never_creates_or_migrates_the_ledger_or_waits_on_its_writer(
    tmp_path, monkeypatch
):
    runner, _, _ = idle_relay(tmp_path, monkeypatch)
    assert runner._scale_to_zero_is_idle() is True
    assert not ledger._db_path().exists()
    with ledger._transaction() as db:
        db.execute("DROP TABLE delivery_obligations")
    assert runner._scale_to_zero_is_idle() is True
    with closing(sqlite3.connect(ledger._db_path())) as db:
        assert not db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='delivery_obligations'"
        ).fetchone()
    ledger._DB_LOCK.acquire()
    try:
        assert runner._scale_to_zero_is_idle() is False
    finally:
        ledger._DB_LOCK.release()


@pytest.mark.asyncio
async def test_waiting_reply_does_not_block_requested_restart_and_recovers_at_boot(
    tmp_path, monkeypatch
):
    runner, adapter, connector = idle_relay(tmp_path, monkeypatch)
    restart, _ = make_restart_runner()
    restart.config, restart.adapters = runner.config, runner.adapters
    restart._async_session_store = runner._async_session_store
    adapter.gateway_runner = restart
    record()
    ledger.mark_failed("owed", "relay transport not connected")
    assert runner._scale_to_zero_is_idle() is False
    restart.stop = AsyncMock()
    assert restart.request_restart(via_service=True)
    assert restart._restart_task is not None
    await asyncio.wait_for(restart._restart_task, 10)
    restart.stop.assert_awaited_once_with(
        restart=True, detached_restart=False, service_restart=True
    )
    assert not connector.sent
    monkeypatch.setattr(ledger, "_owner_stamp", lambda: (424243, 1235.5))
    monkeypatch.setattr(ledger, "_owner_alive", lambda pid, started: False)
    assert await restart._redeliver_pending_obligations() == 1
    assert len(connector.sent) == 1
    with ledger._transaction() as db:
        assert db.execute(
            "SELECT state, owner_pid, attempts FROM delivery_obligations"
        ).fetchone() == ("delivered", 424243, 1)


@pytest.mark.asyncio
async def test_reply_failure_during_quiesce_aborts_suspend_and_releases_redial(
    tmp_path, monkeypatch
):
    runner, adapter, _ = idle_relay(tmp_path, monkeypatch)
    released = []

    async def quiesce():
        record()
        ledger.mark_failed("owed", "relay transport connection lost")
        runner._running = False
        return True

    adapter.release_redial = lambda: released.append(True) or True
    monkeypatch.setattr(adapter, "go_dormant", quiesce)
    monkeypatch.setattr(runner, "_scale_to_zero_status", lambda *args: None)
    suspend = AsyncMock()
    monkeypatch.setattr(runner, "_scale_to_zero_self_suspend", suspend)
    await asyncio.wait_for(runner._scale_to_zero_watcher(interval=0), 10)
    suspend.assert_not_awaited()
    assert released == [True]
