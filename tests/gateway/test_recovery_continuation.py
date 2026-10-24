"""Recovery must finish after transient faults without regenerating an answered turn."""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway import delivery_ledger as ledger
from gateway.config import Platform
from gateway.run_reconnect_recovery import (
    recover_reconnected_sessions,
    schedule_recovery,
)
from gateway.session import SessionEntry, SessionSource, SessionStore
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from tests.gateway.test_offline_boot_delivery_recovery import orphan, recovery_runner
from tests.gateway.test_relay_ledger_recovery import record


def continuation_runner(tmp_path, monkeypatch, *, relay=False, profile="default"):
    runner, adapter, connector = recovery_runner(
        tmp_path, monkeypatch, relay=relay, profile=profile
    )
    runner.config.multiplex_profiles = profile != "default"
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=runner.config)
    runner.session_store = store
    runner._async_session_store._store = store
    runner._async_session_store.clear_resume_pending = AsyncMock(
        side_effect=store.clear_resume_pending
    )
    runner._persist_active_agents = Mock()
    runner._resume_owner_authorized = Mock(return_value=True)
    runner._restart_loop_guard_config = lambda: (100, 60, 0)
    monkeypatch.setattr(
        "gateway.restart_loop_guard.check_and_record", lambda *a, **kw: False
    )
    monkeypatch.setattr(
        "gateway.run_reconnect_recovery.RECOVERY_RETRY_DELAYS", (0, 0, 0)
    )
    resumed = []

    async def dispatch(live, event, key):
        resumed.append((live, key))
        store.clear_resume_pending(key)

    runner._run_startup_resume_event = dispatch
    return runner, adapter, connector, resumed


def add_session(runner, chat, *, owner="default", logical=Platform.SLACK, runtime=None):
    key = f"agent:{runtime or 'main'}:{logical.value}:channel:{chat}"
    source = SessionSource(
        platform=logical, chat_id=chat, user_id="u1", profile=runtime
    )
    entry = SessionEntry(
        session_key=key,
        session_id=chat,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=logical,
        chat_type="channel",
        transport_profile=owner,
        resume_pending=True,
        resume_reason="restart_interrupted",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries[key] = entry
    return entry


async def settle(runner):
    # Tasks may retain a continuation or a resume dispatch while completing.
    while tasks := [t for t in runner._background_tasks if not t.done()]:
        await asyncio.gather(*tasks)


@pytest.mark.asyncio
@pytest.mark.parametrize("relay", [False, True])
@pytest.mark.parametrize("profile", ["default", "secondary"])
@pytest.mark.parametrize("failure", ["clear", "sweep", "refund"])
async def test_transient_fault_eventually_delivers_then_resumes_without_restart(
    tmp_path, monkeypatch, relay, profile, failure
):
    runner, adapter, connector, resumed = continuation_runner(
        tmp_path, monkeypatch, relay=relay, profile=profile
    )
    answered = add_session(runner, "chat", owner=profile)
    unanswered = add_session(runner, "unanswered", owner=profile)
    record(profile=profile)
    orphan()
    clear = runner.session_store.clear_resume_pending
    if failure in ("clear", "refund"):

        async def clear_once(key):
            runner._async_session_store.clear_resume_pending.side_effect = clear
            raise RuntimeError("session store busy")

        runner._async_session_store.clear_resume_pending.side_effect = clear_once
    if failure == "sweep":
        sweep = ledger.sweep_failed_for_runtime
        count = 0

        def sweep_once(*a, **kw):
            nonlocal count
            count += 1
            if count == 1:
                raise RuntimeError("ledger busy")
            return sweep(*a, **kw)

        monkeypatch.setattr(ledger, "sweep_failed_for_runtime", sweep_once)
    if failure == "refund":
        release = ledger.release_runtime_claim
        count = 0

        def release_once(*a, **kw):
            nonlocal count
            count += 1
            if count == 1:
                raise RuntimeError("ledger write busy")
            return release(*a, **kw)

        monkeypatch.setattr(ledger, "release_runtime_claim", release_once)
    platform = Platform.RELAY if relay else Platform.SLACK
    await recover_reconnected_sessions(runner, platform, profile)
    assert not resumed
    await settle(runner)
    with ledger._transaction() as db:
        assert db.execute(
            "SELECT state, attempts FROM delivery_obligations"
        ).fetchone() == ("delivered", 1)
    assert not answered.resume_pending
    assert resumed == [(adapter, unanswered.session_key)]
    assert not runner._delivery_settlements
    assert not runner._reconnect_recovery_tasks
    assert len(connector.sent) == 1 if relay else adapter.send.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["clear", "sweep"])
async def test_inherited_due_flood_refund_keeps_its_reason_and_spacing(
    tmp_path, monkeypatch, failure
):
    runner, adapter, _, _ = continuation_runner(tmp_path, monkeypatch)
    record()
    ledger.mark_failed("owed", "flood_control:60")
    with ledger._transaction() as db:
        db.execute("UPDATE delivery_obligations SET updated_at=updated_at-120")
    orphan()
    if failure == "clear":
        runner._async_session_store.clear_resume_pending.side_effect = RuntimeError(
            "busy"
        )
    else:
        monkeypatch.setattr(
            ledger, "sweep_failed_for_runtime", Mock(side_effect=RuntimeError("busy"))
        )
    assert (
        await runner._redeliver_failed_obligations_for_platform(Platform.SLACK) is None
    )
    adapter.send.assert_not_awaited()
    with ledger._transaction() as db:
        state, attempts, error, updated = db.execute(
            "SELECT state, attempts, last_error, updated_at FROM delivery_obligations"
        ).fetchone()
    assert (state, attempts, error) == ("failed", 0, "flood_control:60")
    assert ledger.retry_not_before(updated, error, attempts) == updated + 60
    assert ledger.pending_retries()[0]["not_before"] > ledger.time.time()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["pending", "attempting", "failed", "flood"])
async def test_auto_resume_never_regenerates_a_ledger_answer_during_other_recovery(
    tmp_path, monkeypatch, phase
):
    runner, _, _, resumed = continuation_runner(tmp_path, monkeypatch)
    answered = add_session(runner, "chat")
    record()
    if phase == "attempting":
        ledger.mark_attempting("owed")
    elif phase in ("failed", "flood"):
        ledger.mark_failed(
            "owed", "flood_control:60" if phase == "flood" else "send_path_degraded"
        )
    assert runner._schedule_resume_pending_sessions(platform=Platform.SLACK) == 0
    await settle(runner)
    assert not resumed
    assert answered.resume_pending


@pytest.mark.asyncio
async def test_overlapping_reconnect_waits_for_claimed_reply_before_resume(
    tmp_path, monkeypatch
):
    runner, adapter, _, resumed = continuation_runner(tmp_path, monkeypatch)
    answered = add_session(runner, "chat")
    record()
    orphan()
    entered, release = asyncio.Event(), asyncio.Event()

    async def hold_clear(key):
        entered.set()
        await release.wait()
        return runner.session_store.clear_resume_pending(key)

    runner._async_session_store.clear_resume_pending.side_effect = hold_clear
    first = asyncio.create_task(recover_reconnected_sessions(runner, Platform.SLACK))
    await entered.wait()
    await recover_reconnected_sessions(runner, Platform.SLACK)
    assert not resumed
    assert answered.resume_pending
    release.set()
    await first
    await settle(runner)
    adapter.send.assert_awaited_once()
    assert not resumed


@pytest.mark.asyncio
async def test_relay_resume_uses_logical_platform_and_exact_receiving_bot(
    tmp_path, monkeypatch
):
    runner, adapter, _, resumed = continuation_runner(tmp_path, monkeypatch, relay=True)
    other = SimpleNamespace(send=AsyncMock())
    runner._profile_adapters = {"other": {Platform.SLACK: other}}
    runner._profile_configs = {"other": runner.config}
    expected = add_session(runner, "primary")
    foreign = add_session(runner, "other", owner="other", runtime="other")
    schedule_recovery(runner, Platform.RELAY)
    schedule_recovery(runner, Platform.RELAY)
    await settle(runner)
    assert resumed == [(adapter, expected.session_key)]
    assert foreign.resume_pending


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stop", ["_running", "_draining", "_restart_requested", "cancel"]
)
async def test_retry_is_retained_bounded_and_stops_before_shutdown_resume(
    tmp_path, monkeypatch, stop, caplog
):
    runner, _, _, resumed = continuation_runner(tmp_path, monkeypatch)
    add_session(runner, "unanswered")
    entered, release = asyncio.Event(), asyncio.Event()
    count = 0

    async def fail(*a, **kw):
        nonlocal count
        count += 1
        if count == 2:
            entered.set()
            await release.wait()

    runner._redeliver_failed_obligations_for_platform = fail
    schedule_recovery(runner, Platform.SLACK)
    await entered.wait()
    task = runner._reconnect_recovery_tasks[(Platform.SLACK, "default")][0]
    assert task in runner._background_tasks
    if stop == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        setattr(runner, stop, stop != "_running")
        release.set()
        await task
    assert not runner._reconnect_recovery_tasks
    assert not resumed
    assert count == 2


@pytest.mark.asyncio
async def test_persistent_fault_warns_and_retains_reply_without_spending_budget(
    tmp_path, monkeypatch, caplog
):
    runner, adapter, _, resumed = continuation_runner(tmp_path, monkeypatch)
    record()
    orphan()
    runner._async_session_store.clear_resume_pending.side_effect = RuntimeError(
        "persistent store fault"
    )
    schedule_recovery(runner, Platform.SLACK)
    await settle(runner)
    with ledger._transaction() as db:
        assert db.execute(
            "SELECT state, attempts FROM delivery_obligations"
        ).fetchone() == ("failed", 0)
    assert "remains incomplete after 3 retries" in caplog.text
    assert not runner._reconnect_recovery_tasks
    assert not resumed
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_boot_clear_failure_is_protected_and_retried_without_another_reconnect(
    tmp_path, monkeypatch
):
    runner, adapter, _, resumed = continuation_runner(tmp_path, monkeypatch)
    answered = add_session(runner, "chat")
    record()
    orphan()

    async def clear_once(key):
        runner._async_session_store.clear_resume_pending.side_effect = (
            runner.session_store.clear_resume_pending
        )
        raise RuntimeError("busy")

    runner._async_session_store.clear_resume_pending.side_effect = clear_once
    assert await runner._claim_pending_obligations() == []
    assert runner._schedule_resume_pending_sessions() == 0
    await settle(runner)
    adapter.send.assert_awaited_once()
    assert not answered.resume_pending
    assert not resumed


def test_failed_resume_clear_is_failure_atomic_and_survives_a_real_store_reload(
    tmp_path, monkeypatch
):
    runner, _, _, _ = continuation_runner(tmp_path, monkeypatch)
    entry = add_session(runner, "chat")
    store = runner.session_store
    monkeypatch.setattr(store, "_routing_db_method", lambda name: None)
    store._save()
    save = store._save_sessions_json
    monkeypatch.setattr(
        store, "_save_sessions_json", Mock(side_effect=OSError("disk unavailable"))
    )
    with pytest.raises(OSError, match="disk unavailable"):
        store.clear_resume_pending(entry.session_key)
    assert entry.resume_pending
    assert entry.resume_reason == "restart_interrupted"
    monkeypatch.setattr(store, "_save_sessions_json", save)
    assert store.clear_resume_pending(entry.session_key)
    loaded = SessionStore(sessions_dir=store.sessions_dir, config=runner.config)
    monkeypatch.setattr(loaded, "_routing_db_method", lambda name: None)
    with loaded._lock:
        loaded._ensure_loaded_locked()
    assert not loaded._entries[entry.session_key].resume_pending


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["delivered", "failed"])
async def test_failed_settlement_retries_only_the_write_before_any_replay(
    tmp_path, monkeypatch, outcome
):
    runner, adapter, _, resumed = continuation_runner(tmp_path, monkeypatch)
    record()
    orphan()
    if outcome == "failed":
        adapter.send.return_value = SimpleNamespace(
            success=False, error="temporary API error"
        )
    name = "mark_delivered" if outcome == "delivered" else "mark_failed"
    persist = getattr(ledger, name)
    calls = 0

    def fail_once(*a, **kw):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("ledger write unavailable")
        return persist(*a, **kw)

    monkeypatch.setattr(ledger, name, fail_once)
    await recover_reconnected_sessions(runner, Platform.SLACK)
    await settle(runner)
    adapter.send.assert_awaited_once()
    assert calls == 2
    with ledger._transaction() as db:
        assert db.execute(
            "SELECT state, attempts FROM delivery_obligations"
        ).fetchone() == (outcome, 1)
    assert not runner._delivery_settlements
    assert not resumed


@pytest.mark.asyncio
async def test_resume_ledger_read_failure_defers_and_retries_enumeration(
    tmp_path, monkeypatch
):
    runner, adapter, _, resumed = continuation_runner(tmp_path, monkeypatch)
    entry = add_session(runner, "unanswered")
    read = ledger.pending_resume_sessions
    calls = 0

    def read_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("ledger read unavailable")
        return read()

    monkeypatch.setattr(ledger, "pending_resume_sessions", read_once)
    await recover_reconnected_sessions(runner, Platform.SLACK)
    assert not resumed
    await settle(runner)
    assert resumed == [(adapter, entry.session_key)]


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["delivered", "failed"])
async def test_original_final_delivery_also_retries_a_failed_settlement(
    tmp_path, monkeypatch, outcome
):
    runner, adapter, _, _ = continuation_runner(tmp_path, monkeypatch)
    record()
    ledger.mark_attempting("owed")
    persist = ledger.mark_delivered if outcome == "delivered" else ledger.mark_failed
    calls = 0

    def fail_once(*a, **kw):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("ledger unavailable")
        return persist(*a, **kw)

    monkeypatch.setattr(
        ledger, "mark_delivered" if outcome == "delivered" else "mark_failed", fail_once
    )
    runner._schedule_flood_redelivery = Mock()
    event = MessageEvent(
        text="question", source=SessionSource(platform=Platform.SLACK, chat_id="chat")
    )
    result = SendResult(success=outcome == "delivered", error="temporary API failure")
    await BasePlatformAdapter._finalize_delivery_obligation(
        adapter, "owed", result, event, adapter
    )
    await settle(runner)
    with ledger._transaction() as db:
        assert db.execute(
            "SELECT state, attempts FROM delivery_obligations"
        ).fetchone() == (outcome, 0)
    assert calls == 2
    assert not runner._delivery_settlements
    adapter.send.assert_not_awaited()
    runner._arm_flood_timers_for_waiting_rows.assert_awaited()


@pytest.mark.asyncio
async def test_installed_healthy_adapter_finishes_a_transient_clear_failure(
    tmp_path, monkeypatch
):
    runner, adapter, _, _ = continuation_runner(tmp_path, monkeypatch)
    add_session(runner, "chat")
    record()
    orphan()
    runner.adapters = {}
    runner._failed_platforms = {Platform.SLACK: {"predecessor": None}}
    runner.delivery_router = SimpleNamespace(adapters={})
    adapter.send_path_degraded = False
    monkeypatch.setattr(
        runner,
        "_publish_primary_adapter",
        lambda platform, live: runner.adapters.update({platform: live}),
    )
    monkeypatch.setattr(runner, "_update_platform_runtime_status", Mock())
    monkeypatch.setattr(runner, "_schedule_planned_restart_replay", Mock())
    monkeypatch.setattr(
        "gateway.channel_directory.build_channel_directory", AsyncMock()
    )

    async def fail_once(key):
        runner._async_session_store.clear_resume_pending.side_effect = (
            runner.session_store.clear_resume_pending
        )
        raise RuntimeError("store busy")

    runner._async_session_store.clear_resume_pending.side_effect = fail_once
    await runner._install_reconnected_adapter(Platform.SLACK, adapter)
    await settle(runner)
    assert runner.adapters[Platform.SLACK] is adapter
    adapter.send.assert_awaited_once()


def test_resume_probe_does_not_create_state_and_fails_closed_on_a_busy_ledger(
    tmp_path, monkeypatch
):
    continuation_runner(tmp_path, monkeypatch)
    assert ledger.pending_resume_sessions() == set()
    assert not ledger._db_path().exists()
    record()
    ledger._DB_LOCK.acquire()
    try:
        with pytest.raises(RuntimeError, match="busy"):
            ledger.pending_resume_sessions()
    finally:
        ledger._DB_LOCK.release()


def test_recovery_passes_record_one_boot_instead_of_tripping_the_restart_breaker(
    tmp_path, monkeypatch
):
    from gateway import restart_loop_guard as guard

    original = guard.check_and_record
    runner, _, _, _ = continuation_runner(tmp_path, monkeypatch)
    monkeypatch.setattr(guard, "check_and_record", original)
    runner._restart_loop_guard_config = lambda: (3, 60, 300)
    add_session(runner, "unanswered")
    guard.clear()
    for _ in range(5):
        assert len(runner._resume_pending_candidates(Platform.SLACK)) == 1
    assert len(guard._load_boots()) == 1


@pytest.mark.asyncio
async def test_timed_worker_recovers_from_a_transient_ledger_read_fault(
    tmp_path, monkeypatch
):
    from gateway.run_startup import GatewayStartupMixin

    runner, _, _, _ = continuation_runner(tmp_path, monkeypatch)
    record()
    ledger.mark_failed("owed", "flood_control:60")
    runner._arm_flood_timers_for_waiting_rows = (
        GatewayStartupMixin._arm_flood_timers_for_waiting_rows.__get__(runner)
    )
    read = ledger.pending_retries
    recovered = asyncio.Event()
    calls = 0

    def fail_once(*a, **kw):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient ledger read failure")
        return read(*a, **kw)

    monkeypatch.setattr(ledger, "pending_retries", fail_once)
    arm = runner._arm_flood_timers_for_waiting_rows

    async def observe_arm():
        await arm()
        recovered.set()

    runner._arm_flood_timers_for_waiting_rows = observe_arm
    runner._schedule_flood_redelivery(Platform.SLACK)
    try:
        await asyncio.wait_for(recovered.wait(), timeout=10)
        assert ("slack", "default") in runner._flood_redelivery_tasks
        with ledger._transaction() as db:
            assert db.execute(
                "SELECT state, attempts FROM delivery_obligations"
            ).fetchone() == ("failed", 0)
    finally:
        runner._running = False
        tasks = list(runner._background_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
