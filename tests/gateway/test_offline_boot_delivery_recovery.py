"""Offline-at-boot transports recover the old process's replies when they reconnect."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway import delivery_ledger as ledger
from gateway.config import Platform, PlatformConfig
from gateway.run_delivery_recovery import schedule_relay_recovery
from tests.gateway.test_relay_ledger_recovery import record, setup_runner


def recovery_runner(tmp_path, monkeypatch, *, relay, profile):
    runner, adapter, connector = setup_runner(tmp_path, monkeypatch)
    runner._background_tasks = set()
    monkeypatch.setattr(ledger, "_owner_alive", lambda pid, started: pid == 424242)
    if not relay:
        adapter = SimpleNamespace(
            send=AsyncMock(return_value=SimpleNamespace(success=True)),
        )
        runner.adapters = {Platform.SLACK: adapter}
    adapter.gateway_runner = runner
    adapter._owner_profile = profile
    if profile != "default":
        runner._profile_configs = {profile: runner.config}
        runner._profile_adapters = {profile: runner.adapters}
        runner.adapters = {}
    return runner, adapter, connector


def orphan():
    with ledger._transaction() as db:
        db.execute(
            "UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("relay", [False, True])
@pytest.mark.parametrize("profile", ["default", "secondary"])
@pytest.mark.parametrize("state", ["pending", "attempting", "failed"])
async def test_offline_boot_then_reconnect_delivers_once_before_resume(
    tmp_path, monkeypatch, relay, profile, state
):
    runner, adapter, connector = recovery_runner(
        tmp_path, monkeypatch, relay=relay, profile=profile
    )
    record(profile=profile)
    actions = {
        "attempting": ledger.mark_attempting,
        "failed": lambda oid: ledger.mark_failed(oid, "relay transport not connected"),
    }
    if state in actions:
        actions[state]("owed")
    orphan()
    live = (
        runner._profile_adapters[profile] if profile != "default" else runner.adapters
    )
    if profile != "default":
        runner._profile_adapters[profile] = {}
    else:
        runner.adapters = {}
    assert await runner._claim_pending_obligations() == []
    if profile != "default":
        runner._profile_adapters[profile] = live
    else:
        runner.adapters = live
    cleared = []

    async def clear(key):
        cleared.append(key)

    runner._async_session_store.clear_resume_pending.side_effect = clear
    original_send = connector.send_outbound if relay else adapter.send

    async def checked_send(*args, **kwargs):
        assert cleared == ["agent:main:slack:channel:chat"]
        return await original_send(*args, **kwargs)

    monkeypatch.setattr(
        connector if relay else adapter,
        "send_outbound" if relay else "send",
        checked_send,
    )
    platform = Platform.RELAY if relay else Platform.SLACK
    assert (
        await runner._redeliver_failed_obligations_for_platform(
            platform, profile=profile
        )
        == 1
    )
    assert (
        await runner._redeliver_failed_obligations_for_platform(
            platform, profile=profile
        )
        == 0
    )
    content = (
        connector.sent[0]["content"]
        if relay
        else original_send.await_args.kwargs["content"]
    )
    assert (
        content
        == ("" if state == "pending" else ledger.RECOVERED_MARKER) + "stored answer"
    )
    with ledger._transaction() as db:
        assert db.execute(
            "SELECT state, attempts, owner_pid FROM delivery_obligations"
        ).fetchone() == ("delivered", 1, 424242)


@pytest.mark.asyncio
async def test_same_relay_handshake_recovers_dead_owner_without_an_inbound_turn(
    tmp_path, monkeypatch
):
    runner, adapter, connector = recovery_runner(
        tmp_path, monkeypatch, relay=True, profile="default"
    )
    record()
    orphan()
    schedule_relay_recovery(adapter)
    await asyncio.gather(*list(runner._background_tasks))
    assert len(connector.sent) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("relay", [False, True])
@pytest.mark.parametrize(
    "case",
    [
        "live_owner",
        "unserved_profile",
        "unadvertised_platform",
        "disabled",
        "expired",
        "exhausted",
    ],
)
async def test_reconnect_preserves_unavailable_or_ineligible_replies(
    tmp_path, monkeypatch, relay, case
):
    runner, adapter, connector = recovery_runner(
        tmp_path, monkeypatch, relay=relay, profile="default"
    )
    record(
        profile="other" if case == "unserved_profile" else "default",
        logical="telegram" if case == "unadvertised_platform" else "slack",
    )
    orphan()
    actions = {
        "live_owner": lambda: monkeypatch.setattr(
            ledger, "_owner_alive", lambda *args: True
        ),
        "disabled": lambda: runner.config.platforms.update(
            {Platform.RELAY if relay else Platform.SLACK: PlatformConfig(enabled=False)}
        ),
    }
    if case in actions:
        actions[case]()
    updates = {
        "expired": ("created_at=?", (0,)),
        "exhausted": ("attempts=?", (ledger.MAX_ATTEMPTS,)),
    }
    if case in updates:
        assignment, values = updates[case]
        with ledger._transaction() as db:
            db.execute(f"UPDATE delivery_obligations SET {assignment}", values)
    assert (
        await runner._redeliver_failed_obligations_for_platform(
            Platform.RELAY if relay else Platform.SLACK
        )
        == 0
    )
    assert not connector.sent
    if not relay:
        adapter.send.assert_not_awaited()
    runner._async_session_store.clear_resume_pending.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconnect_adopts_flood_wait_without_sending_early(tmp_path, monkeypatch):
    runner, _, connector = recovery_runner(
        tmp_path, monkeypatch, relay=True, profile="default"
    )
    record()
    ledger.mark_failed("owed", "flood_control:60")
    orphan()
    assert await runner._redeliver_failed_obligations_for_platform(Platform.RELAY) == 0
    runner._async_session_store.clear_resume_pending.assert_awaited_once()
    assert not connector.sent
    assert ledger.pending_retries()[0]["not_before"] > ledger.time.time()
    with ledger._transaction() as db:
        assert db.execute(
            "SELECT state, attempts, owner_pid FROM delivery_obligations"
        ).fetchone() == ("failed", 0, 424242)


@pytest.mark.asyncio
async def test_overlapping_reconnects_only_send_an_inherited_reply_once(
    tmp_path, monkeypatch
):
    runner, adapter, _ = recovery_runner(
        tmp_path, monkeypatch, relay=False, profile="default"
    )
    record()
    orphan()
    results = await asyncio.gather(
        *(
            runner._redeliver_failed_obligations_for_platform(Platform.SLACK)
            for _ in range(2)
        )
    )
    assert sum(results) == 1
    adapter.send.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["clear_resume", "runtime_sweep"])
async def test_failed_recovery_retains_the_reply_and_refunds_unsent_attempts(
    tmp_path, monkeypatch, failure
):
    runner, adapter, _ = recovery_runner(
        tmp_path, monkeypatch, relay=False, profile="default"
    )
    record()
    orphan()
    actions = {
        "clear_resume": lambda: monkeypatch.setattr(
            runner._async_session_store.clear_resume_pending,
            "side_effect",
            RuntimeError("store unavailable"),
        ),
        "runtime_sweep": lambda: monkeypatch.setattr(
            ledger,
            "sweep_failed_for_runtime",
            Mock(side_effect=RuntimeError("ledger unavailable")),
        ),
    }
    actions[failure]()
    assert (
        await runner._redeliver_failed_obligations_for_platform(Platform.SLACK) is None
    )
    adapter.send.assert_not_awaited()
    with ledger._transaction() as db:
        assert db.execute(
            "SELECT state, attempts FROM delivery_obligations"
        ).fetchone() == ("failed", 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("clear_fails", [False, True])
async def test_primary_install_recovers_before_scheduling_unanswered_turns(
    tmp_path, monkeypatch, clear_fails
):
    runner, adapter, _ = recovery_runner(
        tmp_path, monkeypatch, relay=False, profile="default"
    )
    record()
    orphan()
    runner.adapters = {}
    assert await runner._claim_pending_obligations() == []
    runner._failed_platforms = {Platform.SLACK: {"predecessor": None}}
    if clear_fails:
        runner._async_session_store.clear_resume_pending.side_effect = RuntimeError(
            "store unavailable"
        )
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
    resumed = []

    def resume(*, platform, profile):
        assert profile == "default"
        runner._async_session_store.clear_resume_pending.assert_awaited_once()
        adapter.send.assert_awaited_once()
        resumed.append(platform)
        return 0

    resume_hook = Mock(side_effect=resume)
    monkeypatch.setattr(runner, "_schedule_resume_pending_sessions", resume_hook)
    await runner._install_reconnected_adapter(Platform.SLACK, adapter)
    if clear_fails:
        resume_hook.assert_not_called()
    else:
        assert resumed == [Platform.SLACK]


def test_missing_process_identity_warns_once_and_remains_fail_awake(
    tmp_path, monkeypatch, caplog
):
    recovery_runner(tmp_path, monkeypatch, relay=True, profile="default")
    record()
    ledger.mark_failed("owed", "relay transport not connected")
    monkeypatch.setattr(ledger, "_owner_identity_warning_logged", False)
    monkeypatch.setattr(ledger, "_owner_stamp", lambda: (424242, None))
    with caplog.at_level("WARNING", logger="gateway.delivery_ledger"):
        for _ in range(2):
            with pytest.raises(RuntimeError, match="ownership"):
                ledger.pending_reconnects()
    assert (
        len([r for r in caplog.records if "scale-to-zero will stay awake" in r.message])
        == 1
    )
    monkeypatch.setattr(ledger, "_owner_stamp", lambda: (424242, 1234.5))
    assert ledger.pending_reconnects() == {("slack", "default")}
