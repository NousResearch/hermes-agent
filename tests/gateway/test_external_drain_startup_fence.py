"""A pre-existing drain fences startup until explicit release, not watcher timing."""
import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

import gateway.drain_control as dc
from gateway.config import GatewayConfig
from gateway.run import GatewayRunner
from hermes_state_runtime import get_session_admission
from tests.gateway.test_external_drain_authority_fence import _admit, native_lane as native_lane


@pytest.mark.asyncio
async def test_marker_fences_authority_before_startup(native_lane, monkeypatch):
    runner, authority, _adapter, event = native_lane
    dc.write_drain_request(principal='fixture')
    replacement = GatewayRunner(GatewayConfig(sessions_dir=runner.config.sessions_dir))
    try:
        from gateway.session_authority import SessionAuthority
        from hermes_state_runtime import RuntimeStoreError
        replacement_authority = SessionAuthority(replacement, profile_id=authority.profile_id,
            instance_id=authority.instance_id, db=authority.db, epoch=authority.epoch)
        replacement.session_authority = replacement_authority
        replacement.adapters = runner.adapters
        replacement._wire_adapter_handlers(_adapter)
        monkeypatch.setattr(replacement_authority, '_schedule', lambda ref: None)
        with pytest.raises(RuntimeStoreError, match='runtime_draining'):
            await _admit(replacement, replacement_authority, event)
    finally:
        db = getattr(replacement._session_db, '_db', replacement._session_db)
        if db is not None and db is not authority.db:
            db.close()


def _wire_inert_boot(runner, monkeypatch):
    # Keep _start_impl and _start_finish_wiring, the restore gate, legacy candidate
    # scheduling and native recovery real. Replace only service/transport startup.
    for name in ('_start_install_faulthandler', '_start_startup_warmup', '_wire_teams_pipeline_runtime',
                 '_install_plugin_message_injector', '_start_spawn_background_watchers'):
        monkeypatch.setattr(runner, name, lambda *a, **kw: None)
    for name in ('_start_log_startup_environment', '_start_recover_previous_run', '_start_post_connect_services',
                 '_send_session_db_warning_notifications', '_await_startup_boot_sends'):
        monkeypatch.setattr(runner, name, AsyncMock())
    monkeypatch.setattr(runner, '_start_check_access_policy', lambda: False)
    monkeypatch.setattr(runner, '_start_prefilter_platforms', AsyncMock(return_value=(False, 0, [], [])))
    monkeypatch.setattr(runner, '_start_connect_pending', AsyncMock(return_value=[]))
    monkeypatch.setattr(runner, '_start_aggregate_connect_results', AsyncMock(return_value=0))
    monkeypatch.setattr(runner, '_start_secondary_profiles', AsyncMock(return_value=(False, 0)))
    monkeypatch.setattr('gateway.shutdown_flush.recover_gateway_pending', lambda runner: None)
    monkeypatch.setattr('gateway.cron_store_notices.install_cron_store_notices', lambda *a: None)
    monkeypatch.setattr('gateway.session_hosted_service.ensure_hosted_service', AsyncMock())
    monkeypatch.setattr('hermes_cli.observability.shared_metrics_startup.record_process_ready', lambda *a, **kw: None)


@pytest.mark.asyncio
async def test_startup_holds_actual_restore_and_preserves_native_fifo_until_marker_release(native_lane, monkeypatch):
    runner, authority, adapter, event = native_lane
    receipt = await _admit(runner, authority, event)
    before = get_session_admission(authority.db, admission_id=receipt.admission_id)
    legacy_source = replace(event.source, chat_id='C_LEGACY')
    legacy = runner.session_store.get_or_create_session(legacy_source)
    assert runner.session_store.mark_resume_pending(legacy.session_key)
    legacy_started = asyncio.Event()

    async def inert_legacy_execution(adapter, restored_event, session_key):
        assert session_key == legacy.session_key and restored_event.internal
        legacy_started.set()
        runner._release_running_agent_state(session_key)

    monkeypatch.setattr(runner, '_run_startup_resume_event', inert_legacy_execution)
    authority.sessions.clear()  # real persisted routing entry, not process-local warm state
    dc.write_drain_request(principal='fixture')
    _wire_inert_boot(runner, monkeypatch)
    checked = asyncio.Event()
    loop = asyncio.get_running_loop()
    real_read = dc.drain_requested

    def observed_read():
        result = real_read()
        loop.call_soon_threadsafe(checked.set)
        return result

    monkeypatch.setattr(dc, 'drain_requested', observed_read)
    restore_started = asyncio.Event()
    real_restore = runner._start_finish_wiring

    async def observed_restore(count):
        restore_started.set()
        await real_restore(count)

    monkeypatch.setattr(runner, '_start_finish_wiring', observed_restore)
    task = asyncio.create_task(runner._start_impl())
    try:
        # On the predecessor startup reaches actual restore immediately, without
        # reading the marker. The race ends at whichever real boundary comes first.
        check_task = asyncio.create_task(checked.wait())
        restore_task = asyncio.create_task(restore_started.wait())
        try:
            await asyncio.wait_for(asyncio.wait([check_task, restore_task], return_when=asyncio.FIRST_COMPLETED), 5)
        finally:
            for waiter in (check_task, restore_task):
                waiter.cancel()
            await asyncio.gather(check_task, restore_task, return_exceptions=True)
        assert not restore_started.is_set(), 'startup restoration ran before the existing drain was released'
        assert not task.done()
        assert get_session_admission(authority.db, admission_id=receipt.admission_id) == before
        dc.clear_drain_request()
        assert await asyncio.wait_for(task, 5) is True
        assert restore_started.is_set()
        assert not runner._external_drain_active
        assert receipt.ref.session_id in authority.sessions  # actual native route recovered after release
        assert legacy_started.is_set(), 'legacy resume must occur after the marker is cancelled'
        assert get_session_admission(authority.db, admission_id=receipt.admission_id) == before
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('marker', ['absent', 'expired', 'prior-epoch', 'malformed'])
async def test_startup_marker_semantics_use_existing_drain_contract(native_lane, monkeypatch, marker):
    from datetime import UTC, datetime, timedelta
    import json
    runner, _authority, _adapter, _event = native_lane
    monkeypatch.setattr(dc, 'current_instantiation_epoch', lambda: 'fixture-current')
    if marker == 'malformed':
        dc.drain_request_path().write_text('not-json')
    elif marker != 'absent':
        payload = dc.write_drain_request(principal='fixture')
        if marker == 'expired':
            payload['requested_at'] = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        else:
            payload['epoch'] = 'fixture-prior'
        dc.drain_request_path().write_text(json.dumps(payload))
    replacement = GatewayRunner(GatewayConfig(sessions_dir=runner.config.sessions_dir))
    try:
        assert replacement._external_drain_active is (marker == 'malformed')
        if marker != 'malformed':
            assert await replacement._start_wait_external_drain() is False
            assert not replacement._external_drain_active
    finally:
        db = getattr(replacement._session_db, '_db', replacement._session_db)
        if db is not None:
            db.close()


@pytest.mark.asyncio
async def test_marker_appearing_after_construction_fences_startup(native_lane, monkeypatch):
    runner, _authority, _adapter, _event = native_lane
    assert runner._external_drain_active is False
    dc.write_drain_request(principal='fixture')
    read = asyncio.Event()
    real_update = runner._update_runtime_status

    def update(state, **kwargs):
        real_update(state, **kwargs)
        if state == 'draining':
            read.set()

    monkeypatch.setattr(runner, '_update_runtime_status', update)
    task = asyncio.create_task(runner._start_wait_external_drain())
    try:
        await asyncio.wait_for(read.wait(), 5)
        assert runner._external_drain_active and not task.done()
        dc.clear_drain_request()
        assert await asyncio.wait_for(task, 5) is False
        assert runner._external_drain_active is False
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_shutdown_during_startup_drain_does_not_resurrect_runner(native_lane, monkeypatch):
    runner, _authority, _adapter, _event = native_lane
    dc.write_drain_request(principal='fixture')
    runner._draining = True
    runner._running = False
    runner._shutdown_event.set()
    restore = AsyncMock()
    monkeypatch.setattr(runner, '_start_recover_previous_run', restore)
    assert await runner._start_impl() is True
    assert runner._external_drain_active is True
    assert runner._draining is True
    restore.assert_not_awaited()


@pytest.mark.asyncio
async def test_marker_created_during_connects_fences_before_actual_startup_restore(native_lane, monkeypatch):
    runner, authority, _adapter, event = native_lane
    receipt = await _admit(runner, authority, event)
    before = get_session_admission(authority.db, admission_id=receipt.admission_id)
    _wire_inert_boot(runner, monkeypatch)
    loop = asyncio.get_running_loop()
    marker_seen, restore_started = asyncio.Event(), asyncio.Event()
    real_read = dc.drain_requested
    real_restore = runner._start_finish_wiring

    def observe_read():
        active = real_read()
        if active:
            loop.call_soon_threadsafe(marker_seen.set)
        return active

    async def connects(pending):
        dc.write_drain_request(principal='fixture')
        return []

    async def restore(count):
        restore_started.set()
        await real_restore(count)

    monkeypatch.setattr(dc, 'drain_requested', observe_read)
    monkeypatch.setattr(runner, '_start_connect_pending', connects)
    monkeypatch.setattr(runner, '_start_finish_wiring', restore)
    task = asyncio.create_task(runner._start_impl())
    waiters = [asyncio.create_task(e.wait()) for e in (marker_seen, restore_started)]
    try:
        await asyncio.wait_for(asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED), 5)
        assert not restore_started.is_set(), 'a drain arriving during connect was ignored until after restore'
        assert get_session_admission(authority.db, admission_id=receipt.admission_id) == before
        dc.clear_drain_request()
        assert await asyncio.wait_for(task, 5)
        assert restore_started.is_set()
    finally:
        for waiting in [task, *waiters]:
            if not waiting.done():
                waiting.cancel()
        await asyncio.gather(task, *waiters, return_exceptions=True)
