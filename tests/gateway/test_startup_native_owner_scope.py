"""Native startup reads leave the loop without losing either side's owning profile."""
import asyncio
from contextlib import asynccontextmanager
import threading

import pytest

from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner


@asynccontextmanager
async def _restarted(tmp_path, monkeypatch):
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_authorities import owner_scope
    from gateway.session_ingress_context import native_callback, register_transport_home
    from tests.gateway.test_command_bypass_active_session import _StubAdapter

    root, homes = _reserve_homes(tmp_path, monkeypatch, names=('beta',))
    monkeypatch.setattr('gateway.run._load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    for name, home in homes:
        (home / '.env').write_text(f'TELEGRAM_ALLOWED_USERS={name}-allowed,{name}-revoked\n')
    process_ownership.reserve([home for _, home in homes])
    runner = _runner(root, homes)
    adapters, saved = {}, {}
    try:
        await initialize_gateway_runtime(runner)
        for name, home in homes:
            adapter = _StubAdapter(PlatformConfig(enabled=True, token=f'{name}-fixture', typing_indicator=False),
                                   Platform.TELEGRAM)
            async def send(chat_id, text, **kwargs):
                return SendResult(success=True, message_id='fixture-reply')
            monkeypatch.setattr(adapter, 'send', send)
            adapter.gateway_runner = runner
            if name == 'default':
                runner.adapters[Platform.TELEGRAM] = adapter
                runner._wire_adapter_handlers(adapter)
                register_transport_home(runner, None, home)
            else:
                runner._profile_adapters = {name: {Platform.TELEGRAM: adapter}}
                runner._configure_profile_adapter(adapter, name, Platform.TELEGRAM)
            adapters[name] = adapter
            authority = runner.session_authorities.for_home(home)
            authority._schedule = lambda ref: None  # die after durable admission, before claiming
            saved[name] = {}
            with owner_scope(authority):
                for kind in ('allowed', 'revoked'):
                    source = adapter.build_source(chat_id=f'{name}-{kind}', chat_type='dm', user_id=f'{name}-{kind}')
                    event = MessageEvent(text=f'{name}-{kind}', source=source, message_id=f'{name}-{kind}')
                    with native_callback(runner, event, home, profile=None if name == 'default' else name):
                        receipt = await authority.admit_native(event)
                    saved[name][kind] = receipt.ref.session_id
                    if kind == 'allowed':
                        follower = MessageEvent(text=f'{name}-follower', source=source, message_id=f'{name}-follower')
                        with native_callback(runner, follower, home, profile=None if name == 'default' else name):
                            assert (await authority.admit_native(follower)).ref == receipt.ref
                parent = saved[name]['allowed']
                child = parent + '-compressed'
                authority.db.publish_compression_child(parent_session_id=parent, child_session_id=child,
                    source='telegram', messages=[{'role': 'user', 'content': 'summary'}],
                    require_compression_lease=False)
                route = authority.sessions[parent].route
                assert runner.session_store.advance_compression_session(route, parent, child)
                saved[name]['child'] = child
            (home / '.env').write_text(f'TELEGRAM_ALLOWED_USERS={name}-allowed\n')
        old_epochs = {name: runner.session_authorities.for_home(home).epoch for name, home in homes}
        # Fresh owners and epochs over real durable admissions and the routing store, no live sessions.
        await initialize_gateway_runtime(runner)
        for name, home in homes:
            authority = runner.session_authorities.for_home(home)
            assert authority.epoch > old_epochs[name]
            assert not authority.sessions
        yield runner, homes, adapters, saved
    finally:
        for authority in list(runner.session_authorities):
            tasks = [live.task for live in authority.sessions.values() if live.task is not None]
            if tasks:
                await asyncio.wait_for(asyncio.gather(*tasks), 10)
            authority.db.close()
        runner.session_store.close_all_db_handles()
        for _, home in homes:
            process_ownership.release(home)


@pytest.mark.asyncio
async def test_two_profile_native_recovery_keeps_scope_routes_and_one_drain(tmp_path, monkeypatch):
    from gateway.config import Platform
    from gateway.run_runtime import recover_profile_native_sessions
    from gateway.session_authorities import active_authority, owner_scope
    from gateway.session_authority import SessionAuthority
    from hermes_constants import get_hermes_home
    from hermes_state_runtime import list_session_admissions

    async with _restarted(tmp_path, monkeypatch) as (runner, homes, adapters, saved):
        executed, scheduled, boundaries = [], [], []
        entered, release = asyncio.Event(), asyncio.Event()
        loop_thread = threading.get_ident()
        original_schedule = SessionAuthority._schedule
        original_recover = SessionAuthority.recover_native_sessions
        from gateway import session_detached
        original_detached = session_detached.recover_detached_sessions
        reads = []

        def record_read(kind, bound, owner=None):
            def read(*args, **kwargs):
                reads.append((kind, owner or active_authority(runner).profile_id,
                              str(get_hermes_home()), threading.get_ident()))
                return bound(*args, **kwargs)
            return read

        monkeypatch.setattr(runner.session_store, 'list_sessions',
                            record_read('routes', runner.session_store.list_sessions))
        for authority in runner.session_authorities:
            monkeypatch.setattr(authority, 'logical_owner',
                                record_read('lineage', authority.logical_owner, authority.profile_id))
            monkeypatch.setattr(authority.db, '_read_all',
                                record_read('admissions', authority.db._read_all, authority.profile_id))

        def snapshot(authority, phase):
            boundaries.append((phase, authority.profile_id, str(get_hermes_home()), threading.get_ident()))

        async def recover(authority, bindings):
            snapshot(authority, 'native')
            return await original_recover(authority, bindings)

        def detached(authority, pending):
            snapshot(authority, 'detached')
            return original_detached(authority, pending)

        def schedule(authority, ref):
            snapshot(authority, 'schedule')
            original_schedule(authority, ref)
            scheduled.append(authority.sessions[ref.session_id].task)

        async def handle(event):
            authority = active_authority(runner)
            executed.append((event.text, authority.profile_id, str(get_hermes_home()), threading.get_ident()))
            if event.text.endswith('-allowed'):
                entered.set()
                await asyncio.wait_for(release.wait(), 10)
            return 'fixture completion'

        monkeypatch.setattr(SessionAuthority, 'recover_native_sessions', recover)
        monkeypatch.setattr(SessionAuthority, '_schedule', schedule)
        monkeypatch.setattr(session_detached, 'recover_detached_sessions', detached)
        monkeypatch.setattr(runner, '_handle_message', handle)  # no inference; real authorization/FIFO/settlement

        for name, home in homes:
            authority = runner.session_authorities.for_home(home)
            sid = saved[name]['allowed']
            entered.clear()
            release.clear()
            denied = saved[name]['revoked']
            before = list_session_admissions(authority.db, session_id=denied, pending_only=False)
            if name == 'beta':
                runner._profile_adapters[name].pop(Platform.TELEGRAM)
                absent = await recover_profile_native_sessions(runner, authority)
                assert absent[sid] == 'not_found'
                assert list_session_admissions(authority.db, session_id=sid)[0]['status'] == 'queued'
                assert not executed or all(row[0].startswith('default-') for row in executed)
                runner._profile_adapters[name][Platform.TELEGRAM] = adapters[name]
            # Deliberately enter the OTHER profile, not the target: the coordinator must bind it.
            other = runner.session_authorities.for_home(homes[1 if name == 'default' else 0][1])
            with owner_scope(other):
                outcome = await recover_profile_native_sessions(runner, authority)
                assert str(get_hermes_home()) == other.profile_id
            assert outcome[sid] == 'ready', outcome
            assert outcome[denied] == 'permission_denied', outcome
            task = authority.sessions[sid].task
            try:
                await asyncio.wait_for(entered.wait(), 10)
                # First admission is executing; its queued follower makes the scan select this
                # logical owner again. Recovery must reuse the live drain, not start another.
                repeated = await recover_profile_native_sessions(runner, authority)
                assert repeated[sid] == 'ready'
                assert authority.sessions[sid].task is task
            finally:
                release.set()
            await asyncio.wait_for(task, 10)
            await recover_profile_native_sessions(runner, authority)
            assert list_session_admissions(authority.db, session_id=denied, pending_only=False) == before
            rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
            assert [(r['status'], r['outcome']) for r in rows] == [('terminal', 'completed')] * 2
            assert authority.logical_owner(saved[name]['child']) == sid
            assert authority.physical_target(rows_to_ref(authority, sid)) == saved[name]['child']
        assert executed == [(f'{name}-{kind}', str(home), str(home), loop_thread)
                            for name, home in homes for kind in ('allowed', 'follower')]
        assert len(set(scheduled)) == 2, 'recovery duplicated a drain'
        assert len(scheduled) == 4, 'recovery did not reselect the live owners'
        assert {(phase, owner) for phase, owner, *_ in boundaries} == {
            (phase, str(home)) for phase in ('native', 'detached', 'schedule') for _, home in homes}
        assert all(owner == actual and thread == loop_thread for _, owner, actual, thread in boundaries), boundaries
        worker_reads = [row for row in reads if row[3] != loop_thread]
        assert all(owner == actual for _, owner, actual, _ in worker_reads), worker_reads
        assert {(kind, owner) for kind, owner, *_ in worker_reads} == {
            (kind, str(home)) for kind in ('admissions', 'routes', 'lineage') for _, home in homes}


def rows_to_ref(authority, sid):
    from gateway.session_contract import SessionRef
    return SessionRef(authority.profile_id, sid)


@pytest.mark.asyncio
async def test_native_scan_stall_leaves_loop_and_other_profile_responsive(tmp_path, monkeypatch):
    from gateway.run_runtime import recover_profile_native_sessions
    from hermes_constants import get_hermes_home

    async with _restarted(tmp_path, monkeypatch) as (runner, homes, _adapters, _saved):
        authority = runner.session_authorities.for_home(homes[1][1])
        authority._schedule = lambda ref: None
        loop, loop_thread = asyncio.get_running_loop(), threading.get_ident()
        entered, heartbeat = asyncio.Event(), asyncio.Event()
        release = threading.Event()
        observations = []
        original = authority.db._read_all

        def read(sql, *args, **kwargs):
            if 'SELECT DISTINCT target_session_id' in sql:
                observations.append((threading.get_ident(), str(get_hermes_home())))
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(5), 'store scan blocked the loop; heartbeat could not release it'
            return original(sql, *args, **kwargs)

        monkeypatch.setattr(authority.db, '_read_all', read)
        task = asyncio.create_task(recover_profile_native_sessions(runner, authority))
        try:
            await asyncio.wait_for(entered.wait(), 10)
            # A control callback must run while the store worker remains stalled, in launch scope.
            loop.call_soon(heartbeat.set)
            await asyncio.wait_for(heartbeat.wait(), 5)
            assert not task.done()
            assert get_hermes_home() == homes[0][1]
            assert observations == [(observations[0][0], str(homes[1][1]))]
            assert observations[0][0] != loop_thread
        finally:
            release.set()
            await asyncio.wait_for(task, 10)
