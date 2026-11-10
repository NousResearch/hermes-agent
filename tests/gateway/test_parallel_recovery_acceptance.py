"""Independent continuation review; isolated fixtures, no real providers/services."""
import asyncio
import sqlite3

import pytest

from gateway.session_parallel import route_input, conversation_scope, physical_count
from hermes_state_parallel import active_work
from hermes_state_runtime import claim_session_input
from tests.gateway.test_parallel_native_ingress import setup_owner, event, DiscordFixture
from tests.waiting import async_wait_until
from gateway.session_ingress_context import native_callback
from hermes_constants import get_hermes_home


async def admit_bound(runner, authority, incoming):
    with native_callback(runner, incoming, get_hermes_home()):
        return await authority.admit_native(incoming)


def all_rows(authority):
    with authority.db._read_ctx() as conn:
        return [dict(r) for r in conn.execute('SELECT request_id,status,target_session_id FROM session_admissions ORDER BY seq')]


@pytest.mark.asyncio
async def test_reboot_then_real_scoped_discard_resumes_only_selected_fifo(monkeypatch):
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from hermes_platform.host import facts
    schedule = SessionAuthority._schedule
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda *a: None)
    monkeypatch.setattr(facts, 'kernel_boot_identity', lambda: ('machine', 'new-boot'))
    executed = []

    async def produce(authority, ref, row):
        executed.append(row['request_id'])
        return 'FOLLOWER DONE'

    runner, authority, adapter = await setup_owner(monkeypatch, produce)
    try:
        for mid in ('lost-a', 'lost-b'):
            incoming = event(adapter, mid, mid)
            assert not adapter._drop_unresolved(incoming)
            assert await route_input(runner, incoming) is None
            receipt = await admit_bound(runner, authority, incoming)
            claim_session_input(authority.db, epoch=authority.epoch, session_id=receipt.ref.session_id)
            authority.db._execute_write(lambda conn: conn.execute(
                'INSERT INTO parallel_work_physical(token,work_id,machine_id,boot_id) VALUES(?,?,?,?)',
                (mid, incoming.source.parallel_work.work_id, 'machine', 'old-boot')))
        follow = event(adapter, 'follow-a', 'follow', reply='lost-a')
        assert not adapter._drop_unresolved(follow)
        assert await route_input(runner, follow) is None
        await admit_bound(runner, authority, follow)
        authority.db.close()
        cold_runner = GatewayRunner(GatewayConfig())
        cold = await initialize_session_authority(cold_runner, profile_id='default', instance_id='cold')
        cold_adapter = DiscordFixture()
        cold_adapter.gateway_runner = cold_runner
        cold_runner.adapters = {Platform.DISCORD: cold_adapter}
        cold_runner._wire_adapter_handlers(cold_adapter)
        monkeypatch.setattr(cold_runner, '_effective_busy_input_mode', lambda source: 'parallel')
        monkeypatch.setattr(SessionAuthority, '_schedule', schedule)
        try:
            print('AFTER BOOT', physical_count(cold), all_rows(cold), flush=True)
            assert physical_count(cold) == 0
            assert [r['status'] for r in all_rows(cold)] == ['unknown', 'unknown', 'queued']
            for mid, reply, user in [('bare', None, 'alice'), ('foreign', 'lost-a', 'bob'),
                                     ('missing', 'unrecorded', 'alice')]:
                await cold_adapter.handle_message(event(cold_adapter, mid, '/discard', reply=reply, user=user))
            assert [r['status'] for r in all_rows(cold)] == ['unknown', 'unknown', 'queued']
            await cold_adapter.handle_message(event(cold_adapter, 'discard-a', '/discard', reply='lost-a'))
            await async_wait_until(lambda: not cold_adapter._background_tasks, what='discard finishes')
            print('DISCARD RESPONSE', all_rows(cold), executed, cold_adapter.sent, flush=True)
            await async_wait_until(lambda: executed == ['follow-a'] and all_rows(cold)[2]['status'] == 'terminal', what='only selected FIFO completes')
            print('AFTER DISCARD', physical_count(cold), all_rows(cold), executed, cold_adapter.sent, flush=True)
            assert [r['status'] for r in all_rows(cold)] == ['terminal', 'unknown', 'terminal']
            assert executed == ['follow-a']
            await cold_adapter.handle_message(event(cold_adapter, 'fresh-a', 'fresh followup', reply='lost-a'))
            await async_wait_until(lambda: not cold_adapter._background_tasks, what='post-recovery ordinary reply finishes')
            assert executed == ['follow-a', 'fresh-a']
            assert any('FOLLOWER DONE' in message for message in cold_adapter.sent)
            assert [r['status'] for r in all_rows(cold)] == ['terminal', 'unknown', 'terminal', 'terminal']
        finally:
            await async_wait_until(lambda: not cold_adapter._background_tasks, what='cold cleanup')
            cold.db.close()
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_same_boot_discard_never_releases_physical_capacity(monkeypatch):
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from hermes_platform.host import facts
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda *a: None)
    monkeypatch.setattr(facts, 'kernel_boot_identity', lambda: ('machine', 'same-boot'))
    async def produce(*a):
        raise AssertionError('must not execute')
    runner, authority, adapter = await setup_owner(monkeypatch, produce)
    try:
        incoming = event(adapter, 'lost-live', 'work')
        assert not adapter._drop_unresolved(incoming)
        await route_input(runner, incoming)
        receipt = await admit_bound(runner, authority, incoming)
        claim_session_input(authority.db, epoch=authority.epoch, session_id=receipt.ref.session_id)
        authority.db._execute_write(lambda conn: conn.execute(
            'INSERT INTO parallel_work_physical(token,work_id,machine_id,boot_id) VALUES(?,?,?,?)',
            ('live', incoming.source.parallel_work.work_id, 'machine', 'same-boot')))
        cold = await initialize_session_authority(runner, profile_id='default', instance_id='restart', db=authority.db)
        await adapter.handle_message(event(adapter, 'discard', '/discard', reply='lost-live'))
        await async_wait_until(lambda: not adapter._background_tasks, what='discard complete')
        print('SAME BOOT DISCARD', physical_count(cold), all_rows(cold), flush=True)
        assert physical_count(cold) == 1
        assert all_rows(cold)[0]['status'] == 'terminal'
        assert len(active_work(cold.db, conversation_scope(incoming.source))) == 1
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('policy,accepted,chats', [
    ('', 2, ['room'] * 3),
    ('  parallel_max_runs: 8\n', 8, ['room-a'] * 5 + ['room-b'] * 4),
])
async def test_real_yaml_default_and_profile_ceiling_refuse_before_custody(monkeypatch, policy, accepted, chats):
    from gateway.run import GatewayRunner
    import gateway.run as gateway_run
    monkeypatch.setattr(gateway_run, '_hermes_home', get_hermes_home())
    (get_hermes_home() / 'config.yaml').write_text('display:\n  busy_input_mode: parallel\n' + policy)
    gate = asyncio.Event()
    entered = []
    async def produce(authority, ref, row):
        entered.append(row['request_id'])
        await gate.wait()
        return 'DONE'
    runner, authority, adapter = await setup_owner(monkeypatch, produce)
    # Do not rely on setup_owner's forced parallel mode: exercise the YAML-loaded mode.
    monkeypatch.setattr(runner, '_effective_busy_input_mode', GatewayRunner._effective_busy_input_mode.__get__(runner))
    assert runner._busy_input_mode == 'parallel'
    try:
        for index, chat in enumerate(chats):
            await adapter.handle_message(event(adapter, 'request-' + str(index), 'work', chat=chat))
        await async_wait_until(lambda: len(entered) == accepted and bool(adapter.sent), what='bounded admission and refusal')
        rows = all_rows(authority)
        assert len(rows) == accepted
        assert set(entered) == {'request-' + str(i) for i in range(accepted)}
        assert all(row['status'] == 'started' for row in rows)
    finally:
        gate.set()
        await async_wait_until(lambda: not adapter._background_tasks, what='capacity fixture cleanup')
        authority.db.close()


def test_schema_upgrade_from_original_physical_table_preserves_legacy_lease(tmp_path):
    from hermes_state import SessionDB
    from hermes_state_parallel import reconcile_rebooted_physical
    path = tmp_path / 'legacy.db'
    db = SessionDB(path)
    db.create_session('legacy-session', 'discord')
    db._execute_write(lambda conn: conn.execute(
        'INSERT INTO parallel_work(work_id,scope,route,requester,session_id) VALUES(?,?,?,?,?)',
        ('legacy-work', 'scope', 'legacy-route', 'alice', 'legacy-session')))
    db.close()
    with sqlite3.connect(path) as conn:
        conn.execute('DROP TABLE parallel_work_physical')
        conn.execute('CREATE TABLE parallel_work_physical(token TEXT PRIMARY KEY,work_id TEXT NOT NULL REFERENCES parallel_work(work_id) ON DELETE CASCADE)')
        conn.execute("INSERT INTO parallel_work_physical(token,work_id) VALUES('legacy','legacy-work')")
    upgraded = SessionDB(path)
    try:
        with upgraded._read_ctx() as conn:
            columns = [r[1] for r in conn.execute('PRAGMA table_info(parallel_work_physical)')]
            row = dict(conn.execute('SELECT * FROM parallel_work_physical').fetchone())
        print('UPGRADED', columns, row, flush=True)
        assert columns == ['token', 'work_id', 'machine_id', 'boot_id']
        assert row['machine_id'] == row['boot_id'] == ''
        assert reconcile_rebooted_physical(upgraded, 'machine', 'new-boot') == 0
    finally:
        upgraded.close()
