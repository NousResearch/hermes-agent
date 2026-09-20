"""Real inert FIFO/producer/native Stop lifecycle regressions; no model or runtime."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from gateway import hosted_room_driver as tasks
from gateway.hosted_room_artifacts import RoomArtifactOutbox, RoomArtifactError
from gateway.session_hosted_output_lifecycle import records, key_for
from gateway.session_group_retirement import require_room_retired
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_canonical_hosted_outputs import owner, execute_group_turn
from tests.gateway.test_canonical_output_lifecycle import connection, dispatch


async def stopped_producer(authority, service, runner, root, monkeypatch, *, name, mixed=False):
    from gateway.session_hosted_output import current_output_binding
    from gateway.session_results import execution_result
    from tools.hosted_room_artifact import share_group_file
    shared, release = asyncio.Event(), asyncio.Event()
    captured = []
    output = root / 'cache' / (name + '.txt')
    output.parent.mkdir(exist_ok=True)
    output.write_bytes(b'private output ' + name.encode())
    async def handle(event):
        captured.append(current_output_binding())
        assert json.loads(await asyncio.to_thread(share_group_file, str(output)))['ok']
        shared.set()
        await release.wait()
        execution_result.get().update(result=dict(interrupted=True, final_response='', messages=[]), usage={})
        return ''
    runner._handle_message = handle
    runner._cached_agent_for = lambda _: SimpleNamespace(interrupt=lambda: None)
    manifest = None
    if mixed:
        import base64
        document = service.attachments.put(room_id='room', upload_id=name, name=name+'.txt',
            kind='file', mime='text/plain', data=b'sensitive original input '+name.encode())
        image = service.attachments.put(room_id='room', upload_id=name+'-image', name='pixel.png',
            kind='image', mime='image/png', data=base64.b64decode(
                'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1cAAAAASUVORK5CYII='))
        manifest = [{k: item[k] for k in ('attachment_id', 'name', 'kind', 'mime', 'size')} for item in (document, image)]
    work = asyncio.create_task(execute_group_turn(authority, service, event_id=name, input_manifest=manifest,
                                                  defer_publication=True, thread_id=name))
    client = connection(authority, service)
    try:
        await asyncio.wait_for(shared.wait(), 8)
        answer = await dispatch(client, 'stop', room_id='room', cancel_id=name+'-stop')
        assert answer.get('result') == {'cancelled': 1}, answer
    finally:
        release.set()
        result = await asyncio.wait_for(work, 8)
        service.runtime._thread = None
    task = tasks.get_task(service.db_path, result[3]['identity'])
    service._reconcile_stopped_output(task)
    return task, result[4], captured[0]


def initialize_inputs(authority, root, request):
    from gateway.runtime_ownership import process_ownership
    from gateway.hosted_room_input_reclamation import initialize_working_copies
    process_ownership.reserve([root])
    request.addfinalizer(lambda: process_ownership.release(root))
    initialize_working_copies(authority.db, epoch=authority.epoch)


@pytest.mark.asyncio
@pytest.mark.parametrize('mixed', [False, True], ids=['text', 'mixed'])
async def test_completed_replays_after_real_admission_retirement(tmp_path, monkeypatch, request, mixed):
    from hermes_state_mutation_retirement import retire_prunable
    from gateway.hosted_room_input_custody import custody_holds
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        monkeypatch.setattr('model_tools._resolve_active_context_length', lambda: 32768)
        if mixed:
            initialize_inputs(authority, tmp_path, request)
        task, binding, producer = await stopped_producer(authority, service, runner, tmp_path, monkeypatch,
                                                         name='retirement', mixed=mixed)
        service.prepare_room(binding)
        with authority.db._read_ctx() as conn:
            record, = [r for _, r in records(conn, 'room')]
            assert record['state'] == 'completed'
            assert record['version'] == 2 and 'binding' not in record
            admitted = json.loads(conn.execute('SELECT payload_json FROM session_admissions').fetchone()[0])
            if mixed:
                media, = admitted['attachments_v1']['media']
                assert custody_holds(conn, authority.db.db_path, media)
        assert authority.db._execute_write(lambda c: retire_prunable(c, [producer.ref.session_id])) == [producer.ref.session_id]
        def forbidden(*args, **kwargs):
            raise AssertionError('completed replay must not reconstruct inputs or unlink')
        with monkeypatch.context() as guard:
            guard.setattr('gateway.hosted_room_input_retained.retained_hosted_input', forbidden)
            guard.setattr('gateway.hosted_room_output_discard.os.unlink', forbidden)
            service.prepare_room(binding)
            service.prepare_room(binding)
            assert service._reconcile_stopped_output(task)
            assert not service.output_cleanup_status('room')
            with authority.db._read_ctx() as conn:
                require_room_retired(conn, 'room')
                done, = [r for _, r in records(conn, 'room')]
                assert done['version'] == 2 and done['state'] == 'completed'
                assert not ({'binding', 'original_binding', 'input_binding', 'items', 'blobs'} & done.keys())
                assert admitted['text'] not in json.dumps(done)
                if mixed:
                    assert not custody_holds(conn, authority.db.db_path, media)
            # A compact fact cannot be transplanted to a newly assigned generation.
            authority.db._execute_write(lambda c: c.execute('UPDATE hosted_room_driver_tasks SET execution_generation=execution_generation+1'))
            newer = tasks.get_task(service.db_path, task['identity'])
            authority.db._execute_write(lambda c: c.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                (key_for(newer), json.dumps(done))))
            with pytest.raises(RoomArtifactError):
                service._reconcile_stopped_output(task)
            with pytest.raises(RoomArtifactError):
                service._reconcile_stopped_output(newer)
            authority.db._execute_write(lambda c: c.execute('DELETE FROM state_meta WHERE key=?', (key_for(newer),)))
            authority.db._execute_write(lambda c: c.execute('UPDATE hosted_room_driver_tasks SET execution_generation=execution_generation-1'))
            service.prepare_room(binding)
            assert tasks.prune_published_terminal_tasks(service.db_path, room_id='room', clock=lambda: 10**12, retain=0) == 1
            service.prepare_room(binding)
            with authority.db._read_ctx() as conn:
                require_room_retired(conn, 'room')


@pytest.mark.asyncio
async def test_task_local_input_damage_does_not_starve_later_cleanup(tmp_path, monkeypatch, request):
    from gateway import hosted_room_task_scan as scans
    from gateway.hosted_room_input_reclamation import copy_path
    from hermes_state_mutation_retirement import retire_prunable
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        monkeypatch.setattr('model_tools._resolve_active_context_length', lambda: 32768)
        initialize_inputs(authority, tmp_path, request)
        import os
        original_unlink = os.unlink
        def fail_unlink(*args, **kwargs):
            if kwargs.get('dir_fd') is not None:
                raise OSError('hold real physical cleanup until both producers finish')
            return original_unlink(*args, **kwargs)
        with monkeypatch.context() as hold:
            hold.setattr('gateway.hosted_room_output_discard.os.unlink', fail_unlink)
            pairs = [await stopped_producer(authority, service, runner, tmp_path, monkeypatch,
                                            name=name, mixed=True) for name in ('first', 'second')]
        pairs.sort(key=lambda value: value[0]['identity'].task_id)
        first, binding, producer = pairs[0]
        later = pairs[1][0]
        with authority.db._read_ctx() as conn:
            before = dict(records(conn, 'room'))[key_for(first)]
            assert before['state'] == 'pending'
            input_ref = conn.execute('SELECT * FROM input_custody_copies WHERE copy_id=?',
                (before['binding']['input_binding']['copies'][0]['copy_id'],)).fetchone()
            source = copy_path(authority.db, dict(input_ref))
        original_input = source.read_bytes()
        source.write_bytes(b'damaged retained proof after capture')
        paths = [tmp_path / 'hosted-room-artifact-outbox' / 'blobs' / b['blob_name'] for b in before['blobs']]
        original_bytes = [p.read_bytes() for p in paths]
        with authority.db._read_ctx() as conn:
            intact, _ = service._cleanup_snapshot(conn, later)
        assert not intact.get('unavailable')  # independently intact, different input thread
        def populate(conn):
            row = dict(conn.execute('SELECT * FROM hosted_room_driver_tasks WHERE task_id=?', (first['identity'].task_id,)).fetchone())
            for i in range(scans.BUDGET + 2):
                filler = dict(row, task_id=first['identity'].task_id+f'-inventory-{i:03}',
                    thread_id=f'inventory-{i}', turn_id=f'inventory-{i}', status='queued', execution_generation=0,
                    result_json=None, settlement_id=None, settlement_status=None, terminal_at=None)
                conn.execute('INSERT INTO hosted_room_driver_tasks ('+','.join(filler)+') VALUES ('+
                             ','.join('?' for _ in filler)+')', tuple(filler.values()))
        authority.db._execute_write(populate)
        service._artifact_clock = lambda: 10**12
        sizes, visited = [], []
        original_page = scans.page
        def bounded(*args):
            scan, batch = original_page(*args)
            sizes.append(len(batch))
            visited.extend(t['identity'].task_id for t in batch)
            return scan, batch
        monkeypatch.setattr(scans, 'page', bounded)
        for _ in range(6):
            service._prepare_terminal_tasks(service._room('room'))
        with authority.db._read_ctx() as conn:
            after = dict(records(conn, 'room'))
            blocked = after[key_for(first)]
            assert blocked['blocked'] is True and blocked['reason_code'] == 'input_binding_unavailable'
            assert all(blocked[k] == before[k] for k in ('binding', 'items', 'blobs', 'removed', 'attempts'))
            assert after[key_for(later)]['state'] == 'completed', {
                'reason': after[key_for(later)]['reason_code'],
                'blocked': after[key_for(later)].get('blocked'),
                'next': after[key_for(later)]['next_attempt_at'],
                'later_visits': visited.count(later['identity'].task_id), 'sizes': sizes}
        # Remove only inventory fillers, not either genuine execution. Prove
        # refusal is not merely the fillers' queued status or an incomplete scan.
        authority.db._execute_write(lambda c: c.execute(
            'DELETE FROM hosted_room_driver_tasks WHERE task_id LIKE ?',
            (first['identity'].task_id+'-inventory-%',)))
        service._prepare_terminal_tasks(service._room('room'))
        service._prepare_terminal_tasks(service._room('room'))
        with authority.db._read_ctx() as conn:
            assert not scans.pending(conn, 'room')
            with pytest.raises(RuntimeStoreError, match='output_cleanup_pending'):
                require_room_retired(conn, 'room')
        assert [p.read_bytes() for p in paths] == original_bytes
        assert authority.db._execute_write(lambda c: retire_prunable(c, [producer.ref.session_id])) == []
        assert max(sizes) <= scans.BUDGET and visited.count(first['identity'].task_id) >= 2
        # Owner-wide failure or missing shared input schema must not be converted
        # into another task-local success; the frozen batch cannot advance.
        frozen = authority.db._execute_write(lambda c: scans.page(c, 'room'))[0]
        with monkeypatch.context() as draining:
            draining.setattr(runner, '_draining', True)
            with pytest.raises(RuntimeStoreError):
                service._prepare_terminal_tasks(service._room('room'))
        authority.db._execute_write(lambda c: c.execute('ALTER TABLE input_custody_refs RENAME TO preserved_refs'))
        import sqlite3
        try:
            with pytest.raises(sqlite3.OperationalError):
                service._prepare_terminal_tasks(service._room('room'))
            with authority.db._read_ctx() as conn:
                assert scans.scan_state(conn, 'room') == frozen
        finally:
            authority.db._execute_write(lambda c: c.execute('ALTER TABLE preserved_refs RENAME TO input_custody_refs'))
        # Repair exact original bytes: bounded revisits can discharge, never recapture.
        source.write_bytes(original_input)
        for _ in range(4):
            service._prepare_terminal_tasks(service._room('room'))
        with authority.db._read_ctx() as conn:
            assert dict(records(conn, 'room'))[key_for(first)]['state'] == 'completed'
        assert not any(p.exists() for p in paths)


@pytest.mark.asyncio
@pytest.mark.parametrize(('claim_wins', 'inventory'), [(False, 'ready'), (True, 'ready'),
    (False, 'missing'), (False, 'occupied')], ids=['queued', 'claim-wins', 'missing-inventory', 'unexpected-output'])
async def test_native_stop_before_claim_preserves_exact_canonical_generation(tmp_path, monkeypatch, claim_wins, inventory):
    from gateway.session_results import execution_result
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        monkeypatch.setattr('model_tools._resolve_active_context_length', lambda: 32768)
        outbox = RoomArtifactOutbox(service.db_path)  # explicit initialized inventory
        if inventory == 'missing':
            authority.db._execute_write(lambda c: c.execute('DROP TABLE hosted_room_output_artifacts'))
        queued, allow_claim, started, release, interrupted = (asyncio.Event() for _ in range(5))
        calls = []
        original_drain = authority._drain
        if claim_wins:
            # Pause the real preclaim authorization AFTER its actual check. Stop
            # captures NULL, then the already-authorized claimant wins its CAS.
            import threading
            claim_barrier = threading.Event()
            loop = asyncio.get_running_loop()
            original_check = service.check_admission
            def checked(ref, row):
                result = original_check(ref, row)
                if row['status'] == 'queued':
                    loop.call_soon_threadsafe(queued.set)
                    assert claim_barrier.wait(8)
                return result
            monkeypatch.setattr(service, 'check_admission', checked)
            from gateway import session_finite
            original_execute = session_finite.execute_finite_admission
            async def before_execute(*args):
                started.set()
                await release.wait()
                return await original_execute(*args)
            monkeypatch.setattr(session_finite, 'execute_finite_admission', before_execute)
        else:
            async def drain(ref):
                queued.set()
                await allow_claim.wait()
                return await original_drain(ref)
            monkeypatch.setattr(authority, '_drain', drain)
        async def handle(event):
            calls.append(event.message_id)
            started.set()
            await release.wait()
            execution_result.get().update(result=dict(interrupted=True, final_response='', messages=[]), usage={})
            return ''
        runner._handle_message = handle
        runner._cached_agent_for = lambda _: SimpleNamespace(interrupt=interrupted.set)
        original_cancel = authority.cancel_queued
        async def claim_race(*args):
            claim_barrier.set()
            await asyncio.wait_for(started.wait(), 5)
            return await original_cancel(*args)
        if claim_wins:
            monkeypatch.setattr(authority, 'cancel_queued', claim_race)
        work = asyncio.create_task(execute_group_turn(authority, service))
        client = connection(authority, service)
        stop_succeeded = False
        try:
            await asyncio.wait_for(queued.wait(), 8)
            with authority.db._read_ctx() as conn:
                row = dict(conn.execute('SELECT * FROM session_admissions').fetchone())
                assert row['status'] == 'queued' and row['generation'] is None
            if inventory == 'occupied':
                from gateway.hosted_room_artifacts import RoomArtifactScope
                task, = tasks.list_tasks(service.db_path, room_id='room')
                with authority.db._read_ctx() as conn:
                    snapshot, _ = service._cleanup_snapshot(conn, task)
                scope = RoomArtifactScope.from_mapping(snapshot['scope'])
                path = tmp_path / 'unexpected.txt'
                path.write_bytes(b'unclaimed inventory is NOT unlink authority')
                item = outbox.put_path(scope=scope, path=path)  # inventory-only fixture
            stopped = await dispatch(client, 'stop', room_id='room', cancel_id='before-claim')
            assert stopped.get('result') == {'cancelled': 1}, stopped
            stop_succeeded = True
            with authority.db._read_ctx() as conn:
                current = dict(conn.execute('SELECT * FROM session_admissions').fetchone())
                assert current['admission_id'] == row['admission_id']
                if claim_wins:
                    assert current['status'] == 'started' and type(current['generation']) is int
                    waiting, = [r for _, r in records(conn, 'room')]
                    assert waiting['state'] == 'waiting'
                    assert waiting['binding']['admission']['generation'] == current['generation']
                    assert waiting['original_binding']['admission']['generation'] is None
                else:
                    assert current['status'] == 'terminal' and current['outcome'] == 'cancelled'
                    assert current['generation'] is None and not calls
            if claim_wins:
                assert (interrupted.is_set() or
                        authority.pending_stops.get(current['target_session_id']) == current['generation'])
            repeated = await dispatch(client, 'stop', room_id='room', cancel_id='before-claim')
            assert 'result' in repeated, repeated
        finally:
            allow_claim.set()
            release.set()
            if claim_wins:
                claim_barrier.set()
            if stop_succeeded:
                result = await asyncio.wait_for(work, 8)
            else:
                work.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await work
            service.runtime._thread = None
        service.prepare_room(result[4])
        service.prepare_room(result[4])
        with authority.db._read_ctx() as conn:
            final, = list(conn.execute('SELECT * FROM session_admissions'))
            assert final['status'] == 'terminal'
            assert final['generation'] == current['generation']
            if inventory == 'ready':
                require_room_retired(conn, 'room')
            else:
                with pytest.raises(RuntimeStoreError, match='output_cleanup_pending'):
                    require_room_retired(conn, 'room')
                held, = [r for _, r in records(conn, 'room')]
                assert held['state'] == 'waiting'
                assert held['reason_code'] == ('inventory_unavailable' if inventory == 'missing' else 'unclaimed_output_inventory')
        if inventory == 'occupied':
            assert outbox.read(scope, item['artifact_id'])[1] == path.read_bytes()
        assert not calls  # stopped before handler entry, even when the claim CAS won
        assert tasks.get_task(service.db_path, result[3]['identity'])['execution_generation'] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('boundary', ['replay', 'retirement', 'task-retirement'])
async def test_legacy_completed_fact_compacts_without_reconstructing_input(tmp_path, monkeypatch, boundary):
    from tests.gateway.test_canonical_output_refusals import stranded
    from hermes_state_mutation_retirement import retire_prunable
    async with stranded(tmp_path, monkeypatch) as (authority, service, task, binding, pending, paths):
        service.prepare_room(binding)
        service.prepare_room(binding)
        with authority.db._read_ctx() as conn:
            done = dict(records(conn, 'room'))[key_for(task)]
        assert done['version'] == 2 and done['state'] == 'completed'
        assert not any(p.exists() for p in paths)
        # Version-1 representation of the SAME genuinely completed cleanup, not
        # fabricated terminal execution or removal. Keep its original binding.
        legacy = dict(pending, state='completed', blobs=[], reason_code='completed', next_attempt_at=0)
        authority.db._execute_write(lambda c: c.execute('UPDATE state_meta SET value=? WHERE key=?',
            (json.dumps(legacy), key_for(task))))
        sid = pending['binding']['admission']['target_session_id']
        if boundary == 'retirement':
            assert authority.db._execute_write(lambda c: retire_prunable(c, [sid])) == [sid]
        if boundary == 'task-retirement':
            assert tasks.prune_published_terminal_tasks(service.db_path, room_id='room', clock=lambda: 10**12, retain=0) == 1
        def forbidden(*args, **kwargs):
            raise AssertionError('legacy completed replay must not unlink or inspect live inputs')
        with monkeypatch.context() as guard:
            guard.setattr('gateway.hosted_room_input_retained.retained_hosted_input', forbidden)
            guard.setattr('gateway.hosted_room_output_discard.os.unlink', forbidden)
            if boundary != 'task-retirement':
                assert service._reconcile_stopped_output(task)
        with authority.db._read_ctx() as conn:
            compact = dict(records(conn, 'room'))[key_for(task)]
        assert compact == done and 'binding' not in compact
