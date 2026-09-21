"""Destructive refusals after real producer -> native Stop -> drain/callback."""
import asyncio
import json
import os
import sqlite3
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from gateway import hosted_room_driver as tasks
from gateway.hosted_room_artifacts import RoomArtifactOutbox
from tests.gateway.test_canonical_hosted_outputs import owner, execute_group_turn
from tests.gateway.test_canonical_output_lifecycle import connection, dispatch


@asynccontextmanager
async def stranded(tmp_path, monkeypatch, *, overflow=False):
    from gateway.session_hosted_output import current_output_binding
    from gateway.session_results import execution_result
    from tools.hosted_room_artifact import share_group_file
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        monkeypatch.setattr('model_tools._resolve_active_context_length', lambda: 32768)
        shared, release = asyncio.Event(), asyncio.Event()
        output = tmp_path / 'cache' / 'stranded.txt'
        output.parent.mkdir(exist_ok=True)
        output.write_bytes(b'original private output')
        bindings = []
        async def handle(event):
            bindings.append(current_output_binding())
            assert json.loads(await asyncio.to_thread(share_group_file, str(output)))['ok']
            shared.set()
            await release.wait()
            execution_result.get().update(result=dict(interrupted=True, final_response='', messages=[]), usage={})
            return ''
        runner._handle_message = handle
        work = asyncio.create_task(execute_group_turn(authority, service))
        try:
            await asyncio.wait_for(shared.wait(), 8)
            if overflow:
                # Oversized retained inventory fixture. The real producer above
                # owns the scope; these rows are not an execution/settlement proof.
                def oversized(conn):
                    row = dict(conn.execute('SELECT * FROM hosted_room_output_artifacts').fetchone())
                    root = tmp_path / 'hosted-room-artifact-outbox' / 'blobs'
                    for i in range(64):
                        extra = dict(row, artifact_id=f'overflow-{i}', name=f'overflow-{i}.txt', blob_name=f'overflow-{i}.blob')
                        (root / extra['blob_name']).write_bytes((root / row['blob_name']).read_bytes())
                        conn.execute('INSERT INTO hosted_room_output_artifacts (' + ','.join(extra) + ') VALUES (' +
                                     ','.join('?' for _ in extra) + ')', tuple(extra.values()))
                authority.db._execute_write(oversized)
            client = connection(authority, service)
            stopped = await dispatch(client, 'stop', room_id='room', cancel_id='matrix-stop')
            assert stopped.get('result') == {'cancelled': 1}, stopped
            with monkeypatch.context() as fault:
                def fail_unlink(*args, **kwargs):
                    raise OSError('retained physical obligation')
                fault.setattr('gateway.hosted_room_output_discard.os.unlink', fail_unlink)
                release.set()
                result = await asyncio.wait_for(work, 8)
        finally:
            release.set()
            if not work.done():
                await asyncio.wait_for(work, 8)
            service.runtime._thread = None
        rpc, request, receipt, task, room_binding = result
        task = tasks.get_task(service.db_path, task['identity'])
        from gateway.session_hosted_output_lifecycle import records
        with authority.db._read_ctx() as conn:
            record, = [r for _, r in records(conn, 'room')]
        assert record['state'] == ('waiting' if overflow else 'pending')
        service._artifact_clock = lambda: record['next_attempt_at'] + 1
        paths = (list((tmp_path / 'hosted-room-artifact-outbox' / 'blobs').iterdir()) if overflow else
                 [tmp_path / 'hosted-room-artifact-outbox' / 'blobs' / b['blob_name'] for b in record['blobs']])
        yield authority, service, task, room_binding, record, paths


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['closed', 'replaced', 'quarantined', 'new_owner', 'new_generation'])
async def test_unavailable_owner_and_late_callback_do_not_reopen_or_mutate(tmp_path, monkeypatch, fault):
    async with stranded(tmp_path, monkeypatch) as (authority, service, task, binding, record, paths):
        db = authority.db
        before = [p.read_bytes() for p in paths]
        old_conn = db._conn
        if fault == 'closed':
            db.close()
        elif fault == 'replaced':
            replacement = tmp_path / 'replacement.db'
            with sqlite3.connect(replacement) as other:
                other.execute('CREATE TABLE sentinel(value TEXT)')
                other.execute("INSERT INTO sentinel VALUES('replacement must survive')")
            os.replace(replacement, db.db_path)
            replacement_bytes = db.db_path.read_bytes()
        elif fault == 'quarantined':
            with pytest.raises(Exception):
                db._halt_db_corrupt(sqlite3.DatabaseError('database disk image is malformed'))
            assert db._db_corrupt
        elif fault == 'new_owner':
            authority.db = SimpleNamespace(db_path=db.db_path)
        else:
            db._execute_write(lambda c: c.execute('UPDATE hosted_room_driver_tasks SET execution_generation=execution_generation+1'))
        denied = []
        def forbidden(*args, **kwargs):
            denied.append('cold owner/constructor/unlink')
            raise OSError('forbidden cold cleanup effect')
        changes = old_conn.total_changes if fault != 'closed' else None
        with monkeypatch.context() as guard:
            guard.setattr(RoomArtifactOutbox, '__init__', forbidden)
            guard.setattr(db, '_open_writer_conn', forbidden)
            guard.setattr('gateway.hosted_room_output_discard.os.unlink', forbidden)
            try:
                if fault == 'new_generation':
                    # Invoke the actual late callback with its captured OLD generation.
                    old_attempt = SimpleNamespace(identity=task['identity'], execution_generation=task['execution_generation'])
                    service.runtime._on_terminal(binding, old_attempt, {'status': 'cancelled'})
                else:
                    with pytest.raises(Exception):
                        service.prepare_room(binding)
                with pytest.raises(Exception):
                    service._reconcile_stopped_output(task)
                assert not denied
                assert [p.read_bytes() for p in paths] == before
                if changes is not None:
                    assert old_conn.total_changes == changes
                else:
                    assert db._conn is None
            finally:
                authority.db = db
        if fault == 'replaced':
            assert db.db_path.read_bytes() == replacement_bytes
            # Inspect the exact replacement, not the old owner's still-live WAL.
            with sqlite3.connect(db.db_path.as_uri() + '?immutable=1', uri=True) as replacement:
                assert replacement.execute('SELECT value FROM sentinel').fetchall() == [('replacement must survive',)]


@pytest.mark.asyncio
@pytest.mark.parametrize('missing', ['hosted_room_output_artifacts', 'hosted_room_output_generation_fences', 'uninitialized'])
async def test_missing_outbox_schema_preserves_exact_pending_without_constructor(tmp_path, monkeypatch, missing):
    async with stranded(tmp_path, monkeypatch) as (authority, service, task, binding, record, paths):
        if missing == 'uninitialized':
            authority.db._execute_write(lambda c: c.execute('DROP TABLE hosted_room_output_artifacts'))
            authority.db._execute_write(lambda c: c.execute('CREATE TABLE hosted_room_output_artifacts(scope_key TEXT)'))
        else:
            authority.db._execute_write(lambda c: c.execute('DROP TABLE ' + missing))
        before = [p.read_bytes() for p in paths]
        called = []
        def forbidden(*args, **kwargs):
            called.append(True)
            raise AssertionError('cold outbox constructor')
        monkeypatch.setattr(RoomArtifactOutbox, '__init__', forbidden)
        from gateway.hosted_room_output_discard import OutputCleanupUnavailable
        changes = authority.db._conn.total_changes
        with pytest.raises(OutputCleanupUnavailable):
            service._reconcile_stopped_output(task)
        assert authority.db._conn.total_changes == changes
        assert not called and [p.read_bytes() for p in paths] == before
        with authority.db._read_ctx() as conn:
            if missing == 'uninitialized':
                assert [r['name'] for r in conn.execute('PRAGMA table_info(hosted_room_output_artifacts)')] == ['scope_key']
            else:
                assert not conn.execute('SELECT 1 FROM sqlite_master WHERE name=?', (missing,)).fetchone()
            from gateway.session_hosted_output_lifecycle import records
            retained, = [r for _, r in records(conn, 'room')]
            assert retained['state'] == 'pending' and retained['blobs'] == record['blobs']


@pytest.mark.asyncio
async def test_scope_overflow_is_not_empty_or_partial_retirement(tmp_path, monkeypatch):
    async with stranded(tmp_path, monkeypatch, overflow=True) as (authority, service, task, binding, record, paths):
        assert record['reason_code'] == 'inventory_limit' and len(paths) == 65
        before = [p.read_bytes() for p in paths]
        with authority.db._read_ctx() as conn:
            rows = [dict(r) for r in conn.execute('SELECT * FROM hosted_room_output_artifacts ORDER BY artifact_id')]
            fences = [dict(r) for r in conn.execute('SELECT * FROM hosted_room_output_generation_fences')]
        service.prepare_room(binding)
        assert [p.read_bytes() for p in paths] == before
        from gateway.session_group_retirement import require_room_retired
        from hermes_state_runtime import RuntimeStoreError
        with authority.db._read_ctx() as conn:
            assert [dict(r) for r in conn.execute('SELECT * FROM hosted_room_output_artifacts ORDER BY artifact_id')] == rows
            assert [dict(r) for r in conn.execute('SELECT * FROM hosted_room_output_generation_fences')] == fences
            with pytest.raises(RuntimeStoreError, match='output_cleanup_pending'):
                require_room_retired(conn, 'room')
