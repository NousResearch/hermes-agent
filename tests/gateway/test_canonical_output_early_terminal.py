"""Stop claim race through real RPC/finite/drain/callback, before Stop ACK."""
import asyncio
import base64
import json
from pathlib import Path
import threading

import pytest

from gateway import hosted_room_driver as tasks, hosted_room_task_scan as scans
from gateway.hosted_room_artifacts import RoomArtifactOutbox
from gateway.hosted_room_input_custody import custody_holds
from gateway.session_group_retirement import require_room_retired
from gateway.session_hosted_output_lifecycle import key_for, records
from hermes_state_mutation_retirement import retire_prunable
from tests.gateway.test_canonical_hosted_outputs import owner, execute_group_turn
from tests.gateway.test_canonical_output_lifecycle import connection, dispatch
from tests.gateway.test_canonical_output_stop_corrections import initialize_inputs


@pytest.mark.asyncio
@pytest.mark.parametrize('input_kind', ['text', 'image', 'mixed'])
async def test_claim_terminal_before_stop_ack_retires_exact_capture(tmp_path, monkeypatch, request, input_kind):
    from gateway import session_ingress_media as media
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        monkeypatch.setattr('model_tools._resolve_active_context_length', lambda: 32768)
        initialize_inputs(authority, tmp_path, request)
        RoomArtifactOutbox(service.db_path)  # Real, initialized empty output inventory.
        manifest, original_image, original_document = [], None, None
        if input_kind != 'text':
            original_image = base64.b64decode(
                'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1cAAAAASUVORK5CYII=')
            image = service.attachments.put(room_id='room', upload_id='image', name='pixel.png',
                kind='image', mime='image/png', data=original_image)
            manifest.append({k: image[k] for k in ('attachment_id', 'name', 'kind', 'mime', 'size')})
        if input_kind == 'mixed':
            original_document = b'original document input, unchanged through claim'
            document = service.attachments.put(room_id='room', upload_id='document', name='input.txt',
                kind='file', mime='text/plain', data=original_document)
            manifest.append({k: document[k] for k in ('attachment_id', 'name', 'kind', 'mime', 'size')})

        queued, callback_done = asyncio.Event(), asyncio.Event()
        claim_barrier = threading.Event()
        loop = asyncio.get_running_loop()
        observations, calls = {}, []
        original_check = service.check_admission

        def checked(ref, row):
            result = original_check(ref, row)
            if row['status'] == 'queued':
                loop.call_soon_threadsafe(queued.set)
                assert claim_barrier.wait(10)
            return result

        monkeypatch.setattr(service, 'check_admission', checked)

        async def forbidden_handler(event):
            calls.append(event.message_id)
            raise AssertionError('stopped binding must fail before handler/model entry')

        runner._handle_message = forbidden_handler
        original_release = media.release_admission_media

        def observe_release(db, admission_id):
            # Call the REAL release in its normal drain position, before callback.
            released = original_release(db, admission_id)
            if 'release_admission' in observations:
                return released  # cancel_queued may subsequently replay the release.
            with db._read_ctx() as conn:
                admission = dict(conn.execute('SELECT * FROM session_admissions WHERE admission_id=?',
                                              (admission_id,)).fetchone())
                record, = [value for _, value in records(conn, 'room')]
                observations.update(released=released, release_admission=admission,
                    release_record=record,
                    release_driver=conn.execute('SELECT status FROM hosted_room_driver_tasks').fetchone()[0])
                if input_kind != 'text':
                    reference, = json.loads(admission['payload_json'])['attachments_v1']['media']
                    path = Path(reference['path'])
                    observations.update(reference=reference, image_exists=path.exists(),
                        image_bytes=path.read_bytes() if path.exists() else None,
                        held=custody_holds(conn, db.db_path, reference),
                        ordinary_holders=media._held_media_paths(conn))
            return released

        monkeypatch.setattr(media, 'release_admission_media', observe_release)
        original_terminal = service.runtime._on_terminal

        def observed_terminal(binding, attempt, receipt):
            observations['callback_before'] = tasks.get_task(service.db_path, attempt.identity)['status']
            observations['receipt'] = dict(receipt)
            try:
                return original_terminal(binding, attempt, receipt)
            finally:
                observations['callback_after'] = tasks.get_task(service.db_path, attempt.identity)['status']
                loop.call_soon_threadsafe(callback_done.set)

        monkeypatch.setattr(service.runtime, '_on_terminal', observed_terminal)
        original_cancel = authority.cancel_queued

        async def claim_race(*args):
            with authority.db._read_ctx() as conn:
                captured, = [value for _, value in records(conn, 'room')]
                assert captured['binding']['admission']['generation'] is None
                assert captured['state'] == 'waiting'
            # No executor barrier: actual failed terminal and callback both win ACK.
            claim_barrier.set()
            await asyncio.wait_for(callback_done.wait(), 8)
            observations['before_stop_ack'] = tasks.list_tasks(service.db_path, room_id='room')[0]['status']
            return await original_cancel(*args)

        monkeypatch.setattr(authority, 'cancel_queued', claim_race)
        work = asyncio.create_task(execute_group_turn(authority, service, input_manifest=manifest))
        client = connection(authority, service)
        try:
            await asyncio.wait_for(queued.wait(), 8)
            with authority.db._read_ctx() as conn:
                original, = [dict(row) for row in conn.execute('SELECT * FROM session_admissions')]
                assert original['status'] == 'queued' and original['generation'] is None
            stopped = await dispatch(client, 'stop', room_id='room', cancel_id='early-terminal')
        finally:
            claim_barrier.set()
            result, = await asyncio.wait_for(asyncio.gather(work, return_exceptions=True), 8)
            service.runtime._thread = None

        assert observations['release_driver'] == observations['callback_before'] == 'stopping'
        assert observations['callback_after'] == observations['before_stop_ack'] == 'failed'
        assert observations['receipt']['status'] == 'failed'
        final = observations['release_admission']
        assert final['status'] == 'terminal' and final['outcome'] == 'failed'
        assert type(final['generation']) is int and final['generation'] > 0
        assert all(final[k] == original[k] for k in ('admission_id', 'request_id', 'principal_id',
            'target_session_id', 'owner_epoch', 'payload_json', 'payload_digest', 'intent'))
        capture = observations['release_record']
        assert capture['state'] == 'waiting' and capture['binding']['admission']['generation'] is None
        if input_kind != 'text':
            assert observations['image_exists'], 'normal terminal release lost original native image before callback'
            assert observations['image_bytes'] == original_image
            assert observations['held'] and observations['released'] == 0
            assert observations['ordinary_holders'] == set(), 'independent native holder masks Stop handoff'
            reference = observations['reference']
            # Only the exact waiting original tuple may borrow the handoff hold.
            # Roll back each negative; do not mutate canonical execution evidence.
            def negative_holds(conn):
                key = key_for(tasks.list_tasks(service.db_path, room_id='room')[0])
                conn.execute('SAVEPOINT handoff_negatives')
                try:
                    conn.execute('DELETE FROM state_meta WHERE key=?', (key,))
                    assert not custody_holds(conn, service.db_path, reference), 'unrelated holder masks Stop hold'
                    for field in ('admission_id', 'request_id', 'principal_id', 'target_session_id',
                                  'owner_epoch', 'payload_json', 'payload_digest', 'intent', 'generation'):
                        changed = json.loads(json.dumps(capture))
                        changed['binding']['admission'][field] = (final['generation'] + 1 if field == 'generation'
                            else 'foreign-' + str(changed['binding']['admission'][field]))
                        conn.execute('INSERT OR REPLACE INTO state_meta(key,value) VALUES(?,?)',
                                     (key, json.dumps(changed)))
                        assert not custody_holds(conn, service.db_path, reference), field
                    for state in ('pending', 'completed'):
                        changed = dict(capture, state=state)
                        conn.execute('UPDATE state_meta SET value=? WHERE key=?', (json.dumps(changed), key))
                        assert not custody_holds(conn, service.db_path, reference), state
                finally:
                    conn.execute('ROLLBACK TO handoff_negatives')
                    conn.execute('RELEASE handoff_negatives')
            authority.db._execute_write(negative_holds)
        assert not isinstance(result, BaseException), result
        assert stopped.get('result') == {'cancelled': 1}, stopped
        task = tasks.get_task(service.db_path, result[3]['identity'])
        assert task['status'] == 'failed' and not calls
        assert not task.get('result', {}).get('artifacts')
        service.prepare_room(result[4])
        service.prepare_room(result[4])
        with authority.db._read_ctx() as conn:
            done, = [value for _, value in records(conn, 'room')]
            assert done['state'] == 'completed', 'failed driver stranded its captured cleanup'
            assert done['version'] == 2 and 'binding' not in done
            assert not scans.pending(conn, 'room')
            require_room_retired(conn, 'room')
            if input_kind == 'mixed':
                from gateway.hosted_room_input_reclamation import copy_path
                copies = conn.execute("SELECT * FROM input_custody_copies WHERE namespace='v3'").fetchall()
                assert copies and all(copy_path(authority.db, dict(copy)).read_bytes() == original_document for copy in copies)
        assert not service.output_cleanup_status('room')
        assert authority.db._execute_write(lambda conn: retire_prunable(conn, [final['target_session_id']])) == [final['target_session_id']]
        with authority.db._read_ctx() as conn:
            assert conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 0
            if input_kind != 'text':
                assert Path(reference['path']).read_bytes() == original_image
                assert not custody_holds(conn, service.db_path, reference)
        assert tasks.prune_published_terminal_tasks(service.db_path, room_id='room', clock=lambda: 10**12, retain=0) == 1
        service.prepare_room(result[4])
        with authority.db._read_ctx() as conn:
            require_room_retired(conn, 'room')
        assert not any(event['kind'] == 'message.member' for event in service._events('room'))


@pytest.mark.asyncio
async def test_failed_driver_without_stop_does_not_capture_cleanup(tmp_path, monkeypatch):
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        monkeypatch.setattr('model_tools._resolve_active_context_length', lambda: 32768)
        calls = []

        async def fail(event):
            calls.append(event.message_id)
            raise RuntimeError('inert finite handler failure, no Stop')

        runner._handle_message = fail
        result = await execute_group_turn(authority, service)
        task = tasks.get_task(service.db_path, result[3]['identity'])
        assert task['status'] == 'failed' and len(calls) == 1
        service.prepare_room(result[4])
        service.prepare_room(result[4])
        with authority.db._read_ctx() as conn:
            assert not records(conn, 'room')
        assert not any(event['kind'] == 'room.stop_requested' for event in service._events('room'))
