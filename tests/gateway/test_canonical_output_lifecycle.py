"""Native Stop through actual canonical producer/drain; no driver settlement fixture."""
import asyncio
import json
from types import SimpleNamespace
from pathlib import Path

import pytest
from gateway import hosted_room_driver as tasks
from gateway.session_controls import AuthorityConnection
from tests.gateway.test_canonical_hosted_outputs import owner, execute_group_turn


def connection(authority, service):
    service.runtime._thread = SimpleNamespace(is_alive=lambda: True)
    return AuthorityConnection(authority, SimpleNamespace(), dict(user_id='alice', provider='local',
        capabilities=['session:read', 'session:control', 'session:submit'],
        profile_id=authority.profile_id, instance_id=authority.instance_id))


async def dispatch(client, method, **params):
    return await client.dispatch(dict(id=1, method='groups.' + method, params=params))


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', [None, 'unlink', 'fsync', 'commit',
    'drift_epoch', 'drift_instance', 'drift_cancel', 'drift_admission', 'drift_draining',
    'input_ok', 'input_image_ok', 'input_hold', 'input_bytes', 'input_digest', 'input_recipient', 'input_generation', 'input_session'])
async def test_native_stop_retires_exact_local_cancelled_output(tmp_path, monkeypatch, fault, request):
    from tools import hosted_room_artifact
    from gateway.session_hosted_output import current_output_binding
    from gateway.session_results import execution_result
    from gateway.hosted_room_artifacts import RoomArtifactOutbox, RoomArtifactScope
    from dataclasses import replace
    drift = fault[6:] if fault and fault.startswith('drift_') else None
    if drift:
        fault = 'unlink'
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        monkeypatch.setattr('model_tools._resolve_active_context_length', lambda: 32768)
        shared, release, signalled = asyncio.Event(), asyncio.Event(), asyncio.Event()
        producer, siblings, owned_paths = [], [], []
        runner._cached_agent_for = lambda _: SimpleNamespace(interrupt=signalled.set)
        output = tmp_path / 'cache' / 'private.txt'
        output.parent.mkdir(exist_ok=True)
        output.write_bytes(b'private before interrupted terminal')
        async def handle(event):
            binding = current_output_binding()
            producer.append(binding)
            value = json.loads(await asyncio.to_thread(hosted_room_artifact.share_group_file, str(output)))
            assert value['ok'], value
            with authority.db._read_ctx() as conn:
                owned_paths.extend(tmp_path / 'hosted-room-artifact-outbox' / 'blobs' / row[0]
                    for row in conn.execute('SELECT blob_name FROM hosted_room_output_artifacts WHERE scope_key=?', (binding.scope.key,)))
            # Exact independent scopes are real outbox rows, not settlement witnesses.
            outbox = RoomArtifactOutbox(service.db_path)
            for scope in ((replace(binding.scope, execution_generation=binding.scope.execution_generation + 1),
                          replace(binding.scope, member_id='reviewer', target_profile='reviewer')) if fault is None else ()):
                siblings.append((scope, outbox.put_path(scope=scope, path=output)))
            shared.set()
            await release.wait()
            execution_result.get().update(result=dict(interrupted=True, final_response='', messages=[]), usage={})
            return ''
        runner._handle_message = handle
        input_manifest = None
        if fault and fault.startswith('input_'):
            from gateway.runtime_ownership import process_ownership
            process_ownership.reserve([tmp_path])
            request.addfinalizer(lambda: process_ownership.release(tmp_path))
            from gateway.hosted_room_input_reclamation import initialize_working_copies
            initialize_working_copies(authority.db, epoch=authority.epoch)
            item = service.attachments.put(room_id='room', upload_id='input', name='input.txt',
                kind='file', mime='text/plain', data=b'original canonical input')
            input_manifest = [{k: item[k] for k in ('attachment_id', 'name', 'kind', 'mime', 'size')}]
            if fault == 'input_image_ok':
                import base64
                image = service.attachments.put(room_id='room', upload_id='image-input', name='pixel.png',
                    kind='image', mime='image/png', data=base64.b64decode(
                        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1cAAAAASUVORK5CYII='))
                input_manifest.append({k: image[k] for k in ('attachment_id', 'name', 'kind', 'mime', 'size')})
        work = asyncio.create_task(execute_group_turn(authority, service, input_manifest=input_manifest))

        client = connection(authority, service)
        fail_cleanup = [False]
        if fault in {'unlink', 'fsync'}:
            import os
            original = getattr(os, fault)
            def failing(*args, **kwargs):
                if fail_cleanup[0] and (fault == 'fsync' or kwargs.get('dir_fd') is not None):
                    raise OSError('test-owned physical cleanup fault')
                return original(*args, **kwargs)
            monkeypatch.setattr('gateway.hosted_room_output_discard.os.' + fault, failing)
        try:
            await asyncio.wait_for(shared.wait(), 8)
            if input_manifest:
                with authority.db._read_ctx() as conn:
                    input_before = [dict(r) for r in conn.execute('SELECT * FROM input_custody_refs')]
                    from gateway.hosted_room_input_reclamation import copy_path
                    input_paths = [copy_path(authority.db, dict(r)) for r in conn.execute("SELECT * FROM input_custody_copies WHERE namespace='v3'")]
                    admitted = json.loads(conn.execute('SELECT payload_json FROM session_admissions').fetchone()[0])
                    input_paths.extend(Path(r['path']) for r in admitted.get('attachments_v1', {}).get('media', []))
                assert input_before
                # Original source bytes may have aged out of read authority;
                # cleanup uses retained metadata plus its own input custody only.
                authority.db._execute_write(lambda conn: conn.execute('UPDATE hosted_room_attachments SET expires_at=0'))
                if fault == 'input_hold':
                    # Damaged retained evidence is unavailable, never a fresh capture.
                    authority.db._execute_write(lambda conn: conn.execute('DELETE FROM input_custody_refs'))
                elif fault == 'input_bytes':
                    input_paths[0].write_bytes(b'changed retained input')
                elif fault in {'input_digest', 'input_recipient'}:
                    assignment = "sha256='" + '0' * 64 + "'" if fault == 'input_digest' else "recipient_member_ids_json='[]'"
                    authority.db._execute_write(lambda conn: conn.execute('UPDATE hosted_room_attachments SET ' + assignment))
                elif fault == 'input_generation':
                    authority.db._execute_write(lambda conn: conn.execute('UPDATE input_custody_refs SET generation=generation+1'))
                elif fault == 'input_session':
                    authority.db._execute_write(lambda conn: conn.execute("UPDATE input_custody_refs SET target_session_id='foreign'"))
                input_bytes = [p.read_bytes() for p in input_paths]
                def no_recapture(*args, **kwargs):
                    raise AssertionError('cleanup must not read or recapture source inputs')
                monkeypatch.setattr('gateway.hosted_room_input_preparation.resolve_inputs', no_recapture)
            answer = await dispatch(client, 'stop', room_id='room', cancel_id='exact-stop')
            assert answer.get('result') == {'cancelled': 1}, answer
            await asyncio.wait_for(signalled.wait(), 3)
            state = await dispatch(client, 'state', room_id='room')
            pending = [x for x in state['result']['driver_status']['pending_actions'] if x['kind'] == 'output_cleanup']
            assert pending and pending[0]['state'] == 'waiting', state
            assert list((tmp_path / 'hosted-room-artifact-outbox' / 'blobs').iterdir())
            with pytest.raises(Exception):
                await asyncio.to_thread(producer[0].outbox().put_path, scope=producer[0].scope, path=output)
            fail_cleanup[0] = fault is not None
            if fault == 'commit':
                authority.db._execute_write(lambda conn: conn.execute("""CREATE TRIGGER stop_completion_fault
                    BEFORE UPDATE ON state_meta WHEN NEW.key LIKE 'gateway.hosted.output_cleanup.v1:%'
                    AND json_extract(NEW.value,'$.state')='completed'
                    BEGIN SELECT RAISE(ABORT,'test completion commit'); END"""))
        finally:
            release.set()
            if fault == 'commit':
                import sqlite3
                with pytest.raises(sqlite3.IntegrityError, match='test completion commit'):
                    await asyncio.wait_for(work, 8)
                result = None
            else:
                result = await asyncio.wait_for(work, 8)
            service.runtime._thread = None
        if result is None:
            task, = tasks.list_tasks(service.db_path, room_id='room')
            binding = service.bindings()[0]
        else:
            rpc, request, receipt, task, binding = result
        current = tasks.get_task(service.db_path, task['identity'])
        assert not (current.get('result') or {}).get('artifacts')
        if input_manifest:
            assert [p.read_bytes() for p in input_paths] == input_bytes
        if fault in {'input_ok', 'input_image_ok'}:
            with authority.db._read_ctx() as conn:
                assert [dict(r) for r in conn.execute('SELECT * FROM input_custody_refs')] == input_before
            fault = None
        if fault and fault.startswith('input_'):
            held = [r for r in service.status('room')['pending_actions'] if r['kind'] == 'output_cleanup']
            assert held and held[0]['reason_code'] == 'input_binding_unavailable'
            assert owned_paths and all(path.exists() for path in owned_paths)
            return  # exact damaged evidence remains a hold, never cleanup success
        if fault:
            import time
            from gateway.session_hosted_output_lifecycle import records
            with authority.db._read_ctx() as conn:
                record, = [r for _, r in records(conn, 'room')]
            assert record['state'] == 'pending' and record['removed'] == 1 and len(record['blobs']) == 1
            from hermes_state_mutation_retirement import retire_prunable
            assert authority.db._execute_write(lambda conn: retire_prunable(conn, [producer[0].ref.session_id])) == []
            if drift:
                from gateway.hosted_room_artifacts import RoomArtifactError
                from hermes_state_runtime import RuntimeStoreError
                paths = [tmp_path / 'hosted-room-artifact-outbox' / 'blobs' / b['blob_name'] for b in record['blobs']]
                before = [p.read_bytes() for p in paths]
                fail_cleanup[0] = False
                with monkeypatch.context() as changed:
                    if drift in {'epoch', 'instance'}:
                        changed.setattr(authority, 'epoch' if drift == 'epoch' else 'instance_id',
                                        authority.epoch + 1 if drift == 'epoch' else 'different-owner')
                    elif drift == 'draining':
                        changed.setattr(runner, '_draining', True)
                    elif drift == 'cancel':
                        authority.db._execute_write(lambda conn: conn.execute('UPDATE hosted_room_driver_tasks SET cancel_generation=cancel_generation+1'))
                    else:
                        authority.db._execute_write(lambda conn: conn.execute("UPDATE session_admissions SET principal_id='foreign'"))
                    try:
                        service._artifact_clock = lambda: record['next_attempt_at'] + 1
                        with pytest.raises((RuntimeStoreError, RoomArtifactError)):
                            await asyncio.to_thread(service.prepare_room, binding)
                        assert [p.read_bytes() for p in paths] == before
                    finally:
                        if drift == 'cancel':
                            authority.db._execute_write(lambda conn: conn.execute('UPDATE hosted_room_driver_tasks SET cancel_generation=cancel_generation-1'))
                        elif drift == 'admission':
                            authority.db._execute_write(lambda conn: conn.execute('UPDATE session_admissions SET principal_id=?', (record['binding']['owner'],)))
            # Lose a prunable artifact row, not its independent physical obligation.
            authority.db._execute_write(lambda conn: conn.execute(
                'DELETE FROM hosted_room_output_artifacts WHERE scope_key=?', (producer[0].scope.key,)))
            outbox = RoomArtifactOutbox(service.db_path)
            aged = time.time() + tasks.ARTIFACT_RETRY_RETENTION_SECONDS + 10
            assert tasks.prune_published_terminal_tasks(service.db_path, room_id='room', clock=lambda: aged, retain=0) == 0
            outbox.prune_generation_fences(now=aged)
            service._artifact_clock = lambda: aged
            service._prune_output_retry_metadata('room')
            fail_cleanup[0] = False
            if fault == 'commit':
                authority.db._execute_write(lambda conn: conn.execute('DROP TRIGGER stop_completion_fault'))
            await asyncio.to_thread(service.prepare_room, binding)
            with authority.db._read_ctx() as conn:
                final, = [r for _, r in records(conn, 'room')]
            assert final['state'] == 'completed' and final['removed'] == 1
        with authority.db._read_ctx() as conn:
            assert not conn.execute('SELECT 1 FROM hosted_room_output_artifacts WHERE scope_key=?', (producer[0].scope.key,)).fetchall()
        assert owned_paths and not any(path.exists() for path in owned_paths)
        outbox = RoomArtifactOutbox(service.db_path)
        for scope, item in siblings:
            assert outbox.read(scope, item['artifact_id'])[1] == output.read_bytes()
        # An interrupted scan with intervening task drift needs a fresh revision pass.
        service.prepare_room(binding)
        assert not [x for x in service.status('room')['pending_actions'] if x['kind'] == 'output_cleanup']
        service.prepare_room(binding)
        assert tasks.get_task(service.db_path, task['identity'])['execution_generation'] == current['execution_generation']
        if input_manifest and len(input_manifest) > 1:
            from gateway.hosted_room_input_custody import custody_holds
            from hermes_state_mutation_retirement import retire_prunable
            media, = admitted['attachments_v1']['media']
            with authority.db._read_ctx() as conn:
                assert custody_holds(conn, authority.db.db_path, media)
            assert authority.db._execute_write(lambda conn: retire_prunable(conn, [producer[0].ref.session_id])) == [producer[0].ref.session_id]
            with authority.db._read_ctx() as conn:
                assert not custody_holds(conn, authority.db.db_path, media)

@pytest.mark.asyncio
async def test_unknown_requires_explicit_exact_discard_before_cleanup(tmp_path, monkeypatch):
    from tools import hosted_room_artifact
    from gateway.session_hosted_output import current_output_binding
    from hermes_state_runtime import list_session_admissions
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        monkeypatch.setattr('model_tools._resolve_active_context_length', lambda: 32768)
        shared, release = asyncio.Event(), asyncio.Event()
        captured = []
        output = tmp_path / 'cache' / 'unknown.txt'
        output.parent.mkdir(exist_ok=True)
        output.write_bytes(b'unknown original bytes')
        async def handle(event):
            captured.append(current_output_binding())
            assert json.loads(await asyncio.to_thread(hosted_room_artifact.share_group_file, str(output)))['ok']
            shared.set()
            await release.wait()
            return ''
        runner._handle_message = handle
        work = asyncio.create_task(execute_group_turn(authority, service))
        await asyncio.wait_for(shared.wait(), 8)
        work.cancel()
        with pytest.raises(asyncio.CancelledError):
            await work
        # Explicit lost-receipt fixture, NOT a terminal or settlement witness.
        # The actual producer/drain was interrupted above; only explicit native
        # discard below is permitted to resolve its retained unknown admission.
        authority.db._execute_write(lambda conn: conn.execute("UPDATE session_admissions SET status='unknown' WHERE admission_id=?", (captured[0].row['admission_id'],)))
        authority.db._execute_write(lambda conn: conn.execute("UPDATE hosted_room_driver_tasks SET status='indeterminate' WHERE task_id=?", (captured[0].scope.task_id,)))
        service.runtime.process_generation = 'driver'
        client = connection(authority, service)
        task, = tasks.list_tasks(service.db_path, room_id='room')
        try:
            stopped = await dispatch(client, 'stop', room_id='room', cancel_id='unknown-stop')
            assert 'result' in stopped, stopped
            current = tasks.get_task(service.db_path, task['identity'])
            assert current['status'] == 'indeterminate' and current['execution_generation'] == task['execution_generation']
            state = await dispatch(client, 'state', room_id='room')
            assert any(x['kind'] == 'output_cleanup' and x['reason_code'] == 'unknown_execution'
                       for x in state['result']['driver_status']['pending_actions'])
            exact = dict(room_id='room', member_id='writer', task_id=task['identity'].task_id,
                         execution_generation=task['execution_generation'])
            for changed in ({'member_id': 'reviewer'}, {'execution_generation': True},
                            {'execution_generation': task['execution_generation'] + 1}):
                refused = await dispatch(client, 'discard', **{**exact, **changed})
                assert 'error' in refused, refused
            with authority.db._read_ctx() as conn:
                assert conn.execute('SELECT 1 FROM hosted_room_output_artifacts WHERE scope_key=?', (captured[0].scope.key,)).fetchone()
            discarded = await dispatch(client, 'discard', **exact)
            assert discarded.get('result', {}).get('discarded') is True, discarded
            with authority.db._read_ctx() as conn:
                assert not conn.execute('SELECT 1 FROM hosted_room_output_artifacts WHERE scope_key=?', (captured[0].scope.key,)).fetchone()
            rows = list_session_admissions(authority.db, session_id=captured[0].ref.session_id, pending_only=False)
            assert len(rows) == 1 and rows[0]['status'] == 'terminal' and rows[0]['outcome'] == 'interrupted'
        finally:
            service.runtime._thread = None
