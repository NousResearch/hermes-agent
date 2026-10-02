"""Real hosted image settlement, refused preparation and retained native inputs."""
import asyncio
import hashlib
from pathlib import Path
import time

import pytest

from gateway.hosted_room_driver import TaskIdentity
from gateway.hosted_room_input_preparation import (
    prepare_hosted_input, reconstruct_accepted_payload, reconstruct_attested_payload,
)
from gateway.hosted_room_input_reclamation import collect_native_inputs, collect_working_copies
from gateway.session_contract import Submission
from gateway.session_ingress_media import _media_root
from hermes_state_input_custody import PreparedInputHandle, copy_is_held
from hermes_state_runtime import (
    RuntimeStoreError, claim_session_input, get_session_admission, settle_session_input,
)
from tests.gateway.input_reclamation_fixtures import owned, close, rpc_files, retire_metadata, expire, v3_path


def native_held(db):
    with db._read_ctx() as conn:
        copy, = conn.execute("SELECT * FROM input_custody_copies WHERE namespace='native'").fetchall()
        return copy_is_held(conn, copy, time.time())


@pytest.mark.asyncio
@pytest.mark.parametrize('count', [1, 2])
async def test_terminal_hosted_image_retry_after_real_settlement(tmp_path, monkeypatch, count):
    db, owner = owned(tmp_path, monkeypatch)
    try:
        rpc, bound = rpc_files(tmp_path, owner, count=count, image=True, named=True)
        params = dict(task=TaskIdentity('room', 'task', 'thread', 'turn'), execution_generation=1,
            prompt='read', attachments=[item for item, _ in bound], on_terminal=lambda value: None)
        first = await rpc._submit(params)
        row = get_session_admission(db, admission_id=first['admission_id'])
        image = Path(row['payload']['attachments_v1']['media'][0]['path'])
        executed = []
        async def inert(authority, ref, admission):
            assert image.exists()
            executed.append(admission['admission_id'])
            return 'inert result'
        monkeypatch.setattr('gateway.session_finite.execute_finite_admission', inert)
        await owner._drain(rpc.ref)
        await asyncio.sleep(0)  # Flush the existing terminal waiter, not a timing assertion.
        terminal = get_session_admission(db, admission_id=first['admission_id'])
        assert terminal['status'] == 'terminal' and terminal['outcome'] == 'completed'
        assert not image.exists(), 'real settlement must release the image before retry'
        def no_capture(*args, **kwargs):
            pytest.fail('accepted retry recaptured bytes')
        monkeypatch.setattr('gateway.session_ingress_media.capture_native_media', no_capture)
        monkeypatch.setattr('gateway.hosted_room_input_preparation._copy', no_capture)
        receipts = []
        params['on_terminal'] = receipts.append
        retry = await rpc._submit(params)
        assert retry['admission_id'] == first['admission_id'] and retry['status'] == 'terminal'
        assert receipts[0]['text'] == 'inert result'
        digests = [hashlib.sha256(data).hexdigest() for _, data in bound]
        assert reconstruct_attested_payload(db, 'read', params['attachments'], digests, row) == row['payload']
        for retired in (False, True):
            if retired:
                retire_metadata(db, row['admission_id'])
                from hermes_state_mutation_retirement import RETIRED_PREFIX
                db._execute_write(lambda conn: conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                    (RETIRED_PREFIX + rpc.ref.session_id, '{}')))
                collect_working_copies(db, epoch=owner.epoch)
                prepared = prepare_hosted_input(rpc, request_id=row['request_id'], prompt='read',
                    attachments=params['attachments'])
                receipt = await owner.submit(rpc.principal, Submission(row['request_id'], rpc.ref, prepared.payload, 'queue'),
                    _input_custody=prepared.handle)
                assert receipt.admission_id == first['admission_id'] and receipt.status == 'terminal'
            assert reconstruct_accepted_payload(rpc, 'read', params['attachments'], row) == row['payload']
            assert reconstruct_attested_payload(db, 'read', params['attachments'], digests, row) == row['payload']
            with pytest.raises(RuntimeStoreError, match='admission_conflict'):
                reconstruct_attested_payload(db, 'changed', params['attachments'], digests, row)
            changed = digests[:-1] + ['0' * 64]
            with pytest.raises(RuntimeStoreError, match='admission_conflict'):
                reconstruct_attested_payload(db, 'read', params['attachments'], changed, row)
        await owner._drain(rpc.ref)
        assert executed == [first['admission_id']] and not image.exists()
        rpc.authorizer = lambda *args: False
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            await rpc._dispatch_owned('submit', params)
    finally:
        close(db, tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize('held', [False, True])
async def test_refused_mixed_preparation_native_bytes_are_collectible(tmp_path, monkeypatch, held):
    db, owner = owned(tmp_path, monkeypatch)
    try:
        rpc, bound = rpc_files(tmp_path, owner, count=2, image=True)
        manifest = [item for item, _ in bound]
        image_digest = hashlib.sha256(bound[-1][1]).hexdigest()
        image = _media_root() / image_digest / (image_digest + '.png')
        if held:
            saved = prepare_hosted_input(rpc, request_id='control', prompt='keep', attachments=manifest)
            await owner.submit(rpc.principal, Submission('control', rpc.ref, saved.payload, 'queue'),
                _input_custody=saved.handle)
        def revoke(*args, **kwargs):
            result = prepare_hosted_input(*args, **kwargs)
            assert image.exists(), 'exercise actual canonical capture'
            rpc.authorizer = lambda *args: False
            return result
        monkeypatch.setattr('gateway.hosted_room_input_preparation.prepare_hosted_input', revoke)
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            await rpc._submit(dict(task=TaskIdentity('room', 'task', 'thread', 'turn'), execution_generation=1,
                prompt='read', attachments=manifest, on_terminal=lambda value: None))
        assert db._conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == int(held)
        db._execute_write(lambda conn: conn.execute('UPDATE input_custody_preparations SET expires_at=0'))
        collect_working_copies(db, epoch=owner.epoch)
        # Explicit exclusive pre-ingress opportunity, never online housekeeping.
        collect_native_inputs(db, epoch=owner.epoch)
        assert image.exists() is held
    finally:
        close(db, tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize('runtime', ['starting', 'serving', 'unknown', 'unavailable'])
async def test_owner_startup_reclaims_refused_images_only_before_ingress(tmp_path, monkeypatch, runtime):
    from types import SimpleNamespace
    from gateway import hosted_room_input_reclamation as reclamation
    from gateway.session_authority import initialize_session_authority
    from gateway.session_cron import unbind_owner

    db, owner = owned(tmp_path, monkeypatch)
    successor = None
    try:
        rpc, bound = rpc_files(tmp_path, owner, count=2, image=True)
        prepared = prepare_hosted_input(rpc, request_id='refused', prompt='read',
                                        attachments=[item for item, _ in bound])
        image_digest = hashlib.sha256(bound[-1][1]).hexdigest()
        image = _media_root() / image_digest / (image_digest + '.png')
        assert image.exists()
        expire(db, prepared.handle)  # Never admitted: no admission settlement can release it.
        if runtime == 'unavailable':
            def unavailable(*args, **kwargs):
                raise RuntimeStoreError('storage_unavailable')
            monkeypatch.setattr(reclamation, '_collect', unavailable)
        state = dict(session_runtime_descriptor={'state': 'starting'},
            _running=runtime == 'serving', _draining=False, adapters={}, _profile_adapters={},
            session_api=None, session_control_server=None, session_store=SimpleNamespace())
        if runtime == 'unknown':
            del state['_running']  # A runner that cannot prove it is not serving yet.
        runner = SimpleNamespace(**state)
        successor = await initialize_session_authority(runner, profile_id=str(tmp_path.resolve()),
            instance_id='next', db=db)
        # Collection is never a startup precondition, and it never runs once input is live.
        assert runner.session_authority is successor and successor.epoch == owner.epoch + 1
        assert image.exists() is (runtime != 'starting')
        assert v3_path(prepared).exists()  # Documents belong to the online housekeeping chore.
    finally:
        if successor is not None:
            unbind_owner(successor)
        close(db, tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize('damaged_proof', ['absent', 'foreign'])
async def test_interrupted_consumed_native_image_waits_for_exact_raw_retirement(tmp_path, monkeypatch, damaged_proof):
    from gateway.session_ingress_media import release_admission_media
    from hermes_state_mutation_retirement import retire_prunable
    from hermes_state_terminal import ADMISSION_PREFIX

    db, owner = owned(tmp_path, monkeypatch)
    try:
        rpc, bound = rpc_files(tmp_path, owner, count=2, image=True)
        prepared = prepare_hosted_input(rpc, request_id='hosted:interrupted', prompt='read',
                                        attachments=[item for item, _ in bound])
        assert isinstance(prepared.handle, PreparedInputHandle)
        receipt = await owner.submit(rpc.principal, Submission('hosted:interrupted', rpc.ref,
            prepared.payload, 'queue'), _input_custody=prepared.handle)
        admission = get_session_admission(db, admission_id=receipt.admission_id)
        assert admission is not None
        reference, = admission['payload']['attachments_v1']['media']
        image = Path(reference['path'])
        claimed = claim_session_input(db, epoch=owner.epoch, session_id=rpc.ref.session_id)
        assert claimed is not None and claimed['admission_id'] == receipt.admission_id
        terminal = settle_session_input(db, epoch=owner.epoch, admission_id=receipt.admission_id,
            generation=claimed['generation'], outcome='interrupted')
        assert terminal['status'] == 'terminal' and terminal['outcome'] == 'interrupted'
        db._execute_write(lambda conn: conn.execute(
            'UPDATE input_custody_preparations SET expires_at=0 WHERE preparation_id=?',
            (prepared.handle.preparation_id,)))
        assert native_held(db)
        assert release_admission_media(db, receipt.admission_id) == 0
        assert collect_native_inputs(db, epoch=owner.epoch)['removed'] == 0
        assert image.read_bytes() == bound[-1][1]

        # Only the raw-retirement writer emits the proof that releases the image.
        assert db._execute_write(lambda conn: retire_prunable(conn, [rpc.ref.session_id])) == [rpc.ref.session_id]
        key = ADMISSION_PREFIX + receipt.admission_id
        with db._read_ctx() as conn:
            saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()[0]
        def damage(conn):
            if damaged_proof == 'absent':
                conn.execute('DELETE FROM state_meta WHERE key=?', (key,))
            else:
                conn.execute("UPDATE state_meta SET value=json_set(value,'$.principal_id','foreign') WHERE key=?", (key,))
        db._execute_write(damage)
        assert native_held(db)
        assert collect_native_inputs(db, epoch=owner.epoch)['removed'] == 0
        assert image.read_bytes() == bound[-1][1]
        db._execute_write(lambda conn: conn.execute('INSERT OR REPLACE INTO state_meta(key,value) VALUES(?,?)',
                                                   (key, saved)))
        assert not native_held(db)
        assert collect_native_inputs(db, epoch=owner.epoch)['removed'] == 1
        assert not image.exists()
    finally:
        close(db, tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize('uncertainty', ['live', 'foreign_admission'])
async def test_consumed_native_image_rejects_live_or_foreign_admission(tmp_path, monkeypatch, uncertainty):
    db, owner = owned(tmp_path, monkeypatch)
    try:
        rpc, bound = rpc_files(tmp_path, owner, count=2, image=True)
        prepared = prepare_hosted_input(rpc, request_id='hosted:uncertain', prompt='read',
                                        attachments=[item for item, _ in bound])
        assert isinstance(prepared.handle, PreparedInputHandle)
        receipt = await owner.submit(rpc.principal, Submission('hosted:uncertain', rpc.ref,
            prepared.payload, 'queue'), _input_custody=prepared.handle)
        row = get_session_admission(db, admission_id=receipt.admission_id)
        assert row is not None
        reference, = row['payload']['attachments_v1']['media']
        image = Path(reference['path'])
        if uncertainty == 'foreign_admission':
            claimed = claim_session_input(db, epoch=owner.epoch, session_id=rpc.ref.session_id)
            assert claimed is not None
            settle_session_input(db, epoch=owner.epoch, admission_id=receipt.admission_id,
                                 generation=claimed['generation'], outcome='completed')
            db._execute_write(lambda conn: conn.execute(
                "UPDATE session_admissions SET principal_id='foreign' WHERE admission_id=?",
                (receipt.admission_id,)))
        db._execute_write(lambda conn: conn.execute(
            'UPDATE input_custody_preparations SET expires_at=0 WHERE preparation_id=?',
            (prepared.handle.preparation_id,)))
        with db._read_ctx() as conn:
            copy = conn.execute("SELECT * FROM input_custody_copies WHERE namespace='native'").fetchone()
            assert copy is not None and copy_is_held(conn, copy, float('inf'))
        assert collect_native_inputs(db, epoch=owner.epoch)['removed'] == 0
        assert image.read_bytes() == bound[-1][1]
    finally:
        close(db, tmp_path)

@pytest.mark.asyncio
async def test_recreated_document_generation_can_replay_terminal_metadata_but_cannot_restart_it(tmp_path, monkeypatch):
    db, owner = owned(tmp_path, monkeypatch)
    try:
        rpc, bound = rpc_files(tmp_path, owner, named=True)
        params = dict(task=TaskIdentity('room', 'task', 'thread', 'turn'), execution_generation=1,
            prompt='read', attachments=[item for item, _ in bound], on_terminal=lambda value: None)
        first = await rpc._submit(params)
        executions = []
        async def execute(authority, ref, admission):
            executions.append(admission['admission_id'])
            return 'completed original turn'
        monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
        await owner._drain(rpc.ref)
        await asyncio.sleep(0)
        original = get_session_admission(db, admission_id=first['admission_id'])
        from types import SimpleNamespace
        path = v3_path(SimpleNamespace(payload=original['payload']))
        retire_metadata(db, original['admission_id'])
        from hermes_state_mutation_retirement import RETIRED_PREFIX
        db._execute_write(lambda conn: conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
            (RETIRED_PREFIX + rpc.ref.session_id, '{}')))
        assert collect_working_copies(db, epoch=owner.epoch)['removed'] == 1
        assert not path.exists()
        # Another request prepares the same name/digest as a new private generation.
        prepare_hosted_input(rpc, request_id='hosted:other', prompt='other', attachments=params['attachments'])
        with db._read_ctx() as conn:
            ref = conn.execute('SELECT generation FROM input_custody_refs WHERE admission_id=?',
                               (original['admission_id'],)).fetchone()
            copy = conn.execute("SELECT generation FROM input_custody_copies WHERE namespace='v3'").fetchone()
            assert copy['generation'] > ref['generation']
        assert path.read_bytes() == bound[0][1]
        # Terminal replay uses metadata only; it may not read the replacement bytes.
        def no_read(*args, **kwargs):
            pytest.fail('terminal replay must not consume a newer private generation')
        monkeypatch.setattr('gateway.hosted_room_input_preparation.verified_identity', no_read)
        rebuilt = reconstruct_attested_payload(db, 'read', params['attachments'],
            [hashlib.sha256(data).hexdigest() for _, data in bound], original)
        assert rebuilt == original['payload']
        replay = await owner.submit(rpc.principal, Submission(original['request_id'], rpc.ref, rebuilt, 'queue'))
        assert replay.admission_id == original['admission_id'] and replay.status == 'terminal'
        await owner._drain(rpc.ref)
        assert executions == [original['admission_id']]
        assert claim_session_input(db, epoch=owner.epoch, session_id=rpc.ref.session_id) is None
    finally:
        close(db, tmp_path)
