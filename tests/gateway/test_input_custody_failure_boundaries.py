"""Failure and uncertainty controls for input custody lifetimes."""
import hashlib
from pathlib import Path

import pytest

from gateway.hosted_room_input_preparation import prepare_hosted_input, reconstruct_accepted_payload
from gateway.hosted_room_input_reclamation import collect_native_inputs, collect_working_copies
from gateway.session_contract import Submission
from gateway.session_ingress_media import _media_root
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.input_reclamation_fixtures import owned, close, rpc_files, expire


@pytest.mark.asyncio
@pytest.mark.parametrize('boundary', ['callback', 'partial_publish', 'rollback', 'expired', 'live_preparation'])
async def test_mixed_failed_capture_keeps_lease_until_safe_collection(tmp_path, monkeypatch, boundary):
    from gateway import session_ingress_media as native
    db, owner = owned(tmp_path, monkeypatch)
    try:
        rpc, bound = rpc_files(tmp_path, owner, count=2, image=True)
        args = dict(request_id='hosted:failed', prompt='read', attachments=[item for item, _ in bound])
        digest = hashlib.sha256(bound[-1][1]).hexdigest()
        image = _media_root() / digest / (digest + '.png')
        if boundary in {'callback', 'partial_publish'}:
            with monkeypatch.context() as patch:
                if boundary == 'callback':
                    from gateway.session_submission_payload import normalize_submission_payload
                    def fail_callback(*args, **kwargs):
                        normalize_submission_payload(*args, **kwargs)
                        assert image.exists()
                        raise RuntimeError('fixture callback')
                    patch.setattr('gateway.session_submission_payload.normalize_submission_payload', fail_callback)
                else:
                    sync = native._sync_directory
                    def fail_after_publish(path):
                        sync(path)
                        if image.exists():
                            raise RuntimeError('fixture partial publication')
                    patch.setattr(native, '_sync_directory', fail_after_publish)
                with pytest.raises(RuntimeError, match='fixture'):
                    prepare_hosted_input(rpc, **args)
        else:
            prepared = prepare_hosted_input(rpc, **args)
            if boundary == 'live_preparation':
                another = prepare_hosted_input(rpc, **(args | {'request_id': 'another'}))
                expire(db, prepared.handle)
                collect_native_inputs(db, epoch=owner.epoch)
                assert image.exists()
                expire(db, another.handle)
            else:
                with monkeypatch.context() as patch:
                    if boundary == 'rollback':
                        from hermes_state_input_custody import accept_prepared_input
                        def rollback(*args, **kwargs):
                            accept_prepared_input(*args, **kwargs)
                            raise RuntimeStoreError('fixture rollback')
                        patch.setattr('hermes_state_input_custody.accept_prepared_input', rollback)
                    else:
                        expire(db, prepared.handle)
                    with pytest.raises(RuntimeStoreError, match='fixture rollback|input_preparation_expired'):
                        await owner.submit(rpc.principal, Submission(args['request_id'], rpc.ref, prepared.payload, 'queue'),
                            _input_custody=prepared.handle)
        assert image.exists()
        assert db._conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 0
        assert db._conn.execute('SELECT count(*) FROM input_custody_refs').fetchone()[0] == 0
        db._execute_write(lambda conn: conn.execute('UPDATE input_custody_preparations SET expires_at=0'))
        # A new connection reads the committed intent after the failed publisher is gone.
        from hermes_state import SessionDB
        db.close()
        db = SessionDB(db_path=tmp_path / 'state.db')
        owner.db = db
        collect_working_copies(db, epoch=owner.epoch)
        for _ in range(4):
            assert collect_native_inputs(db, epoch=owner.epoch, limit=1)['scanned'] <= 1
        assert not image.exists()
    finally:
        close(db, tmp_path)


@pytest.mark.asyncio
async def test_caller_terminal_hint_cannot_skip_live_image_validation(tmp_path, monkeypatch):
    from hermes_state_runtime import get_session_admission
    db, owner = owned(tmp_path, monkeypatch)
    try:
        rpc, bound = rpc_files(tmp_path, owner, image=True)
        manifest = [item for item, _ in bound]
        prepared = prepare_hosted_input(rpc, request_id='hosted:live', prompt='read', attachments=manifest)
        receipt = await owner.submit(rpc.principal, Submission('hosted:live', rpc.ref, prepared.payload, 'queue'))
        row = get_session_admission(db, admission_id=receipt.admission_id)
        Path(row['payload']['attachments_v1']['media'][0]['path']).write_bytes(b'corrupt')
        with pytest.raises(RuntimeStoreError, match='storage_unavailable'):
            reconstruct_accepted_payload(rpc, 'read', manifest, {**row, 'status': 'terminal'}, retired=True)
    finally:
        close(db, tmp_path)


@pytest.mark.asyncio
async def test_native_preparation_transfers_branch_custody_before_terminal_release(tmp_path, monkeypatch):
    from hermes_state_input_custody import copy_branch_input_refs
    from hermes_state_mutation_retirement import RETIRED_PREFIX
    db, owner = owned(tmp_path, monkeypatch)
    try:
        rpc, bound = rpc_files(tmp_path, owner, count=2, image=True)
        prepared = prepare_hosted_input(rpc, request_id='hosted:branch', prompt='read',
            attachments=[item for item, _ in bound])
        await owner.submit(rpc.principal, Submission('hosted:branch', rpc.ref, prepared.payload, 'queue'),
            _input_custody=prepared.handle)
        digest = hashlib.sha256(bound[-1][1]).hexdigest()
        image = _media_root() / digest / (digest + '.png')
        db.create_session('branch', source='cli', model_config={'_branched_from': rpc.ref.session_id})
        db._execute_write(lambda conn: copy_branch_input_refs(conn, rpc.ref.session_id, 'branch'))
        async def inert(*args):
            return 'done'
        monkeypatch.setattr('gateway.session_finite.execute_finite_admission', inert)
        await owner._drain(rpc.ref)
        assert image.exists(), 'branch ownership must survive original settlement'
        collect_native_inputs(db, epoch=owner.epoch)
        assert image.exists()
        def retired_branch(conn):
            conn.execute('DELETE FROM sessions WHERE id=?', ('branch',))
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (RETIRED_PREFIX + 'branch', '{}'))
        db._execute_write(retired_branch)
        collect_native_inputs(db, epoch=owner.epoch)
        assert not image.exists()
    finally:
        close(db, tmp_path)
