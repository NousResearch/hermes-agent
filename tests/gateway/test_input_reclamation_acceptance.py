"""Typed custody effects are atomic with canonical admission and exact retries."""
import asyncio
import os

import pytest

from gateway.hosted_room_driver import TaskIdentity
from gateway.hosted_room_input_preparation import prepare_hosted_input
from gateway.hosted_room_input_reclamation import collect_working_copies
from gateway.session_contract import Submission
from gateway.session_ingress_media import capture_native_media, release_admission_media
from hermes_state_runtime import RuntimeStoreError, get_session_admission
from tests.gateway.input_reclamation_fixtures import owned, close, rpc_files, retire_metadata, expire, v3_path
from tests.gateway.test_input_custody_migration import candidate


@pytest.mark.asyncio
@pytest.mark.parametrize('named', [False, True])
async def test_new_rpc_has_independent_v3_bytes_and_readonly_accepted_retry(tmp_path, monkeypatch, named):
    home = tmp_path / 'profiles' / 'member' if named else tmp_path
    home.mkdir(parents=True, exist_ok=True)
    db, owner = owned(home, monkeypatch)
    try:
        rpc, bound = rpc_files(home, owner, named=named)
        native, alias = candidate(home, owner, '0.txt', b'A' * 32)
        params = dict(task=TaskIdentity('room', 'task', 'thread', 'turn'), execution_generation=1,
            prompt='read', attachments=[item for item, _ in bound], on_terminal=lambda value: None)
        receipt = await rpc._submit(params)
        row = get_session_admission(db, admission_id=receipt['admission_id'])
        path = v3_path(type('Prepared', (), {'payload': row['payload']}))
        assert 'working-documents-v3' in str(path) and not os.path.samefile(path, alias)
        assert release_admission_media(db, native['admission_id']) == 1
        assert path.read_bytes() == b'A' * 32
        captured = capture_native_media([path])[0]
        assert not os.path.samefile(path, captured['path'])
        def no_writes(*args, **kwargs):
            raise AssertionError('accepted retry must not publish copies')
        monkeypatch.setattr('gateway.hosted_room_input_preparation._copy', no_writes)
        assert (await rpc._submit(params))['admission_id'] == receipt['admission_id']
        assert get_session_admission(db, admission_id=receipt['admission_id'])['payload'] == row['payload']
        assert collect_working_copies(db, epoch=owner.epoch)['removed'] == 0
    finally:
        close(db, home)


@pytest.mark.asyncio
async def test_lease_expiry_racing_prepares_and_atomic_failure(tmp_path, monkeypatch):
    db, owner = owned(tmp_path, monkeypatch)
    try:
        rpc, bound = rpc_files(tmp_path, owner)
        args = dict(request_id='hosted:fixed', prompt='read', attachments=[item for item, _ in bound])
        first = prepare_hosted_input(rpc, **args)
        expire(db, first.handle)
        with pytest.raises(RuntimeStoreError, match='input_preparation_expired'):
            await owner.submit(rpc.principal, Submission(args['request_id'], rpc.ref, first.payload, 'queue'), _input_custody=first.handle)
        second, third = await asyncio.gather(*(asyncio.to_thread(prepare_hosted_input, rpc, **args) for _ in range(2)))
        assert second.payload == third.payload == first.payload
        assert second.handle.token != third.handle.token
        from hermes_state_input_custody import accept_prepared_input
        def fail_after_refs(*args, **kwargs):
            accept_prepared_input(*args, **kwargs)
            raise RuntimeStoreError('fixture_rollback')
        with monkeypatch.context() as patch:
            patch.setattr('hermes_state_input_custody.accept_prepared_input', fail_after_refs)
            with pytest.raises(RuntimeStoreError, match='fixture_rollback'):
                await owner.submit(rpc.principal, Submission(args['request_id'], rpc.ref, second.payload, 'queue'), _input_custody=second.handle)
        with db._read_ctx() as conn:
            assert conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 0
            assert conn.execute('SELECT count(*) FROM input_custody_refs').fetchone()[0] == 0
        receipts = [await owner.submit(rpc.principal, Submission(args['request_id'], rpc.ref, prepared.payload, 'queue'),
            _input_custody=prepared.handle) for prepared in (second, third)]
        assert receipts[0].admission_id == receipts[1].admission_id
        expire(db, third.handle)
        assert collect_working_copies(db, epoch=owner.epoch)['removed'] == 0
        retire_metadata(db, receipts[0].admission_id)
        assert collect_working_copies(db, epoch=owner.epoch)['removed'] == 1
        assert not v3_path(second).exists()
        retired = prepare_hosted_input(rpc, **args)
        assert retired.handle.admission_id == receipts[0].admission_id
        assert not v3_path(second).exists()
    finally:
        close(db, tmp_path)


@pytest.mark.asyncio
async def test_authorization_after_background_preparation_refuses_acceptance(tmp_path, monkeypatch):
    db, owner = owned(tmp_path, monkeypatch)
    try:
        rpc, bound = rpc_files(tmp_path, owner)
        def revoke_after_copy(*args, **kwargs):
            result = prepare_hosted_input(*args, **kwargs)
            rpc.authorizer = lambda *args: False
            return result
        monkeypatch.setattr('gateway.hosted_room_input_preparation.prepare_hosted_input', revoke_after_copy)
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            await rpc._submit(dict(task=TaskIdentity('room', 'task', 'thread', 'turn'), execution_generation=1,
                prompt='read', attachments=[item for item, _ in bound], on_terminal=lambda value: None))
        with db._read_ctx() as conn:
            assert conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 0
            assert conn.execute('SELECT count(*) FROM input_custody_refs').fetchone()[0] == 0
        db._execute_write(lambda conn: conn.execute('UPDATE input_custody_preparations SET expires_at=0'))
        assert collect_working_copies(db, epoch=owner.epoch)['removed'] == 1
    finally:
        close(db, tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize('named', [False, True])
async def test_mixed_inputs_reconstruct_from_exact_refs_without_staging(tmp_path, monkeypatch, named):
    from gateway.hosted_room_input_preparation import reconstruct_accepted_payload
    db, owner = owned(tmp_path, monkeypatch)
    try:
        rpc, bound = rpc_files(tmp_path, owner, named=named, count=2, image=True)
        prepared = prepare_hosted_input(rpc, request_id='hosted:mixed', prompt='read', attachments=[item for item, _ in bound])
        receipt = await owner.submit(rpc.principal, Submission('hosted:mixed', rpc.ref, prepared.payload, 'queue'),
                                     _input_custody=prepared.handle)
        row = get_session_admission(db, admission_id=receipt.admission_id)
        assert row['payload']['attachments_v1']['media_types'] == ['image/png']
        def no_capture(*args, **kwargs):
            raise AssertionError('preclaim and exact retries must be read-only')
        monkeypatch.setattr('gateway.session_ingress_media.capture_native_media', no_capture)
        monkeypatch.setattr('gateway.hosted_room_input_preparation._copy', no_capture)
        assert reconstruct_accepted_payload(rpc, 'read', [item for item, _ in bound], row) == row['payload']
        again = prepare_hosted_input(rpc, request_id='hosted:mixed', prompt='read', attachments=[item for item, _ in bound])
        assert (await owner.submit(rpc.principal, Submission('hosted:mixed', rpc.ref, again.payload, 'queue'),
                                   _input_custody=again.handle)).admission_id == receipt.admission_id
    finally:
        close(db, tmp_path)
