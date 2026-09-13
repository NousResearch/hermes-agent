"""Bounded seal/drain behavior through real SQLite and private temporary files."""
from pathlib import Path
import pytest

from gateway.hosted_room_input_preparation import prepare_hosted_input
from gateway.hosted_room_input_reclamation import collect_working_copies
from gateway.session_contract import Submission
from hermes_state_input_custody import copy_branch_input_refs
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.input_reclamation_fixtures import owned, close, rpc_files, expire, v3_path, copy_records, retire_metadata


@pytest.mark.asyncio
async def test_committed_seal_survives_failed_unlink_transaction(tmp_path, monkeypatch):
    db, owner = owned(tmp_path, monkeypatch)
    try:
        rpc, bound = rpc_files(tmp_path, owner)
        args = dict(request_id='hosted:prepare', prompt='read', attachments=[item for item, _ in bound])
        prepared = prepare_hosted_input(rpc, **args)
        path = v3_path(prepared)
        expire(db, prepared.handle)
        execute = db._execute_write
        calls = 0
        def fail_second_writer(fn, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                def fail_after_unlink(conn):
                    result = fn(conn)
                    assert not path.exists()
                    raise RuntimeStoreError('fixture_commit_refusal')
                return execute(fail_after_unlink, *args, **kwargs)
            return execute(fn, *args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(db, '_execute_write', fail_second_writer)
            assert collect_working_copies(db, epoch=owner.epoch)['removed'] == 0
        assert copy_records(db)[0]['state'] == 'sealed'
        assert not path.exists()
        with pytest.raises(RuntimeStoreError, match='input_preparation_busy'):
            prepare_hosted_input(rpc, **args)
        with pytest.raises(RuntimeStoreError, match='input_preparation_expired'):
            await owner.submit(rpc.principal, Submission(args['request_id'], rpc.ref, prepared.payload, 'queue'),
                               _input_custody=prepared.handle)
        assert collect_working_copies(db, epoch=owner.epoch)['removed'] == 1
        fresh = prepare_hosted_input(rpc, **args)
        assert fresh.payload == prepared.payload and v3_path(fresh).exists()
        assert fresh.handle.token != prepared.handle.token
    finally:
        close(db, tmp_path)


@pytest.mark.asyncio
async def test_bounded_cursor_passes_held_prefix_and_branch_refs_protect_copy(tmp_path, monkeypatch):
    db, owner = owned(tmp_path, monkeypatch)
    try:
        rpc, bound = rpc_files(tmp_path, owner, count=3)
        prepared = [prepare_hosted_input(rpc, request_id=f'hosted:{i}', prompt='read', attachments=[item])
                    for i, (item, _) in enumerate(bound)]
        with db._read_ctx() as conn:
            items = {r['copy_id']: r['preparation_id'] for r in conn.execute('SELECT * FROM input_custody_items')}
        ordered = copy_records(db)
        victim = next(p for p in prepared if p.handle.preparation_id == items[ordered[-1]['copy_id']])
        expire(db, victim.handle)
        results = [collect_working_copies(db, epoch=owner.epoch, limit=1) for _ in range(3)]
        assert all(r['scanned'] <= 1 for r in results)
        assert sum(r['removed'] for r in results) == 1
        assert not v3_path(victim).exists()
        survivor = next(p for p in prepared if p is not victim)
        request_id = next(f'hosted:{i}' for i, p in enumerate(prepared) if p is survivor)
        receipt = await owner.submit(rpc.principal, Submission(request_id, rpc.ref, survivor.payload, 'queue'),
                                     _input_custody=survivor.handle)
        db.create_session('branch', source='cli', model_config={'_branched_from': rpc.ref.session_id})
        db._execute_write(lambda conn: copy_branch_input_refs(conn, rpc.ref.session_id, 'branch'))
        retire_metadata(db, receipt.admission_id)
        assert collect_working_copies(db, epoch=owner.epoch)['removed'] == 0
        assert v3_path(survivor).exists()
    finally:
        close(db, tmp_path)


@pytest.mark.asyncio
async def test_expired_copy_honors_structured_api_holder_and_epoch(tmp_path, monkeypatch):
    from gateway.session_ingress_media import capture_native_media
    from hermes_state_runtime import admit_session_input
    db, owner = owned(tmp_path, monkeypatch)
    try:
        rpc, bound = rpc_files(tmp_path, owner)
        prepared = prepare_hosted_input(rpc, request_id='hosted:prepare', prompt='read', attachments=[bound[0][0]])
        path = v3_path(prepared)
        captured = capture_native_media([path])
        # Native capture is physically independent; its retained API reference
        # must survive while the exclusively owned abandoned v3 entry is removed.
        admit_session_input(db, epoch=owner.epoch, principal_id='api', session_id='s', request_id='api',
            payload={'text': 'api', 'api_turn_v1': {'media': captured}})
        expire(db, prepared.handle)
        with pytest.raises(RuntimeStoreError, match='stale_epoch'):
            collect_working_copies(db, epoch=owner.epoch + 1)
        assert path.exists()
        assert collect_working_copies(db, epoch=owner.epoch)['removed'] == 1
        assert Path(captured[0]['path']).read_bytes() == b'A' * 32
    finally:
        close(db, tmp_path)
