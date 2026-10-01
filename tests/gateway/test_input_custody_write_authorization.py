"""Custody and authorization share the accepting writer, including exact replay."""
import pytest

from gateway.hosted_room_driver import TaskIdentity
from gateway.hosted_room_input_reclamation import collect_working_copies
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.input_reclamation_fixtures import close, owned, rpc_files


@pytest.mark.asyncio
@pytest.mark.parametrize('revoked', [False, True])
async def test_revocation_after_payload_normalization_fences_custody(tmp_path, monkeypatch, revoked):
    from gateway import session_submission_payload

    db, owner = owned(tmp_path, monkeypatch)
    try:
        rpc, bound = rpc_files(tmp_path, owner)
        db._execute_write(lambda conn: conn.execute(
            "INSERT INTO state_meta(key,value) VALUES('custody-authorized','yes')"))
        original = session_submission_payload.normalize_submission_payload
        normalizations = []

        def normalize(*args, **kwargs):
            payload = original(*args, **kwargs)
            normalizations.append(payload)
            # The first pass prepares bytes in the background; the second happens
            # in submit after the RPC's post-preparation authorization check.
            if revoked and len(normalizations) == 2:
                db._execute_write(lambda conn: conn.execute(
                    "DELETE FROM state_meta WHERE key='custody-authorized'"))
            return payload

        calls = []

        def authorize_write(conn, task, generation):
            assert conn.in_transaction
            assert task.task_id == 'task' and generation == 1
            calls.append(task)
            if conn.execute("SELECT 1 FROM state_meta WHERE key='custody-authorized'").fetchone() is None:
                raise RuntimeStoreError('permission_denied')

        rpc.authorize_write = authorize_write
        monkeypatch.setattr(session_submission_payload, 'normalize_submission_payload', normalize)
        params = dict(task=TaskIdentity('room', 'task', 'thread', 'turn'), execution_generation=1,
                      prompt='read', attachments=[item for item, _ in bound], on_terminal=lambda value: None)
        if revoked:
            with pytest.raises(RuntimeStoreError, match='permission_denied'):
                await rpc._submit(params)
            with db._read_ctx() as conn:
                assert conn.execute('SELECT COUNT(*) FROM session_admissions').fetchone()[0] == 0
                assert conn.execute('SELECT COUNT(*) FROM input_custody_refs').fetchone()[0] == 0
            db._execute_write(lambda conn: conn.execute('UPDATE input_custody_preparations SET expires_at=0'))
            assert collect_working_copies(db, epoch=owner.epoch)['removed'] == 1
        else:
            receipt = await rpc._submit(params)
            db._execute_write(lambda conn: conn.execute(
                "DELETE FROM state_meta WHERE key='custody-authorized'"))
            replay = await rpc._submit(params)
            assert replay['admission_id'] == receipt['admission_id']
            with db._read_ctx() as conn:
                assert conn.execute('SELECT COUNT(*) FROM session_admissions').fetchone()[0] == 1
                assert conn.execute('SELECT COUNT(*) FROM input_custody_refs').fetchone()[0] == 1
        assert len(calls) == 1
        assert len(normalizations) == 2
    finally:
        close(db, tmp_path)
