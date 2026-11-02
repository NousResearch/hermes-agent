"""Delete of a compressed + branched root (NF4): the /branch child survives as its own conversation,
its ``_branched_from`` naming the deleted segment, and the delete's tombstones age out after the retry
horizon. The retired-session marker does not, so that name can never be issued to a new session the
surviving branch would then claim as its parent."""
import time

import pytest

import hermes_state_runtime as rt
from gateway.session_results import retain_result
from hermes_state import SessionDB


def _keys(db, prefix):
    with db._read_ctx() as conn:
        return [row[0] for row in conn.execute('SELECT key FROM state_meta WHERE key GLOB ?', (prefix + '*',))]


def test_a_deleted_compressed_root_leaves_its_branch_and_no_reusable_id(tmp_path):
    from hermes_state_mutation_retirement import RETIRED_PREFIX
    from hermes_state_terminal import ADMISSION_PREFIX, IDENTITY_PREFIX, RESULT_PREFIX
    with SessionDB(tmp_path / 'state.db') as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        db.create_session('root', source='cli')
        rt.admit_session_input(db, epoch=epoch, principal_id='owner', session_id='root', request_id='r1',
                               payload={'text': 'turn'})
        row = rt.claim_session_input(db, epoch=epoch, session_id='root')
        retain_result(db, epoch=epoch, row=row, result={'result': {'final_response': 'reply'}, 'usage': {}})
        db.append_message('root', 'user', 'turn')
        assert db.try_acquire_compression_lock('root', 'holder')
        db.publish_compression_child(parent_session_id='root', child_session_id='tip', source='cli',
                                     messages=[{'role': 'user', 'content': 'summary'}], compression_lock_holder='holder')
        db.release_compression_lock('root', 'holder')
        db.create_session('branch', source='cli', parent_session_id='tip', model_config={'_branched_from': 'tip'})
        db.append_message('branch', 'user', 'copied turn')
        snap, at = db.get_session('root'), time.time() - 8 * 86400
        with pytest.MonkeyPatch.context() as clock:
            clock.setattr(time, 'time', lambda: at)
            receipt = rt.mutate_runtime_session(db, epoch=epoch, principal_id='human', session_id='root',
                                                request_id='del', operation='delete', payload={},
                                                expected_revision=snap['runtime_revision'],
                                                expected_generation=snap['runtime_generation'])
        assert set(receipt['deleted_ids']) == {'root', 'tip'}
        db.prune_sessions(older_than_days=90)

        branch = db.get_session('branch')
        assert branch['parent_session_id'] is None and '"_branched_from": "tip"' in branch['model_config']
        assert any(s['id'] == 'branch' for s in db.list_sessions_rich(limit=50))
        assert [k for p in (ADMISSION_PREFIX, IDENTITY_PREFIX, RESULT_PREFIX) for k in _keys(db, p)] == []
        assert sorted(_keys(db, RETIRED_PREFIX)) == [RETIRED_PREFIX + 'root', RETIRED_PREFIX + 'tip']
        for sid in ('root', 'tip'):
            with pytest.raises(rt.RuntimeStoreError, match='not_found'):
                db.ensure_session(sid, source='unknown')  # a late backfill naming the deleted id
            assert db.get_session(sid) is None
