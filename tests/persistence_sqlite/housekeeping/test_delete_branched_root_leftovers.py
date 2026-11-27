"""Delete of a compressed + branched root on a SQLite backend store (NF4): the /branch child survives as its own conversation,
its ``_branched_from`` naming the deleted segment, and the delete's tombstones age out after the retry
horizon. The retired-session marker does not, so that name can never be issued to a new session the
surviving branch would then claim as its parent."""
import time

import pytest

from gateway.session_results import retain_result
from persistence.core.errors import AuthorityRefusal
from persistence.sqlite.authority._admission_store import admit_session_input, claim_session_input
from persistence.sqlite.authority._mutation import mutate_runtime_session
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from persistence.sqlite.conversation._store.compression import publish_compression_child
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.retirement import RETIRED_PREFIX
from persistence.sqlite.conversation._store.sessions import create_session, get_session, list_sessions_rich
from persistence.sqlite.conversation._store.terminal import ADMISSION_PREFIX, IDENTITY_PREFIX, RESULT_PREFIX
from tests.contracts._scopes import MEDIA_ROOT


def _keys(db, prefix):
    with db._read_ctx() as conn:
        return [row[0] for row in conn.execute('SELECT key FROM state_meta WHERE key GLOB ?', (prefix + '*',))]


def test_a_deleted_compressed_root_leaves_its_branch_and_no_reusable_id(backend, scope_a):
    store = backend.open(scope_a)
    db = store.holder()
    epoch = begin_runtime_epoch(db, instance_id='owner')
    create_session(db, 'root', source='cli')
    admit_session_input(db, epoch=epoch, principal_id='owner', session_id='root', request_id='r1',
                        payload={'text': 'turn'})
    row = claim_session_input(db, epoch=epoch, session_id='root')
    retain_result(store, epoch=epoch, row=row, result={'result': {'final_response': 'reply'}, 'usage': {}})
    append_message(db, 'root', 'user', 'turn')
    assert store.coordination.try_acquire_compression_lock(session_id='root', holder='holder')
    publish_compression_child(db, parent_session_id='root', child_session_id='tip', source='cli',
                              messages=[{'role': 'user', 'content': 'summary'}], compression_lock_holder='holder')
    store.coordination.release_compression_lock(session_id='root', holder='holder')
    create_session(db, 'branch', source='cli', parent_session_id='tip', model_config={'_branched_from': 'tip'})
    append_message(db, 'branch', 'user', 'copied turn')
    snap, at = get_session(db, 'root'), time.time() - 8 * 86400
    with pytest.MonkeyPatch.context() as clock:
        clock.setattr(time, 'time', lambda: at)
        receipt = mutate_runtime_session(db, epoch=epoch, principal_id='human', session_id='root',
                                         request_id='del', operation='delete', payload={},
                                         expected_revision=snap['runtime_revision'],
                                         expected_generation=snap['runtime_generation'], media_root=MEDIA_ROOT)[0]
    assert set(receipt['deleted_ids']) == {'root', 'tip'}
    store.maintenance.prune_sessions(older_than_days=90, media_root=MEDIA_ROOT)

    branch = get_session(db, 'branch')
    assert branch['parent_session_id'] is None and '"_branched_from": "tip"' in branch['model_config']
    assert any(s['id'] == 'branch' for s in list_sessions_rich(db, limit=50))
    assert [k for p in (ADMISSION_PREFIX, IDENTITY_PREFIX, RESULT_PREFIX) for k in _keys(db, p)] == []
    assert sorted(_keys(db, RETIRED_PREFIX)) == [RETIRED_PREFIX + 'root', RETIRED_PREFIX + 'tip']
    for sid in ('root', 'tip'):
        with pytest.raises(AuthorityRefusal, match='not_found'):
            create_session(db, sid, source='unknown')  # a late backfill naming the deleted id
        assert get_session(db, sid) is None
