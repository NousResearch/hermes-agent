"""A compression segment id gets a terminal refusal, not a retryable one.

The receipt lives on the conversation's logical owner, which is not always the compression root
(a conversation adopted mid-chain keeps its receipt where it was adopted), so a client that holds a
later segment's id would otherwise poll ``storage_unavailable`` forever.
"""
import json
from types import SimpleNamespace

import pytest


def _db(tmp_path):
    """ancient -> owner (holds the receipt) -> mid -> tip."""
    from hermes_state import SessionDB
    from hermes_state_local import POLICY_PREFIX
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('ancient', source='gui')
    db.end_session('ancient', 'compression')
    db.create_session('owner', source='gui', parent_session_id='ancient')
    db._write_sql('INSERT INTO state_meta(key,value) VALUES(?,?)',
                  (POLICY_PREFIX + 'owner', json.dumps({'entry': {'session_id': 'tip'},
                                                        'lineage': ['owner', 'mid', 'tip']})))
    db.end_session('owner', 'compression')
    db.create_session('mid', source='gui', parent_session_id='owner')
    db.end_session('mid', 'compression')
    db.create_session('tip', source='gui', parent_session_id='mid')
    return db


@pytest.mark.parametrize('reader', ['owned_receipt', 'restore_local_session'])
@pytest.mark.parametrize('sid, reason', [
    ('mid', 'not_found'),
    ('tip', 'not_found'),
    ('ancient', 'storage_unavailable'),  # before the receipt's lineage: not listed, stays transient
    ('never-existed', 'storage_unavailable'),
])
def test_receipt_refusal_terminal_only_for_a_listed_segment(tmp_path, reader, sid, reason):
    from gateway import session_local_recovery
    from hermes_state_runtime import RuntimeStoreError
    with _db(tmp_path) as db:
        authority = SimpleNamespace(db=db)
        with pytest.raises(RuntimeStoreError) as refused:
            getattr(session_local_recovery, reader)(authority, sid)
        assert refused.value.reason == reason
