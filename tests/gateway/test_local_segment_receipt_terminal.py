"""A compression segment id gets a terminal refusal, not a retryable one.

The receipt lives on the lineage root, so a client that holds a segment id would otherwise poll
``storage_unavailable`` forever.
"""
import json
from types import SimpleNamespace

import pytest


def _db(tmp_path):
    from hermes_state import SessionDB
    from hermes_state_local import POLICY_PREFIX
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('root', source='gui')
    db._write_sql('INSERT INTO state_meta(key,value) VALUES(?,?)',
                  (POLICY_PREFIX + 'root', json.dumps({'entry': {'session_id': 'root'}})))
    db.end_session('root', 'compression')
    db.create_session('tip', source='gui', parent_session_id='root')
    return db


@pytest.mark.parametrize('reader', ['owned_receipt', 'restore_local_session'])
@pytest.mark.parametrize('case', ['segment', 'unknown'])
def test_receipt_refusal_terminal_only_for_a_segment(tmp_path, reader, case):
    from gateway import session_local_recovery
    from hermes_state_runtime import RuntimeStoreError
    with _db(tmp_path) as db:
        sid = 'tip' if case == 'segment' else 'never-existed'
        authority = SimpleNamespace(db=db)
        with pytest.raises(RuntimeStoreError) as refused:
            getattr(session_local_recovery, reader)(authority, sid)
        assert refused.value.reason == ('not_found' if case == 'segment' else 'storage_unavailable')
