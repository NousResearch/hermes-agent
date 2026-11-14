"""Deleted local receipts are terminal refusals, not retryable storage failures (#155582)."""
import json
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('reader', ['owned_receipt', 'restore_local_session'])
@pytest.mark.parametrize('deleted', [True, False], ids=['retired', 'missing-live-receipt'])
def test_local_receipt_refusal_distinguishes_retirement(tmp_path, reader, deleted):
    from gateway import session_local_recovery
    from hermes_state import SessionDB
    from hermes_state_local import POLICY_PREFIX
    from hermes_state_mutation_retirement import retired_session
    from hermes_state_runtime import RuntimeStoreError

    sid = 'local-deleted-conversation'
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session(sid, source='gui')
        # A deletion removes the private creation receipt along with the transcript.
        db._write_sql('INSERT INTO state_meta(key,value) VALUES(?,?)',
                      (POLICY_PREFIX + sid, json.dumps({'entry': {'session_id': sid}})))
        if deleted:
            assert db.delete_session(sid)
            assert retired_session(db, sid)
        else:
            db._write_sql('DELETE FROM state_meta WHERE key=?', (POLICY_PREFIX + sid,))
        authority = SimpleNamespace(db=db)
        with pytest.raises(RuntimeStoreError) as refused:
            getattr(session_local_recovery, reader)(authority, sid)
        assert refused.value.reason == ('not_found' if deleted else 'storage_unavailable')
