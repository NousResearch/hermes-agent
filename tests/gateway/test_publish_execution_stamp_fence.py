"""publish_execution (per token / tool event, under event_stream.lock) fences on the claim's execution
stamp, not a per-delta ledger read (get_session + pending-admission decode) (B1-11)."""
import asyncio

import pytest

from tests.gateway.test_session_authority_cancel_settlement import _authority, _submit


@pytest.mark.asyncio
async def test_publish_execution_reads_no_ledger_per_delta_and_still_fences_stale_generations(tmp_path, monkeypatch):
    from gateway import session_finite

    db, authority = _authority(tmp_path, monkeypatch)
    published = {}

    async def execute(authority, ref, row):
        generation = row['generation']
        reads = []
        real_get, real_list = authority.db.get_session, authority._pending_rows
        authority.db.get_session = lambda *a, **k: reads.append('get_session') or real_get(*a, **k)
        authority._pending_rows = lambda *a, **k: reads.append('pending') or real_list(*a, **k)
        try:
            published['live'] = [authority.publish_execution('s', generation, 'message.delta', {'text': str(i)})
                                 for i in range(50)]
            published['stale'] = authority.publish_execution('s', generation - 1, 'message.delta', {'text': 'x'})
        finally:
            authority.db.get_session, authority._pending_rows = real_get, real_list
        published['reads'] = reads
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    with db:
        await _submit(authority, 'turn')
        await asyncio.wait_for(authority.sessions['s'].task, 5)
        generation = db.get_session('s')['runtime_generation']
        after = authority.publish_execution('s', generation, 'message.delta', {'text': 'late'})
    assert published['reads'] == [], f"{len(published['reads'])} ledger reads for 51 deltas under the stream lock"
    assert published['stale'] is False and after is False, 'a callback outlived its exact running claim'
