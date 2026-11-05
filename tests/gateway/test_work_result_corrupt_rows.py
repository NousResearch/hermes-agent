"""A malformed persisted identifier must not starve healthy obligations."""
import json

import pytest

from hermes_state_runtime import list_session_admissions
from tests.gateway.test_process_work_results import enrolled, owner
from tools import process_work_results as work


@pytest.mark.asyncio
@pytest.mark.parametrize('field', ['work_id', 'incarnation', 'process_id', 'result_id', 'route', 'parent'])
async def test_blob_identifier_preserved_without_starving_sibling(monkeypatch, field):
    runner, authority, key, sid = await owner(monkeypatch)
    registry, source = enrolled(key, sid, work=[
        {'work_id': 'bad', 'incarnation': '1'},
        {'work_id': 'good', 'incarnation': '1'},
    ])
    work.finish_without_result(source)
    authority.db._execute_write(lambda conn: conn.execute(
        'UPDATE process_work_results SET ' + field + "=? WHERE work_id='bad'", (b'\xff',)))
    for _ in range(2):
        await runner._drain_watch_notifications(registry.completion_queue)
    admissions = list_session_admissions(authority.db, session_id=sid, pending_only=False)
    assert len(admissions) == 2
    assert json.loads(admissions[-1]['payload']['text'].splitlines()[1])['work_id'] == 'good'
    corrupted = [row for row in work.records(authority.db) if isinstance(row[field], bytes)]
    assert len(corrupted) == 1
    assert corrupted[0]['disposition'] == 'supervision_failed'
    assert corrupted[0]['payload'] is None
