"""Deleting a logical local owner retires its current physical transcript too."""
import asyncio
import json
import sqlite3
import httpx
import pytest
from tests.gateway.test_native_http_auth import daemon, ticket, headers
from tests.gateway.test_session_mutation_reset import reset_round


@pytest.mark.linux_only
def test_delete_after_reset_covers_physical_route_and_cache_generation(tmp_path):
    with daemon(tmp_path) as (home, descriptor), httpx.Client(base_url=descriptor['api_origin'], trust_env=False) as client:
        params, reset = asyncio.run(reset_round(home, descriptor))
        sid = params['session_id']
        target = reset['target_session_id']
        with sqlite3.connect(home / 'state.db') as db:
            route, source = db.execute('SELECT session_key,source FROM sessions WHERE id=?', (target,)).fetchone()
            before = db.execute('SELECT generation FROM conversation_generations WHERE source=? AND session_key=?', (source, route)).fetchone()[0]
        response = client.delete('/api/sessions/' + sid, params={'request_id': 'delete-reset',
            'expected_revision': reset['revision'], 'expected_generation': reset['execution_generation']}, headers=headers(ticket(home, descriptor)))
        assert response.status_code == 200, response.text
        assert set(response.json()['deleted_ids']) == {sid, target}
        with sqlite3.connect(home / 'state.db') as db:
            assert db.execute('SELECT COUNT(*) FROM sessions WHERE id IN (?,?)', (sid, target)).fetchone()[0] == 0
            assert not any(json.loads(row[0])['session_id'] in {sid,target} for row in db.execute('SELECT entry_json FROM gateway_routing'))
            assert db.execute('SELECT generation FROM conversation_generations WHERE source=? AND session_key=?', (source, route)).fetchone()[0] > before
