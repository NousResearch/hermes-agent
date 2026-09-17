"""Cold history authorization does not manufacture an execution route."""
import asyncio
import httpx
import pytest
from tests.gateway.test_native_http_auth import daemon, ticket, headers
from tests.gateway.test_session_mutation_retirement import create


@pytest.mark.linux_only
def test_cold_import_mutation_binds_owner_across_restart(tmp_path):
    with daemon(tmp_path) as (home, descriptor), httpx.Client(base_url=descriptor['api_origin'], trust_env=False) as client:
        # Native creation remains independently authorized while cold history binds.
        asyncio.run(create(home, descriptor))
        response = client.post('/api/sessions/import', headers=headers(ticket(home, descriptor)), json={
            'request_id': 'import', 'expected_revision': 0, 'sessions': [
                {'id': 'cold', 'source': 'cli', 'messages': [{'role': 'user', 'content': 'history'}]}]})
        assert response.status_code == 200, response.text
        snapshot = client.get('/api/sessions/cold/mutation-snapshot', headers=headers(ticket(home, descriptor)))
        assert snapshot.status_code == 200, snapshot.text
        body = {'request_id': 'edit', 'expected_revision': snapshot.json()['runtime_revision'], 'title': 'Cold history'}
        edited = client.patch('/api/sessions/cold', headers=headers(ticket(home, descriptor)), json=body)
        assert edited.status_code == 200, edited.text
        stale = client.patch('/api/sessions/cold', headers=headers(ticket(home, descriptor)), json=body | {'request_id': 'competitor', 'title': 'stale'})
        assert stale.status_code == 409 and 'revision_conflict' in stale.text
    with daemon(tmp_path) as (home, descriptor), httpx.Client(base_url=descriptor['api_origin'], trust_env=False) as client:
        assert client.patch('/api/sessions/cold', headers=headers(ticket(home, descriptor)), json=body).json() == edited.json()
        assert client.get('/api/sessions/cold/mutation-snapshot', headers=headers(ticket(home, descriptor))).status_code == 200
        denied = client.patch('/api/sessions/cold', headers={'Authorization': 'Bearer normal-http-owner'}, json=body | {'request_id': 'foreign'})
        assert denied.status_code == 403, denied.text
