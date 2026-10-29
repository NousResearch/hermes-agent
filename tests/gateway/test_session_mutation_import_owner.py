"""The importer owns cold control in its receipt transaction. Hermes is single-user: the dashboard
login is the machine owner, so it continues the owner's import (an exact retry), never a stranger."""
import httpx
import pytest
from tests.gateway.test_native_http_auth import daemon, ticket, headers


@pytest.mark.platforms("linux")
def test_import_atomically_binds_each_new_history_to_its_importer(tmp_path):
    with daemon(tmp_path) as (home, descriptor), httpx.Client(base_url=descriptor['api_origin'], trust_env=False) as client:
        response = client.post('/api/sessions/import', headers=headers(ticket(home, descriptor)), json={
            'request_id': 'owned-import', 'expected_revision': 0, 'sessions': [
                {'id': 'cli-import', 'source': 'cli'}, {'id': 'foreign-import', 'source': 'telegram'}]})
        assert response.status_code == 200, response.text
        for sid in ('cli-import', 'foreign-import'):
            body = {'request_id': 'edit', 'expected_revision': 1, 'title': sid}
            login = client.patch('/api/sessions/' + sid, headers={'Authorization': 'Bearer normal-http-owner'}, json=body)
            assert login.status_code == 200, login.text
            owned = client.patch('/api/sessions/' + sid, headers=headers(ticket(home, descriptor)), json=body)
            assert owned.status_code == 200 and owned.json() == login.json(), owned.text
