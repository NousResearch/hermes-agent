"""A dedicated same-user grant creates profiles without widening profile HTTP grants."""
import asyncio

import httpx
import pytest

from tests.gateway.test_native_http_auth import daemon, headers, reject_ws, ticket


@pytest.mark.platforms('linux')
def test_native_profile_create_grant_is_route_bound_and_single_use(tmp_path):
    with daemon(tmp_path) as (home, descriptor), httpx.Client(
            base_url=descriptor['api_origin'], trust_env=False, timeout=20) as client:
        body = {'name': 'gui-created', 'clone_from': 'default', 'soul': 'A GUI-created persona'}
        assert client.post('/api/profiles', json=body,
            headers=headers(ticket(home, descriptor))).status_code == 403
        assert not (home / 'profiles' / 'gui-created').exists()
        assert client.get('/api/config?profile=other',
            headers=headers(ticket(home, descriptor))).status_code == 403
        grant = ticket(home, descriptor, 'native-profile-create')
        for method, path in [('GET', '/api/config'), ('GET', '/api/profiles'),
                             ('POST', '/api/profiles/import'), ('PATCH', '/api/profiles/other'),
                             ('DELETE', '/api/profiles/other')]:
            response = client.request(method, path, json=body, headers=headers(grant))
            assert response.status_code == 401, (method, path, response.text)
        asyncio.run(reject_ws(descriptor['api_origin'], grant))
        for origin in ('', 'https://evil.example'):
            response = client.post('/api/profiles', json=body, headers={**headers(grant), 'Origin': origin})
            assert response.status_code == 401
        created = client.post('/api/profiles', json=body, headers=headers(grant))
        assert created.status_code == 200, created.text
        assert created.json()['name'] == 'gui-created'
        assert (home / 'profiles' / 'gui-created' / 'config.yaml').is_file()
        assert (home / 'profiles' / 'gui-created' / 'SOUL.md').read_text() == body['soul']
        assert created.json()['soul_set'] is True
        listed = client.get('/api/profiles', headers=headers(ticket(home, descriptor)))
        assert 'gui-created' in {p['name'] for p in listed.json()['profiles']}
        assert client.post('/api/profiles', json={**body, 'name': 'replay'},
            headers=headers(grant)).status_code == 401
        assert not (home / 'profiles' / 'replay').exists()
