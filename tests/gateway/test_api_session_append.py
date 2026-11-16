"""Authenticated record-only HTTP writes use the same canonical owner as API runs."""
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


@pytest.mark.asyncio
async def test_batch_append_replay_validation_and_readback(api, owner, monkeypatch):
    monkeypatch.setattr('gateway.run._load_gateway_config', lambda: {'memory': {'nudge_interval': 0}})
    owner.db.create_session('speech', source='api_server')
    app = web.Application()
    app.router.add_post('/api/sessions/{session_id}/messages', api._handle_append_session_messages)
    app.router.add_get('/api/sessions/{session_id}/messages', api._handle_session_messages)
    app.router.add_get('/v1/capabilities', api._handle_capabilities)
    batch = {'messages': [
        {'client_id': 'a', 'role': 'user', 'content': 'First thought'},
        {'client_id': 'b', 'role': 'user', 'content': 'A correction'},
        {'client_id': 'c', 'role': 'assistant', 'content': 'What the user heard'}]}
    async with TestClient(TestServer(app)) as client:
        url = '/api/sessions/speech/messages'
        response = await client.post(url, json=batch)
        result = await response.json()
        assert response.status == 200, result
        assert result['object'] == 'hermes.session.messages'
        assert len({row['id'] for row in result['messages']}) == 3
        repeat = await client.post(url, json=batch)
        assert await repeat.json() == result
        conflict = await client.post(url, json={'messages': [
            {'client_id': 'd', 'role': 'user', 'content': 'Must roll back'},
            {'client_id': 'a', 'role': 'user', 'content': 'Changed'}]})
        assert conflict.status == 409
        assert (await conflict.json())['error']['code'] == 'client_id_conflict'
        for invalid in ({'messages': []}, {'messages': [{'role': 'tool', 'content': 'x'}]}, {'messages': batch['messages'], 'run': True}):
            assert (await client.post(url, json=invalid)).status == 400
        assert (await client.post('/api/sessions/missing/messages', json=batch)).status == 404
        read = await client.get(url)
        assert [row['content'] for row in (await read.json())['data']] == [m['content'] for m in batch['messages']]
        caps = await (await client.get('/v1/capabilities')).json()
        assert 'session_append_messages' in caps['endpoints']
    assert owner.db.get_session('speech')['message_count'] == 3
    assert owner.db._conn.execute('SELECT COUNT(*) FROM session_admissions').fetchone()[0] == 0


@pytest.mark.asyncio
async def test_append_auth_and_source_ownership(api, owner):
    api._api_key = 'test-key'
    owner.db.create_session('local', source='cli')
    app = web.Application()
    app.router.add_post('/api/sessions/{session_id}/messages', api._handle_append_session_messages)
    async with TestClient(TestServer(app)) as client:
        body = {'messages': [{'role': 'user', 'content': 'Unauthorized'}]}
        url = '/api/sessions/local/messages'
        assert (await client.post(url, json=body)).status == 401
        response = await client.post(url, json=body, headers={'Authorization': 'Bearer test-key'})
        assert response.status == 409
    assert owner.db.get_messages('local') == []
