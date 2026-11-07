"""Every API door admits a compression continuation's id on its lineage root (the admission owner)."""
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.session_api_turn import admit_api_turn

KEY = 'logical-owner-key-long-enough'


def _compressed(api, owner):
    admit_api_turn(api, session_id='root', request_id='first', user_message='hello', conversation_history=[])
    owner.db.publish_compression_child(parent_session_id='root', child_session_id='tip', source='api_server',
        messages=[{'role': 'assistant', 'content': 'summary'}], require_compression_lease=False)
    owner.runner.session_store.advance_compression_session(owner.sessions['root'].route, 'root', 'tip')


def _chat(client):
    return client.post('/v1/chat/completions', json={'messages': [{'role': 'user', 'content': 'next'}]},
                       headers={'Authorization': 'Bearer ' + KEY, 'X-Hermes-Session-Id': 'tip'})


def _runs(client):
    return client.post('/v1/runs', json={'input': 'next', 'session_id': 'tip'},
                       headers={'Authorization': 'Bearer ' + KEY})


def _session_chat(client):
    return client.post('/api/sessions/tip/chat', json={'message': 'next'},
                       headers={'Authorization': 'Bearer ' + KEY})


@pytest.mark.asyncio
@pytest.mark.parametrize('door,expected', [(_chat, 200), (_runs, 202), (_session_chat, 200)])
async def test_api_doors_admit_a_continuation_on_its_root(api, owner, monkeypatch, door, expected):
    from gateway.platforms import api_server_runs
    from hermes_state_runtime import list_session_admissions

    async def handle(event):
        from gateway.session_results import execution_result
        execution_result.get()['result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'

    async def no_execution(*args, **kwargs):
        pass
    owner.runner._handle_message = handle
    monkeypatch.setattr(api_server_runs, '_execute_run', no_execution)
    _compressed(api, owner)
    api._api_key = KEY
    app = web.Application()
    app.router.add_post('/v1/chat/completions', api._handle_chat_completions)
    app.router.add_post('/v1/runs', api._handle_runs)
    app.router.add_post('/api/sessions/{session_id}/chat', api._handle_session_chat)
    async with TestClient(TestServer(app)) as client:
        response = await door(client)
        assert response.status == expected, await response.text()
    rows = list_session_admissions(owner.db, session_id='root', pending_only=False)
    assert [row['payload']['text'] for row in rows] == ['hello', 'next']
    assert not list_session_admissions(owner.db, session_id='tip', pending_only=False)
