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


@pytest.mark.asyncio
@pytest.mark.parametrize('path', ['/api/sessions/s1/chat', '/api/sessions/s1/chat/stream'])
async def test_session_chat_turn_sees_a_turn_finished_after_the_request_read_history(api, owner, monkeypatch, path):
    """The session-chat doors read the transcript before admission; a turn another writer finishes
    in between must still be in this turn's context, as on the other API doors (#84235)."""
    from gateway.session_api_turn import api_execution
    from gateway.session_results import execution_result

    owner.db.create_session('s1', 'api_server')
    owner.db.append_message('s1', 'user', 'first question')
    seen = {}

    async def handle(event):
        # What _load_turn_history runs with: the admitted snapshot, else the transcript at turn time.
        snapshot = api_execution.get()['history']
        seen['history'] = snapshot if snapshot is not None else owner.db.get_messages_as_conversation('s1')
        execution_result.get()['result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'
    owner.runner._handle_message = handle
    real_read = api._conversation_history_for_session

    async def read_then_another_writer_finishes(session_id):
        history = await real_read(session_id)
        owner.db.append_message('s1', 'assistant', 'answer the other writer persisted')
        return history
    monkeypatch.setattr(api, '_conversation_history_for_session', read_then_another_writer_finishes)
    api._api_key = KEY
    app = web.Application()
    app.router.add_post('/api/sessions/{session_id}/chat', api._handle_session_chat)
    app.router.add_post('/api/sessions/{session_id}/chat/stream', api._handle_session_chat_stream)
    async with TestClient(TestServer(app)) as client:
        response = await client.post(path, json={'message': 'next'}, headers={'Authorization': 'Bearer ' + KEY})
        await response.read()

    assert [m['content'] for m in seen['history']] == ['first question', 'answer the other writer persisted']
