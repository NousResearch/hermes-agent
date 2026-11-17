"""The opt-in-free openai-codex route uses Hermes' model loop and the shared intent."""
import asyncio
import json

import pytest

from gateway.platforms.event import MessageEvent
from hermes_cli.runtime_provider import frozen_runtime_config, resolve_runtime_provider
from hermes_state_runtime import list_session_admissions
from tests.fakes.providers.openai_responses import FakeResponsesServer, FunctionCall, Message, Turn
from tests.gateway.fresh_session.test_admitted_turn import admitted as admitted_fixture, provider

admitted = admitted_fixture
from tests.waiting import async_wait_until

CHECKPOINT = ('Objective: complete synthetic workspace task. Authorization: local edits, no publishing. '
              'Decision: inspect synthetic.txt. Completed: reproduction. Checks: focused test pending. '
              'Next: inspect the file and run the focused test.')


def isolated_route(base_url):
    # A non-secret sentinel bypasses credential stores. Only the isolated config and public resolver run.
    config = {'model': {'provider': 'openai-codex', 'default': 'gpt-5.4'}}
    assert 'openai_runtime' not in config['model']
    with frozen_runtime_config(config):
        runtime = resolve_runtime_provider(requested='openai-codex', target_model='gpt-5.4',
            explicit_api_key='synthetic-not-a-credential', explicit_base_url=base_url)
    assert runtime['provider'] == 'openai-codex'
    assert runtime['api_mode'] == 'codex_responses'
    return runtime


@pytest.mark.asyncio
async def test_openai_codex_resolves_without_app_server_and_composes_handoff(admitted, monkeypatch):
    a = admitted
    key = a.runner._session_key_for_source(a.source)
    parent = a.runner.session_store.get_or_create_session(a.source).session_id
    def no_credentials():
        raise AssertionError('Isolated explicit route must not load Codex credentials')
    monkeypatch.setattr('hermes_cli.runtime_provider.resolve_codex_runtime_credentials', no_credentials)
    def no_app_server(*args, **kwargs):
        raise AssertionError('openai-codex without opt-in must use the Hermes loop')
    monkeypatch.setattr('agent.codex_runtime._ensure_codex_session', no_app_server)
    with FakeResponsesServer([
        Turn([FunctionCall('fresh_session', {'handoff': CHECKPOINT})]),
        Turn([Message('Responses final answer.')]),
        Turn([Message('Responses following answer.')]),
    ]) as server:
        runtime = isolated_route(server.base_url)
        monkeypatch.setattr(a.runner, '_resolve_session_agent_runtime', lambda **kwargs: ('gpt-5.4', runtime))
        await asyncio.wait_for(a.adapter._process_message_background(MessageEvent(
            text='Finish the synthetic task and choose a boundary.', source=a.source, message_id='responses-1'), key), 30)
        assert await async_wait_until(lambda: any('Continuing in a fresh session' in t for t, _ in a.adapter.sent))
        rows = list_session_admissions(a.authority.db, session_id=parent, pending_only=False)
        assert len(rows) == 1 and rows[0]['status'] == 'terminal' and rows[0]['outcome'] == 'completed'
        child = a.runner.session_store.peek_session_id(key)
        assert child != parent
        assert a.authority.db.get_session(parent)['end_reason'] == 'session_handoff'
        assert CHECKPOINT in a.authority.db.get_messages(child)[0]['content']
        assert [t for t, _ in a.adapter.sent][-2:] == ['Responses final answer.', 'Continuing in a fresh session in this topic.']
        await asyncio.wait_for(a.runner._handle_message(MessageEvent(
            text='Proceed with the next action.', source=a.source, message_id='responses-2')), 30)
        requests = server.main_requests()
        assert any(t.get('name') == 'fresh_session' for t in requests[0]['tools'])
        assert CHECKPOINT in json.dumps(requests[-1]['input'])
        assert not any(item.get('type') == 'function_call_output' for item in requests[-1]['input'])
        assert a.runner.session_store.peek_session_id(key) == child
        assert not server.invalid_requests(), '\n'.join(error for _, error in server.invalid_requests())
