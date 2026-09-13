"""Private API ingress preserves caller content and conversation authority."""
import json
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.session_api_turn import admit_api_turn, api_execution
from tests.gateway.test_api_source_binding import owner


@pytest.fixture
def api(owner, tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter.gateway_runner = owner.runner
    adapter._session_db = owner.db
    owner.runner._adapter_for_source = lambda source: adapter
    yield adapter
    adapter._response_store.close()
    adapter._run_idempotency_store.close()


def test_declared_key_is_durable_and_cannot_rebind(api, owner):
    def admit(sid, key, bind=True):
        return admit_api_turn(api, session_id=sid, gateway_session_key=key,
                              bind_declared_conversation=bind,
                              user_message='hello', conversation_history=[])
    _, ref, _ = admit('first', 'conversation')
    assert api._declared_conversation_session('conversation') == ref.session_id
    owner.sessions.clear()
    _, retried, _ = admit('second', 'conversation')
    assert retried == ref
    # Explicit and chained selections never transfer their identity to a header.
    admit(ref.session_id, 'other', bind=False)
    assert api._declared_conversation_session('other') is None
    from hermes_state_runtime import RuntimeStoreError
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        admit(ref.session_id, 'other')
    assert api._declared_conversation_session('conversation') == ref.session_id


@pytest.mark.asyncio
async def test_structured_content_bypasses_text_parser_without_losing_parts(api, owner, monkeypatch):
    from gateway.session_ingress import execute_admission
    from gateway.run_turn_runner import TurnRunner
    from gateway.session_results import execution_result
    content = [{'type': 'text', 'text': '/literal prompt'},
               {'type': 'image_url', 'image_url': {'url': 'https://example.com/image.png', 'detail': 'high'}}]
    _, ref, row = admit_api_turn(api, session_id='image', user_message=content, conversation_history=[])
    async def handle(event):
        assert isinstance(event.text, str)
        assert event.get_command() is None
        turn = object.__new__(TurnRunner)
        turn._ctx = SimpleNamespace(message=event.text, session_key=owner.sessions[ref.session_id].route)
        turn._runner = SimpleNamespace(_consume_pending_native_image_paths=lambda key: [])
        # The transient image survives; the text part now carries the durable reference hint.
        message = turn._native_image_run_message()
        assert message[1] == content[1]
        assert message[0]['text'] == '/literal prompt\n\n[Image attached: https://example.com/image.png]'
        assert api_execution.get()['history'] == []
        execution_result.get()['result'] = {'final_response': 'ok'}
        return 'ok'
    owner.runner._handle_message = handle
    assert await execute_admission(owner, ref, row) == 'ok'



def test_api_author_is_admission_scoped_not_session_identity(api, owner):
    from gateway.session_api_turn import prepare_api_execution
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext
    from hermes_state_runtime import RuntimeStoreError
    author = {'id': 'bot:peer', 'name': 'Peer', 'is_bot': True}
    args = dict(session_id='author', request_id='one', user_message='hello', conversation_history=[], turn_author=author)
    _, ref, row = admit_api_turn(api, **args)
    assert row['payload']['api_turn_v1']['turn_author'] == author
    author['name'] = 'changed'
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        admit_api_turn(api, **args)
    owner.sessions.clear()
    from gateway.session_api import restore_api_session
    restore_api_session(owner, ref.session_id)
    seen = []
    def run(message, *, turn_author=None, **kwargs):
        seen.append(turn_author)
        return {'final_response': 'ok'}
    agent = SimpleNamespace(run_conversation=run)
    turn = TurnRunner(SimpleNamespace(session_authority=None, _consume_pending_native_image_paths=lambda key: []),
                      TurnContext(message='hello', session_id=ref.session_id, session_key='api',
                                  source=SimpleNamespace(user_id='authenticated', user_name='Owner', is_bot=False)))
    for payload in (row['payload'], {'text': 'human'}):
        prepared = prepare_api_execution(owner, ref, payload)
        token = api_execution.set(prepared)
        try:
            turn._run_conversation_with_approval(agent, [], None, None, None)
        finally:
            api_execution.reset(token)
    assert seen == [row['payload']['api_turn_v1']['turn_author'], None]


def test_persisted_owner_scope_is_private_strict_and_legacy_safe(api, owner):
    from gateway.platforms.api_server_authority_runs import run_projection
    from gateway.session_api_turn import owns_api_run, prepare_api_execution
    from hermes_state_runtime import RuntimeStoreError

    scope = 'a' * 64
    authority, ref, row = admit_api_turn(
        api, session_id='private-owner', active_run_id='run_private_owner',
        user_message='hello', conversation_history=[], run_owner_scope=scope)
    assert row['payload']['api_turn_v1']['run_owner_scope'] == scope
    assert 'run_owner_scope' not in prepare_api_execution(authority, ref, row['payload'])
    assert 'run_owner_scope' not in json.dumps(run_projection(api, 'run_private_owner'))
    assert owns_api_run(api, 'run_private_owner', scope)

    for malformed in ('A' * 64, 'a' * 63, 'g' * 64, 7):
        with pytest.raises(RuntimeStoreError, match='invalid_params'):
            admit_api_turn(
                api, session_id='invalid-owner', user_message='x',
                conversation_history=[], run_owner_scope=malformed)

    admit_api_turn(
        api, session_id='legacy-ownerless', active_run_id='run_legacy_ownerless',
        user_message='old', conversation_history=[])
    assert not owns_api_run(api, 'run_legacy_ownerless', scope)
