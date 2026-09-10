"""Private API ingress preserves caller content and conversation authority."""
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
        assert turn._native_image_run_message() == content
        assert api_execution.get()['history'] == []
        execution_result.get()['result'] = {'final_response': 'ok'}
        return 'ok'
    owner.runner._handle_message = handle
    assert await execute_admission(owner, ref, row) == 'ok'
