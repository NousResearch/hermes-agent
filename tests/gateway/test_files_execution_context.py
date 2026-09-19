"""Lower execution companion: private prepared content, ordinary API and native turns."""
import copy
from types import SimpleNamespace

import pytest

from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext
from gateway.session_api_turn import api_execution


@pytest.mark.parametrize('prepared', [
    {'content': 'prompt\nverified document reference', 'files_persist_user_message': 'prompt\n[Attached file: "report.txt"]'},
    {'content': [{'type': 'text', 'text': 'prompt'}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}],
     'files_persist_user_message': 'prompt\n[Attached image: "pixel.png"]'},
])
def test_private_files_content_and_persistence_are_separate(prepared):
    calls = []
    def run(message, *, turn_author=None, **kwargs):
        calls.append((message, kwargs))
        return {}
    runner = SimpleNamespace(session_authority=None, _consume_pending_native_image_paths=lambda key: [])
    ctx = TurnContext(message='prompt', source=SimpleNamespace(user_id='u', user_name='U'), session_key='turn', session_id='s')
    turn = TurnRunner(runner, ctx)
    saved = copy.deepcopy(prepared)
    token = api_execution.set(prepared)
    try:
        turn._run_conversation_with_approval(SimpleNamespace(run_conversation=run), [], None, None, None)
    finally:
        api_execution.reset(token)
    assert calls == [(prepared['content'], {'conversation_history': [], 'task_id': 's',
                                           'persist_user_message': prepared['files_persist_user_message']})]
    assert prepared == saved


@pytest.mark.parametrize('api', [True, False])
@pytest.mark.parametrize('recover', [True, False])
def test_ordinary_text_keeps_pending_and_recovery_presentation(api, recover):
    calls = []
    history = [{'role': 'system', 'content': 'stable system prefix'},
               {'role': 'user', 'content': 'earlier request'},
               {'role': 'assistant', 'content': 'prior'}]
    if recover:
        history[-1]['tool_calls'] = [{'id': 'call'}]
        history.append({'role': 'tool', 'content': 'prior tool', 'tool_call_id': 'call'})
    frozen = copy.deepcopy(history)
    runner = SimpleNamespace(session_authority=None, _consume_pending_native_image_paths=lambda key: [],
        _pending_model_notes={'turn': 'model notice'}, _pending_skills_reload_notes={'turn': 'skills notice'},
        session_store=SimpleNamespace(_entries={}))
    ctx = TurnContext(message='current presentation', history=history, user_config={},
        source=SimpleNamespace(user_id='u', user_name='U'), session_key='turn', session_id='s')
    turn = TurnRunner(runner, ctx)
    def run(message, *, turn_author=None, **kwargs):
        calls.append((message, kwargs))
        return {}
    token = api_execution.set({'content': 'raw payload', 'turn_author': None} if api else None)
    try:
        persist, timestamp = turn._prepare_turn_message(history)
        turn._run_conversation_with_approval(SimpleNamespace(run_conversation=run), history,
                                             'observed context', persist, timestamp)
    finally:
        api_execution.reset(token)
    message, kwargs = calls[0]
    assert 'current presentation' in message and 'raw payload' not in message
    assert 'model notice' in message and 'skills notice' in message
    assert 'observed context' in message
    if recover:
        assert 'IGNORE those pending results' in message
        assert kwargs['persist_user_message'] == 'model notice\n\ncurrent presentation'
    else:
        assert kwargs['persist_user_message'] == ctx.message
    assert history == frozen
    assert runner._pending_model_notes == runner._pending_skills_reload_notes == {}


@pytest.mark.parametrize('api', [True, False])
def test_old_native_image_then_text_does_not_reuse_image(tmp_path, api):
    import base64
    image = tmp_path / 'pixel.png'
    image.write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aXioAAAAASUVORK5CYII='))
    native = [{'type': 'text', 'text': 'caption'},
              {'type': 'image_url', 'image_url': {'url': 'https://example.test/pixel.png'}}]
    pending = [] if api else [str(image)]
    def consume(key):
        result = pending[:]
        pending.clear()
        return result
    runner = SimpleNamespace(session_authority=None, _consume_pending_native_image_paths=consume)
    turn = TurnRunner(runner, TurnContext(message='caption', session_key='turn'))
    token = api_execution.set({'content': native} if api else None)
    try:
        content = turn._native_image_run_message()
        assert isinstance(content, list) and content[1]['type'] == 'image_url'
        if api:
            assert content == native
    finally:
        api_execution.reset(token)
    turn._ctx.message = 'next text'
    assert turn._native_image_run_message() == 'next text'
