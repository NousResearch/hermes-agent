"""Use a real captured SDK response and real compressor/session scopes for accounting and handoff."""
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture(autouse=True)
def isolated_home(tmp_path):
    previous = os.environ.get('HERMES_HOME')
    os.environ['HERMES_HOME'] = str(tmp_path)
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(tmp_path)
    from hermes_cli.config import atomic_config_write

    atomic_config_write(tmp_path / 'config.yaml', {'lsp': {'enabled': False}})
    try:
        yield
    finally:
        reset_hermes_home_override(token)
        if previous is None:
            os.environ.pop('HERMES_HOME')
        else:
            os.environ['HERMES_HOME'] = previous


def recorded_response():
    from openai.types.responses import Response

    path = Path(__file__).with_name('actual-provider-responses-response.json')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == 'd112958b14b06961dfcc1f4116478f106a43f23a82d2e17066b02ce25692163d'
    return Response.model_validate(json.loads(path.read_bytes()))


def test_real_response_keeps_cached_input_when_adapted_to_chat_usage():
    from agent.auxiliary_codex_response import _parse_codex_final_response
    from agent.usage_pricing import normalize_usage

    response = recorded_response()
    text, calls, chat_usage, finish = _parse_codex_final_response(response)
    assert ''.join(text) == response.output_text and not calls and finish == 'stop'
    original = normalize_usage(response.usage, api_mode='codex_responses')
    converted = normalize_usage(chat_usage)
    assert original.cache_read_tokens == 4096 and original.input_tokens == 150
    assert original.output_tokens == 7
    assert converted.input_tokens == original.input_tokens
    assert converted.cache_read_tokens == original.cache_read_tokens
    assert converted.cache_write_tokens == original.cache_write_tokens
    assert converted.output_tokens == original.output_tokens
    assert converted.reasoning_tokens == original.reasoning_tokens


def test_real_compressor_accepts_only_its_bound_sessions_runtime(tmp_path):
    from agent.auxiliary_client import scoped_runtime_main
    from agent.context_compressor import ContextCompressor
    from hermes_state import SessionDB

    response = recorded_response()
    policy = {'enabled': True, 'effort': 'low'}
    compressor = ContextCompressor(response.model, config_context_length=872000, quiet_mode=True, reasoning_config=policy)
    policy['effort'] = 'high'
    assert compressor.reasoning_config['effort'] == 'low'
    owner, foreign = uuid4().hex, uuid4().hex
    db = SessionDB(tmp_path / 'state.db')
    try:
        for sid in (owner, foreign):
            db.create_session(sid, source='cli', model=response.model)
        compressor.bind_session_state(db, owner)
        runtime = {'model': response.model, 'provider': 'openai-codex', 'session_id': owner,
                   'reasoning_config': {'enabled': True, 'effort': 'high'}}
        with scoped_runtime_main(runtime):
            runtime['reasoning_config']['effort'] = 'low'
            inherited = compressor._summary_main_runtime()
            assert inherited['session_id'] == owner and inherited['reasoning_config']['effort'] == 'high'
            inherited['reasoning_config']['effort'] = 'none'
            assert compressor._summary_main_runtime()['reasoning_config']['effort'] == 'high'
        runtime['session_id'] = foreign
        with scoped_runtime_main(runtime):
            assert compressor._summary_main_runtime()['reasoning_config']['effort'] == 'low'
        assert compressor._summary_main_runtime()['reasoning_config']['effort'] == 'low'
    finally:
        db.close()


@pytest.mark.parametrize('explicit', [False, True])
def test_actual_sdk_request_builder_preserves_explicit_reasoning_precedence(explicit):
    from agent.auxiliary_client import _CodexCompletionsAdapter, resolve_provider_client

    response = recorded_response()
    client, _model = resolve_provider_client('ollama', model='qwen3:8b', explicit_base_url='http://localhost:11434/v1')
    assert client is not None
    try:
        adapter = _CodexCompletionsAdapter(client, response.model)
        kwargs = {'messages': [{'role': 'user', 'content': response.output_text}], 'reasoning_effort': 'low'}
        if explicit:
            kwargs['extra_body'] = {'reasoning': {'effort': 'high'}}
        built, model, _timeout = adapter._build_responses_kwargs(kwargs)
        assert model == response.model
        assert built['reasoning']['effort'] == ('high' if explicit else 'low')
    finally:
        client.close()
