"""Replay two actual provider responses through a real cached agent's accounting."""
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def local_agent(tmp_path):
    previous = os.environ.get('HERMES_HOME')
    os.environ['HERMES_HOME'] = str(tmp_path)
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(tmp_path)
    from hermes_cli.config import atomic_config_write

    atomic_config_write(tmp_path / 'config.yaml', {'lsp': {'enabled': False}})
    from agent.auxiliary_client import resolve_provider_client
    from hermes_state import SessionDB
    from run_agent import AIAgent

    client, _model = resolve_provider_client('ollama', model='qwen3:8b', explicit_base_url='http://localhost:11434/v1')
    assert client is not None
    db = SessionDB(tmp_path / 'state.db')
    agent = AIAgent(
        api_key=client.api_key, base_url=str(client.base_url), provider='custom',
        model='gpt-6.1-sol', api_mode='codex_responses', enabled_toolsets=[],
        quiet_mode=True, session_id=uuid4().hex, session_db=db, skip_context_files=True,
        skip_memory=True, skip_background_review=True, save_trajectories=False,
        max_iterations=1, reasoning_config={'enabled': True, 'effort': 'high'},
    )
    try:
        yield agent
    finally:
        agent.close()
        db.close()
        client.close()
        reset_hermes_home_override(token)
        if previous is None:
            os.environ.pop('HERMES_HOME')
        else:
            os.environ['HERMES_HOME'] = previous


def captured_response(sample):
    from openai.types.responses import Response

    filename, expected = {
        1: ('actual-provider-responses-response.json', 'd112958b14b06961dfcc1f4116478f106a43f23a82d2e17066b02ce25692163d'),
        2: ('actual-provider-responses-response-2.json', '0e002e6793388c1d70ecc062e54077d1edf5f2336ddea7a8b0c81bb4df5bfc79'),
    }[sample]
    data = Path(__file__).with_name(filename).read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected
    return Response.model_validate(json.loads(data))


def test_cached_agent_reports_each_actual_response_as_one_turn(local_agent):
    from agent.turn_usage import record_response_usage
    from agent.usage_pricing import normalize_usage
    from gateway.runtime_footer import (
        _gateway_turn_runtime_metadata, capture_turn_usage_baseline, format_runtime_footer,
    )

    agent = local_agent
    seen = set()
    totals = {'input': 0, 'output': 0, 'cache': 0}
    for sample in (1, 2):
        response = captured_response(sample)
        assert response.id not in seen
        seen.add(response.id)
        usage = normalize_usage(response.usage, provider=agent.provider, api_mode=agent.api_mode)
        baseline = capture_turn_usage_baseline(agent)
        messages = [{'role': 'user', 'content': response.output_text}]
        record_response_usage(
            agent, response, messages=messages, api_call_count=1, api_duration=0,
            compression_attempts=0, max_compression_attempts=1,
        )
        metadata = _gateway_turn_runtime_metadata(agent, **baseline, result_api_calls=1)
        assert metadata['turn_input_tokens'] == usage.input_tokens == 150
        assert metadata['turn_output_tokens'] == usage.output_tokens == 7
        assert metadata['turn_cache_read_tokens'] == usage.cache_read_tokens == 4096
        assert metadata['turn_cache_write_tokens'] == usage.cache_write_tokens == 0
        assert metadata['last_prompt_tokens'] == usage.prompt_tokens == 4246
        assert metadata['token_usage_status'] == metadata['cache_usage_status'] == 'reported'
        assert metadata['context_usage_status'] == 'reported'
        assert metadata['reasoning_effort'] == 'high'
        totals['input'] += usage.input_tokens
        totals['output'] += usage.output_tokens
        totals['cache'] += usage.cache_read_tokens
        assert metadata['uncached_input_tokens'] == totals['input']
        assert metadata['output_tokens'] == totals['output']
        assert metadata['cache_read_tokens'] == totals['cache']
        footer = format_runtime_footer(
            model=metadata['model'], context_tokens=metadata['last_prompt_tokens'],
            context_length=metadata['context_length'], tokens_in=metadata['turn_input_tokens'],
            tokens_out=metadata['turn_output_tokens'], cache_read_tokens=metadata['turn_cache_read_tokens'],
            cache_write_tokens=metadata['turn_cache_write_tokens'], token_usage_status=metadata['token_usage_status'],
            cache_usage_status=metadata['cache_usage_status'], fields=['tokens_turn', 'cache_hit'],
        )
        assert 'tokens(turn,uncached):150 in/7 out' in footer and '96%' in footer
        assert '300 in' not in footer and '14 out' not in footer


def test_real_agent_without_a_provider_response_omits_unreported_usage(local_agent):
    from gateway.runtime_footer import (
        _gateway_turn_runtime_metadata, capture_turn_usage_baseline, format_runtime_footer,
    )

    metadata = _gateway_turn_runtime_metadata(
        local_agent, **capture_turn_usage_baseline(local_agent), result_api_calls=0,
    )
    for field in ('turn_input_tokens', 'turn_output_tokens', 'turn_cache_read_tokens',
                  'turn_cache_write_tokens', 'token_usage_status', 'cache_usage_status', 'context_usage_status'):
        assert metadata[field] is None
    footer = format_runtime_footer(
        model=metadata['model'], context_tokens=metadata['last_prompt_tokens'],
        context_length=metadata['context_length'], tokens_in=metadata['turn_input_tokens'],
        tokens_out=metadata['turn_output_tokens'], token_usage_status=metadata['token_usage_status'],
        context_usage_status=metadata['context_usage_status'], fields=['tokens_turn', 'context_window'],
    )
    assert footer == ''
