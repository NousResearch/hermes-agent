"""Behavioral wire-boundary regression tests; no network or real credentials."""
import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.tools.test_azure_schema import tool, get_pattern, PATTERN
from tools.azure_schema import AzureToolSchemaError


@pytest.fixture
def agent():
    from run_agent import AIAgent
    a = AIAgent(api_key='test-key', base_url='https://test.openai.azure.com/openai/v1',
                provider='azure-foundry', model='gpt-6-astra', api_mode='chat_completions',
                enabled_toolsets=[], quiet_mode=True, skip_context_files=True, skip_memory=True)
    a._interrupt_requested = False
    return a


def completed():
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='OK',tool_calls=None),finish_reason='stop')],usage=None)


@pytest.mark.parametrize('stream', [False, True])
def test_main_agent_sanitizes_at_sdk_send_after_overrides(agent, stream):
    from tests.agent.test_streaming import _make_stream_chunk
    client = MagicMock()
    client.base_url = 'https://test.openai.azure.com/openai/v1'
    client.chat.completions.create.return_value = iter([_make_stream_chunk(content='OK',finish_reason='stop')]) if stream else completed()
    kwargs = {'model':'gpt-6-astra','messages':[{'role':'user','content':'test'}], 'tools':[tool()], 'extra_body':{'tools':[tool()]}}
    before = copy.deepcopy(kwargs)
    with patch.object(agent, '_create_request_openai_client',return_value=client), patch.object(agent,'_close_request_openai_client'):
        if stream: result = agent._interruptible_streaming_api_call(kwargs)
        else: result = agent._interruptible_api_call(kwargs)
    assert result.choices[0].message.content == 'OK'
    sent = client.chat.completions.create.call_args.kwargs
    assert get_pattern(sent['tools'][0]) != PATTERN
    assert get_pattern(sent['extra_body']['tools'][0]) != PATTERN
    assert kwargs == before
    assert sent['model'] == 'gpt-6-astra'


def test_main_agent_bad_schema_never_reaches_sdk(agent):
    client = MagicMock()
    with patch.object(agent,'_create_request_openai_client',return_value=client), patch.object(agent,'_close_request_openai_client'):
        with pytest.raises(AzureToolSchemaError):
            agent._interruptible_api_call({'model':'gpt-6-astra', 'tools':[tool(r'\p{Unknown}')]})
    client.chat.completions.create.assert_not_called()


def test_responses_runtime_preserves_flat_constraints(agent):
    from agent.codex_runtime import run_codex_stream
    client = MagicMock()
    client.responses.create.return_value = SimpleNamespace(output=[],usage=None)
    original = {'model':'gpt-6-astra','tools':[tool(flat=True)]}
    run_codex_stream(agent,original,client=client)
    assert get_pattern(client.responses.create.call_args.kwargs['extra_body']['tools'][0]) != PATTERN
    assert get_pattern(original['tools'][0]) == PATTERN


@pytest.mark.parametrize('async_mode', [False, True])
def test_auxiliary_calls_sanitize_real_builder_before_send(async_mode):
    import asyncio
    from unittest.mock import AsyncMock
    from agent.auxiliary_client import call_llm, async_call_llm
    client = MagicMock()
    client.base_url = 'https://test.openai.azure.com/openai/v1'
    client.chat.completions.create = AsyncMock(return_value=completed()) if async_mode else MagicMock(return_value=completed())
    kwargs: dict = dict(model='gpt-6-astra',provider='azure-foundry',messages=[{'role':'user','content':'test'}],tools=[tool()])
    with patch("agent.auxiliary_client._get_cached_client", return_value=(client,"gpt-6-astra")):
        if async_mode: result = asyncio.run(async_call_llm(**kwargs))
        else: result = call_llm(**kwargs)
    assert result.choices[0].message.content == 'OK'
    assert get_pattern(client.chat.completions.create.call_args.kwargs['tools'][0]) != PATTERN
    assert get_pattern(kwargs['tools'][0]) == PATTERN


@pytest.mark.parametrize('host,provider', [('https://test.openai.azure.com/openai/v1',''), ('https://private-proxy.test/v1','azure-foundry')])
def test_auxiliary_responses_does_not_strip_azure_patterns_formats_enums(host,provider):
    from agent.auxiliary_client import _CodexCompletionsAdapter
    client = MagicMock()
    client.base_url = host
    client.responses.create.return_value = iter([])
    original = [tool()]
    original[0]['function']['parameters']['properties']['url'] = {'type':'string','format':'uri','enum':['a/b']}
    adapter = _CodexCompletionsAdapter(client,'gpt-6-astra',provider=provider)
    with patch('agent.codex_runtime._consume_codex_event_stream',return_value=SimpleNamespace(output=[],usage=None)):
        adapter.create(messages=[{'role':'user','content':'test'}], tools=original)
    sent = client.responses.create.call_args.kwargs['extra_body']['tools'][0]
    assert get_pattern(sent) != PATTERN
    assert sent['parameters']['properties']['url'] == {'type':'string','format':'uri','enum':['a/b']}
    assert get_pattern(original[0]) == PATTERN
