"""An explicitly configured OpenRouter route keeps reasoning behind a proxy (#156806)."""
from types import SimpleNamespace

import pytest

from agent.chat_completion_helpers import _chat_summary_attempt, _iteration_summary_api_messages
from run_agent import AIAgent


@pytest.mark.parametrize(('provider', 'host', 'model', 'capable', 'expected'), [
    ('openrouter', 'http://127.0.0.1:1/api/v1', 'google/gemini-4-carbon', True,
     {'enabled': True, 'effort': 'high'}),
    ('openrouter', 'http://127.0.0.1:1/api/v1', 'google/gemini-4-carbon', False, None),
    ('openrouter', 'http://127.0.0.1:1/api/v1', 'deepseek/deepseek-chat', None,
     {'enabled': True, 'effort': 'high'}),
    ('openrouter', 'http://127.0.0.1:1/api/v1', 'unknown/model', None, None),
    ('custom', 'http://127.0.0.1:1/api/v1', 'google/gemini-4-carbon', True, None),
    ('openrouter', 'https://api.mistral.ai/v1', 'google/gemini-4-carbon', True, None),
])
def test_proxy_reasoning_on_normal_and_summary_requests(monkeypatch, provider, host, model, capable, expected):
    from hermes_cli import models

    monkeypatch.setattr(models, '_openrouter_reasoning_caps_cache', {
        model: None if capable is None else {
            'supports_reasoning': capable, 'supported_efforts': ['low', 'medium', 'high']}})
    monkeypatch.setattr(models, '_openrouter_caps_warm_started', True)
    agent = AIAgent(
        api_key='local-test', base_url=host, provider=provider,
        api_mode='chat_completions', model=model,
        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[],
        reasoning_config={'enabled': True, 'effort': 'high'})
    agent._cached_system_prompt = 'Reply briefly.'
    messages = [{'role': 'user', 'content': 'hello'}]
    sent = [agent._build_api_kwargs(messages)]

    def send_summary(_agent, _request_id, kwargs, _callback, **_options):
        sent.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='summary', tool_calls=None), finish_reason='stop')], usage=None)

    monkeypatch.setattr('agent.chat_completion_helpers._managed_summary_call', send_summary)
    summary = _chat_summary_attempt(agent, _iteration_summary_api_messages(agent, messages), 'summary-request')
    assert summary(0) == 'summary'
    assert len(sent) == 2
    for kwargs in sent:
        assert (kwargs.get('extra_body') or {}).get('reasoning') == expected
