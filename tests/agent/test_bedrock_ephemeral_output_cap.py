"""Bedrock Converse must honour the one-shot output-cap override.

The truncation recovery paths in ``agent/conversation_loop.py`` (length
continuation boost, truncated tool-call boost, and the output-cap clamp) all
signal the next request's cap by setting ``_ephemeral_max_output_tokens`` on the
agent.  Every other transport branch in ``build_api_kwargs`` consumes it; the
``bedrock_converse`` branch used to ignore it, so retries re-requested the same
static cap and re-truncated identically — the agentic loop broke with
"Response truncated due to output length limit".
"""

from types import SimpleNamespace

import pytest

from agent.chat_completion_helpers import build_api_kwargs


class _RecordingTransport:
    """Captures the max_tokens the caller asked for."""

    def __init__(self):
        self.seen_max_tokens = None

    def build_kwargs(self, model, messages, tools=None, **params):
        self.seen_max_tokens = params.get("max_tokens")
        return {"modelId": model, "messages": messages}


@pytest.fixture
def bedrock_agent():
    transport = _RecordingTransport()
    agent = SimpleNamespace(
        api_mode="bedrock_converse",
        model="us.anthropic.claude-opus-5",
        tools=[],
        max_tokens=8192,
        _bedrock_region="us-west-2",
        _bedrock_guardrail_config=None,
        _get_transport=lambda: transport,
    )
    return agent, transport


def test_ephemeral_cap_overrides_static_max_tokens(bedrock_agent):
    agent, transport = bedrock_agent
    agent._ephemeral_max_output_tokens = 32768

    build_api_kwargs(agent, [{"role": "user", "content": "hi"}])

    assert transport.seen_max_tokens == 32768


def test_ephemeral_cap_is_consumed_after_one_request(bedrock_agent):
    """The override is one-shot: the following request falls back to the static cap."""
    agent, transport = bedrock_agent
    agent._ephemeral_max_output_tokens = 32768

    build_api_kwargs(agent, [{"role": "user", "content": "hi"}])
    assert agent._ephemeral_max_output_tokens is None

    build_api_kwargs(agent, [{"role": "user", "content": "again"}])
    assert transport.seen_max_tokens == 8192


def test_static_cap_used_when_no_override_present(bedrock_agent):
    agent, transport = bedrock_agent
    agent._ephemeral_max_output_tokens = None

    build_api_kwargs(agent, [{"role": "user", "content": "hi"}])

    assert transport.seen_max_tokens == 8192


def test_falls_back_to_default_when_nothing_configured(bedrock_agent):
    agent, transport = bedrock_agent
    agent._ephemeral_max_output_tokens = None
    agent.max_tokens = None

    build_api_kwargs(agent, [{"role": "user", "content": "hi"}])

    assert transport.seen_max_tokens == 4096
