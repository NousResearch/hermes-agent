"""Regression for #110480: Bedrock Converse's request builder must consume the one-shot
truncation-retry boost (``_ephemeral_max_output_tokens``) the same way every other api_mode's
builder does. ``turn_truncation.py`` sets that field to escalate ``max_tokens`` on a truncated
tool-call/continuation retry; ``_build_bedrock_kwargs`` read ``agent.max_tokens`` directly and
ignored it, so a Bedrock Converse truncation retry re-sent the SAME cap four times and always
re-truncated identically instead of escalating.
"""

from unittest.mock import MagicMock

from agent.chat_completion_helpers import _build_bedrock_kwargs


def _agent(max_tokens=32000, ephemeral=None):
    agent = MagicMock()
    agent.model = "eu.anthropic.claude-sonnet-5"
    agent.max_tokens = max_tokens
    agent._bedrock_region = "eu-west-1"
    agent._bedrock_guardrail_config = None
    agent._ephemeral_max_output_tokens = ephemeral
    agent._get_transport.return_value.build_kwargs.side_effect = (
        lambda **kwargs: kwargs
    )
    return agent


def test_ephemeral_boost_overrides_configured_max_tokens():
    """A truncation-retry boost must reach the wire, not the stale configured cap."""
    agent = _agent(max_tokens=32000, ephemeral=64000)
    kwargs = _build_bedrock_kwargs(agent, [], [])
    assert kwargs["max_tokens"] == 64000


def test_ephemeral_boost_is_consumed_once():
    """Like every other api_mode's builder, the one-shot field must be cleared after use."""
    agent = _agent(max_tokens=32000, ephemeral=64000)
    _build_bedrock_kwargs(agent, [], [])
    assert agent._ephemeral_max_output_tokens is None


def test_falls_back_to_configured_max_tokens_when_no_boost_is_armed():
    """No regression on the normal (non-retry) path."""
    agent = _agent(max_tokens=32000, ephemeral=None)
    kwargs = _build_bedrock_kwargs(agent, [], [])
    assert kwargs["max_tokens"] == 32000
