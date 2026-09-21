"""An unset ``agent.reasoning_effort`` resolves to ``medium`` on custom / OpenAI-compatible routes.

The Nous, OpenRouter and AI Gateway profiles already fill medium when no effort is configured; the
custom profile (``providers.<name>`` blocks, ``--provider custom``) omitted the field, so the
endpoint's own default applied — for kimi-k3 that is ``max``: 3x the reasoning tokens and ~3x the
latency of medium. The default is decided at request time in ``_reasoning_config_for_wire`` so the
rejection ladder sees exactly what went out, and it never overrides an explicit effort.
"""
from unittest.mock import patch

import pytest

from agent.chat_completion_helpers import _build_api_kwargs_for_mode
from agent.models_dev import ModelCapabilities
from agent.transports.chat_completions import ChatCompletionsTransport
from providers import get_provider_profile


class _Agent:
    """The attributes ``_build_api_kwargs_for_mode`` reads before handing off to the transport builder."""

    def __init__(self, reasoning_config, *, provider="custom:relay", api_mode="chat_completions"):
        self.reasoning_config = reasoning_config
        self.provider = provider
        self.model = "moonshotai/kimi-k3"
        self.api_mode = api_mode
        self.base_url = "http://relay.example/v1"
        self.tools = []
        self.request_overrides = None
        self.service_tier = None
        self._fast_until = 0.0
        self._ephemeral_reasoning_off = False
        self._reasoning_effort_rejected = False
        self._reasoning_disable_rejected = False
        self._ollama_num_ctx: int | None = None

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _wire_reasoning_config(agent):
    """The ``reasoning_config`` the main loop hands the transport for this agent's next request."""
    def _capture(agent, api_messages, tools_for_api, reasoning_config, request_overrides, cache_scope_id):
        return {"reasoning_config": reasoning_config}

    with patch("agent.chat_completion_helpers._build_chat_completions_kwargs", _capture):
        return _build_api_kwargs_for_mode(agent, [], [])["reasoning_config"]


def _custom_wire_field(reasoning_config):
    kwargs = ChatCompletionsTransport().build_kwargs(
        "moonshotai/kimi-k3", [{"role": "user", "content": "hi"}], tools=None,
        provider_profile=get_provider_profile("custom:relay"), reasoning_config=reasoning_config,
        base_url="http://relay.example/v1",
    )
    return kwargs.get("reasoning_effort")


@pytest.mark.parametrize(
    "configured, expected_wire",
    [(None, "medium"), ({"enabled": True, "effort": "low"}, "low"), ({"enabled": False}, "none")],
)
def test_unset_effort_goes_out_as_medium_and_explicit_efforts_are_untouched(configured, expected_wire):
    agent = _Agent(configured)
    with patch("agent.models_dev.get_model_capabilities", return_value=None):
        sent = _wire_reasoning_config(agent)
    assert _custom_wire_field(sent) == expected_wire
    # The rejection ladder classifies the NEXT 400 from what was recorded as sent.
    assert agent._wire_reasoning_config == sent


def test_unset_effort_default_keeps_the_field_off_where_it_would_be_wrong():
    # A model the catalog / model_overrides mark as non-reasoning.
    with patch("agent.models_dev.get_model_capabilities", return_value=ModelCapabilities(supports_reasoning=False)):
        assert _wire_reasoning_config(_Agent(None)) is None
    with patch("agent.models_dev.get_model_capabilities", return_value=None):
        # The route already rejected the reasoning field this session (#112781 ladder → route default).
        rejected = _Agent(None)
        rejected._reasoning_effort_rejected = True
        assert _wire_reasoning_config(rejected) is None
        # A local Ollama model pulled without the thinking capability 400s on reasoning_effort.
        ollama = _Agent(None)
        ollama._ollama_num_ctx = 8192
        ollama._ollama_supports_thinking_cached = lambda: False
        assert _wire_reasoning_config(ollama) is None
        # Profiles that decide inside build_api_kwargs_extras (Nous fills medium itself) get None as before.
        assert _wire_reasoning_config(_Agent(None, provider="nous")) is None
        # Not a chat-completions wire: the Anthropic adapter's "unset = no thinking kwargs" stands.
        from agent.reasoning_params import unset_reasoning_default
        assert unset_reasoning_default(_Agent(None, api_mode="anthropic_messages")) is None
