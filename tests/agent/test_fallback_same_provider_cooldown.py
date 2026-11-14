"""A fallback's 429 benches the route that 429'd, not a primary that shares its provider name (#141025).

Two OpenAI-compatible servers are both ``provider: custom``. The primary fails (5xx), the
fallback on the other ``base_url`` answers 429: that throttle says nothing about the primary, so
the next turn must go back to it instead of staying pinned to the rate-limited fallback.
"""

from unittest.mock import MagicMock, patch

import pytest

from agent.error_classifier import FailoverReason
from run_agent import AIAgent

_PRIMARY_URL = "http://127.0.0.1:9101/v1"
_FALLBACK_URL = "http://127.0.0.1:9102/v1"


def _agent(fallback):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.context_compressor.get_model_context_length", return_value=200_000),
    ):
        agent = AIAgent(
            api_key="sk-test-primary", provider="custom", base_url=_PRIMARY_URL, model="primary-model",
            quiet_mode=True, skip_context_files=True, skip_memory=True, fallback_model=fallback,
        )
    agent.client = MagicMock()
    return agent


def _fallback_client(base_url):
    client = MagicMock()
    client.api_key, client.base_url = "sk-test-fallback", base_url
    return client


@pytest.mark.parametrize(("fb_model", "fb_url"), [
    ("fallback-model", _FALLBACK_URL),   # same provider name, other endpoint
    ("fallback-model", _PRIMARY_URL),    # same provider and endpoint, other model
    ("primary-model", _FALLBACK_URL),    # same provider and model name, other endpoint
])
def test_fallback_429_does_not_bench_a_primary_sharing_its_provider_name(fb_model, fb_url):
    agent = _agent([{"provider": "custom", "model": fb_model, "base_url": fb_url}])
    with patch("agent.auxiliary_client.resolve_provider_client",
               return_value=(_fallback_client(fb_url), fb_model)):
        assert agent._try_activate_fallback(reason=FailoverReason.server_error) is True
        assert (agent.provider, agent.model) == ("custom", fb_model)
        # The fallback now 429s; the chain is exhausted.
        agent._try_activate_fallback(reason=FailoverReason.rate_limit)

    assert not getattr(agent, "_rate_limited_until", 0), "the fallback's 429 armed the primary's cooldown"
    assert agent._restore_primary_runtime() is True
    assert (agent.provider, agent.model, agent.base_url.rstrip("/")) == ("custom", "primary-model", _PRIMARY_URL)


def test_primary_429_still_benches_the_primary():
    """The guard narrows only chain-switches: leaving the primary on a 429 still arms its cooldown."""
    agent = _agent([{"provider": "custom", "model": "fallback-model", "base_url": _FALLBACK_URL}])
    with patch("agent.auxiliary_client.resolve_provider_client",
               return_value=(_fallback_client(_FALLBACK_URL), "fallback-model")):
        assert agent._try_activate_fallback(reason=FailoverReason.rate_limit) is True

    assert getattr(agent, "_rate_limited_until", 0) > 0
    assert agent._restore_primary_runtime() is False
