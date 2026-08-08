"""Regression tests for the model-routing mixin extraction (shard s1, c23).

The 7 URL / provider classification helpers moved VERBATIM from
``run_agent.py`` into ``plugins.agent.mixins.model_routing_mixin``.
These tests pin the pure classification behavior and the MRO wiring
(``AIAgent`` must still expose the exact same function objects, so
callers inside run_agent.py keep working without any change).
"""

from __future__ import annotations

import run_agent
from plugins.agent.mixins.model_routing_mixin import ModelRoutingMixin

MOVED = (
    "_is_direct_openai_url",
    "_is_azure_openai_url",
    "_is_github_copilot_url",
    "_is_openrouter_url",
    "_is_copilot_url",
    "_is_copilot_provider",
    "_is_codex_backend",
)


def _bare_agent(**attrs):
    """Object.__new__-based bare adapter (no __init__ side effects)."""
    agent = object.__new__(run_agent.AIAgent)
    for key, value in attrs.items():
        setattr(agent, key, value)
    return agent


def test_methods_are_wired_through_mro():
    for name in MOVED:
        assert getattr(run_agent.AIAgent, name) is getattr(ModelRoutingMixin, name), name


def test_is_direct_openai_url():
    agent = _bare_agent(_base_url_lower="https://api.openai.com/v1", _base_url_hostname="api.openai.com")
    assert agent._is_direct_openai_url() is True
    agent = _bare_agent(_base_url_lower="https://api.openai.com/v1", _base_url_hostname="api.openai.com")
    assert agent._is_direct_openai_url("https://api.openai.com/v1") is True
    agent = _bare_agent(_base_url_lower="https://other.example.com/v1", _base_url_hostname="other.example.com")
    assert agent._is_direct_openai_url() is False
    # Explicit base_url argument wins over instance state.
    agent = _bare_agent(_base_url_lower="https://other.example.com/v1", _base_url_hostname="other.example.com")
    assert agent._is_direct_openai_url("https://api.openai.com/v1") is True


def test_is_azure_openai_url():
    agent = _bare_agent(_base_url_lower="https://my-resource.openai.azure.com/openai/v1")
    assert agent._is_azure_openai_url() is True
    agent = _bare_agent(_base_url_lower="https://api.openai.com/v1")
    assert agent._is_azure_openai_url() is False
    agent = _bare_agent(_base_url_lower="")
    assert agent._is_azure_openai_url("https://res.openai.azure.com/") is True


def test_is_github_copilot_url():
    agent = _bare_agent(_base_url_lower="https://api.githubcopilot.com/chat/completions", _base_url_hostname="api.githubcopilot.com")
    assert agent._is_github_copilot_url() is True
    agent = _bare_agent(_base_url_lower="https://api.openai.com/v1", _base_url_hostname="api.openai.com")
    assert agent._is_github_copilot_url() is False
    # Subdomain hosts count.
    agent = _bare_agent(_base_url_lower="", _base_url_hostname="")
    assert agent._is_github_copilot_url("https://copilot-proxy.githubcopilot.com/v1") is True
    # No hostname at all -> False, never an exception.
    agent = _bare_agent(_base_url_lower="", _base_url_hostname="")
    assert agent._is_github_copilot_url() is False


def test_is_openrouter_url():
    agent = _bare_agent(_base_url_lower="https://openrouter.ai/api/v1")
    assert agent._is_openrouter_url() is True
    agent = _bare_agent(_base_url_lower="https://api.openai.com/v1")
    assert agent._is_openrouter_url() is False


def test_is_copilot_url():
    agent = _bare_agent(_base_url_lower="https://api.githubcopilot.com/chat/completions")
    assert agent._is_copilot_url() is True
    agent = _bare_agent(_base_url_lower="https://models.github.ai/v1")
    assert agent._is_copilot_url() is True
    agent = _bare_agent(_base_url_lower="https://api.openai.com/v1")
    assert agent._is_copilot_url() is False


def test_is_copilot_provider_spellings_and_url_fallback():
    for spelling in ("copilot", "github-copilot", "github", "  Copilot  "):
        agent = _bare_agent(provider=spelling, _base_url_lower="https://api.openai.com/v1")
        assert agent._is_copilot_provider() is True, spelling
    agent = _bare_agent(provider="anthropic", _base_url_lower="https://api.githubcopilot.com/v1")
    assert agent._is_copilot_provider() is True  # URL fallback signal
    agent = _bare_agent(provider="anthropic", _base_url_lower="https://api.openai.com/v1")
    assert agent._is_copilot_provider() is False
    agent = _bare_agent(provider=None, _base_url_lower="https://api.openai.com/v1")
    assert agent._is_copilot_provider() is False  # None provider must not raise


def test_is_codex_backend():
    agent = _bare_agent(api_mode="codex_responses", _base_url_hostname="chatgpt.com",
                        _base_url_lower="https://chatgpt.com/backend-api/codex")
    assert agent._is_codex_backend() is True
    agent = _bare_agent(api_mode="responses", _base_url_hostname="chatgpt.com",
                        _base_url_lower="https://chatgpt.com/backend-api/codex")
    assert agent._is_codex_backend() is False
    agent = _bare_agent(api_mode="codex_responses", _base_url_hostname="api.openai.com",
                        _base_url_lower="https://api.openai.com/v1")
    assert agent._is_codex_backend() is False
