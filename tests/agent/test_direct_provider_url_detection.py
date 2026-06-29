from __future__ import annotations
from agent.providers.openai_adapter import is_direct_openai_url, is_azure_openai_url, is_github_copilot_url, max_tokens_param

from run_agent import AIAgent


def _agent_with_base_url(base_url: str) -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.base_url = base_url
    return agent


def test_direct_openai_url_requires_openai_host():
    agent = _agent_with_base_url("https://api.openai.com.example/v1")

    assert is_direct_openai_url(getattr(agent, "base_url", None), getattr(agent, "_base_url_hostname", "")) is False


def test_direct_openai_url_ignores_path_segment_match():
    agent = _agent_with_base_url("https://proxy.example.test/api.openai.com/v1")

    assert is_direct_openai_url(getattr(agent, "base_url", None), getattr(agent, "_base_url_hostname", "")) is False


def test_direct_openai_url_accepts_native_host():
    agent = _agent_with_base_url("https://api.openai.com/v1")

    assert is_direct_openai_url(getattr(agent, "base_url", None), getattr(agent, "_base_url_hostname", "")) is True
