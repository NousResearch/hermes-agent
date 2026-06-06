from types import SimpleNamespace

import pytest

import agent.agent_init as agent_init
import agent.auxiliary_client as auxiliary_client
from agent.model_metadata import MINIMUM_CONTEXT_LENGTH
import hermes_cli.config as config_module


class DummyCompressor:
    def __init__(self, *args, **kwargs):
        self.context_length = self._context_length


def make_agent():
    agent = SimpleNamespace()
    agent._base_url_hostname = ""
    agent._base_url_lower = ""
    agent._transport_cache = {}
    agent._get_transport = lambda *args, **kwargs: None
    agent._is_openrouter_url = lambda *args, **kwargs: False
    agent._is_azure_openai_url = lambda *args, **kwargs: False
    agent._is_direct_openai_url = lambda *args, **kwargs: False
    agent._provider_model_requires_responses_api = lambda *args, **kwargs: False
    agent._anthropic_prompt_cache_policy = lambda *args, **kwargs: (False, False)
    agent._create_openai_client = lambda *args, **kwargs: SimpleNamespace()
    agent._ensure_lmstudio_runtime_loaded = lambda _config_context_length: None
    return agent


def test_init_agent_rejects_model_below_user_config_override(monkeypatch):
    config = {"model": {"context_length": 60000}}
    monkeypatch.setattr(config_module, "load_config", lambda: config)
    monkeypatch.setattr(agent_init, "_install_safe_stdio", lambda: None)
    monkeypatch.setattr(
        auxiliary_client,
        "resolve_provider_client",
        lambda *args, **kwargs: (SimpleNamespace(api_key="test-key", base_url="https://example.invalid/v1"), "test/model"),
    )

    context_length = 50000

    class Compressor(DummyCompressor):
        _context_length = context_length

    monkeypatch.setattr(agent_init, "ContextCompressor", Compressor)

    agent = make_agent()

    with pytest.raises(ValueError, match="user-configured override"):
        agent_init.init_agent(agent, model="test/model")


def test_init_agent_rejects_model_below_minimum_without_override(monkeypatch):
    monkeypatch.setattr(config_module, "load_config", lambda: {})
    monkeypatch.setattr(agent_init, "_install_safe_stdio", lambda: None)
    monkeypatch.setattr(
        auxiliary_client,
        "resolve_provider_client",
        lambda *args, **kwargs: (SimpleNamespace(api_key="test-key", base_url="https://example.invalid/v1"), "test/model"),
    )

    context_length = 50000

    class Compressor(DummyCompressor):
        _context_length = context_length

    monkeypatch.setattr(agent_init, "ContextCompressor", Compressor)

    agent = make_agent()

    with pytest.raises(ValueError, match=f"below the minimum {MINIMUM_CONTEXT_LENGTH:,}"):
        agent_init.init_agent(agent, model="test/model")
