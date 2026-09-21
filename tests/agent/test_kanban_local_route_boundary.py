from types import SimpleNamespace

import pytest

from agent.agent_init import _init_fallback_chain, _routed_client_kwargs


def test_local_worker_never_activates_remote_init_or_turn_fallback(monkeypatch):
    from agent import auxiliary_client

    fallback = [{"provider": "openrouter", "model": "remote-model"}]
    calls = []

    def resolve(provider, **kwargs):
        calls.append(provider)
        return None, None

    monkeypatch.setattr(auxiliary_client, "resolve_provider_client", resolve)
    monkeypatch.setenv("HERMES_KANBAN_LOCAL_ONLY", "1")
    agent = SimpleNamespace(provider="ollama", model="local-model", quiet_mode=True)
    _init_fallback_chain(agent, fallback)
    assert agent._fallback_chain == []
    with pytest.raises(RuntimeError, match="No LLM provider|Provider 'ollama'"):
        _routed_client_kwargs(agent, fallback, 1)
    assert calls == ["ollama"]

    monkeypatch.delenv("HERMES_KANBAN_LOCAL_ONLY")
    _init_fallback_chain(agent, fallback)
    assert agent._fallback_chain == fallback
