"""A fully-refusing init-time fallback ladder must say why at WARNING before dying.

One burned primary (an exhausted single-entry OpenRouter pool, for example) plus fallback
entries that cannot resolve credentials raises the generic ``No LLM provider configured``
RuntimeError. Without a WARNING naming each refused entry and its reason, the failure is
indistinguishable from actual config loss in gateway.log.
"""

import logging
from types import SimpleNamespace

import pytest


def _agent():
    return SimpleNamespace(provider="openrouter", model="z-ai/glm-5.3-flash", base_url=None,
                           api_key=None, _fallback_activated=False)


_LADDER = [
    {"provider": "deepseek", "model": "deepseek-v4-pro"},
    {"provider": "kimi", "model": "kimi-k3"},
]


def _refusing_router(monkeypatch):
    """Primary resolves no client; deepseek resolves none (missing key), kimi raises."""
    def _fake_resolve(provider, model=None, **kwargs):
        if provider == "kimi":
            raise RuntimeError("kimi auth handshake refused")
        return (None, None)

    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", _fake_resolve)
    monkeypatch.setattr("hermes_cli.fallback_config.resolve_entry_api_key", lambda entry: None)


def test_fully_refusing_ladder_warns_with_each_reason(tmp_path, monkeypatch, caplog):
    from agent import agent_init

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _refusing_router(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="run_agent"):
        with pytest.raises(RuntimeError, match="No LLM provider configured"):
            agent_init._routed_client_kwargs(_agent(), _LADDER, 60)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    text = warnings[0].getMessage()
    assert "openrouter" in text
    assert "deepseek" in text and "no usable credentials" in text
    assert "kimi" in text and "kimi auth handshake refused" in text


def test_recovered_ladder_does_not_warn(tmp_path, monkeypatch, caplog):
    """An entry refusing while a later one serves the init is degraded config, not a failure."""
    from agent import agent_init

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    class _Client:
        pass

    def _fake_resolve(provider, model=None, **kwargs):
        if provider == "kimi":
            return (_Client(), model)
        return (None, None)

    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", _fake_resolve)
    monkeypatch.setattr("hermes_cli.fallback_config.resolve_entry_api_key", lambda entry: None)
    monkeypatch.setattr(agent_init, "_client_kwargs_from_routed",
                        lambda client, timeout: {"api_key": "k"})

    agent = _agent()
    with caplog.at_level(logging.WARNING, logger="run_agent"):
        kwargs = agent_init._routed_client_kwargs(agent, _LADDER, 60)

    assert kwargs == {"api_key": "k"}
    assert agent.provider == "kimi"
    assert agent._fallback_activated
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
