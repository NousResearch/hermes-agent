"""A fully-refusing init-time fallback ladder must say why at WARNING before dying.

One burned primary (an exhausted single-entry OpenRouter pool, for example) plus fallback
entries that cannot resolve credentials raises the generic ``No LLM provider configured``
RuntimeError. Without a WARNING naming each refused entry and its reason, the failure is
indistinguishable from actual config loss in gateway.log.
"""

import logging
from types import SimpleNamespace

import pytest


def _agent(provider="openrouter"):
    return SimpleNamespace(
        provider=provider,
        model="z-ai/glm-5.3-flash",
        base_url=None,
        api_key=None,
        _fallback_activated=False,
    )


_LADDER = [
    {"provider": "deepseek", "model": "deepseek-v4-pro"},
    {"provider": "kimi", "model": "kimi-k3"},
]


_LADDER_REASONS = (
    "entry[0]",
    "no usable credentials",
    "entry[1]",
    "RuntimeError",
)


class _Pool:
    def __init__(self, available):
        self._available = available

    def has_credentials(self):
        return True

    def has_available(self, *, model=None):
        return self._available


@pytest.mark.parametrize(
    "primary, ladder, pool_available, raise_match, expected",
    [
        ("openrouter", _LADDER, True, "No LLM provider configured", _LADDER_REASONS),
        # Explicit non-OpenRouter primary: provider-specific missing-credentials error.
        ("anthropic", _LADDER, True, "no API key was found", _LADDER_REASONS),
        # #119533: a burned primary pool with no ladder at all must still be named.
        ("openrouter", [], False, "No LLM provider configured", ("credential pool exhausted",)),
    ],
)
def test_fully_refusing_ladder_warns_with_each_reason(
    monkeypatch, caplog, primary, ladder, pool_available, raise_match, expected
):
    """Primary resolves no client; deepseek resolves none (missing key), kimi raises."""
    from agent import agent_init

    def _fake_resolve(provider, model=None, **kwargs):
        if provider == "kimi":
            raise RuntimeError("kimi auth handshake refused")
        return (None, None)

    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", _fake_resolve)
    monkeypatch.setattr(
        "hermes_cli.fallback_config.resolve_entry_api_key", lambda entry: None
    )
    monkeypatch.setattr(
        "agent.credential_pool.load_pool", lambda provider: _Pool(pool_available)
    )

    with caplog.at_level(logging.WARNING, logger="run_agent"):
        with pytest.raises(RuntimeError, match=raise_match):
            agent_init._routed_client_kwargs(_agent(primary), ladder, 60)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    text = warnings[0].getMessage()
    assert primary in text
    for fragment in expected:
        assert fragment in text


def test_recovered_ladder_does_not_warn(monkeypatch, caplog):
    """An entry refusing while a later one serves the init is degraded config, not a failure."""
    from agent import agent_init

    class _Client:
        pass

    def _fake_resolve(provider, model=None, **kwargs):
        if provider == "kimi":
            return (_Client(), model)
        return (None, None)

    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", _fake_resolve)
    monkeypatch.setattr(
        "hermes_cli.fallback_config.resolve_entry_api_key", lambda entry: None
    )
    monkeypatch.setattr(
        agent_init,
        "_client_kwargs_from_routed",
        lambda client, timeout: {"api_key": "k"},
    )

    agent = _agent()
    with caplog.at_level(logging.WARNING, logger="run_agent"):
        kwargs = agent_init._routed_client_kwargs(agent, _LADDER, 60)

    assert kwargs == {"api_key": "k"}
    assert agent.provider == "kimi"
    assert agent._fallback_activated
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_refused_init_ladder_never_logs_route_or_exception_values(monkeypatch, caplog):
    from agent import agent_init

    secret = "CANARY_INIT_SECRET_117816"
    ladder = [{"provider": "custom", "model": f"model-{secret}"}]

    def _fail(provider, *args, **kwargs):
        if provider == "custom":
            raise RuntimeError(f"credential failure {secret}")
        return (None, None)

    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", _fail)
    monkeypatch.setattr(
        "hermes_cli.fallback_config.resolve_entry_api_key", lambda entry: None
    )
    monkeypatch.setattr(
        "agent.credential_pool.load_pool", lambda provider: _Pool(True)
    )

    with caplog.at_level(logging.DEBUG, logger="run_agent"):
        with pytest.raises(RuntimeError, match="No LLM provider configured"):
            agent_init._routed_client_kwargs(_agent(), ladder, 60)

    assert "entry[0] (RuntimeError)" in caplog.text
    assert secret not in caplog.text
