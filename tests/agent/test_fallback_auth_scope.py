"""An exhausted key is probed once per provider route, without suppressing independent keys."""

from unittest.mock import patch

import pytest

from agent.error_classifier import FailoverReason
from tests.agent.test_provider_fallback import _make_agent, _mock_client


@pytest.mark.parametrize("reason", [FailoverReason.auth, FailoverReason.auth_permanent])
def test_auth_skips_sibling_models_but_uses_explicit_local_fallback(reason):
    agent = _make_agent(fallback_model=[
        *({"provider": "openrouter", "model": f"other-{i}"} for i in range(20)),
        {"provider": "custom", "model": "local/model", "base_url": "http://127.0.0.1:8080/v1"},
    ])
    agent.provider = "openrouter"
    agent._credential_pool = None
    destinations = []

    def resolve(provider, **kwargs):
        destinations.append(provider)
        return _mock_client(base_url=kwargs.get("explicit_base_url") or "https://openrouter.ai/api/v1", api_key=agent.api_key), kwargs["model"]

    with patch("agent.auxiliary_client.resolve_provider_client", side_effect=resolve):
        assert agent._try_activate_fallback(reason)
    assert agent.provider == "custom" and agent.model == "local/model"
    assert destinations == ["openrouter"] * 20 + ["custom"]


@pytest.mark.parametrize("independent", ["key", "endpoint", "pool", "non_auth"])
def test_independent_auth_routes_are_still_reachable(independent):
    entry = {"provider": "openrouter", "model": "alternative"}
    if independent == "key":
        entry["api_key"] = "independent-test-key"
    if independent == "endpoint":
        entry["base_url"] = "http://127.0.0.1:8081/v1"
    agent = _make_agent(fallback_model=[entry])
    agent.provider = "openrouter"
    if independent != "pool":
        agent._credential_pool = None
    else:
        agent._credential_pool = object()
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(_mock_client(api_key=entry.get("api_key") or agent.api_key, base_url=entry.get("base_url") or "https://openrouter.ai/api/v1"), "alternative")):
        assert agent._try_activate_fallback(FailoverReason.rate_limit if independent == "non_auth" else FailoverReason.auth)
    assert agent.model == "alternative"


def test_repaired_credentials_are_retried_on_the_next_turn():
    agent = _make_agent(fallback_model=[
        {"provider": "openrouter", "model": "alternative"},
        {"provider": "custom", "model": "local/model", "base_url": "http://127.0.0.1:8080/v1"},
    ])
    agent.provider = "openrouter"
    agent._credential_pool = None
    agent._current_turn_id = "first-turn"
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(_mock_client(api_key=agent.api_key), "local/model")):
        assert agent._try_activate_fallback(FailoverReason.auth)
    agent.provider = "openrouter"
    agent._current_turn_id = "next-turn"
    agent._fallback_index = 0
    agent.api_key = "repaired-test-key"
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(_mock_client(api_key=agent.api_key), "alternative")):
        assert agent._try_activate_fallback(FailoverReason.rate_limit)
    assert agent.model == "alternative"


@pytest.mark.parametrize("difference", ["key", "endpoint"])
def test_implicit_healthy_provider_defaults_survive_a_failed_override(difference):
    agent = _make_agent(fallback_model=[{"provider": "openrouter", "model": "healthy-default"}])
    agent.provider = "openrouter"
    agent._credential_pool = None
    client = _mock_client(api_key=agent.api_key)
    if difference == "key":
        client.api_key = "healthy-default-test-key"
    else:
        agent.base_url = "http://127.0.0.1:8089/v1"
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(client, "healthy-default")):
        assert agent._try_activate_fallback(FailoverReason.auth)
    assert agent.model == "healthy-default"


def test_model_dependent_default_endpoint_is_not_quarantined():
    agent = _make_agent(fallback_model=[
        {"provider": "openrouter", "model": "rejected-model"},
        {"provider": "openrouter", "model": "healthy-model"},
    ])
    agent.provider = "openrouter"
    agent._credential_pool = None

    def resolve(_provider, **kwargs):
        endpoint = agent.base_url if kwargs["model"] == "rejected-model" else "http://127.0.0.1:8091/v1"
        return _mock_client(base_url=endpoint, api_key=agent.api_key), kwargs["model"]

    with patch("agent.auxiliary_client.resolve_provider_client", side_effect=resolve):
        assert agent._try_activate_fallback(FailoverReason.auth)
    assert agent.model == "healthy-model"
