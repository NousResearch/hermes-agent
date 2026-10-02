"""A configured custom provider keeps its pool rows under the ``providers.<key>`` slug or the
legacy ``custom:<name>`` key. Startup resolution finds rows under either; fallback activation, the
per-turn primary restore and a mid-session ``/model`` switch must bind the same pool, or 429/401
rotation silently stops for the rest of the session."""

from unittest.mock import MagicMock, patch

import pytest
from ruamel.yaml import YAML

from agent.chat_completion_helpers import try_activate_fallback
from agent.error_classifier import FailoverReason
from hermes_cli.auth import write_credential_pool
from hermes_constants import get_hermes_home
from run_agent import AIAgent

RELAY_URL = "http://127.0.0.1:9/v1"
FALLBACK = [{"provider": "openrouter", "model": "x/fb", "base_url": "http://127.0.0.1:9/or/v1"}]


@pytest.fixture
def legacy_relay_rows(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-or-key")
    config = {
        "model": {"provider": "relay", "default": "relay-model"},
        "providers": {"relay": {"name": "Relay Display", "base_url": RELAY_URL, "api_mode": "chat_completions"}},
        "fallback_providers": FALLBACK,
    }
    with open(get_hermes_home() / "config.yaml", "w") as fh:
        YAML().dump(config, fh)
    write_credential_pool("custom:relay-display", [
        {"id": f"r{i}", "label": f"k{i}", "auth_type": "api_key", "priority": i, "source": "manual",
         "access_token": f"fake-relay-key-{i}", "base_url": RELAY_URL}
        for i in (1, 2)
    ])


def _agent(provider, model, base_url, api_key, credential_pool=None, fallback=None):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.context_compressor.get_model_context_length", return_value=200_000),
    ):
        return AIAgent(
            model=model, provider=provider, api_key=api_key, base_url=base_url,
            api_mode="chat_completions", credential_pool=credential_pool, quiet_mode=True,
            skip_context_files=True, skip_memory=True, enabled_toolsets=[], fallback_model=fallback,
        )


def _ctx_len():
    return patch("agent.model_metadata.get_model_context_length", return_value=200_000)


def _away_and_back_via_fallback(agent, rt):
    with _ctx_len():
        assert try_activate_fallback(agent, FailoverReason.server_error)
    agent._rate_limited_until = 0
    agent.client = MagicMock()
    with _ctx_len():
        assert agent._restore_primary_runtime()


def _away_and_back_via_model_switch(agent, rt):
    fb = FALLBACK[0]
    with _ctx_len():
        agent.switch_model(fb["model"], fb["provider"], api_key="fake-or-key",
                           base_url=fb["base_url"], api_mode="chat_completions")
        agent.switch_model("relay-model", rt["provider"], api_key=rt["api_key"],
                           base_url=rt["base_url"], api_mode="chat_completions")


@pytest.mark.parametrize("away_and_back", [_away_and_back_via_fallback, _away_and_back_via_model_switch],
                         ids=["fallback", "model_switch"])
def test_primary_keeps_its_pool_across_a_round_trip(legacy_relay_rows, away_and_back):
    from hermes_cli.runtime_provider import resolve_runtime_provider

    rt = resolve_runtime_provider(requested="relay")
    agent = _agent(rt["provider"], "relay-model", rt["base_url"], rt["api_key"],
                   credential_pool=rt["credential_pool"], fallback=FALLBACK)
    rows_before = {e.id for e in agent._credential_pool.entries()}

    away_and_back(agent, rt)

    assert {e.id for e in agent._credential_pool.entries()} == rows_before == {"r1", "r2"}


def test_fallback_to_named_custom_attaches_the_pool_holding_its_rows(legacy_relay_rows):
    agent = _agent("openrouter", "x/primary", "http://127.0.0.1:9/primary/v1", "fake-or-key",
                   fallback=[{"provider": "relay", "model": "relay-model"}])

    with _ctx_len():
        assert try_activate_fallback(agent, FailoverReason.rate_limit)

    assert agent.provider == "relay"
    assert {e.id for e in agent._credential_pool.entries()} == {"r1", "r2"}
