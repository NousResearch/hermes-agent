"""Auxiliary Anthropic client on an Azure Foundry endpoint must use the Azure key, never the Claude
Code OAuth token the ``anthropic`` pool (or the legacy resolver) holds (#154242).

Same defect as the main runtime ladder: Foundry rejects Claude Code OAuth tokens, and pairing one
with a ``*.azure.com`` host presents the subscription token to a non-Anthropic endpoint.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import agent.auxiliary_client as aux

FOUNDRY_URL = "https://myres.services.ai.azure.com/anthropic"
OAUTH_TOKEN = "claude-code-oauth-token"


@pytest.fixture
def claude_code_pool(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "AZURE_ANTHROPIC_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    entry = SimpleNamespace(provider="anthropic", access_token=OAUTH_TOKEN, runtime_api_key=OAUTH_TOKEN,
                            source="claude_code", base_url="")
    monkeypatch.setattr(aux, "_select_pool_entry", lambda *_a, **_k: (True, entry))
    built = {}

    def _fake_build(token, base_url):
        built.update(token=token, base_url=base_url)
        return MagicMock()

    monkeypatch.setattr("agent.anthropic_adapter.build_anthropic_client", _fake_build)
    return built


def _config(monkeypatch, **model_cfg):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"model": dict(model_cfg)})


def test_foundry_model_api_key_wins_over_claude_code_pool(claude_code_pool, monkeypatch):
    _config(monkeypatch, provider="anthropic", base_url=FOUNDRY_URL, api_key="KAZ")
    client, _ = aux._try_anthropic()
    assert client is not None
    assert claude_code_pool == {"token": "KAZ", "base_url": FOUNDRY_URL}


def test_foundry_azure_env_key_wins_over_claude_code_pool(claude_code_pool, monkeypatch):
    monkeypatch.setenv("AZURE_ANTHROPIC_KEY", "KAZ-ENV")
    _config(monkeypatch, provider="anthropic", base_url=FOUNDRY_URL)
    aux._try_anthropic()
    assert claude_code_pool == {"token": "KAZ-ENV", "base_url": FOUNDRY_URL}


def test_explicit_foundry_url_never_gets_the_oauth_token(claude_code_pool, monkeypatch):
    _config(monkeypatch)
    client, model = aux._try_anthropic(explicit_base_url=FOUNDRY_URL)
    assert (client, model) == (None, None)
    assert claude_code_pool == {}


def test_plain_anthropic_endpoint_still_uses_the_pool(claude_code_pool, monkeypatch):
    _config(monkeypatch, provider="anthropic")
    aux._try_anthropic()
    assert claude_code_pool["token"] == OAUTH_TOKEN
