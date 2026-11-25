"""Auxiliary clients for OpenCode Zen/Go follow the per-model wire table the main runtime uses (#98799).

The relay serves Responses-only, Anthropic-wire and chat/completions models behind one provider, so
the transport must be derived from the resolved model — a provider-level or persisted ``api_mode``
is stale for every other model.
"""

from __future__ import annotations

import pytest
from openai import OpenAI

from agent import auxiliary_client as aux


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "sk-go-test")
    monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "sk-zen-test")
    (home / "config.yaml").write_text(
        "custom_providers:\n"
        "  - name: opencode-go-bridge\n"
        "    base_url: https://opencode.ai/zen/go/v1\n"
        "    api_key: sk-bridge\n"
        "  - name: opencode-go-pinned\n"
        "    base_url: https://opencode.ai/zen/go/v1\n"
        "    api_key: sk-pinned\n"
        "    api_mode: chat_completions\n"
    )
    return home


_WIRE_BY_MODEL = [
    ("gpt-5.6-luna", aux.CodexAuxiliaryClient),
    ("minimax-m2.5", aux.AnthropicAuxiliaryClient),
    ("glm-5", OpenAI),
]


@pytest.mark.parametrize("model, expected", _WIRE_BY_MODEL)
@pytest.mark.parametrize("stale_api_mode", [None, "chat_completions"])
def test_builtin_opencode_go_client_follows_the_model_not_the_persisted_mode(model, expected, stale_api_mode):
    client, resolved = aux.resolve_provider_client("opencode-go", model=model, api_mode=stale_api_mode)
    assert resolved == model
    assert type(client) is expected
    async_client, _ = aux.resolve_provider_client("opencode-go", model=model, api_mode=stale_api_mode, async_mode=True)
    assert (type(async_client) is aux.AsyncCodexAuxiliaryClient) == (expected is aux.CodexAuxiliaryClient)


@pytest.mark.parametrize("model, expected", _WIRE_BY_MODEL)
def test_named_custom_opencode_family_entry_follows_the_model(model, expected):
    """An ``opencode-go-*`` custom entry without an api_mode of its own gets each model's wire, like main."""
    client, resolved = aux.resolve_provider_client("custom:opencode-go-bridge", model=model)
    assert resolved == model
    assert type(client) is expected
    if expected is aux.AnthropicAuxiliaryClient:
        assert str(client._real_client.base_url).rstrip("/") == "https://opencode.ai/zen/go"
    else:
        base = client._real_client.base_url if expected is aux.CodexAuxiliaryClient else client.base_url
        assert str(base).rstrip("/") == "https://opencode.ai/zen/go/v1"


def test_named_custom_opencode_family_entry_with_declared_api_mode_is_honoured():
    """An entry that declares ``api_mode`` keeps it — the main runtime only re-derives when the entry has none."""
    client, resolved = aux.resolve_provider_client("custom:opencode-go-pinned", model="gpt-5.6-luna")
    assert resolved == "gpt-5.6-luna"
    assert type(client) is OpenAI
    assert str(client.base_url).rstrip("/") == "https://opencode.ai/zen/go/v1"
