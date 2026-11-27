"""Tests for agent.auxiliary_client._try_custom_endpoint's anthropic_messages branch.

When a user configures a custom endpoint with ``api_mode: anthropic_messages``
(e.g. MiniMax, Zhipu GLM, LiteLLM in Anthropic-proxy mode), auxiliary tasks
(compression, web_extract, session_search, title generation) must use the
native Anthropic transport rather than being silently downgraded to an
OpenAI-wire client that speaks the wrong protocol.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        "OPENAI_API_KEY", "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)


def _install_anthropic_adapter_mocks():
    """Patch build_anthropic_client so the test doesn't need the SDK."""
    fake_client = MagicMock(name="anthropic_client")
    return patch(
        "agent.anthropic_adapter.build_anthropic_client",
        return_value=fake_client,
    ), fake_client


def test_auxiliary_main_follows_live_custom_transport(tmp_path, monkeypatch):
    from openai import OpenAI
    from agent import auxiliary_client as aux

    pytest.importorskip("anthropic")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("model:\n  provider: custom\n", encoding="utf-8")
    runtime = {"provider": "custom", "base_url": "https://relay.example/v1",
               "api_key": "test-key", "api_mode": "chat_completions"}

    def client(mode=None):
        return aux._get_cached_client("main", "title-model", api_mode=mode,
                                      main_runtime=runtime, task="title_generation")[0]

    assert isinstance(client(), OpenAI)
    runtime["api_mode"] = "anthropic_messages"
    messages_client = client()
    assert isinstance(messages_client, aux.AnthropicAuxiliaryClient)
    assert messages_client.base_url == runtime["base_url"]
    assert isinstance(client("chat_completions"), OpenAI)


def test_custom_endpoint_anthropic_messages_builds_anthropic_wrapper():
    """api_mode=anthropic_messages → returns AnthropicAuxiliaryClient, not OpenAI."""
    from agent.auxiliary_client import _try_custom_endpoint, AnthropicAuxiliaryClient

    with patch(
        "agent.auxiliary_client._resolve_custom_runtime",
        return_value=(
            "https://api.minimax.io/anthropic",
            "minimax-key",
            "anthropic_messages",
        ),
    ), patch(
        "agent.auxiliary_client._read_main_model",
        return_value="claude-sonnet-4-6",
    ):
        adapter_patch, _fake_client = _install_anthropic_adapter_mocks()
        with adapter_patch:
            client, model = _try_custom_endpoint()

    assert isinstance(client, AnthropicAuxiliaryClient), (
        "Custom endpoint with api_mode=anthropic_messages must return the "
        f"native Anthropic wrapper, got {type(client).__name__}"
    )
    assert model == "claude-sonnet-4-6"
    # Wrapper should NOT be marked as OAuth — third-party endpoints are
    # always API-key authenticated.
    assert client.api_key == "minimax-key"
    assert client.base_url == "https://api.minimax.io/anthropic"


def test_custom_endpoint_anthropic_messages_falls_back_when_sdk_missing():
    """Graceful degradation when anthropic SDK is unavailable."""
    from agent.auxiliary_client import _try_custom_endpoint

    import_error = ImportError("anthropic package not installed")

    with patch(
        "agent.auxiliary_client._resolve_custom_runtime",
        return_value=("https://api.minimax.io/anthropic", "k", "anthropic_messages"),
    ), patch(
        "agent.auxiliary_client._read_main_model",
        return_value="claude-sonnet-4-6",
    ), patch(
        "agent.anthropic_adapter.build_anthropic_client",
        side_effect=import_error,
    ):
        client, model = _try_custom_endpoint()

    # Should fall back to an OpenAI-wire client rather than returning
    # (None, None) — the tool still needs to do *something*.
    assert client is not None
    assert model == "claude-sonnet-4-6"
    # OpenAI client, not AnthropicAuxiliaryClient.
    from agent.auxiliary_client import AnthropicAuxiliaryClient
    assert not isinstance(client, AnthropicAuxiliaryClient)


def test_custom_endpoint_chat_completions_still_uses_openai_wire():
    """Regression: default path (no api_mode) must remain OpenAI client."""
    from agent.auxiliary_client import _try_custom_endpoint, AnthropicAuxiliaryClient

    with patch(
        "agent.auxiliary_client._resolve_custom_runtime",
        return_value=("https://api.example.com/v1", "key", None),
    ), patch(
        "agent.auxiliary_client._read_main_model",
        return_value="my-model",
    ):
        client, model = _try_custom_endpoint()

    assert client is not None
    assert model == "my-model"
    assert not isinstance(client, AnthropicAuxiliaryClient)
