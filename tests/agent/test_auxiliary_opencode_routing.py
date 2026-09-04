"""OpenCode Go/Zen auxiliary clients must honor per-model API routing (#98799).

``resolve_provider_client()`` builds auxiliary clients (compression, title
generation, vision, embeddings, session search) through its API-key
registry branch. OpenCode Go and Zen mix API surfaces per model —
``gpt-5.6-luna``/``grok-*``/``muse-spark*`` are served via ``/v1/responses``,
``minimax-*``/``qwen*``/``claude-*`` via ``/v1/messages``, everything else
via ``/v1/chat/completions`` — but the registry branch never consulted
``opencode_model_api_mode()``, so every auxiliary request went to
``/chat/completions`` and Responses-only models failed with HTTP 500
(#98799). The main-agent path (``hermes_cli/runtime_provider.py``)
re-derives the mode per model; auxiliary must match.

These tests drive the real ``resolve_provider_client`` with a real
``OPENCODE_GO_API_KEY`` env credential and mock only the OpenAI SDK
constructor, so the full production resolution path (registry lookup,
credential resolution, base-URL handling, wrap decision) runs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.auxiliary_client import (
    AnthropicAuxiliaryClient,
    CodexAuxiliaryClient,
    resolve_provider_client,
)

GO_BASE = "https://opencode.ai/zen/go/v1"
ZEN_BASE = "https://opencode.ai/zen/v1"


@pytest.fixture(autouse=True)
def _opencode_env(monkeypatch, tmp_path):
    """Isolate credential env and give opencode-go a key so the registry
    branch is reachable without touching the real environment."""
    for key in (
        "OPENCODE_GO_API_KEY", "OPENCODE_ZEN_API_KEY",
        "OPENCODE_GO_BASE_URL", "OPENCODE_ZEN_BASE_URL",
        "OPENAI_API_KEY", "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "go-test-key")
    monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "zen-test-key")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


def _fake_openai_factory():
    """Mock the OpenAI SDK constructor; capture base_url per constructed client."""
    constructed = []

    def _factory(**kwargs):
        client = MagicMock()
        client.api_key = kwargs.get("api_key", "")
        client.base_url = kwargs.get("base_url", "")
        client._construction_kwargs = kwargs
        constructed.append(client)
        return client

    return _factory, constructed


class TestOpenCodeGoResponsesModels:
    """Layer 1: Responses-only models on Go must get CodexAuxiliaryClient."""

    @pytest.mark.parametrize("model", ["gpt-5.6-luna", "grok-4.5", "muse-spark"])
    def test_go_responses_model_wraps_codex(self, model):
        factory, constructed = _fake_openai_factory()
        with patch("agent.auxiliary_client.OpenAI", side_effect=factory), \
             patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            client, resolved = resolve_provider_client("opencode-go", model)
        assert resolved == model, "model must pass through unchanged"
        assert isinstance(client, CodexAuxiliaryClient), (
            f"{model} on Go is Responses-only; plain chat.completions 500s (#98799)"
        )
        # Underlying OpenAI client keeps the /v1 base (SDK appends /responses).
        assert str(client.base_url).rstrip("/") == GO_BASE

    def test_go_responses_model_async_mode(self):
        """async consumers (web_extract, session_search) get the async codex wrapper."""
        from agent.auxiliary_client import AsyncCodexAuxiliaryClient

        factory, _ = _fake_openai_factory()
        with patch("agent.auxiliary_client.OpenAI", side_effect=factory), \
             patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            client, resolved = resolve_provider_client(
                "opencode-go", "gpt-5.6-luna", async_mode=True
            )
        assert isinstance(client, AsyncCodexAuxiliaryClient)


class TestOpenCodeGoAnthropicModels:
    """Layer 2: anthropic_messages models on Go must get AnthropicAuxiliaryClient."""

    @pytest.mark.parametrize("model", ["minimax-m2.7", "qwen3.7-max"])
    def test_go_anthropic_model_wraps_anthropic(self, model):
        factory, _ = _fake_openai_factory()
        with patch("agent.auxiliary_client.OpenAI", side_effect=factory), \
             patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            client, resolved = resolve_provider_client("opencode-go", model)
        assert resolved == model
        assert isinstance(client, AnthropicAuxiliaryClient), (
            f"{model} on Go is Anthropic-wire; chat.completions 404s (#98799)"
        )
        # Anthropic SDK appends /v1/messages itself — base must be /v1-stripped.
        assert str(client.base_url).rstrip("/") == "https://opencode.ai/zen/go"


class TestOpenCodeGoChatCompletionsModels:
    """Layer 3: chat_completions models on Go stay plain — no regression."""

    @pytest.mark.parametrize("model", ["glm-5", "kimi-k2.5", "deepseek-v4-flash"])
    def test_go_chat_model_stays_plain(self, model):
        factory, constructed = _fake_openai_factory()
        with patch("agent.auxiliary_client.OpenAI", side_effect=factory), \
             patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            client, resolved = resolve_provider_client("opencode-go", model)
        assert resolved == model
        assert not isinstance(client, (CodexAuxiliaryClient, AnthropicAuxiliaryClient)), (
            f"{model} on Go is chat_completions; wrapping would break it"
        )
        assert constructed, "underlying OpenAI client must be constructed"
        assert constructed[0].base_url.rstrip("/") == GO_BASE


class TestOpenCodeZenParity:
    """Layer 4: opencode-zen (and zen-hosted opencode-free) get the same routing."""

    @pytest.mark.parametrize("model,wrapper", [
        ("claude-sonnet-4-6", AnthropicAuxiliaryClient),
        ("gpt-5.2", CodexAuxiliaryClient),
    ])
    def test_zen_models_routed_per_model(self, model, wrapper):
        factory, _ = _fake_openai_factory()
        with patch("agent.auxiliary_client.OpenAI", side_effect=factory), \
             patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            client, resolved = resolve_provider_client("opencode-zen", model)
        assert resolved == model
        assert isinstance(client, wrapper), (
            f"{model} on Zen must resolve to {wrapper.__name__}"
        )


class TestOpenCodeCustomFamilyProvider:
    """Layer 5: custom providers extending a family slug (opencode-go-bridge,
    #85589) must re-derive api_mode per model, not trust the entry's stale
    declared api_mode."""

    def test_custom_family_entry_derives_mode_from_model(self, monkeypatch):
        factory, _ = _fake_openai_factory()
        entry = {
            "name": "opencode-go-bridge",
            "base_url": "https://opencode.ai/zen/go/v1",
            "api_key": "bridge-key",
            "api_mode": "chat_completions",  # stale: persisted for glm-5
            "model": "gpt-5.6-luna",  # target model needs Responses
        }
        monkeypatch.setattr(
            "hermes_cli.runtime_provider._get_named_custom_provider",
            lambda name: dict(entry) if name == "opencode-go-branch" or name == entry["name"] else None,
        )
        with patch("agent.auxiliary_client.OpenAI", side_effect=factory), \
             patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            client, resolved = resolve_provider_client("opencode-go-bridge", "gpt-5.6-luna")
        assert resolved == "gpt-5.6-luna"
        assert isinstance(client, CodexAuxiliaryClient), (
            "custom family provider must re-derive api_mode per model (#98799, #85589)"
        )


class TestOpenCodeBaseUrlOverrides:
    """Layer 6: base-URL env overrides must not be mangled by /v1 handling."""

    def test_go_env_base_url_override_untouched(self, monkeypatch):
        """A custom proxy override keeps its URL exactly (no opencode.ai
        /v1 re-suffix) while still wrapping for Responses models."""
        monkeypatch.setenv("OPENCODE_GO_BASE_URL", "https://proxy.example/go")
        factory, constructed = _fake_openai_factory()
        with patch("agent.auxiliary_client.OpenAI", side_effect=factory), \
             patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            client, resolved = resolve_provider_client("opencode-go", "gpt-5.6-luna")
        assert isinstance(client, CodexAuxiliaryClient)
        assert constructed[0].base_url.rstrip("/") == "https://proxy.example/go"


class TestOpenCodeAuxRoutingIntegration:
    """End-to-end through the task-level resolver — the exact repro from the
    issue: auxiliary.compression.provider=opencode-go + gpt-5.6-luna."""

    def test_compression_task_resolution_routes_responses(self):
        factory, _ = _fake_openai_factory()
        with patch("agent.auxiliary_client.OpenAI", side_effect=factory), \
             patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            client, resolved = resolve_provider_client(
                "opencode-go", "gpt-5.6-luna", task="compression"
            )
        assert resolved == "gpt-5.6-luna"
        assert isinstance(client, CodexAuxiliaryClient)


class TestOpenCodeNoRegressionOtherProviders:
    """Guard: non-OpenCode registry providers keep their existing behavior."""

    def test_zai_unchanged_by_opencode_fix(self, monkeypatch):
        monkeypatch.setenv("ZAI_API_KEY", "zai-key")
        factory, constructed = _fake_openai_factory()
        with patch("agent.auxiliary_client.OpenAI", side_effect=factory), \
             patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            client, resolved = resolve_provider_client("zai", "glm-4.5-flash")
        assert resolved == "glm-4.5-flash"
        assert not isinstance(client, (CodexAuxiliaryClient, AnthropicAuxiliaryClient))
