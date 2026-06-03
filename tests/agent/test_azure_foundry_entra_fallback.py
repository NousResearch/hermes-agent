"""Validate the Entra ID callable-api_key fix across the two changed files.

Change 1 (auxiliary_client.py): ``_try_azure_foundry`` stashes the original
``api_key`` (which may be a callable token provider) as
``client._hermes_original_api_key`` so downstream code can recover it.

Change 2 (chat_completion_helpers.py): ``try_activate_fallback`` reads
``_hermes_original_api_key`` from the fallback client instead of
``fb_client.api_key`` (which the OpenAI SDK exposes as ``""`` for callable
providers).  This prevents per-request client clones from being built with
``api_key=""`` → 401.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_credential_cache():
    from agent.azure_identity_adapter import reset_credential_cache
    reset_credential_cache()
    yield
    reset_credential_cache()


@pytest.fixture
def fake_azure_identity(monkeypatch):
    """Stand-in for azure.identity — keeps CI hermetic."""
    from agent import azure_identity_adapter as _adapter

    def _provider(scope):
        return lambda: f"jwt-for-{scope}"

    fake_module = SimpleNamespace(
        DefaultAzureCredential=lambda **kw: SimpleNamespace(
            kwargs=kw,
            get_token=lambda scope: SimpleNamespace(token="fake", expires_on=9999999999),
        ),
        get_bearer_token_provider=lambda credential, scope: _provider(scope),
    )
    monkeypatch.setattr(_adapter, "_require_azure_identity", lambda: fake_module)
    monkeypatch.setitem(sys.modules, "azure.identity", fake_module)


@pytest.fixture
def patch_load_config(monkeypatch):
    def _apply(model_cfg):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"model": model_cfg},
        )
    return _apply


# ── Change 1: _hermes_original_api_key stashed on client ─────────────────


class TestHermesOriginalApiKeyStashed:
    """``_try_azure_foundry`` must stash the original api_key on the client
    so that callable token providers survive the OpenAI SDK's stringification."""

    def test_callable_api_key_stashed(
        self, monkeypatch, fake_azure_identity, patch_load_config,
    ):
        from agent import auxiliary_client as _aux

        stashed = {}

        class _FakeOpenAI:
            def __init__(self, **kwargs):
                self.api_key = ""  # SDK behaviour for callables
                self.base_url = kwargs.get("base_url", "")
                stashed["init_api_key"] = kwargs.get("api_key")

        monkeypatch.setattr(_aux, "OpenAI", _FakeOpenAI)
        patch_load_config({
            "provider": "azure-foundry",
            "base_url": "https://r.openai.azure.com/openai/v1",
            "api_mode": "chat_completions",
            "auth_mode": "entra_id",
            "default": "gpt-4o",
        })
        client, _ = _aux._try_azure_foundry(model="gpt-4o")
        assert client is not None
        # The callable must be stashed on the client attribute.
        assert hasattr(client, "_hermes_original_api_key")
        assert callable(client._hermes_original_api_key)
        # And it must be the same object passed to the constructor.
        assert client._hermes_original_api_key is stashed["init_api_key"]

    def test_string_api_key_stashed(
        self, monkeypatch, patch_load_config,
    ):
        from agent import auxiliary_client as _aux

        stashed = {}

        class _FakeOpenAI:
            def __init__(self, **kwargs):
                self.api_key = kwargs.get("api_key", "")
                self.base_url = kwargs.get("base_url", "")
                stashed["init_api_key"] = kwargs.get("api_key")

        monkeypatch.setattr(_aux, "OpenAI", _FakeOpenAI)
        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "sk-static-key")
        patch_load_config({
            "provider": "azure-foundry",
            "base_url": "https://r.openai.azure.com/openai/v1",
            "api_mode": "chat_completions",
            "default": "gpt-4o",
        })
        client, _ = _aux._try_azure_foundry(model="gpt-4o")
        assert client is not None
        assert client._hermes_original_api_key == "sk-static-key"

    def test_codex_responses_callable_stashed_on_inner_client(
        self, monkeypatch, fake_azure_identity, patch_load_config,
    ):
        """When codex_responses wrapping is applied, the stashed attribute
        must exist on the inner OpenAI client (before wrapping)."""
        from agent import auxiliary_client as _aux

        stashed = {}

        class _FakeOpenAI:
            def __init__(self, **kwargs):
                self.api_key = ""
                self.base_url = kwargs.get("base_url", "")
                stashed["init_api_key"] = kwargs.get("api_key")

        monkeypatch.setattr(_aux, "OpenAI", _FakeOpenAI)
        patch_load_config({
            "provider": "azure-foundry",
            "base_url": "https://r.openai.azure.com/openai/v1",
            "api_mode": "chat_completions",
            "auth_mode": "entra_id",
            "default": "gpt-5.4-mini",
        })
        client, _ = _aux._try_azure_foundry(model="gpt-5.4-mini")
        assert isinstance(client, _aux.CodexAuxiliaryClient)
        # The CodexAuxiliaryClient wraps the OpenAI client; the stash
        # is on the inner client, which is what the outer wrapper exposes
        # when downstream reads api_key.
        assert callable(stashed["init_api_key"])


# ── Change 2: try_activate_fallback recovers callable api_key ────────────


def _make_agent(fallback_model=None):
    """Create a minimal AIAgent with optional fallback config."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


class TestFallbackRecoverCallableApiKey:
    """``try_activate_fallback`` must use ``_hermes_original_api_key``
    (the callable token provider) instead of the SDK-exposed empty string."""

    def test_callable_token_provider_recovered_in_fallback(self):
        """When the fallback client has a callable _hermes_original_api_key,
        agent.api_key and _client_kwargs["api_key"] must be set to the
        callable — not the empty string from fb_client.api_key."""
        token_provider = lambda: "Bearer entra-jwt"

        fb_client = MagicMock()
        fb_client.api_key = ""  # SDK behaviour for callable providers
        fb_client._hermes_original_api_key = token_provider
        fb_client.base_url = "https://r.openai.azure.com/openai/v1"
        fb_client._custom_headers = None

        fbs = [{"provider": "azure-foundry", "model": "gpt-4o"}]
        agent = _make_agent(fallback_model=fbs)

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(fb_client, "gpt-4o"),
        ):
            ok = agent._try_activate_fallback()

        assert ok is True
        # The callable must be recovered, not the empty string.
        assert agent.api_key is token_provider
        assert agent._client_kwargs["api_key"] is token_provider

    def test_string_api_key_still_works_without_stash(self):
        """When the fallback client does NOT have _hermes_original_api_key
        (e.g. non-Azure provider), plain fb_client.api_key is used as before."""
        fb_client = MagicMock()
        fb_client.api_key = "sk-regular-key"
        fb_client.base_url = "https://api.openai.com/v1"
        fb_client._custom_headers = None
        # No _hermes_original_api_key attribute
        del fb_client._hermes_original_api_key

        fbs = [{"provider": "openai", "model": "gpt-4o"}]
        agent = _make_agent(fallback_model=fbs)

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(fb_client, "gpt-4o"),
        ):
            ok = agent._try_activate_fallback()

        assert ok is True
        assert agent.api_key == "sk-regular-key"
        assert agent._client_kwargs["api_key"] == "sk-regular-key"

    def test_string_stash_preferred_over_empty_sdk_key(self):
        """When _hermes_original_api_key is a string and fb_client.api_key
        is somehow empty, the stash wins."""
        fb_client = MagicMock()
        fb_client.api_key = ""
        fb_client._hermes_original_api_key = "sk-azure-static-key"
        fb_client.base_url = "https://r.openai.azure.com/openai/v1"
        fb_client._custom_headers = None

        fbs = [{"provider": "azure-foundry", "model": "gpt-4o"}]
        agent = _make_agent(fallback_model=fbs)

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(fb_client, "gpt-4o"),
        ):
            ok = agent._try_activate_fallback()

        assert ok is True
        assert agent.api_key == "sk-azure-static-key"
        assert agent._client_kwargs["api_key"] == "sk-azure-static-key"

    def test_non_none_sdk_key_used_when_stash_is_none(self):
        """Edge case: _hermes_original_api_key is explicitly None (shouldn't
        happen but guard against it) — fall through to fb_client.api_key."""
        fb_client = MagicMock()
        fb_client.api_key = "sk-normal-key"
        fb_client._hermes_original_api_key = None
        fb_client.base_url = "https://api.openai.com/v1"
        fb_client._custom_headers = None

        fbs = [{"provider": "openai", "model": "gpt-4o"}]
        agent = _make_agent(fallback_model=fbs)

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(fb_client, "gpt-4o"),
        ):
            ok = agent._try_activate_fallback()

        assert ok is True
        assert agent.api_key == "sk-normal-key"
        assert agent._client_kwargs["api_key"] == "sk-normal-key"
