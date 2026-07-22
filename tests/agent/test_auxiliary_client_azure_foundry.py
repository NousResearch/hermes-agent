"""Tests for auxiliary client routing of the ``azure-foundry`` provider.

Covers the dedicated branch in ``agent.auxiliary_client.resolve_provider_client``
that delegates to :func:`hermes_cli.runtime_provider._resolve_azure_foundry_runtime`
instead of falling into the generic ``resolve_api_key_provider_credentials``
path (which only knows about ``AZURE_FOUNDRY_API_KEY`` and would 401 for
Entra ID users and miss ``model.base_url`` overrides for api-key users
with non-standard Foundry-projects endpoints).

Pinned scenarios:

  * ``auth_mode: api_key`` → plain OpenAI client with the static string
    key for ``chat_completions``.
  * ``auth_mode: entra_id`` + ``chat_completions`` → plain OpenAI
    client with a callable ``api_key`` (the bearer-token provider) —
    confirms the callable survives the auxiliary path end-to-end.
  * ``auth_mode: entra_id`` + GPT-5.x model → CodexAuxiliaryClient
    wrapping the OpenAI client (api_mode auto-upgrades to
    codex_responses).
  * Anthropic-style + entra_id → rejected at the runtime resolver,
    so the aux path returns ``(None, None)``.
  * Failure path when no model is configured returns ``(None, None)``
    cleanly so the auto chain falls through.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _reset_credential_cache():
    from agent.azure_identity_adapter import reset_credential_cache
    reset_credential_cache()
    yield
    reset_credential_cache()


@pytest.fixture
def fake_azure_identity(monkeypatch):
    """Stand-in for azure.identity (keeps CI hermetic when the SDK is
    not installed)."""
    from agent import azure_identity_adapter as _adapter

    last = {"scope": None}

    def _provider(scope):
        return lambda: f"jwt-for-{scope}"

    fake_module = SimpleNamespace(
        DefaultAzureCredential=lambda **kw: SimpleNamespace(
            kwargs=kw,
            get_token=lambda scope: SimpleNamespace(token="fake", expires_on=9999999999),
        ),
        get_bearer_token_provider=lambda credential, scope: (
            last.__setitem__("scope", scope),
            _provider(scope),
        )[-1],
    )
    monkeypatch.setattr(_adapter, "_require_azure_identity", lambda: fake_module)
    monkeypatch.setitem(sys.modules, "azure.identity", fake_module)
    return last


@pytest.fixture
def patch_load_config(monkeypatch):
    """Helper to set model_cfg seen by _try_azure_foundry."""
    def _apply(model_cfg):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"model": model_cfg},
        )
    return _apply


# ---------------------------------------------------------------------------
# auth_mode: api_key (default) — regression for the legacy path
# ---------------------------------------------------------------------------


class TestAuxAzureFoundryApiKey:
    def test_chat_completions_returns_plain_openai_client(self, monkeypatch, patch_load_config):
        from agent.auxiliary_client import _try_azure_foundry
        from openai import OpenAI as _OpenAI

        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "sk-azure-static-key")
        patch_load_config({
            "provider": "azure-foundry",
            "base_url": "https://r.openai.azure.com/openai/v1",
            "api_mode": "chat_completions",
            "default": "gpt-4o",
        })
        client, resolved = _try_azure_foundry(model="gpt-4o")
        assert client is not None
        assert resolved == "gpt-4o"
        assert isinstance(client, _OpenAI)
        assert client.api_key == "sk-azure-static-key"

    def test_chat_completions_apim_domain_adds_api_key_header(
        self, monkeypatch, patch_load_config,
    ):
        from agent import auxiliary_client as _aux

        captured = {}

        class _FakeOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.api_key = kwargs.get("api_key", "")
                self.base_url = kwargs.get("base_url", "")

        monkeypatch.setattr(_aux, "_create_openai_client", lambda **kwargs: _FakeOpenAI(**kwargs))
        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "sk-azure-static-key")
        patch_load_config({
            "provider": "azure-foundry",
            "base_url": "https://apim-n1ai-usw2-89fbd882c.azure-api.net/openai/v1",
            "api_mode": "chat_completions",
            "default": "gpt-4o",
        })
        client, resolved = _aux._try_azure_foundry(model="gpt-4o")
        assert client is not None
        assert resolved == "gpt-4o"
        assert captured["default_headers"]["api-key"] == "sk-azure-static-key"

    def test_codex_responses_wraps_in_codex_aux_client(self, monkeypatch, patch_load_config):
        from agent.auxiliary_client import _try_azure_foundry, CodexAuxiliaryClient

        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "sk-azure-static-key")
        patch_load_config({
            "provider": "azure-foundry",
            "base_url": "https://r.openai.azure.com/openai/v1",
            "api_mode": "chat_completions",
            "default": "gpt-5.4-mini",
        })
        # GPT-5.x → runtime auto-upgrades to codex_responses
        client, resolved = _try_azure_foundry(model="gpt-5.4-mini")
        assert resolved == "gpt-5.4-mini"
        assert isinstance(client, CodexAuxiliaryClient)
        assert client.api_key == "sk-azure-static-key"

    def test_no_key_returns_none(self, monkeypatch, patch_load_config):
        from agent.auxiliary_client import _try_azure_foundry

        monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
        patch_load_config({
            "provider": "azure-foundry",
            "base_url": "https://r.openai.azure.com/openai/v1",
            "api_mode": "chat_completions",
            "default": "gpt-4o",
        })
        client, resolved = _try_azure_foundry(model="gpt-4o")
        assert client is None
        assert resolved is None

    def test_no_model_returns_none(self, monkeypatch, patch_load_config):
        """Azure has no fallback aux model — fail soft so the auto chain
        can try other providers."""
        from agent.auxiliary_client import _try_azure_foundry

        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "sk-azure-static-key")
        patch_load_config({
            "provider": "azure-foundry",
            "base_url": "https://r.openai.azure.com/openai/v1",
            "api_mode": "chat_completions",
            # No default model
        })
        client, resolved = _try_azure_foundry()
        assert client is None
        assert resolved is None


# ---------------------------------------------------------------------------
# auth_mode: entra_id — callable api_key survives end-to-end
# ---------------------------------------------------------------------------


class TestAuxAzureFoundryEntra:
    def test_callable_api_key_reaches_openai_constructor(
        self, monkeypatch, fake_azure_identity, patch_load_config,
    ):
        """The token provider callable must arrive at ``OpenAI(api_key=...)``
        intact — never stringified to ``"no-key-required"`` or to the
        SDK-internal empty-string representation BEFORE we hand it off.

        We assert on the public SDK contract (constructor receives the
        callable) rather than ``client.api_key``, because OpenAI 2.24.0
        stores callable api_keys in a private attribute and exposes
        ``client.api_key`` as ``""``. The SDK still calls the callable
        per request to mint ``Authorization: Bearer <token>``; that
        behaviour is the documented Microsoft/OpenAI contract we rely on.
        """
        from agent import auxiliary_client as _aux

        received = {}

        class _FakeOpenAI:
            def __init__(self, **kwargs):
                received.update(kwargs)
                # Mirror the fields downstream callers read.
                self.api_key = kwargs.get("api_key", "")
                self.base_url = kwargs.get("base_url", "")

        monkeypatch.setattr(_aux, "OpenAI", _FakeOpenAI)
        patch_load_config({
            "provider": "azure-foundry",
            "base_url": "https://r.openai.azure.com/openai/v1",
            "api_mode": "chat_completions",
            "auth_mode": "entra_id",
            "default": "gpt-4o",
        })
        client, resolved = _aux._try_azure_foundry(model="gpt-4o")
        assert client is not None
        assert resolved == "gpt-4o"
        # Public-contract assertion: the OpenAI SDK constructor saw the
        # callable, exactly as Microsoft's Foundry sample requires.
        assert callable(received["api_key"])
        assert not isinstance(received["api_key"], str)
        assert received["api_key"]().startswith("jwt-for-")
        # Base URL forwarded verbatim (no /responses suffix stripping
        # in this path — that's a separate concern handled by the
        # runtime resolver only when the user re-saves config).
        assert received["base_url"] == "https://r.openai.azure.com/openai/v1"

    def test_codex_responses_with_entra_wraps_correctly(
        self, monkeypatch, fake_azure_identity, patch_load_config,
    ):
        """GPT-5.x deployment on Entra ID — auto-upgraded to
        codex_responses, wrapped in CodexAuxiliaryClient, callable
        api_key handed to the underlying OpenAI SDK."""
        from agent import auxiliary_client as _aux

        received = {}

        class _FakeOpenAI:
            def __init__(self, **kwargs):
                received.update(kwargs)
                self.api_key = kwargs.get("api_key", "")
                self.base_url = kwargs.get("base_url", "")

        monkeypatch.setattr(_aux, "OpenAI", _FakeOpenAI)
        patch_load_config({
            "provider": "azure-foundry",
            "base_url": "https://r.openai.azure.com/openai/v1",
            "api_mode": "chat_completions",
            "auth_mode": "entra_id",
            "default": "gpt-5.4-mini",
        })
        client, resolved = _aux._try_azure_foundry(model="gpt-5.4-mini")
        assert resolved == "gpt-5.4-mini"
        assert isinstance(client, _aux.CodexAuxiliaryClient)
        # The Codex wrapper received an OpenAI client built with the
        # callable api_key — verify against the SDK constructor record,
        # not the wrapper attribute (which mirrors the SDK's empty-
        # string representation).
        assert callable(received["api_key"])
        assert received["api_key"]().startswith("jwt-for-")

    def test_entra_anthropic_messages_uses_bearer_hook(
        self, monkeypatch, fake_azure_identity, patch_load_config,
    ):
        """Entra ID + anthropic_messages: runtime returns a callable
        api_key; ``_maybe_wrap_anthropic`` → ``build_anthropic_client``
        detects the callable and installs the bearer-injecting httpx
        event hook on a custom ``httpx.Client`` passed to the
        Anthropic SDK via ``http_client=``."""
        from agent import auxiliary_client as _aux
        from agent import anthropic_adapter as _anthropic

        received = {}

        class _FakeOpenAI:
            def __init__(self, **kwargs):
                received["openai"] = kwargs
                self.api_key = kwargs.get("api_key", "")
                self.base_url = kwargs.get("base_url", "")

        class _FakeAnthropicSDK:
            class Anthropic:
                def __init__(self, **kwargs):
                    received["anthropic"] = kwargs

        monkeypatch.setattr(_aux, "OpenAI", _FakeOpenAI)
        monkeypatch.setattr(_anthropic, "_get_anthropic_sdk", lambda: _FakeAnthropicSDK)

        patch_load_config({
            "provider": "azure-foundry",
            "base_url": "https://r.services.ai.azure.com/anthropic",
            "api_mode": "anthropic_messages",
            "auth_mode": "entra_id",
            "default": "claude-sonnet-4-5",
        })
        client, resolved = _aux._try_azure_foundry(model="claude-sonnet-4-5")
        assert client is not None
        assert resolved == "claude-sonnet-4-5"
        # The Anthropic SDK constructor received a custom http_client
        # (the bearer-injecting hook) and a placeholder auth_token.
        anthropic_kwargs = received.get("anthropic") or {}
        assert "http_client" in anthropic_kwargs, (
            "build_anthropic_client must pass a custom http_client when "
            "given a callable api_key, otherwise the SDK cannot mint "
            "fresh tokens per request"
        )
        assert anthropic_kwargs.get("auth_token") == "entra-id-bearer-via-http-hook"
        # Verify the http_client actually has our event hook installed.
        http_client = anthropic_kwargs["http_client"]
        hooks = getattr(http_client, "event_hooks", {})
        assert "request" in hooks and len(hooks["request"]) >= 1


# ---------------------------------------------------------------------------
# resolve_provider_client → azure-foundry dispatch
# ---------------------------------------------------------------------------


class TestResolveProviderClientAzureFoundry:
    def test_dispatches_to_azure_branch_not_generic_api_key_path(
        self, monkeypatch, fake_azure_identity, patch_load_config,
    ):
        """End-to-end: the public ``resolve_provider_client`` entry
        point must take the dedicated azure-foundry branch, NOT the
        generic api-key registry path that would call
        ``resolve_api_key_provider_credentials`` and return None for
        Entra users."""
        from agent import auxiliary_client as _aux

        received = {}

        class _FakeOpenAI:
            def __init__(self, **kwargs):
                received.update(kwargs)
                self.api_key = kwargs.get("api_key", "")
                self.base_url = kwargs.get("base_url", "")

        monkeypatch.setattr(_aux, "OpenAI", _FakeOpenAI)
        patch_load_config({
            "provider": "azure-foundry",
            "base_url": "https://r.openai.azure.com/openai/v1",
            "api_mode": "chat_completions",
            "auth_mode": "entra_id",
            "default": "gpt-4o",
        })
        client, resolved = _aux.resolve_provider_client("azure-foundry", "gpt-4o")
        assert client is not None
        assert resolved == "gpt-4o"
        # The callable made it through resolve_provider_client → _try_azure_foundry
        # → OpenAI(api_key=...).
        assert callable(received["api_key"])

    def test_warns_and_returns_none_on_failure(
        self, monkeypatch, patch_load_config, caplog,
    ):
        """When azure-foundry is requested but cannot be resolved
        (e.g. no model + no key), we return (None, None) and log a
        clear warning pointing at ``hermes doctor``."""
        import logging
        from agent.auxiliary_client import resolve_provider_client

        monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
        patch_load_config({
            "provider": "azure-foundry",
            "base_url": "https://r.openai.azure.com/openai/v1",
            "api_mode": "chat_completions",
            # No default → resolver yields no model → bail
        })
        with caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
            client, resolved = resolve_provider_client("azure-foundry")
        assert client is None
        assert resolved is None
        assert any(
            "azure-foundry" in rec.message and "hermes doctor" in rec.message
            for rec in caplog.records
        )
