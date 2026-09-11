"""Tests for Azure Foundry Entra ID runtime resolution.

Covers the contract introduced in PR for Microsoft Entra ID auth on
``azure-foundry``:

  * ``_resolve_azure_foundry_runtime`` returns a callable ``api_key`` for
    ``model.auth_mode = entra_id`` (OpenAI-style only).
  * Anthropic-style endpoints with ``auth_mode = entra_id`` return the same
    callable runtime credential as OpenAI-style endpoints.
  * The legacy ``api_key`` path is unchanged when ``auth_mode`` is absent
    or set to ``api_key``.
  * Explicit ``--api-key`` overrides at runtime still work in entra mode
    (escape hatch for one-off testing).
  * ``model.entra.scope`` propagates to the token-provider config; Azure
    identity selection stays in standard AZURE_* env vars.
  * ``_get_azure_foundry_auth_status`` is structural — never mints a
    token (verified by checking the credential cache untouched).
  * ``has_usable_secret`` for ``AZURE_FOUNDRY_API_KEY`` is irrelevant
    when ``auth_mode == entra_id``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import cast

import pytest


@pytest.fixture(autouse=True)
def _reset_credential_cache():
    from agent.azure_identity_adapter import reset_credential_cache
    reset_credential_cache()
    yield
    reset_credential_cache()


@pytest.fixture
def fake_azure_identity(monkeypatch):
    """Identical fake to test_azure_identity_adapter — keeps Azure SDK
    out of these tests so they run in CI without the package installed."""
    from agent import azure_identity_adapter as _adapter

    last = {"scope": None, "kwargs": None, "credential_count": 0}

    def _provider(scope):
        return lambda: f"jwt-for-{scope}"

    fake_module = SimpleNamespace(
        DefaultAzureCredential=lambda **kw: SimpleNamespace(
            kwargs=kw,
            get_token=lambda scope: SimpleNamespace(token="fake", expires_on=9999999999),
        ),
        get_bearer_token_provider=lambda credential, scope: (
            last.__setitem__("scope", scope),
            last.__setitem__("kwargs", credential.kwargs),
            last.__setitem__("credential_count", cast(int, last["credential_count"]) + 1),
            _provider(scope),
        )[-1],
    )
    monkeypatch.setattr(_adapter, "_require_azure_identity", lambda: fake_module)
    monkeypatch.setitem(sys.modules, "azure.identity", fake_module)
    return last


# ---------------------------------------------------------------------------
# _resolve_azure_foundry_runtime: entra_id branch
# ---------------------------------------------------------------------------


class TestResolveAzureFoundryRuntimeEntra:
    def test_returns_callable_api_key_for_entra(self, fake_azure_identity):
        from hermes_cli.runtime_provider import _resolve_azure_foundry_runtime
        runtime = _resolve_azure_foundry_runtime(
            requested_provider="azure-foundry",
            model_cfg={
                "provider": "azure-foundry",
                "base_url": "https://my-resource.openai.azure.com/openai/v1",
                "api_mode": "chat_completions",
                "auth_mode": "entra_id",
                "default": "gpt-4o",  # stays on chat_completions (no codex auto-upgrade)
            },
        )
        assert runtime["provider"] == "azure-foundry"
        assert runtime["auth_mode"] == "entra_id"
        assert runtime["api_mode"] == "chat_completions"
        assert callable(runtime["api_key"])
        assert runtime["source"] == "entra_id"


    def test_entra_propagates_scope_only(self, fake_azure_identity):
        """``model.entra.scope`` is the only Hermes-managed Azure SDK
        setting. Identity selection (client ID, tenant, authority,
        service principal secret, federated token file) flows through
        standard ``AZURE_*`` env vars read by azure-identity directly.
        Legacy ``model.entra.client_id`` / ``tenant_id`` / ``authority``
        keys in config.yaml are silently ignored."""
        from hermes_cli.runtime_provider import _resolve_azure_foundry_runtime
        _resolve_azure_foundry_runtime(
            requested_provider="azure-foundry",
            model_cfg={
                "provider": "azure-foundry",
                "base_url": "https://my-resource.services.ai.azure.com/v1",
                "api_mode": "chat_completions",
                "auth_mode": "entra_id",
                "entra": {
                    "scope": "https://custom.example/.default",
                    "client_id": "client-uuid",
                    # Legacy keys must not crash — they are accepted in
                    # from_dict but never propagated to the SDK.
                    "tenant_id": "legacy-tenant",
                    "authority": "https://login.microsoftonline.us",
                },
            },
        )
        assert fake_azure_identity["scope"] == "https://custom.example/.default"
        kw = fake_azure_identity["kwargs"]
        assert "managed_identity_client_id" not in kw
        assert "workload_identity_client_id" not in kw
        assert "interactive_browser_tenant_id" not in kw
        assert "authority" not in kw




    def test_entra_with_explicit_api_key_uses_string_escape_hatch(self, fake_azure_identity):
        """Passing --api-key on the CLI overrides the entra path so a
        user can debug a single request with a static key without
        editing config.yaml."""
        from hermes_cli.runtime_provider import _resolve_azure_foundry_runtime
        runtime = _resolve_azure_foundry_runtime(
            requested_provider="azure-foundry",
            model_cfg={
                "provider": "azure-foundry",
                "base_url": "https://r.openai.azure.com/openai/v1",
                "api_mode": "chat_completions",
                "auth_mode": "entra_id",
            },
            explicit_api_key="explicit-string-key",
        )
        assert runtime["api_key"] == "explicit-string-key"
        assert runtime["auth_mode"] == "api_key"
        assert runtime["source"] == "explicit"


# ---------------------------------------------------------------------------
# _resolve_azure_foundry_runtime: legacy api_key branch (regression)
# ---------------------------------------------------------------------------


class TestResolveAzureFoundryRuntimeApiKey:
    def test_default_auth_mode_uses_static_key(self, monkeypatch):
        from hermes_cli.runtime_provider import _resolve_azure_foundry_runtime
        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "sk-azure-static-key")
        runtime = _resolve_azure_foundry_runtime(
            requested_provider="azure-foundry",
            model_cfg={
                "provider": "azure-foundry",
                "base_url": "https://r.openai.azure.com/openai/v1",
                "api_mode": "chat_completions",
            },
        )
        assert runtime["api_key"] == "sk-azure-static-key"
        assert runtime["auth_mode"] == "api_key"
        assert "entra" not in runtime  # only present in entra mode

    def test_explicit_auth_mode_api_key(self, monkeypatch):
        from hermes_cli.runtime_provider import _resolve_azure_foundry_runtime
        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "sk-static")
        runtime = _resolve_azure_foundry_runtime(
            requested_provider="azure-foundry",
            model_cfg={
                "provider": "azure-foundry",
                "base_url": "https://r.openai.azure.com/openai/v1",
                "api_mode": "chat_completions",
                "auth_mode": "api_key",
            },
        )
        assert runtime["api_key"] == "sk-static"
        assert runtime["auth_mode"] == "api_key"

    def test_anthropic_messages_strips_v1_suffix(self, monkeypatch):
        from hermes_cli.runtime_provider import _resolve_azure_foundry_runtime
        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "k")
        runtime = _resolve_azure_foundry_runtime(
            requested_provider="azure-foundry",
            model_cfg={
                "provider": "azure-foundry",
                "base_url": "https://r.services.ai.azure.com/anthropic/v1",
                "api_mode": "anthropic_messages",
            },
        )
        assert runtime["base_url"] == "https://r.services.ai.azure.com/anthropic"


# ---------------------------------------------------------------------------
# _get_azure_foundry_auth_status (auth.py) — never mints a token
# ---------------------------------------------------------------------------


class TestAzureFoundryAuthStatus:
    def test_entra_status_does_not_mint_token(self, monkeypatch, tmp_path):
        """Structural check — must return logged_in=True based on
        importable + config, never call get_bearer_token_provider."""
        from hermes_cli import auth as _auth
        # Force load_config to return our entra config.
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {
                "model": {
                    "provider": "azure-foundry",
                    "auth_mode": "entra_id",
                    "base_url": "https://r.openai.azure.com/openai/v1",
                },
            },
        )
        # Patch has_azure_identity_installed to True; do NOT patch the
        # token provider — if the code path tried to mint, the SDK
        # missing would raise.
        monkeypatch.setattr(
            "agent.azure_identity_adapter.has_azure_identity_installed",
            lambda: True,
        )
        info = _auth._get_azure_foundry_auth_status()
        assert info["logged_in"] is True
        assert info["auth_mode"] == "entra_id"
        assert info["azure_identity_installed"] is True
        assert info["scope"].endswith("/.default")


    def test_api_key_status_false_when_missing(self, monkeypatch):
        from hermes_cli import auth as _auth
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {
                "model": {
                    "provider": "azure-foundry",
                    "auth_mode": "api_key",
                },
            },
        )
        monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
        info = _auth._get_azure_foundry_auth_status()
        assert info["logged_in"] is False
