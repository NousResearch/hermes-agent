"""Tests for empty model fallback — when provider is configured but model is missing."""

from unittest.mock import patch


class TestGetDefaultModelForProvider:
    """Unit tests for hermes_cli.models.get_default_model_for_provider."""

    def test_known_provider_returns_first_model(self):
        from hermes_cli.models import get_default_model_for_provider
        result = get_default_model_for_provider("openai-codex")
        # Should return first model from _PROVIDER_MODELS["openai-codex"]
        assert result
        assert isinstance(result, str)

    def test_openrouter_returns_empty(self):
        """OpenRouter uses dynamic model fetch, no static catalog entry."""
        from hermes_cli.models import get_default_model_for_provider
        # OpenRouter is not in _PROVIDER_MODELS — it uses live fetching
        result = get_default_model_for_provider("openrouter")
        assert result == ""

    def test_unknown_provider_returns_empty(self):
        from hermes_cli.models import get_default_model_for_provider
        assert get_default_model_for_provider("nonexistent-provider") == ""

    def test_custom_provider_returns_empty(self):
        """Custom provider has no model catalog — should return empty."""
        from hermes_cli.models import get_default_model_for_provider
        # Custom providers don't have entries in _PROVIDER_MODELS
        assert get_default_model_for_provider("some-random-custom") == ""

    def test_nous_silent_default_is_not_the_expensive_flagship(self):
        """Nous Portal is a metered aggregator whose curated list is ordered
        most-capable-first, so entry [0] is the priciest flagship
        (anthropic/claude-opus-4.8). The silent fallback (provider set, no model)
        must NOT escalate to it — otherwise an unconfigured profile silently
        bills the most expensive model. Regression for the billing footgun.
        """
        from hermes_cli.models import (
            _PROVIDER_MODELS,
            _PROVIDER_SILENT_DEFAULT_OVERRIDES,
            get_default_model_for_provider,
        )

        result = get_default_model_for_provider("nous")
        assert result, "nous must resolve to a usable default model"
        assert "opus" not in result.lower(), (
            f"silent default escalated to an expensive flagship: {result!r}"
        )
        assert result != _PROVIDER_MODELS["nous"][0], (
            "silent default must not be the most-capable/priciest catalog entry"
        )
        # The override must point at a model that actually exists in the catalog.
        assert result == _PROVIDER_SILENT_DEFAULT_OVERRIDES["nous"]
        assert result in _PROVIDER_MODELS["nous"]

    def test_override_falls_back_to_catalog_when_missing(self):
        """If an override model is no longer in the catalog, fall back to [0]
        rather than returning a stale/absent id."""
        from unittest.mock import patch

        from hermes_cli import models as models_mod

        with patch.dict(
            models_mod._PROVIDER_SILENT_DEFAULT_OVERRIDES,
            {"openai-codex": "does-not-exist-model"},
            clear=False,
        ):
            result = models_mod.get_default_model_for_provider("openai-codex")
            assert result == models_mod._PROVIDER_MODELS["openai-codex"][0]


class TestGatewayEmptyModelFallback:
    """Test that _resolve_session_agent_runtime fills in empty model from provider catalog."""

    def test_empty_model_filled_from_provider(self):
        """When config has no model but provider is openai-codex, use first codex model."""
        from gateway.run import GatewayRunner
 
        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}
 
        # Mock _resolve_gateway_model to return empty string and openai-codex provider
        # Mock _resolve_runtime_agent_kwargs to return openai-codex provider
        with patch("gateway.run._resolve_gateway_model", return_value=("", "openai-codex")), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "openai-codex",
                 "api_key": "test-key",
                 "base_url": "https://chatgpt.com/backend-api/codex",
                 "api_mode": "codex_responses",
             }):
            model, kwargs = runner._resolve_session_agent_runtime()
 
        # Model should have been filled in from provider catalog
        assert model, "Model should not be empty when provider is known"
        assert isinstance(model, str)
        assert kwargs["provider"] == "openai-codex"

    def test_nonempty_model_not_overridden(self):
        """When config has a model set, don't override it."""
        from gateway.run import GatewayRunner
 
        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}
 
        with patch("gateway.run._resolve_gateway_model", return_value=("gpt-5.4", "openai-codex")), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "openai-codex",
                 "api_key": "test-key",
                 "base_url": "https://chatgpt.com/backend-api/codex",
                 "api_mode": "codex_responses",
             }):
            model, kwargs = runner._resolve_session_agent_runtime()
 
        assert model == "gpt-5.4", "Explicit model should not be overridden"

    def test_empty_model_no_provider_stays_empty(self):
        """When both model and provider are empty, model stays empty."""
        from gateway.run import GatewayRunner
 
        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}
 
        with patch("gateway.run._resolve_gateway_model", return_value=("", "")), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "",
                 "api_key": "test-key",
                 "base_url": "https://example.com",
                 "api_mode": "chat_completions",
             }):
            model, kwargs = runner._resolve_session_agent_runtime()
 
        # Can't fill in a default without knowing the provider
        assert model == ""


class TestResolveGatewayModel:
    """Test _resolve_gateway_model reads model from config correctly."""

    def test_returns_default_key(self):
        from gateway.run import _resolve_gateway_model
        assert _resolve_gateway_model({"model": {"default": "gpt-5.4"}}) == ("gpt-5.4", "")

    def test_returns_model_key_fallback(self):
        from gateway.run import _resolve_gateway_model
        assert _resolve_gateway_model({"model": {"model": "gpt-5.4"}}) == ("gpt-5.4", "")

    def test_returns_empty_when_missing(self):
        from gateway.run import _resolve_gateway_model
        assert _resolve_gateway_model({"model": {}}) == ("", "")

    def test_returns_empty_when_no_model_section(self):
        from gateway.run import _resolve_gateway_model
        assert _resolve_gateway_model({}) == ("", "")

    def test_string_model_config(self):
        from gateway.run import _resolve_gateway_model
        assert _resolve_gateway_model({"model": "my-model"}) == ("my-model", "")
