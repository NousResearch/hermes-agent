"""Tests for empty model fallback — when provider is configured but model is missing."""

from unittest.mock import patch

from gateway.config import Platform
from gateway.session import SessionSource


class TestGetDefaultModelForProvider:
    """Unit tests for hermes_cli.models.get_default_model_for_provider."""

    def test_known_provider_returns_first_model(self):
        from hermes_cli.models import get_default_model_for_provider
        result = get_default_model_for_provider("openai-codex")
        # Should return first model from _PROVIDER_MODELS["openai-codex"]
        assert result
        assert isinstance(result, str)





    def test_catalog_label_overrides_constant(self):
        """A ``"default": true`` label in the cached catalog manifest wins over
        the in-repo constant, so maintainers can rotate the silent default
        without shipping a release."""
        from unittest.mock import patch

        from hermes_cli import models as models_mod

        with patch(
            "hermes_cli.model_catalog.get_default_model_from_cache",
            return_value="qwen/qwen3.8-max",
        ):
            assert (
                models_mod.get_preferred_silent_default_model("nous")
                == "qwen/qwen3.8-max"
            )
            # nous catalog carries qwen3.8-max, so the full resolver follows.
            assert (
                models_mod.get_default_model_for_provider("nous")
                == "qwen/qwen3.8-max"
            )






def _make_telegram_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


class TestGatewayEmptyModelFallback:
    """Test that _resolve_session_agent_runtime fills in empty model from provider catalog."""

    def test_empty_model_filled_from_provider(self):
        """When config has no model but provider is openai-codex, use first codex model."""
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}

        # Mock _resolve_gateway_model to return empty string
        # Mock _resolve_runtime_agent_kwargs to return openai-codex provider
        with patch("gateway.run._resolve_gateway_model", return_value=""), \
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

        with patch("gateway.run._resolve_gateway_model", return_value="gpt-5.4"), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "openai-codex",
                 "api_key": "***",
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

        with patch("gateway.run._resolve_gateway_model", return_value=""), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "",
                 "api_key": "***",
                 "base_url": "https://example.com",
                 "api_mode": "chat_completions",
             }):
            model, kwargs = runner._resolve_session_agent_runtime()

        # Can't fill in a default without knowing the provider
        assert model == ""

    def test_platform_provider_override_requests_matching_runtime_provider(self):
        """Platform config should resolve credentials for its own provider, not the global one."""
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}

        cfg = {
            "model": {"default": "gpt-5.4", "provider": "openai-codex"},
            "platforms": {
                "telegram": {
                    "extra": {
                        "model": "claude-sonnet-4",
                        "provider": "anthropic",
                    }
                }
            },
        }

        seen = {}

        def _runtime_for_platform(*, requested_provider=None, explicit_api_key=None, explicit_base_url=None, requested=None):
            seen["requested"] = requested_provider if requested_provider is not None else requested
            seen["explicit_api_key"] = explicit_api_key
            seen["explicit_base_url"] = explicit_base_url
            return {
                "provider": "anthropic",
                "api_key": "ant-key",
                "base_url": "https://api.anthropic.com",
                "api_mode": "anthropic_messages",
            }

        with patch("gateway.run._resolve_runtime_agent_kwargs", side_effect=_runtime_for_platform):
            model, kwargs = runner._resolve_session_agent_runtime(
                source=_make_telegram_source(),
                user_config=cfg,
            )

        assert seen == {
            "requested": "anthropic",
            "explicit_api_key": None,
            "explicit_base_url": None,
        }
        assert model == "claude-sonnet-4"
        assert kwargs["provider"] == "anthropic"
        assert kwargs["api_key"] == "ant-key"
        assert kwargs["base_url"] == "https://api.anthropic.com"
        assert kwargs["api_mode"] == "anthropic_messages"

    def test_platform_explicit_runtime_fields_override_resolved_credentials(self):
        """Platform config should be able to pin endpoint credentials/runtime per platform."""
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}

        cfg = {
            "model": {"default": "gpt-5.4", "provider": "openai-codex"},
            "platforms": {
                "telegram": {
                    "extra": {
                        "provider": "custom",
                        "model": "local-model",
                        "api_key": "telegram-key",
                        "base_url": "https://llm.internal/v1",
                        "api_mode": "chat_completions",
                    }
                }
            },
        }

        def _runtime_for_platform(*, requested_provider=None, explicit_api_key=None, explicit_base_url=None, requested=None):
            provider = requested_provider if requested_provider is not None else requested
            return {
                "provider": provider,
                "api_key": explicit_api_key,
                "base_url": explicit_base_url,
                "api_mode": "should-be-overridden",
            }

        with patch("gateway.run._resolve_runtime_agent_kwargs", side_effect=_runtime_for_platform):
            model, kwargs = runner._resolve_session_agent_runtime(
                source=_make_telegram_source(),
                user_config=cfg,
            )

        assert model == "local-model"
        assert kwargs["provider"] == "custom"
        assert kwargs["api_key"] == "telegram-key"
        assert kwargs["base_url"] == "https://llm.internal/v1"
        assert kwargs["api_mode"] == "chat_completions"


class TestResolveGatewayModel:


    def test_returns_default_key(self):
        from gateway.run import _resolve_gateway_model
        assert _resolve_gateway_model({"model": {"default": "gpt-5.4"}}) == "gpt-5.4"

    def test_returns_model_key_fallback(self):
        from gateway.run import _resolve_gateway_model
        assert _resolve_gateway_model({"model": {"model": "gpt-5.4"}}) == "gpt-5.4"

    def test_returns_empty_when_missing(self):
        from gateway.run import _resolve_gateway_model
        assert _resolve_gateway_model({"model": {}}) == ""

    def test_returns_empty_when_no_model_section(self):
        from gateway.run import _resolve_gateway_model
        assert _resolve_gateway_model({}) == ""

    def test_string_model_config(self):
        from gateway.run import _resolve_gateway_model
        assert _resolve_gateway_model({"model": "my-model"}) == "my-model"

    def test_platform_model_override_wins_over_global_default(self):
        from gateway.run import _resolve_gateway_model

        cfg = {
            "model": {"default": "gpt-5.4"},
            "platforms": {
                "telegram": {
                    "extra": {"model": "claude-sonnet-4"}
                }
            },
        }

        assert _resolve_gateway_model(cfg, platform=Platform.TELEGRAM) == "claude-sonnet-4"
        assert _resolve_gateway_model(cfg, platform=Platform.SLACK) == "gpt-5.4"
