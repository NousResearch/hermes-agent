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

        with patch("gateway.run._resolve_gateway_model", return_value=""), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "",
                 "api_key": "test-key",
                 "base_url": "https://example.com",
                 "api_mode": "chat_completions",
             }):
            model, kwargs = runner._resolve_session_agent_runtime()

        # Can't fill in a default without knowing the provider
        assert model == ""


class TestPlatformModelsRouting:
    """Test config.platform_models — per-platform default model/provider routing."""

    def test_platform_models_routes_to_correct_provider_and_model(self):
        """When platform_models matches the session platform, use that provider+model."""
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}

        cfg = {
            "model": {"default": "gpt-5.4"},
            "platform_models": {
                "weixin": {"provider": "qwen", "model": "qwen3.6-plus"},
            },
        }

        fake_runtime = {
            "provider": "custom:qwen",
            "api_key": "qwen-key",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_mode": "chat_completions",
            "model": "qwen3.6-plus",
        }

        with patch(
            "gateway.run._resolve_gateway_model", return_value="gpt-5.4"
        ), patch(
            "gateway.run._resolve_runtime_agent_kwargs",
            return_value={"provider": "openai-codex"},
        ), patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value=fake_runtime,
        ):
            model, kwargs = runner._resolve_session_agent_runtime(
                session_key="agent:main:weixin:dm:u1",
                user_config=cfg,
            )

        assert model == "qwen3.6-plus"
        assert kwargs["provider"] == "custom:qwen"
        assert kwargs["api_key"] == "qwen-key"
        assert kwargs["base_url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"
        assert kwargs["api_mode"] == "chat_completions"

    def test_session_override_wins_over_platform_models(self):
        """Session /model override takes precedence over platform_models."""
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {
            "agent:main:weixin:dm:u1": {
                "model": "override-model",
                "provider": "override-prov",
                "api_key": "override-key",
            }
        }

        cfg = {
            "model": {"default": "gpt-5.4"},
            "platform_models": {
                "weixin": {"provider": "qwen", "model": "qwen3.6-plus"},
            },
        }

        with patch("gateway.run._resolve_gateway_model", return_value="gpt-5.4"):
            model, kwargs = runner._resolve_session_agent_runtime(
                session_key="agent:main:weixin:dm:u1",
                user_config=cfg,
            )

        # Session override should win — platform_models ignored
        assert model == "override-model"
        assert kwargs["provider"] == "override-prov"
        assert kwargs["api_key"] == "override-key"

    def test_platform_models_missing_key_is_silent_noop(self):
        """When session platform isn't in platform_models, fall through gracefully."""
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}

        cfg = {
            "model": {"default": "gpt-5.4"},
            "platform_models": {
                "weixin": {"provider": "qwen", "model": "qwen3.6-plus"},
            },
        }

        with patch("gateway.run._resolve_gateway_model", return_value="gpt-5.4"), patch(
            "gateway.run._resolve_runtime_agent_kwargs",
            return_value={"provider": "openai-codex"},
        ):
            model, kwargs = runner._resolve_session_agent_runtime(
                session_key="agent:main:dingtalk:dm:u1",
                user_config=cfg,
            )

        # Should fall through to global default — no error
        assert model == "gpt-5.4"
        assert kwargs["provider"] == "openai-codex"

    def test_no_platform_models_section_is_noop(self):
        """When config has no platform_models key, behave as before."""
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}

        cfg = {"model": {"default": "gpt-5.4"}}

        with patch("gateway.run._resolve_gateway_model", return_value="gpt-5.4"), patch(
            "gateway.run._resolve_runtime_agent_kwargs",
            return_value={"provider": "openai-codex"},
        ):
            model, kwargs = runner._resolve_session_agent_runtime(
                session_key="agent:main:weixin:dm:u1",
                user_config=cfg,
            )

        assert model == "gpt-5.4"

    def test_platform_models_resolution_failure_is_caught(self):
        """If resolve_runtime_provider explodes, the block degrades silently."""
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}

        cfg = {
            "model": {"default": "gpt-5.4"},
            "platform_models": {
                "weixin": {"provider": "nonexistent-prov", "model": "should-not-matter"},
            },
        }

        with patch("gateway.run._resolve_gateway_model", return_value="gpt-5.4"), patch(
            "gateway.run._resolve_runtime_agent_kwargs",
            return_value={"provider": "openai-codex"},
        ), patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=RuntimeError("simulated credential failure"),
        ):
            model, kwargs = runner._resolve_session_agent_runtime(
                session_key="agent:main:weixin:dm:u1",
                user_config=cfg,
            )

        # Exception caught — falls through to global default
        assert model == "gpt-5.4"
        assert kwargs["provider"] == "openai-codex"


class TestResolveGatewayModel:
    """Test _resolve_gateway_model reads model from config correctly."""

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
