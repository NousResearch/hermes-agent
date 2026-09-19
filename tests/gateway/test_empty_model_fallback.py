"""Tests for empty model fallback — when provider is configured but model is missing."""

from types import SimpleNamespace
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
            return_value="qwen/qwen3.8-max-0902",
        ):
            assert (
                models_mod.get_preferred_silent_default_model("nous")
                == "qwen/qwen3.8-max-0902"
            )
            # nous catalog carries qwen3.8-max-0902, so the full resolver follows.
            assert (
                models_mod.get_default_model_for_provider("nous")
                == "qwen/qwen3.8-max-0902"
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

    def test_empty_model_policy_denies_catalog_default_before_agent_construction(self):
        """A model deny policy must see the model selected by the gateway's silent default fill."""
        import pytest
        from gateway.run import GatewayRunner
        from hermes_cli.routing_policy import RoutingPolicyError

        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}
        runner.config = {"routing_policy": {"enabled": True, "deny": {"models": ["z-ai/*"]}}}

        with patch("gateway.run._resolve_gateway_model", return_value=""), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "openrouter",
                 "api_key": "test-key",
                 "base_url": "https://openrouter.ai/api/v1",
                 "api_mode": "chat_completions",
             }), \
             patch("hermes_cli.models.get_default_model_for_provider", return_value="z-ai/glm-5.2"):
            with pytest.raises(RoutingPolicyError, match="selected model"):
                runner._resolve_session_agent_runtime()

    def test_require_explicit_rejects_before_catalog_default_or_recovery(self):
        """Strict policy sees missing intent before gateway-derived model recovery."""
        import pytest
        from gateway.run import GatewayRunner
        from hermes_cli.routing_policy import RoutingPolicyError

        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}
        runner.config = {"routing_policy": {"enabled": True, "require_explicit": True}}

        with patch("gateway.run._resolve_gateway_model", return_value=""), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "requested_provider": "openai-codex", "provider": "openai-codex",
                 "base_url": "https://chatgpt.com/backend-api/codex",
             }), \
             patch("hermes_cli.models.get_default_model_for_provider", side_effect=pytest.fail) as catalog:
            with pytest.raises(RoutingPolicyError, match="explicit model"):
                runner._resolve_session_agent_runtime()
        assert not catalog.called

    def test_persisted_fast_override_is_denied_before_return(self):
        """A durable override with its own API key cannot bypass the gateway policy gate."""
        import pytest
        from gateway.run import GatewayRunner
        from hermes_cli.routing_policy import RoutingPolicyError

        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}
        runner.config = {"routing_policy": {"enabled": True, "deny": {"models": ["z-ai/*"]}}}
        runner._session_state = lambda _key: SimpleNamespace(
            conversation=SimpleNamespace(model_override={
                "model": "z-ai/glm-5.2", "provider": "openrouter", "api_key": "test-key",
                "base_url": "https://openrouter.ai/api/v1", "capabilities": {},
            })
        )
        runner._peek_session_state = runner._session_state

        with patch("gateway.run._resolve_gateway_model", return_value="allowed-model"):
            with pytest.raises(RoutingPolicyError):
                runner._resolve_session_agent_runtime(session_key="s1")

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
