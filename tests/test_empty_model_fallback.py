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


class TestAdoptRuntimeModel:
    """Bundled custom_providers model is fill-in (CLI rules); fallback overrides."""

    def test_fills_empty_current_model(self):
        from gateway.run import _adopt_runtime_model

        model, kwargs = _adopt_runtime_model(
            "",
            {"model": "qwen3:32b", "provider": "custom", "requested_provider": "local-ollama"},
        )
        assert model == "qwen3:32b"
        assert "model" not in kwargs
        assert "_runtime_model_override" not in kwargs

    def test_config_model_wins_over_bundled_model(self):
        from gateway.run import _adopt_runtime_model

        model, kwargs = _adopt_runtime_model(
            "llama3:8b",
            {"model": "qwen3:32b", "provider": "custom", "requested_provider": "local-ollama"},
        )
        assert model == "llama3:8b"
        assert "model" not in kwargs

    def test_provider_slug_is_replaced_with_bundled_model(self):
        from gateway.run import _adopt_runtime_model

        model, _kwargs = _adopt_runtime_model(
            "local-ollama",
            {"model": "qwen3:32b", "provider": "custom", "requested_provider": "local-ollama"},
        )
        assert model == "qwen3:32b"

    def test_fallback_flag_overrides_nonempty_config_model(self):
        from gateway.run import _adopt_runtime_model

        model, kwargs = _adopt_runtime_model(
            "llama3:8b",
            {
                "model": "fallback/model",
                "provider": "openrouter",
                "_runtime_model_override": True,
            },
        )
        assert model == "fallback/model"
        assert "model" not in kwargs
        assert "_runtime_model_override" not in kwargs


class TestGatewayRuntimeCustomProviderModel:
    """A custom_providers entry with a ``model`` field propagates it to the
    gateway agent (#9702): the runtime resolver emits the model and
    _resolve_session_agent_runtime pops it before AIAgent construction."""

    def _resolve(self, config_model, runtime_extra=None):
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}
        runtime = {
            "provider": "custom",
            "requested_provider": "local-ollama",
            "api_key": "no-key-required",
            "base_url": "https://ollama.local/v1",
            "api_mode": "chat_completions",
            "model": "qwen3:32b",
        }
        if runtime_extra:
            runtime.update(runtime_extra)

        with patch("gateway.run._resolve_gateway_model", return_value=config_model), \
             patch("hermes_cli.runtime_provider._get_model_config", return_value={}), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=runtime):
            return runner._resolve_session_agent_runtime()

    def test_runtime_model_used_when_config_model_empty(self):
        model, kwargs = self._resolve("")
        assert model == "qwen3:32b"
        assert "model" not in kwargs
        assert kwargs["provider"] == "custom"

    def test_config_model_beats_runtime_bundled_model(self):
        """custom_providers.model fills in an empty default; it does not
        replace a user-selected model.default (CLI fill-in rules)."""
        model, kwargs = self._resolve("llama3:8b")
        assert model == "llama3:8b"
        assert "model" not in kwargs

    def test_runtime_model_used_when_config_model_is_provider_slug(self):
        model, kwargs = self._resolve("local-ollama")
        assert model == "qwen3:32b"
        assert "model" not in kwargs

    def test_fallback_runtime_model_overrides_config_model(self):
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}
        with patch("gateway.run._resolve_gateway_model", return_value="llama3:8b"), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "openrouter",
                 "api_key": "sk-fb",
                 "base_url": "https://openrouter.ai/api/v1",
                 "api_mode": "chat_completions",
                 "model": "fallback/model",
                 "_runtime_model_override": True,
             }):
            model, kwargs = runner._resolve_session_agent_runtime()
        assert model == "fallback/model"
        assert "model" not in kwargs
        assert "_runtime_model_override" not in kwargs
