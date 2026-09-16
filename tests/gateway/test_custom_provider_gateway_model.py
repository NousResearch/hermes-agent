"""A ``custom_providers`` entry's bundled model reaches gateway-created agents (#9702).

``resolve_runtime_provider()`` puts the entry's ``model`` / ``default_model`` on the runtime dict,
but ``gateway.run._runtime_agent_kwargs()`` used to drop it, so every gateway surface fell back to
``model.default`` or the provider catalog. Adoption is fill-in, matching the CLI
(``hermes_cli/cli_agent_setup_mixin.py``): a real ``model.default`` still wins. Only a fallback
entry, which names its model explicitly, replaces a configured model.
"""

from unittest.mock import patch

from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.run import (
    _RUNTIME_MODEL_OVERRIDE_KEY, _RUNTIME_MODEL_SOURCE_KEY, GatewayRunner, _adopt_runtime_model,
    _resolve_runtime_agent_kwargs)
from gateway.session import SessionSource


def _custom_runtime(**extra):
    """A resolved named-custom-provider runtime, as runtime_provider_custom builds it."""
    return {
        "name": "ollama-local",
        "provider": "custom",
        "requested_provider": "ollama-local",
        "api_key": "no-key-required",
        "base_url": "https://ollama.example/v1",
        "api_mode": "chat_completions",
        "model": "qwen2.5-coder:32b",
        **extra}


class TestAdoptRuntimeModel:
    """Fill-in vs override, and that the private keys never survive."""

    def test_fills_in_when_no_model_configured(self):
        model, kwargs = _adopt_runtime_model("", {"model": "qwen2.5-coder:32b", "provider": "custom"})
        assert model == "qwen2.5-coder:32b"

    def test_configured_model_wins(self):
        model, _ = _adopt_runtime_model("llama3.3:70b", {"model": "qwen2.5-coder:32b", "provider": "custom"})
        assert model == "llama3.3:70b"

    def test_provider_slug_as_model_is_replaced(self):
        """``--model ollama-local`` names the provider, not a model; sending it verbatim 400s."""
        model, _ = _adopt_runtime_model(
            "ollama-local",
            {"model": "qwen2.5-coder:32b", "provider": "custom", "requested_provider": "ollama-local"})
        assert model == "qwen2.5-coder:32b"

    def test_entry_name_as_model_is_replaced(self):
        model, _ = _adopt_runtime_model(
            "Ollama-Local",
            {"model": "qwen2.5-coder:32b", "provider": "custom",
             _RUNTIME_MODEL_SOURCE_KEY: "ollama-local"})
        assert model == "qwen2.5-coder:32b"

    def test_fallback_entry_overrides_configured_model(self):
        model, _ = _adopt_runtime_model(
            "llama3.3:70b",
            {"model": "meta-llama/llama-4-maverick", "provider": "openrouter",
             _RUNTIME_MODEL_OVERRIDE_KEY: True})
        assert model == "meta-llama/llama-4-maverick"

    def test_no_runtime_model_keeps_current(self):
        model, kwargs = _adopt_runtime_model("llama3.3:70b", {"provider": "custom"})
        assert model == "llama3.3:70b"
        assert kwargs == {"provider": "custom"}

    def test_blank_runtime_model_keeps_current(self):
        model, _ = _adopt_runtime_model("llama3.3:70b", {"model": "   ", "provider": "custom"})
        assert model == "llama3.3:70b"

    def test_private_keys_are_stripped(self):
        """The result is splatted into ``AIAgent(model=..., **kwargs)``; a leftover key is a TypeError."""
        _, kwargs = _adopt_runtime_model(
            "", {"model": "qwen2.5-coder:32b", "provider": "custom", "api_key": "k",
                 _RUNTIME_MODEL_OVERRIDE_KEY: True, _RUNTIME_MODEL_SOURCE_KEY: "ollama-local"})
        assert kwargs == {"provider": "custom", "api_key": "k"}


class TestResolveRuntimeAgentKwargsEmitsModel:
    """The resolvers must emit the key their callers already pop."""

    def test_bundled_model_is_emitted(self):
        with patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=_custom_runtime()):
            kwargs = _resolve_runtime_agent_kwargs()
        assert kwargs["model"] == "qwen2.5-coder:32b"
        assert kwargs[_RUNTIME_MODEL_SOURCE_KEY] == "ollama-local"

    def test_runtime_without_model_emits_no_key(self):
        runtime = _custom_runtime()
        del runtime["model"]
        with patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=runtime):
            kwargs = _resolve_runtime_agent_kwargs()
        assert "model" not in kwargs
        assert _RUNTIME_MODEL_SOURCE_KEY not in kwargs

    def test_for_provider_emits_model(self):
        from gateway.run import _resolve_runtime_agent_kwargs_for_provider
        with patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=_custom_runtime()):
            kwargs = _resolve_runtime_agent_kwargs_for_provider("ollama-local")
        assert kwargs["model"] == "qwen2.5-coder:32b"


class TestSessionRuntimeAdoptsBundledModel:
    """End to end through ``_resolve_session_agent_runtime``."""

    def _runner(self, channel_overrides=None):
        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}
        runner.config = GatewayConfig(
            platforms={Platform.DISCORD: PlatformConfig(
                enabled=True, channel_overrides=channel_overrides or {})})
        return runner

    def test_bundled_model_used_when_config_model_empty(self):
        runner = self._runner()
        with patch("gateway.run._resolve_gateway_model", return_value=""), \
             patch("gateway.run._resolve_runtime_agent_kwargs",
                   return_value=dict(_custom_runtime(), **{_RUNTIME_MODEL_SOURCE_KEY: "ollama-local"})):
            model, runtime = runner._resolve_session_agent_runtime()
        assert model == "qwen2.5-coder:32b"
        assert "model" not in runtime
        assert _RUNTIME_MODEL_SOURCE_KEY not in runtime

    def test_config_model_wins_over_bundled_model(self):
        runner = self._runner()
        with patch("gateway.run._resolve_gateway_model", return_value="llama3.3:70b"), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value=_custom_runtime()):
            model, _ = runner._resolve_session_agent_runtime(
                user_config={"model": {"default": "llama3.3:70b"}})
        assert model == "llama3.3:70b"

    def test_provider_only_channel_override_adopts_that_providers_model(self):
        runner = self._runner({"chan_1": ChannelOverride(provider="ollama-local")})
        source = SessionSource(platform=Platform.DISCORD, chat_id="chan_1", user_id="u1")
        with patch("gateway.run._resolve_gateway_model", return_value="global/model"), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "anthropic", "api_key": "k", "base_url": "https://api.anthropic.com",
                 "api_mode": "chat_completions"}), \
             patch("gateway.run._resolve_runtime_agent_kwargs_for_provider",
                   return_value=dict(_custom_runtime(), **{_RUNTIME_MODEL_SOURCE_KEY: "ollama-local"})):
            model, runtime = runner._resolve_session_agent_runtime(
                source=source, user_config={"model": {"default": "global/model"}})
        assert model == "qwen2.5-coder:32b"
        assert runtime["base_url"] == "https://ollama.example/v1"

    def test_explicit_channel_model_beats_bundled_model(self):
        runner = self._runner(
            {"chan_1": ChannelOverride(model="channel/model", provider="ollama-local")})
        source = SessionSource(platform=Platform.DISCORD, chat_id="chan_1", user_id="u1")
        with patch("gateway.run._resolve_gateway_model", return_value="global/model"), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "anthropic", "api_key": "k", "base_url": "https://api.anthropic.com",
                 "api_mode": "chat_completions"}), \
             patch("gateway.run._resolve_runtime_agent_kwargs_for_provider",
                   return_value=_custom_runtime()):
            model, _ = runner._resolve_session_agent_runtime(
                source=source, user_config={"model": {"default": "global/model"}})
        assert model == "channel/model"
