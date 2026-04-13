"""Tests for custom provider model switching fixes.

Covers three bugs:
1. api_mode not passed to model validation/probe (wrong catalog for Anthropic proxies)
2. custom:<name>:<model> triple syntax broken when currently on a custom provider
3. /model picker shows 0 models for custom_providers with models: dict
"""

import pytest
from unittest.mock import patch, MagicMock


# =============================================================================
# 1. parse_model_input — triple syntax
# =============================================================================

class TestParseModelInputTripleSyntax:
    """Regression: custom:<name>:<model> must resolve correctly."""

    def test_triple_syntax_returns_named_provider(self):
        from hermes_cli.models import parse_model_input
        provider, model = parse_model_input("custom:cch_anthropic:glm-5-turbo", "custom:cch_openai")
        assert provider == "custom:cch_anthropic"
        assert model == "glm-5-turbo"

    def test_triple_syntax_from_non_custom_provider(self):
        from hermes_cli.models import parse_model_input
        provider, model = parse_model_input("custom:cch_openai:MiniMax-M2.1", "openai")
        assert provider == "custom:cch_openai"
        assert model == "MiniMax-M2.1"

    def test_single_colon_custom_still_works(self):
        from hermes_cli.models import parse_model_input
        provider, model = parse_model_input("custom:glm-5-turbo", "openai")
        assert provider == "custom"
        assert model == "glm-5-turbo"

    def test_non_custom_input_returns_current_provider(self):
        from hermes_cli.models import parse_model_input
        provider, model = parse_model_input("glm-5-turbo", "openai")
        assert provider == "openai"
        assert model == "glm-5-turbo"


# =============================================================================
# 2. validate_requested_model — api_mode propagation
# =============================================================================

class TestValidateRequestedModelApiMode:
    """Regression: api_mode must be forwarded to probe_api_models."""

    def test_custom_provider_with_anthropic_api_mode_sends_header(self):
        """When api_mode=anthropic_messages, probe must include anthropic-version header."""
        from hermes_cli.models import validate_requested_model

        fake_models = ["glm-5-turbo", "MiniMax-M2.1"]
        probe_result = {"models": fake_models, "probed_url": "http://localhost:23000/v1/models"}

        with patch("hermes_cli.models.probe_api_models", return_value=probe_result) as mock_probe:
            result = validate_requested_model(
                "glm-5-turbo",
                "custom:cch_anthropic",
                api_key="sk-test",
                base_url="http://localhost:23000/v1",
                api_mode="anthropic_messages",
            )

        mock_probe.assert_called_once()
        call_kwargs = mock_probe.call_args
        assert call_kwargs.kwargs.get("api_mode") == "anthropic_messages", \
            "probe_api_models should receive api_mode='anthropic_messages'"
        assert result["accepted"] is True
        assert result["recognized"] is True

    def test_named_custom_provider_recognized(self):
        """custom:<name> prefix must be recognized as custom, not fall through to OpenAI."""
        from hermes_cli.models import validate_requested_model

        fake_models = ["glm-5-turbo"]
        probe_result = {"models": fake_models, "probed_url": "http://localhost:23000/v1/models"}

        with patch("hermes_cli.models.probe_api_models", return_value=probe_result) as mock_probe:
            result = validate_requested_model(
                "glm-5-turbo",
                "custom:cch_anthropic",
                api_key="sk-test",
                base_url="http://localhost:23000/v1",
            )

        # Must call probe (custom provider path), not skip to generic OpenAI
        mock_probe.assert_called_once()
        assert result["accepted"] is True

    def test_plain_custom_without_api_mode(self):
        """Plain custom provider (no api_mode) should still work."""
        from hermes_cli.models import validate_requested_model

        fake_models = ["my-local-model"]
        probe_result = {"models": fake_models, "probed_url": "http://localhost:8000/v1/models"}

        with patch("hermes_cli.models.probe_api_models", return_value=probe_result) as mock_probe:
            result = validate_requested_model(
                "my-local-model",
                "custom",
                api_key="sk-test",
                base_url="http://localhost:8000/v1",
            )

        mock_probe.assert_called_once()
        call_kwargs = mock_probe.call_args
        assert call_kwargs.kwargs.get("api_mode") is None, \
            "Plain custom should not pass api_mode"
        assert result["accepted"] is True


# =============================================================================
# 3. probe_api_models — anthropic-version header
# =============================================================================

class TestProbeApiModelsAnthropicHeader:
    """Ensure anthropic-version header is added when api_mode=anthropic_messages."""

    def test_anthropic_mode_includes_header(self):
        from hermes_cli.models import probe_api_models

        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.read.return_value = b'{"data": [{"id": "glm-5-turbo"}]}'
        fake_response.__enter__ = MagicMock(return_value=fake_response)
        fake_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=fake_response) as mock_urlopen:
            result = probe_api_models(
                api_key="sk-test",
                base_url="http://localhost:23000/v1",
                api_mode="anthropic_messages",
            )

        # Check the request was made with anthropic-version header
        req = mock_urlopen.call_args[0][0]
        # urllib.request.Request stores headers with original casing
        header_found = any(
            k.lower() == "anthropic-version" and v == "2023-06-01"
            for k, v in req.headers.items()
        )
        assert header_found, \
            "anthropic-version header must be present for anthropic_messages mode"
        assert "glm-5-turbo" in (result.get("models") or [])

    def test_non_anthropic_mode_omits_header(self):
        from hermes_cli.models import probe_api_models

        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.read.return_value = b'{"data": [{"id": "gpt-5"}]}'
        fake_response.__enter__ = MagicMock(return_value=fake_response)
        fake_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=fake_response) as mock_urlopen:
            result = probe_api_models(
                api_key="sk-test",
                base_url="http://localhost:8000/v1",
            )

        req = mock_urlopen.call_args[0][0]
        has_anthropic_header = any(
            k.lower() == "anthropic-version" for k in req.headers
        )
        assert not has_anthropic_header, \
            "anthropic-version header must NOT be present for non-anthropic mode"


# =============================================================================
# 4. list_authenticated_providers — models: dict from custom_providers
# =============================================================================

class TestCustomProvidersModelsDict:
    """Regression: models: dict in custom_providers must appear in picker listing."""

    def test_models_dict_format(self, monkeypatch):
        """models: {name: {context_length: N}} should list all model names."""
        from hermes_cli.model_switch import list_authenticated_providers

        monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
        monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

        custom_providers = [
            {
                "name": "CCH Anthropic",
                "base_url": "http://localhost:23000/v1",
                "api_key": "sk-test",
                "api_mode": "anthropic_messages",
                "models": {
                    "glm-5-turbo": {"context_length": 202752},
                    "MiniMax-M2.1": {"context_length": 196608},
                },
            }
        ]

        providers = list_authenticated_providers(
            current_provider="custom",
            custom_providers=custom_providers,
        )

        custom = next((p for p in providers if p.get("is_user_defined")), None)
        assert custom is not None, "Custom provider should appear in listing"
        assert custom["total_models"] == 2, f"Expected 2 models, got {custom['total_models']}"
        assert "glm-5-turbo" in custom["models"]
        assert "MiniMax-M2.1" in custom["models"]

    def test_models_list_format(self, monkeypatch):
        """models: [name, ...] should list all model names."""
        from hermes_cli.model_switch import list_authenticated_providers

        monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
        monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

        custom_providers = [
            {
                "name": "My Local",
                "base_url": "http://localhost:8000/v1",
                "api_key": "sk-test",
                "models": ["model-a", "model-b", "model-c"],
            }
        ]

        providers = list_authenticated_providers(
            current_provider="custom",
            custom_providers=custom_providers,
        )

        custom = next((p for p in providers if p.get("is_user_defined")), None)
        assert custom is not None
        assert custom["total_models"] == 3
        assert "model-a" in custom["models"]
        assert "model-b" in custom["models"]
        assert "model-c" in custom["models"]

    def test_models_dict_dedupes_with_default_model(self, monkeypatch):
        """If model (singular) is also in models dict, don't duplicate."""
        from hermes_cli.model_switch import list_authenticated_providers

        monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
        monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

        custom_providers = [
            {
                "name": "Dup Test",
                "base_url": "http://localhost:8000/v1",
                "api_key": "sk-test",
                "model": "glm-5-turbo",  # default, also in models dict
                "models": {
                    "glm-5-turbo": {"context_length": 202752},
                    "other-model": {"context_length": 8192},
                },
            }
        ]

        providers = list_authenticated_providers(
            current_provider="custom",
            custom_providers=custom_providers,
        )

        custom = next((p for p in providers if p.get("is_user_defined")), None)
        assert custom is not None
        assert custom["total_models"] == 2, \
            f"Expected 2 unique models (deduped), got {custom['total_models']}"
        assert custom["models"].count("glm-5-turbo") == 1

    def test_no_models_field_still_works(self, monkeypatch):
        """Providers without models: field should still work (backward compat)."""
        from hermes_cli.model_switch import list_authenticated_providers

        monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
        monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

        custom_providers = [
            {
                "name": "Simple Provider",
                "base_url": "http://localhost:8000/v1",
                "api_key": "sk-test",
                "model": "default-model",
            }
        ]

        providers = list_authenticated_providers(
            current_provider="custom",
            custom_providers=custom_providers,
        )

        custom = next((p for p in providers if p.get("is_user_defined")), None)
        assert custom is not None
        assert custom["total_models"] == 1
        assert custom["models"] == ["default-model"]


# =============================================================================
# 5. fetch_api_models — api_mode passthrough
# =============================================================================

class TestFetchApiModelsApiMode:
    """Ensure fetch_api_models forwards api_mode to probe_api_models."""

    def test_passes_api_mode_through(self):
        from hermes_cli.models import fetch_api_models

        probe_result = {"models": ["glm-5-turbo"]}
        with patch("hermes_cli.models.probe_api_models", return_value=probe_result) as mock_probe:
            result = fetch_api_models(
                api_key="sk-test",
                base_url="http://localhost:23000/v1",
                api_mode="anthropic_messages",
            )

        mock_probe.assert_called_once_with(
            "sk-test",
            "http://localhost:23000/v1",
            timeout=5.0,
            api_mode="anthropic_messages",
        )
        assert result == ["glm-5-turbo"]

    def test_default_no_api_mode(self):
        from hermes_cli.models import fetch_api_models

        probe_result = {"models": ["gpt-5"]}
        with patch("hermes_cli.models.probe_api_models", return_value=probe_result) as mock_probe:
            fetch_api_models(api_key="sk-test", base_url="http://localhost:8000/v1")

        call_kwargs = mock_probe.call_args.kwargs
        assert "api_mode" not in call_kwargs or call_kwargs.get("api_mode") is None
