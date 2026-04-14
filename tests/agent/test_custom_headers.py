"""Tests for custom_headers feature — user-defined HTTP headers injected into LLM clients."""

from unittest.mock import patch, MagicMock

import pytest

import hermes_cli.config  # ensure module is loaded for patching
import agent.anthropic_adapter  # noqa: F401
import agent.auxiliary_client  # noqa: F401


# ---------------------------------------------------------------------------
# get_custom_headers (config layer)
# ---------------------------------------------------------------------------


class TestGetCustomHeaders:
    def test_returns_empty_dict_when_not_configured(self):
        with patch("hermes_cli.config.load_config", return_value={}):
            assert hermes_cli.config.get_custom_headers() == {}

    def test_returns_headers_from_config(self):
        cfg = {"custom_headers": {"X-Request-Source": "my-agent", "X-Team": "infra"}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            result = hermes_cli.config.get_custom_headers()
            assert result == {"X-Request-Source": "my-agent", "X-Team": "infra"}

    def test_coerces_values_to_strings(self):
        cfg = {"custom_headers": {"X-Retry": 3, "X-Debug": True}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            result = hermes_cli.config.get_custom_headers()
            assert result == {"X-Retry": "3", "X-Debug": "True"}

    def test_ignores_non_dict_custom_headers(self):
        cfg = {"custom_headers": "not-a-dict"}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert hermes_cli.config.get_custom_headers() == {}

    def test_returns_empty_on_load_config_failure(self):
        with patch("hermes_cli.config.load_config", side_effect=RuntimeError("boom")):
            assert hermes_cli.config.get_custom_headers() == {}


# ---------------------------------------------------------------------------
# build_anthropic_client — custom headers injection
# ---------------------------------------------------------------------------


class TestAnthropicClientCustomHeaders:
    """Verify custom_headers are merged into the Anthropic client's default_headers."""

    def test_custom_headers_merged_with_api_key_auth(self):
        custom = {"X-Request-Source": "my-agent"}
        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk, \
             patch("hermes_cli.config.get_custom_headers", return_value=custom):
            agent.anthropic_adapter.build_anthropic_client("sk-ant-api03-something")
            kwargs = mock_sdk.Anthropic.call_args[1]
            assert kwargs["default_headers"]["X-Request-Source"] == "my-agent"
            # Provider headers must still be present
            assert "anthropic-beta" in kwargs["default_headers"]

    def test_custom_headers_merged_with_oauth_token(self):
        custom = {"X-Audit": "yes"}
        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk, \
             patch("hermes_cli.config.get_custom_headers", return_value=custom):
            agent.anthropic_adapter.build_anthropic_client("sk-ant-oat01-" + "x" * 60)
            kwargs = mock_sdk.Anthropic.call_args[1]
            assert kwargs["default_headers"]["X-Audit"] == "yes"
            assert "anthropic-beta" in kwargs["default_headers"]
            assert "x-app" in kwargs["default_headers"]

    def test_custom_headers_merged_with_bearer_auth(self):
        custom = {"X-Source": "internal"}
        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk, \
             patch("hermes_cli.config.get_custom_headers", return_value=custom):
            agent.anthropic_adapter.build_anthropic_client(
                "minimax-key", base_url="https://api.minimax.io/anthropic")
            kwargs = mock_sdk.Anthropic.call_args[1]
            assert kwargs["default_headers"]["X-Source"] == "internal"

    def test_no_custom_headers_when_empty(self):
        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk, \
             patch("hermes_cli.config.get_custom_headers", return_value={}):
            agent.anthropic_adapter.build_anthropic_client("sk-ant-api03-something")
            kwargs = mock_sdk.Anthropic.call_args[1]
            # Should still work normally — no extra keys injected
            assert "X-Request-Source" not in kwargs.get("default_headers", {})

    def test_custom_headers_override_provider_headers(self):
        """User-defined headers should take precedence over provider defaults."""
        custom = {"x-app": "my-custom-app"}
        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk, \
             patch("hermes_cli.config.get_custom_headers", return_value=custom):
            agent.anthropic_adapter.build_anthropic_client("sk-ant-oat01-" + "x" * 60)
            kwargs = mock_sdk.Anthropic.call_args[1]
            assert kwargs["default_headers"]["x-app"] == "my-custom-app"


# ---------------------------------------------------------------------------
# _merge_custom_headers (auxiliary client layer)
# ---------------------------------------------------------------------------


class TestMergeCustomHeaders:
    def test_merges_into_existing_headers(self):
        custom = {"X-Request-Source": "test"}
        with patch("hermes_cli.config.get_custom_headers", return_value=custom):
            result = agent.auxiliary_client._merge_custom_headers({"Existing": "value"})
            assert result == {"Existing": "value", "X-Request-Source": "test"}

    def test_returns_custom_only_when_no_base(self):
        custom = {"X-Request-Source": "test"}
        with patch("hermes_cli.config.get_custom_headers", return_value=custom):
            result = agent.auxiliary_client._merge_custom_headers()
            assert result == {"X-Request-Source": "test"}

    def test_returns_empty_when_nothing_configured(self):
        with patch("hermes_cli.config.get_custom_headers", return_value={}):
            result = agent.auxiliary_client._merge_custom_headers()
            assert result == {}

    def test_custom_overrides_base(self):
        custom = {"X-OpenRouter-Title": "My Agent"}
        with patch("hermes_cli.config.get_custom_headers", return_value=custom):
            result = agent.auxiliary_client._merge_custom_headers(
                {"X-OpenRouter-Title": "Hermes Agent"})
            assert result["X-OpenRouter-Title"] == "My Agent"

    def test_graceful_on_import_failure(self):
        with patch("hermes_cli.config.get_custom_headers", side_effect=ImportError):
            result = agent.auxiliary_client._merge_custom_headers({"Keep": "this"})
            assert result == {"Keep": "this"}
