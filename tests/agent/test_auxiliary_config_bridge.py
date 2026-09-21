"""Tests for auxiliary model config bridging — verifies that config.yaml values
are properly mapped to environment variables by both CLI and gateway loaders.

Also tests the vision_tools and browser_tool model override env vars.
"""

import os
import sys
from unittest.mock import patch, MagicMock, AsyncMock

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _run_auxiliary_bridge(config_dict, monkeypatch):
    """Exercise the CLI's real config merge and auxiliary environment bridge."""
    # Clear env vars
    for key in (
        "AUXILIARY_VISION_PROVIDER", "AUXILIARY_VISION_MODEL",
        "AUXILIARY_VISION_BASE_URL", "AUXILIARY_VISION_API_KEY",
        "AUXILIARY_APPROVAL_PROVIDER", "AUXILIARY_APPROVAL_MODEL",
        "AUXILIARY_APPROVAL_BASE_URL", "AUXILIARY_APPROVAL_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    from hermes_cli.cli_config_load import (
        _cli_config_defaults,
        _merge_file_config,
        _mirror_config_to_env,
    )

    defaults = _cli_config_defaults()
    _merge_file_config(defaults, config_dict)
    _mirror_config_to_env(defaults, False)


# ── Config bridging tests ────────────────────────────────────────────────────


class TestAuxiliaryConfigBridge:
    """Verify the config.yaml → env var bridging logic used by CLI and gateway."""


    def test_vision_model_bridged(self, monkeypatch):
        config = {
            "auxiliary": {
                "vision": {"provider": "auto", "model": "openai/gpt-4o"},
            }
        }
        _run_auxiliary_bridge(config, monkeypatch)
        assert os.environ.get("AUXILIARY_VISION_MODEL") == "openai/gpt-4o"
        # auto provider should not be set
        assert os.environ.get("AUXILIARY_VISION_PROVIDER") is None

    def test_approval_bridged(self, monkeypatch):
        config = {
            "auxiliary": {
                "approval": {"provider": "nous", "model": "gemini-2.5-flash"},
            }
        }
        _run_auxiliary_bridge(config, monkeypatch)
        assert os.environ.get("AUXILIARY_APPROVAL_PROVIDER") == "nous"
        assert os.environ.get("AUXILIARY_APPROVAL_MODEL") == "gemini-2.5-flash"





    def test_mixed_tasks(self, monkeypatch):
        config = {
            "auxiliary": {
                "vision": {"provider": "openrouter", "model": ""},
                "approval": {"provider": "auto", "model": "custom-llm"},
            }
        }
        _run_auxiliary_bridge(config, monkeypatch)
        assert os.environ.get("AUXILIARY_VISION_PROVIDER") == "openrouter"
        assert os.environ.get("AUXILIARY_VISION_MODEL") is None
        assert os.environ.get("AUXILIARY_APPROVAL_PROVIDER") is None
        assert os.environ.get("AUXILIARY_APPROVAL_MODEL") == "custom-llm"





# ── Gateway bridge parity test ───────────────────────────────────────────────


class TestGatewayBridgeBehavior:
    """Exercise the gateway's actual auxiliary config-to-environment bridge."""

    def test_gateway_bridges_builtin_and_plugin_auxiliary_tasks(self, monkeypatch):
        for key in (
            "AUXILIARY_VISION_PROVIDER", "AUXILIARY_VISION_MODEL",
            "AUXILIARY_APPROVAL_PROVIDER", "AUXILIARY_APPROVAL_MODEL",
            "AUXILIARY_REVIEW_PROVIDER", "AUXILIARY_REVIEW_MODEL",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(
            "hermes_cli.plugins.get_plugin_auxiliary_tasks",
            lambda: [{"key": "review"}],
        )

        from gateway.run import _bridge_auxiliary_config_to_env

        _bridge_auxiliary_config_to_env({
            "vision": {"provider": "openai", "model": "vision-model"},
            "approval": {"provider": "nous", "model": "approval-model"},
            "review": {"provider": "openrouter", "model": "review-model"},
        })

        assert os.environ["AUXILIARY_VISION_PROVIDER"] == "openai"
        assert os.environ["AUXILIARY_VISION_MODEL"] == "vision-model"
        assert os.environ["AUXILIARY_APPROVAL_PROVIDER"] == "nous"
        assert os.environ["AUXILIARY_APPROVAL_MODEL"] == "approval-model"
        assert os.environ["AUXILIARY_REVIEW_PROVIDER"] == "openrouter"
        assert os.environ["AUXILIARY_REVIEW_MODEL"] == "review-model"

    def test_gateway_keeps_compression_settings_out_of_auxiliary_environment(self, monkeypatch):
        monkeypatch.delenv("CONTEXT_COMPRESSION_PROVIDER", raising=False)
        monkeypatch.delenv("CONTEXT_COMPRESSION_MODEL", raising=False)

        from gateway.run import _bridge_config_to_env

        _bridge_config_to_env({
            "compression": {"provider": "openai", "model": "compression-model"}
        })

        assert "CONTEXT_COMPRESSION_PROVIDER" not in os.environ
        assert "CONTEXT_COMPRESSION_MODEL" not in os.environ


# ── Vision model override tests ──────────────────────────────────────────────


class TestVisionModelOverride:
    """Test that AUXILIARY_VISION_MODEL env var overrides the default model in the handler."""

    @pytest.mark.asyncio
    async def test_env_var_overrides_default(self, monkeypatch):
        monkeypatch.setenv("AUXILIARY_VISION_MODEL", "openai/gpt-4o")
        from tools.vision_tools import _handle_vision_analyze
        with (
            patch("tools.vision_tools.vision_analyze_tool", new_callable=AsyncMock) as mock_tool,
            patch("tools.vision_tools._should_use_native_vision_fast_path", return_value=False),
        ):
            mock_tool.return_value = '{"success": true}'
            await _handle_vision_analyze({"image_url": "http://test.jpg", "question": "test"})
            call_args = mock_tool.call_args
            # 3rd positional arg = model
            assert call_args[0][2] == "openai/gpt-4o"

    @pytest.mark.asyncio
    async def test_default_model_when_no_override(self, monkeypatch):
        monkeypatch.delenv("AUXILIARY_VISION_MODEL", raising=False)
        from tools.vision_tools import _handle_vision_analyze
        with (
            patch("tools.vision_tools.vision_analyze_tool", new_callable=AsyncMock) as mock_tool,
            patch("tools.vision_tools._should_use_native_vision_fast_path", return_value=False),
        ):
            mock_tool.return_value = '{"success": true}'
            await _handle_vision_analyze({"image_url": "http://test.jpg", "question": "test"})
            call_args = mock_tool.call_args
            # With no AUXILIARY_VISION_MODEL env var, model should be None
            # (the centralized call_llm router picks the provider default)
            assert call_args[0][2] is None


# ── DEFAULT_CONFIG shape tests ───────────────────────────────────────────────


class TestDefaultConfigShape:
    """Verify the DEFAULT_CONFIG in hermes_cli/config.py has correct auxiliary structure."""

    def test_auxiliary_section_exists(self):
        from hermes_cli.config import DEFAULT_CONFIG
        assert "auxiliary" in DEFAULT_CONFIG

    def test_vision_task_structure(self):
        from hermes_cli.config import DEFAULT_CONFIG
        vision = DEFAULT_CONFIG["auxiliary"]["vision"]
        assert "provider" in vision
        assert "model" in vision
        assert vision["provider"] == "auto"
        assert vision["model"] == ""

    def test_web_extract_task_removed(self):
        """web_extract no longer summarizes via LLM — no aux slot."""
        from hermes_cli.config import DEFAULT_CONFIG
        assert "web_extract" not in DEFAULT_CONFIG["auxiliary"]


# ── CLI defaults parity ─────────────────────────────────────────────────────


class TestCLIDefaultsHaveAuxiliaryKeys:
    """Verify CLI config values reach the environment through the real loader helpers."""

    def test_cli_config_file_auxiliary_values_are_bridged(self, monkeypatch):
        for key in (
            "AUXILIARY_APPROVAL_PROVIDER", "AUXILIARY_APPROVAL_MODEL",
            "AUXILIARY_APPROVAL_BASE_URL", "AUXILIARY_APPROVAL_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)

        _run_auxiliary_bridge({
            "auxiliary": {
                "approval": {
                    "provider": "nous",
                    "model": "gemini-2.5-flash",
                    "base_url": "https://example.test/v1",
                    "api_key": "test-key",
                }
            }
        }, monkeypatch)

        assert os.environ["AUXILIARY_APPROVAL_PROVIDER"] == "nous"
        assert os.environ["AUXILIARY_APPROVAL_MODEL"] == "gemini-2.5-flash"
        assert os.environ["AUXILIARY_APPROVAL_BASE_URL"] == "https://example.test/v1"
        assert os.environ["AUXILIARY_APPROVAL_API_KEY"] == "test-key"
