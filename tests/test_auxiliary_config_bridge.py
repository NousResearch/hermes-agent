"""Tests for auxiliary model config bridging — verifies that config.yaml values
are properly mapped to environment variables by both CLI and gateway loaders.

Also tests the vision_tools and browser_tool model override env vars.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _run_auxiliary_bridge(config_dict, monkeypatch):
    """Simulate the auxiliary config → env var bridging logic shared by CLI and gateway.

    This mirrors the code in cli.py load_cli_config() and gateway/run.py.
    Both use the same pattern; we test it once here.
    """
    # Clear env vars
    for key in (
        "AUXILIARY_VISION_PROVIDER", "AUXILIARY_VISION_MODEL",
        "AUXILIARY_WEB_EXTRACT_PROVIDER", "AUXILIARY_WEB_EXTRACT_MODEL",
        "CONTEXT_COMPRESSION_PROVIDER", "CONTEXT_COMPRESSION_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)

    # Compression bridge
    compression_cfg = config_dict.get("compression", {})
    if compression_cfg and isinstance(compression_cfg, dict):
        compression_env_map = {
            "enabled": "CONTEXT_COMPRESSION_ENABLED",
            "threshold": "CONTEXT_COMPRESSION_THRESHOLD",
            "summary_model": "CONTEXT_COMPRESSION_MODEL",
            "summary_provider": "CONTEXT_COMPRESSION_PROVIDER",
        }
        for cfg_key, env_var in compression_env_map.items():
            if cfg_key in compression_cfg:
                os.environ[env_var] = str(compression_cfg[cfg_key])

    # Auxiliary bridge
    auxiliary_cfg = config_dict.get("auxiliary", {})
    if auxiliary_cfg and isinstance(auxiliary_cfg, dict):
        aux_task_env = {
            "vision":      ("AUXILIARY_VISION_PROVIDER",      "AUXILIARY_VISION_MODEL"),
            "web_extract": ("AUXILIARY_WEB_EXTRACT_PROVIDER",  "AUXILIARY_WEB_EXTRACT_MODEL"),
        }
        for task_key, (prov_env, model_env) in aux_task_env.items():
            task_cfg = auxiliary_cfg.get(task_key, {})
            if not isinstance(task_cfg, dict):
                continue
            prov = str(task_cfg.get("provider", "")).strip()
            model = str(task_cfg.get("model", "")).strip()
            if prov and prov != "auto":
                os.environ[prov_env] = prov
            if model:
                os.environ[model_env] = model


# ── Config bridging tests ────────────────────────────────────────────────────


class TestAuxiliaryConfigBridge:
    """Verify the config.yaml → env var bridging logic used by CLI and gateway."""

    def test_vision_provider_bridged(self, monkeypatch):
        config = {
            "auxiliary": {
                "vision": {"provider": "openrouter", "model": ""},
                "web_extract": {"provider": "auto", "model": ""},
            }
        }
        _run_auxiliary_bridge(config, monkeypatch)
        assert os.environ.get("AUXILIARY_VISION_PROVIDER") == "openrouter"
        # auto should not be set
        assert os.environ.get("AUXILIARY_WEB_EXTRACT_PROVIDER") is None

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

    def test_web_extract_bridged(self, monkeypatch):
        config = {
            "auxiliary": {
                "web_extract": {"provider": "nous", "model": "gemini-2.5-flash"},
            }
        }
        _run_auxiliary_bridge(config, monkeypatch)
        assert os.environ.get("AUXILIARY_WEB_EXTRACT_PROVIDER") == "nous"
        assert os.environ.get("AUXILIARY_WEB_EXTRACT_MODEL") == "gemini-2.5-flash"

    def test_compression_provider_bridged(self, monkeypatch):
        config = {
            "compression": {
                "summary_provider": "nous",
                "summary_model": "gemini-3-flash",
            }
        }
        _run_auxiliary_bridge(config, monkeypatch)
        assert os.environ.get("CONTEXT_COMPRESSION_PROVIDER") == "nous"
        assert os.environ.get("CONTEXT_COMPRESSION_MODEL") == "gemini-3-flash"

    def test_empty_values_not_bridged(self, monkeypatch):
        config = {
            "auxiliary": {
                "vision": {"provider": "auto", "model": ""},
            }
        }
        _run_auxiliary_bridge(config, monkeypatch)
        assert os.environ.get("AUXILIARY_VISION_PROVIDER") is None
        assert os.environ.get("AUXILIARY_VISION_MODEL") is None

    def test_missing_auxiliary_section_safe(self, monkeypatch):
        """Config without auxiliary section should not crash."""
        config = {"model": {"default": "test-model"}}
        _run_auxiliary_bridge(config, monkeypatch)
        assert os.environ.get("AUXILIARY_VISION_PROVIDER") is None

    def test_non_dict_task_config_ignored(self, monkeypatch):
        """Malformed task config (e.g. string instead of dict) is safely ignored."""
        config = {
            "auxiliary": {
                "vision": "openrouter",  # should be a dict
            }
        }
        _run_auxiliary_bridge(config, monkeypatch)
        assert os.environ.get("AUXILIARY_VISION_PROVIDER") is None

    def test_mixed_tasks(self, monkeypatch):
        config = {
            "auxiliary": {
                "vision": {"provider": "openrouter", "model": ""},
                "web_extract": {"provider": "auto", "model": "custom-llm"},
            }
        }
        _run_auxiliary_bridge(config, monkeypatch)
        assert os.environ.get("AUXILIARY_VISION_PROVIDER") == "openrouter"
        assert os.environ.get("AUXILIARY_VISION_MODEL") is None
        assert os.environ.get("AUXILIARY_WEB_EXTRACT_PROVIDER") is None
        assert os.environ.get("AUXILIARY_WEB_EXTRACT_MODEL") == "custom-llm"

    def test_all_tasks_with_overrides(self, monkeypatch):
        config = {
            "compression": {
                "summary_provider": "main",
                "summary_model": "local-model",
            },
            "auxiliary": {
                "vision": {"provider": "openrouter", "model": "google/gemini-2.5-flash"},
                "web_extract": {"provider": "nous", "model": "gemini-3-flash"},
            }
        }
        _run_auxiliary_bridge(config, monkeypatch)
        assert os.environ.get("CONTEXT_COMPRESSION_PROVIDER") == "main"
        assert os.environ.get("CONTEXT_COMPRESSION_MODEL") == "local-model"
        assert os.environ.get("AUXILIARY_VISION_PROVIDER") == "openrouter"
        assert os.environ.get("AUXILIARY_VISION_MODEL") == "google/gemini-2.5-flash"
        assert os.environ.get("AUXILIARY_WEB_EXTRACT_PROVIDER") == "nous"
        assert os.environ.get("AUXILIARY_WEB_EXTRACT_MODEL") == "gemini-3-flash"

    def test_whitespace_in_values_stripped(self, monkeypatch):
        config = {
            "auxiliary": {
                "vision": {"provider": "  openrouter  ", "model": "  my-model  "},
            }
        }
        _run_auxiliary_bridge(config, monkeypatch)
        assert os.environ.get("AUXILIARY_VISION_PROVIDER") == "openrouter"
        assert os.environ.get("AUXILIARY_VISION_MODEL") == "my-model"

    def test_empty_auxiliary_dict_safe(self, monkeypatch):
        config = {"auxiliary": {}}
        _run_auxiliary_bridge(config, monkeypatch)
        assert os.environ.get("AUXILIARY_VISION_PROVIDER") is None
        assert os.environ.get("AUXILIARY_WEB_EXTRACT_PROVIDER") is None


# ── Gateway bridge parity test ───────────────────────────────────────────────


class TestGatewayBridgeCodeParity:
    """Verify the gateway/run.py config bridge contains the auxiliary section."""

    def test_gateway_has_auxiliary_bridge(self):
        """The gateway config bridge must include auxiliary.* bridging."""
        gateway_path = Path(__file__).parent.parent / "gateway" / "run.py"
        content = gateway_path.read_text()
        # Check for key patterns that indicate the bridge is present
        assert "AUXILIARY_VISION_PROVIDER" in content
        assert "AUXILIARY_VISION_MODEL" in content
        assert "AUXILIARY_WEB_EXTRACT_PROVIDER" in content
        assert "AUXILIARY_WEB_EXTRACT_MODEL" in content

    def test_gateway_has_compression_provider(self):
        """Gateway must bridge compression.summary_provider."""
        gateway_path = Path(__file__).parent.parent / "gateway" / "run.py"
        content = gateway_path.read_text()
        assert "summary_provider" in content
        assert "CONTEXT_COMPRESSION_PROVIDER" in content


# ── Vision model override tests ──────────────────────────────────────────────


class TestVisionModelOverride:
    """Test that AUXILIARY_VISION_MODEL env var overrides the default model in the handler."""

    def test_env_var_overrides_default(self, monkeypatch):
        monkeypatch.setenv("AUXILIARY_VISION_MODEL", "openai/gpt-4o")
        from tools.vision_tools import _handle_vision_analyze
        with patch("tools.vision_tools.vision_analyze_tool", new_callable=MagicMock) as mock_tool:
            mock_tool.return_value = '{"success": true}'
            _handle_vision_analyze({"image_url": "http://test.jpg", "question": "test"})
            call_args = mock_tool.call_args
            # 3rd positional arg = model
            assert call_args[0][2] == "openai/gpt-4o"

    def test_default_model_when_no_override(self, monkeypatch):
        monkeypatch.delenv("AUXILIARY_VISION_MODEL", raising=False)
        from tools.vision_tools import _handle_vision_analyze, DEFAULT_VISION_MODEL
        with patch("tools.vision_tools.vision_analyze_tool", new_callable=MagicMock) as mock_tool:
            mock_tool.return_value = '{"success": true}'
            _handle_vision_analyze({"image_url": "http://test.jpg", "question": "test"})
            call_args = mock_tool.call_args
            expected = DEFAULT_VISION_MODEL or "google/gemini-3-flash-preview"
            assert call_args[0][2] == expected


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

    def test_web_extract_task_structure(self):
        from hermes_cli.config import DEFAULT_CONFIG
        web = DEFAULT_CONFIG["auxiliary"]["web_extract"]
        assert "provider" in web
        assert "model" in web
        assert web["provider"] == "auto"
        assert web["model"] == ""

    def test_compression_provider_default(self):
        from hermes_cli.config import DEFAULT_CONFIG
        compression = DEFAULT_CONFIG["compression"]
        assert "summary_provider" in compression
        assert compression["summary_provider"] == "auto"


# ── CLI defaults parity ─────────────────────────────────────────────────────


class TestCLIDefaultsHaveAuxiliaryKeys:
    """Verify cli.py load_cli_config() defaults dict does NOT include auxiliary
    (it comes from config.yaml deep merge, not hardcoded defaults)."""

    def test_cli_defaults_can_merge_auxiliary(self):
        """The load_cli_config deep merge logic handles keys not in defaults.
        Verify auxiliary would be picked up from config.yaml."""
        # This is a structural assertion: cli.py's second-pass loop
        # carries over keys from file_config that aren't in defaults.
        # So auxiliary config from config.yaml gets merged even though
        # cli.py's defaults dict doesn't define it.
        import cli as _cli_mod
        source = Path(_cli_mod.__file__).read_text()
        assert "auxiliary_config = defaults.get(\"auxiliary\"" in source
        assert "AUXILIARY_VISION_PROVIDER" in source
        assert "AUXILIARY_VISION_MODEL" in source
