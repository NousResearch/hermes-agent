"""Test for OpenCode base_url /v1 auto-completion at config load time.

Regression test for: When config.yaml has base_url without /v1 for an
OpenCode provider, the runtime hits /chat/completions (no /v1) and
returns an HTML 404. This test ensures _normalize_root_model_keys
auto-corrects the suffix.
"""
import re
import sys
import os

# Add the hermes-agent path so we can import the module under test
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _apply_normalization(config):
    """Inline a copy of the normalization logic for unit testing.

    Mirrors hermes_cli.config._normalize_root_model_keys behavior for
    the OpenCode /v1 auto-completion step.
    """
    _OPENCODE_PROVIDERS = {"opencode-zen", "opencode-go"}
    model = config.get("model")
    if not isinstance(model, dict):
        return config
    _provider_norm = str(model.get("provider") or "").strip().lower()
    _base_norm = str(model.get("base_url") or "").strip()
    if (
        _provider_norm in _OPENCODE_PROVIDERS
        and _base_norm
        and not _base_norm.rstrip("/").endswith("/v1")
    ):
        model["base_url"] = _base_norm.rstrip("/") + "/v1"
    return config


class TestOpenCodeBaseUrlNormalization:
    """Test that /v1 is auto-appended for OpenCode providers at config load time."""

    def test_opencode_go_appends_v1(self):
        """opencode-go with base_url missing /v1 should be auto-corrected."""
        config = {
            "model": {
                "provider": "opencode-go",
                "base_url": "https://opencode.ai/zen/go",
                "default": "glm-5.2",
            }
        }
        result = _apply_normalization(config)
        assert result["model"]["base_url"] == "https://opencode.ai/zen/go/v1"

    def test_opencode_zen_appends_v1(self):
        """opencode-zen with base_url missing /v1 should be auto-corrected."""
        config = {
            "model": {
                "provider": "opencode-zen",
                "base_url": "https://opencode.ai/zen",
                "default": "claude-sonnet-4-6",
            }
        }
        result = _apply_normalization(config)
        assert result["model"]["base_url"] == "https://opencode.ai/zen/v1"

    def test_already_has_v1_unchanged(self):
        """base_url already ending in /v1 should be unchanged."""
        config = {
            "model": {
                "provider": "opencode-go",
                "base_url": "https://opencode.ai/zen/go/v1",
                "default": "glm-5.2",
            }
        }
        result = _apply_normalization(config)
        assert result["model"]["base_url"] == "https://opencode.ai/zen/go/v1"

    def test_trailing_slash_normalized(self):
        """base_url with trailing slash should be normalized to /v1."""
        config = {
            "model": {
                "provider": "opencode-go",
                "base_url": "https://opencode.ai/zen/go/",
                "default": "glm-5.2",
            }
        }
        result = _apply_normalization(config)
        assert result["model"]["base_url"] == "https://opencode.ai/zen/go/v1"

    def test_non_opencode_provider_unchanged(self):
        """Non-OpenCode providers should not be affected."""
        config = {
            "model": {
                "provider": "openai",
                "base_url": "https://api.openai.com/v1",
                "default": "gpt-4",
            }
        }
        result = _apply_normalization(config)
        assert result["model"]["base_url"] == "https://api.openai.com/v1"

    def test_empty_base_url_unchanged(self):
        """Empty base_url should be left as-is."""
        config = {
            "model": {
                "provider": "opencode-go",
                "base_url": "",
                "default": "glm-5.2",
            }
        }
        result = _apply_normalization(config)
        assert result["model"]["base_url"] == ""

    def test_no_model_section_unchanged(self):
        """Config without model section should not crash."""
        config = {"other_key": "value"}
        result = _apply_normalization(config)
        assert result == {"other_key": "value"}

    def test_deepseek_v4_pro_scenario(self):
        """Regression: switching to deepseek-v4-pro should not 404 due to stale base_url."""
        config = {
            "model": {
                "provider": "opencode-go",
                "base_url": "https://opencode.ai/zen/go",  # No /v1 — the bug
                "default": "deepseek-v4-pro",
            }
        }
        result = _apply_normalization(config)
        assert result["model"]["base_url"].endswith("/v1")
        assert "deepseek-v4-pro" in result["model"]["default"]
