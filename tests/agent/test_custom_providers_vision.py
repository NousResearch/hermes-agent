"""Tests for custom_providers[].models[].supports_vision override (#41036).

When a named custom provider declares per-model supports_vision via the
legacy list-style custom_providers config, image_routing should honor it
and route images natively instead of falling through to models.dev or
the auxiliary vision_analyze path.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# _supports_vision_override — custom_providers lookup
# ---------------------------------------------------------------------------


class TestCustomProvidersVisionOverride:
    """_supports_vision_override should check custom_providers list entries."""

    def test_custom_providers_supports_vision_true(self):
        """custom_providers entry with supports_vision=true → native routing."""
        from agent.image_routing import _supports_vision_override
        cfg = {
            "custom_providers": [
                {
                    "name": "9router-anthropic",
                    "models": {
                        "mimoanth/mimo-v2.5": {
                            "supports_vision": True,
                        }
                    }
                }
            ]
        }
        result = _supports_vision_override(
            cfg, "9router-anthropic", "mimoanth/mimo-v2.5"
        )
        assert result is True

    def test_custom_providers_supports_vision_false(self):
        """custom_providers entry with supports_vision=False → explicit false."""
        from agent.image_routing import _supports_vision_override
        cfg = {
            "custom_providers": [
                {
                    "name": "my-llm",
                    "models": {
                        "some-model": {
                            "supports_vision": False,
                        }
                    }
                }
            ]
        }
        result = _supports_vision_override(cfg, "my-llm", "some-model")
        assert result is False

    def test_custom_providers_custom_prefix(self):
        """Provider name at runtime may be 'custom:<name>'."""
        from agent.image_routing import _supports_vision_override
        cfg = {
            "custom_providers": [
                {
                    "name": "9router-anthropic",
                    "models": {
                        "mimoanth/mimo-v2.5": {
                            "supports_vision": True,
                        }
                    }
                }
            ]
        }
        # Runtime provider is "custom:9router-anthropic"
        result = _supports_vision_override(
            cfg, "custom:9router-anthropic", "mimoanth/mimo-v2.5"
        )
        assert result is True

    def test_custom_providers_no_match_returns_none(self):
        """No matching custom_providers entry → falls through (returns None)."""
        from agent.image_routing import _supports_vision_override
        cfg = {
            "custom_providers": [
                {
                    "name": "other-provider",
                    "models": {
                        "other-model": {
                            "supports_vision": True,
                        }
                    }
                }
            ]
        }
        result = _supports_vision_override(
            cfg, "my-provider", "my-model"
        )
        assert result is None

    def test_custom_providers_model_not_listed(self):
        """Entry exists but model is not listed → falls through."""
        from agent.image_routing import _supports_vision_override
        cfg = {
            "custom_providers": [
                {
                    "name": "my-provider",
                    "models": {
                        "other-model": {
                            "supports_vision": True,
                        }
                    }
                }
            ]
        }
        result = _supports_vision_override(
            cfg, "my-provider", "unlisted-model"
        )
        assert result is None

    def test_custom_providers_ignores_non_dict_entries(self):
        """Non-dict entries in custom_providers list are skipped."""
        from agent.image_routing import _supports_vision_override
        cfg = {
            "custom_providers": [
                "not-a-dict",
                123,
                None,
                {
                    "name": "my-provider",
                    "models": {
                        "my-model": {
                            "supports_vision": True,
                        }
                    }
                }
            ]
        }
        result = _supports_vision_override(
            cfg, "my-provider", "my-model"
        )
        assert result is True

    def test_custom_providers_empty_list(self):
        """Empty custom_providers list → no override."""
        from agent.image_routing import _supports_vision_override
        cfg = {"custom_providers": []}
        result = _supports_vision_override(cfg, "any", "any")
        assert result is None

    def test_custom_providers_no_models_key(self):
        """Entry without models key → skipped gracefully."""
        from agent.image_routing import _supports_vision_override
        cfg = {
            "custom_providers": [
                {"name": "my-provider"}  # no models key
            ]
        }
        result = _supports_vision_override(
            cfg, "my-provider", "my-model"
        )
        assert result is None

    def test_custom_providers_empty_name(self):
        """Entry with empty name → skipped."""
        from agent.image_routing import _supports_vision_override
        cfg = {
            "custom_providers": [
                {
                    "name": "",
                    "models": {"m": {"supports_vision": True}},
                }
            ]
        }
        result = _supports_vision_override(cfg, "any", "m")
        assert result is None


# ---------------------------------------------------------------------------
# decide_image_input_mode integration
# ---------------------------------------------------------------------------


class TestDecideImageInputMode:
    """End-to-end: custom_providers overrides should produce 'native' mode."""

    def test_custom_providers_true_returns_native(self):
        from agent.image_routing import decide_image_input_mode
        cfg = {
            "custom_providers": [
                {
                    "name": "9router-anthropic",
                    "models": {
                        "mimoanth/mimo-v2.5": {
                            "supports_vision": True,
                        }
                    }
                }
            ]
        }
        result = decide_image_input_mode(
            "9router-anthropic", "mimoanth/mimo-v2.5", cfg
        )
        assert result == "native"

    def test_custom_providers_false_returns_text(self):
        from agent.image_routing import decide_image_input_mode
        cfg = {
            "custom_providers": [
                {
                    "name": "my-provider",
                    "models": {
                        "my-model": {
                            "supports_vision": False,
                        }
                    }
                }
            ]
        }
        result = decide_image_input_mode("my-provider", "my-model", cfg)
        assert result == "text"

    def test_top_level_supports_vision_takes_precedence(self):
        """Top-level model.supports_vision still wins over custom_providers."""
        from agent.image_routing import decide_image_input_mode
        cfg = {
            "model": {"supports_vision": False},
            "custom_providers": [
                {
                    "name": "my-provider",
                    "models": {
                        "my-model": {
                            "supports_vision": True,
                        }
                    }
                }
            ]
        }
        result = decide_image_input_mode("my-provider", "my-model", cfg)
        assert result == "text"

    def test_providers_dict_takes_precedence(self):
        """providers.<name>.models takes precedence over custom_providers."""
        from agent.image_routing import decide_image_input_mode
        cfg = {
            "providers": {
                "my-provider": {
                    "models": {
                        "my-model": {"supports_vision": False}
                    }
                }
            },
            "custom_providers": [
                {
                    "name": "my-provider",
                    "models": {
                        "my-model": {"supports_vision": True}
                    }
                }
            ]
        }
        result = decide_image_input_mode("my-provider", "my-model", cfg)
        assert result == "text"
