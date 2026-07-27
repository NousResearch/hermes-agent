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

    def test_cli_named_provider_identity_survives_custom_runtime_resolution(self):
        """The CLI-selected name must drive lookup after runtime canonicalizes it."""
        from agent.image_routing import decide_image_input_mode

        cfg = {
            "model": {"provider": "default-proxy"},
            "custom_providers": [
                {
                    "name": "custom",
                    "models": {"shared-model": {"supports_vision": False}},
                },
                {
                    "name": "default-proxy",
                    "models": {"shared-model": {"supports_vision": False}},
                },
                {
                    "name": "my-vision-provider",
                    "models": {"shared-model": {"supports_vision": True}},
                },
            ],
        }
        assert decide_image_input_mode(
            "custom",
            "shared-model",
            cfg,
            requested_provider="my-vision-provider",
        ) == "native"

    def test_cli_named_provider_explicit_false_is_not_shadowed_by_default(self):
        """A selected false override wins even when the configured default is true."""
        from agent.image_routing import decide_image_input_mode

        cfg = {
            "model": {"provider": "default-proxy"},
            "custom_providers": [
                {
                    "name": "default-proxy",
                    "models": {"shared-model": {"supports_vision": True}},
                },
                {
                    "name": "text-only-provider",
                    "models": {"shared-model": {"supports_vision": False}},
                },
            ],
        }
        assert decide_image_input_mode(
            "custom",
            "shared-model",
            cfg,
            requested_provider="text-only-provider",
        ) == "text"

    def test_runtime_provider_identity_does_not_leak_to_another_model(self):
        """Context identity is only evidence for its exact runtime provider/model."""
        from agent.auxiliary_client import clear_runtime_main, set_runtime_main
        from agent.image_routing import decide_image_input_mode

        cfg = {
            "custom_providers": [
                {
                    "name": "my-vision-provider",
                    "models": {
                        "selected-model": {"supports_vision": True},
                        "other-model": {"supports_vision": True},
                    },
                }
            ]
        }
        clear_runtime_main()
        try:
            set_runtime_main(
                "custom",
                "selected-model",
                requested_provider="my-vision-provider",
            )
            assert decide_image_input_mode("custom", "other-model", cfg) == "text"
        finally:
            clear_runtime_main()
