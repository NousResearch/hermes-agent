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






    def test_custom_providers_empty_list(self):
        """Empty custom_providers list → no override."""
        from agent.image_routing import _supports_vision_override
        cfg = {"custom_providers": []}
        result = _supports_vision_override(cfg, "any", "any")
        assert result is None




# ---------------------------------------------------------------------------
# decide_image_input_mode integration
# ---------------------------------------------------------------------------


class TestDecideImageInputMode:
    """End-to-end: custom_providers overrides should produce 'native' mode."""




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
