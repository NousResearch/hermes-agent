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


# ---------------------------------------------------------------------------
# Background task (/background) must carry the named custom-provider identity
# so per-model vision capability survives — #69894.
# ---------------------------------------------------------------------------


class TestBackgroundTaskCustomProviderIdentity:
    def _run_background_and_capture_agent_kwargs(self, runtime: dict) -> dict:
        """Invoke ``/background`` with a mocked route and return the kwargs the
        handler passed to ``AIAgent``. Runs the background thread synchronously
        and aborts right after construction so no real turn executes."""
        from unittest.mock import MagicMock, patch
        import hermes_cli.cli_commands_mixin as ccm

        captured: dict = {}

        class _Recorder:
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)
                raise RuntimeError("captured — stop before running the turn")

        class _SyncThread:
            def __init__(self, target=None, **_kw):
                self._target = target

            def start(self):
                try:
                    self._target()
                except Exception:
                    pass

        shell = MagicMock()
        shell._background_task_counter = 0
        shell._ensure_runtime_credentials.return_value = True
        shell._resolve_turn_agent_config.return_value = {
            "model": "qwen3.8-max-preview",
            "runtime": runtime,
            "request_overrides": None,
        }

        with patch.object(ccm, "threading") as mock_threading, patch(
            "cli.AIAgent", _Recorder
        ):
            mock_threading.Thread = _SyncThread
            ccm.CLICommandsMixin._handle_background_command.__get__(shell)(
                "/background inspect this screenshot"
            )
        return captured

    def test_background_agent_receives_named_custom_provider(self):
        """A ``--provider custom:<name>`` session's background task must forward
        the routable ``custom:<name>`` identity to the AIAgent; otherwise
        init_agent defaults requested_provider to the bare ``custom`` and native
        vision breaks for the custom provider (#69894)."""
        runtime = {
            "provider": "custom",
            "requested_provider": "custom:qwen-token-plan",
            "api_key": "sk-test",
            "base_url": "https://token-plan.example/v1",
            "api_mode": "chat_completions",
            "command": None,
            "args": [],
            "max_tokens": None,
            "credential_pool": None,
        }
        captured = self._run_background_and_capture_agent_kwargs(runtime)
        assert captured.get("requested_provider") == "custom:qwen-token-plan"
