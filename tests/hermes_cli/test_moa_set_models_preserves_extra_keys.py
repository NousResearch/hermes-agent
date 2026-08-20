"""Regression tests for ``set_moa_models`` preserving undeclared config keys.

Issue #58819: ``MoaConfigPayload`` does not declare ``save_traces`` or
``trace_dir``, so a GUI save via ``PUT /api/model/moa`` silently drops
these hand-edited keys from ``config.yaml``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hermes_cli.web_server import MoaConfigPayload, MoaModelSlot, MoaPresetPayload, set_moa_models


def _base_payload(**overrides) -> MoaConfigPayload:
    """Return a minimal valid MoaConfigPayload."""
    defaults = dict(
        default_preset="default",
        active_preset="",
        presets={
            "default": MoaPresetPayload(
                reference_models=[
                    MoaModelSlot(provider="openai-codex", model="gpt-5.5"),
                ],
                aggregator=MoaModelSlot(provider="openrouter", model="anthropic/claude-opus-4.8"),
                max_tokens=4096,
                enabled=True,
            ),
        },
    )
    defaults.update(overrides)
    return MoaConfigPayload(**defaults)


class TestSetMoaModelsPreservesUndeclaredKeys:
    """save_traces / trace_dir must survive a GUI save."""

    def test_save_traces_preserved(self, tmp_path):
        """Hand-edited ``moa.save_traces: true`` must not be dropped."""
        existing_cfg = {
            "moa": {
                "save_traces": True,
                "trace_dir": "/custom/traces",
                "default_preset": "default",
                "presets": {
                    "default": {
                        "reference_models": [
                            {"provider": "openai-codex", "model": "gpt-5.5"},
                        ],
                        "aggregator": {"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
                        "max_tokens": 4096,
                        "enabled": True,
                    },
                },
            },
        }

        saved_cfg = {}

        def fake_load_config():
            return dict(existing_cfg)  # shallow copy

        def fake_save_config(cfg):
            saved_cfg.update(cfg)

        payload = _base_payload()

        with (
            patch("hermes_cli.web_server.load_config", side_effect=fake_load_config),
            patch("hermes_cli.web_server.save_config", side_effect=fake_save_config),
            patch("hermes_cli.web_server._profile_scope"),
        ):
            set_moa_models(payload)

        moa = saved_cfg["moa"]
        assert moa.get("save_traces") is True, (
            "save_traces was dropped by set_moa_models"
        )
        assert moa.get("trace_dir") == "/custom/traces", (
            "trace_dir was dropped by set_moa_models"
        )

    def test_advisory_controls_round_trip_through_gui_payload(self):
        saved_cfg = {}
        payload = _base_payload()
        payload.presets["default"].advisory_context = "none"
        payload.presets["default"].advisory_max_chars = 50_000

        with (
            patch("hermes_cli.web_server.load_config", return_value={}),
            patch(
                "hermes_cli.web_server.save_config",
                side_effect=lambda cfg: saved_cfg.update(cfg),
            ),
            patch("hermes_cli.web_server._profile_scope"),
        ):
            result = set_moa_models(payload)

        preset = saved_cfg["moa"]["presets"]["default"]
        assert preset["advisory_context"] == "none"
        assert preset["advisory_max_chars"] == 50_000
        assert result["presets"]["default"]["advisory_context"] == "none"
        assert result["presets"]["default"]["advisory_max_chars"] == 50_000

    @pytest.mark.parametrize("value", [True, 0, -1, 1.5, "not-a-number"])
    def test_advisory_max_chars_rejects_invalid_write_values(self, value):
        with pytest.raises(ValueError, match="advisory_max_chars"):
            MoaPresetPayload(advisory_max_chars=value)


