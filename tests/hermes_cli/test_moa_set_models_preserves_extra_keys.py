"""Regression tests for ``set_moa_models`` preserving undeclared config keys.

Issue #58819: ``MoaConfigPayload`` does not declare ``save_traces`` or
``trace_dir``, so a GUI save via ``PUT /api/model/moa`` silently drops
these hand-edited keys from ``config.yaml``.
"""

from __future__ import annotations

from unittest.mock import patch

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

    def test_trace_dir_empty_string_preserved(self, tmp_path):
        """Even an empty-string ``trace_dir`` must survive."""
        existing_cfg = {
            "moa": {
                "save_traces": True,
                "trace_dir": "",
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
            return dict(existing_cfg)

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
        assert moa.get("save_traces") is True
        assert moa.get("trace_dir") == ""

    def test_no_existing_moa_key_still_works(self, tmp_path):
        """When ``moa`` key is absent from config, the endpoint must not crash."""
        existing_cfg: dict = {}

        saved_cfg = {}

        def fake_load_config():
            return dict(existing_cfg)

        def fake_save_config(cfg):
            saved_cfg.update(cfg)

        payload = _base_payload()

        with (
            patch("hermes_cli.web_server.load_config", side_effect=fake_load_config),
            patch("hermes_cli.web_server.save_config", side_effect=fake_save_config),
            patch("hermes_cli.web_server._profile_scope"),
        ):
            result = set_moa_models(payload)

        assert result["ok"] is True
        assert "default_preset" in saved_cfg["moa"]
