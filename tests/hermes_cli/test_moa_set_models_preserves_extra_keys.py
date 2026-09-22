"""Regression tests for ``set_moa_models`` preserving undeclared config keys.

Issue #58819: ``MoaConfigPayload`` does not declare ``save_traces`` or
``trace_dir``, so a GUI save via ``PUT /api/model/moa`` silently drops
these hand-edited keys from ``config.yaml``.
"""

from __future__ import annotations

from unittest.mock import patch

from hermes_cli.web_models import MoaConfigPayload, MoaModelSlot, MoaPresetPayload
from hermes_cli.web_routers.models import set_moa_models


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

        def fake_save_config(cfg, **_kwargs):
            saved_cfg.update(cfg)

        payload = _base_payload()

        with (
            patch("hermes_cli.config.load_config", side_effect=fake_load_config),
            patch("hermes_cli.config.save_config", side_effect=fake_save_config),
            patch("hermes_cli.web_server_profiles._profile_scope"),
        ):
            set_moa_models(payload)

        moa = saved_cfg["moa"]
        assert moa.get("save_traces") is True, (
            "save_traces was dropped by set_moa_models"
        )
        assert moa.get("trace_dir") == "/custom/traces", (
            "trace_dir was dropped by set_moa_models"
        )




def test_moa_save_writes_only_the_moa_section(tmp_path, monkeypatch):
    """#89184: a MoA autosave must not re-persist the rest of the effective-config snapshot.

    ``load_config()`` is a default-expanded snapshot; saving it whole after a chain was written
    out-of-band (or was simply stale) rewrote ``fallback_providers`` too. Real config pipeline,
    temp HERMES_HOME.
    """
    import yaml
    from hermes_cli.config import get_config_path, load_config, read_raw_config

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    get_config_path().write_text(yaml.safe_dump({"model": {"default": "m1", "provider": "custom"}}), encoding="utf-8")
    stale = load_config()                       # snapshot BEFORE the chain exists
    stale["fallback_providers"] = []            # what the default-expanded snapshot carries
    chain = [{"provider": "custom", "model": "glm-5.08", "base_url": "http://gw:8080/v1"}]
    raw = read_raw_config()
    raw["fallback_providers"] = chain
    get_config_path().write_text(yaml.safe_dump(raw), encoding="utf-8")

    with (
        patch("hermes_cli.config.load_config", return_value=stale),
        patch("hermes_cli.web_server_profiles._profile_scope"),
    ):
        set_moa_models(_base_payload())

    on_disk = read_raw_config()
    assert on_disk["fallback_providers"] == chain, "MoA save clobbered fallback_providers"
    # The MoA edit itself landed (a non-default value, so default stripping leaves it on disk).
    assert on_disk["moa"]["presets"]["default"]["reference_models"][0]["model"] == "gpt-5.5"
