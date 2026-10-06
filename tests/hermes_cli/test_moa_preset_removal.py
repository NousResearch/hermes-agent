"""Regression tests for deleting an MoA preset through ``PUT /api/model/moa``.

The desktop MoA settings pane deletes a preset by PUTting the *remaining* presets
(``apps/desktop/src/app/settings/model-settings.tsx``, the ``Delete`` button). The route
persisted that payload with ``save_config(..., merge_existing=True)``, whose deep merge
cannot express a key removal — ``hermes_cli/config.py::_merge_partial_save`` documents the
limitation itself ("Key REMOVALS are not supported here") and restores the omitted preset
from disk. The UI's optimistic state and the route's ``{"ok": true}`` both claimed success,
and the preset reappeared on the next read.
"""

from __future__ import annotations

from unittest.mock import patch

import yaml

from hermes_cli.config import get_config_path, read_raw_config
from hermes_cli.web_models import MoaConfigPayload, MoaModelSlot, MoaPresetPayload
from hermes_cli.web_routers.models import set_moa_models

_FALLBACK_CHAIN = [{"provider": "custom", "model": "glm-5.08", "base_url": "http://gw:8080/v1"}]


def _slot(model: str) -> MoaModelSlot:
    return MoaModelSlot(provider="openai-codex", model=model)


def _payload_preset(model: str) -> MoaPresetPayload:
    """A preset as the desktop sends it."""
    return MoaPresetPayload(
        reference_models=[_slot(model)],
        aggregator=MoaModelSlot(provider="openrouter", model="anthropic/claude-opus-4.8"),
        max_tokens=4096,
        enabled=True,
    )


def _raw_preset(model: str) -> dict:
    """The same preset as it sits in config.yaml."""
    return {
        "reference_models": [{"provider": "openai-codex", "model": model}],
        "aggregator": {"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
        "max_tokens": 4096,
        "enabled": True,
    }


def _seed_config() -> None:
    """default + Heavy + Light, a hand-edited undeclared key, and an unrelated section."""
    get_config_path().write_text(
        yaml.safe_dump(
            {
                "moa": {
                    "default_preset": "Heavy",
                    # Undeclared by MoaConfigPayload — must survive the save (#58819).
                    "save_traces": True,
                    "trace_dir": "/custom/traces",
                    "presets": {
                        "default": _raw_preset("gpt-5.5"),
                        "Heavy": _raw_preset("deepseek-v4-pro"),
                        "Light": _raw_preset("deepseek-v4.1-flash"),
                    },
                },
                # Written by another surface — the MoA save must not clobber it (#89184).
                "fallback_providers": _FALLBACK_CHAIN,
            }
        ),
        encoding="utf-8",
    )


def test_preset_removal_is_persisted(tmp_path, monkeypatch):
    """A preset absent from the payload is deleted from disk, not restored by the merge."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _seed_config()

    # Exactly what the desktop sends after deleting "Heavy".
    payload = MoaConfigPayload(
        default_preset="default",
        active_preset="",
        presets={"default": _payload_preset("gpt-5.5"), "Light": _payload_preset("deepseek-v4.1-flash")},
    )

    with patch("hermes_cli.web_server_profiles._profile_scope"):
        set_moa_models(payload)

    on_disk = read_raw_config()
    assert "Heavy" not in on_disk["moa"]["presets"], "deleted preset was restored from disk"
    assert set(on_disk["moa"]["presets"]) == {"default", "Light"}

    # The #58819 contract this route exists to protect still holds through the same write.
    assert on_disk["moa"]["save_traces"] is True, "undeclared hand-edited key was dropped"
    assert on_disk["moa"]["trace_dir"] == "/custom/traces"
    # ...as does the #89184 one: other sections are re-persisted exactly as they were on disk.
    assert on_disk["fallback_providers"] == _FALLBACK_CHAIN, "MoA save clobbered another section"


def test_removing_the_last_non_default_preset_is_persisted(tmp_path, monkeypatch):
    """Deleting the active preset persists both the removal and the payload's new routing."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _seed_config()

    payload = MoaConfigPayload(
        default_preset="Light",
        active_preset="",
        presets={"default": _payload_preset("gpt-5.5"), "Light": _payload_preset("deepseek-v4.1-flash")},
    )

    with patch("hermes_cli.web_server_profiles._profile_scope"):
        set_moa_models(payload)

    on_disk = read_raw_config()["moa"]
    assert set(on_disk["presets"]) == {"default", "Light"}
    assert on_disk["default_preset"] == "Light", "payload's default_preset was not honored"
    assert on_disk["presets"]["Light"]["reference_models"][0]["model"] == "deepseek-v4.1-flash"
