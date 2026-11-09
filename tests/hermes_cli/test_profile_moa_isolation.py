"""#80318 witness correction: root presets are not implicit cross-profile grants."""
import json
from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def test_profile_moa_requires_its_own_declared_preset(tmp_path, monkeypatch):
    from agent.errors import MoAPresetNotFoundError
    from hermes_cli.config import load_config
    from hermes_cli.moa_config import resolve_moa_preset

    root = tmp_path / ".hermes"
    home = root / "profiles" / "reviewer"
    home.mkdir(parents=True)
    preset = {
        "reference_models": [{"provider": "openai", "model": "fixture-reference"}],
        "aggregator": {"provider": "openai", "model": "root-only-model"},
    }
    root_config = root / "config.yaml"
    root_bytes = json.dumps({"moa": {"presets": {"Root Only": preset}}})
    root_config.write_text(root_bytes, encoding="utf-8")
    assignment = {"model": {"default": "Root Only", "provider": "moa"}}
    profile_config = home / "config.yaml"
    profile_config.write_text(json.dumps(assignment), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    token = set_hermes_home_override(home)
    try:
        with pytest.raises(MoAPresetNotFoundError, match="Root Only"):
            resolve_moa_preset(load_config().get("moa"), "Root Only")
        # Explicit owner config, not an ambient root fallback, authorizes this preset.
        own_preset = {**preset, "aggregator": {"provider": "openai", "model": "profile-owned-model"}}
        assignment["moa"] = {"presets": {"Root Only": own_preset}}
        profile_config.write_text(json.dumps(assignment), encoding="utf-8")
        resolved = resolve_moa_preset(load_config().get("moa"), "Root Only")
        assert resolved["aggregator"]["model"] == "profile-owned-model"
        assert root_config.read_text(encoding="utf-8") == root_bytes
    finally:
        reset_hermes_home_override(token)
