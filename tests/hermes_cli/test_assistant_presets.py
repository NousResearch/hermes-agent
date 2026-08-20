"""Tests for hermes_cli.assistant_presets (Energy-inspired role presets)."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.assistant_presets import (
    PRESETS,
    apply_preset_files,
    format_preset_catalog,
    get_preset,
    preset_keys,
    suggested_automation_commands,
)
from hermes_cli.profiles import create_profile, read_profile_meta


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    """Isolated profile root (mirrors tests/hermes_cli/test_profiles.py)."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return tmp_path


class TestCatalog:
    def test_catalog_is_nonempty_with_unique_keys(self):
        keys = preset_keys()
        assert keys
        assert len(keys) == len(set(keys))

    def test_every_preset_is_complete(self):
        for p in PRESETS:
            assert p.title and p.tagline and p.description and p.soul, p.key

    def test_automation_keys_resolve_against_blueprint_catalog(self):
        """Preset automations must reference real blueprints AND fill cleanly
        with their declared slot values + blueprint defaults (contract, not
        snapshot: catalog edits that break a preset fail here)."""
        from cron.blueprint_catalog import fill_blueprint, get_blueprint

        for p in PRESETS:
            for key, values in p.automations:
                bp = get_blueprint(key)
                assert bp is not None, f"preset {p.key} references unknown blueprint {key}"
                spec = fill_blueprint(bp, values)
                assert spec["prompt"] and spec["schedule"]

    def test_get_preset_is_case_insensitive_and_none_on_unknown(self):
        first = PRESETS[0]
        assert get_preset(first.key.upper()) is first
        assert get_preset("no-such-preset") is None
        assert get_preset("") is None

    def test_format_catalog_mentions_every_key(self):
        text = format_preset_catalog()
        for p in PRESETS:
            assert p.key in text


class TestApplyPresetFiles:
    def test_writes_soul_and_description(self, tmp_path):
        preset = PRESETS[0]
        apply_preset_files(tmp_path, preset)
        soul = (tmp_path / "SOUL.md").read_text(encoding="utf-8")
        assert preset.soul in soul
        meta = read_profile_meta(tmp_path)
        assert meta.get("description") == preset.description
        assert meta.get("description_auto") is False


class TestCreateProfileWithPreset:
    def test_unknown_preset_fails_before_creating_directory(self, profile_env):
        with pytest.raises(ValueError, match="Unknown preset"):
            create_profile("scout", preset="not-a-preset")
        assert not (profile_env / ".hermes" / "profiles" / "scout").exists()

    def test_preset_overrides_default_soul_and_sets_description(self, profile_env):
        preset = get_preset("research-scout")
        profile_dir = create_profile("scout", no_alias=True, preset="research-scout")
        soul = (profile_dir / "SOUL.md").read_text(encoding="utf-8")
        assert preset.soul in soul
        assert read_profile_meta(profile_dir).get("description") == preset.description

    def test_explicit_description_wins_over_preset(self, profile_env):
        profile_dir = create_profile(
            "scout2", no_alias=True, preset="research-scout",
            description="My custom router text",
        )
        assert read_profile_meta(profile_dir).get("description") == "My custom router text"

    def test_no_preset_keeps_default_soul(self, profile_env):
        from hermes_cli.default_soul import DEFAULT_SOUL_MD

        profile_dir = create_profile("plain", no_alias=True)
        assert (profile_dir / "SOUL.md").read_text(encoding="utf-8") == DEFAULT_SOUL_MD


class TestSuggestedCommands:
    def test_commands_render_for_presets_with_automations(self):
        for p in PRESETS:
            cmds = suggested_automation_commands(p, "myrole")
            assert len(cmds) == len(p.automations)
            for cmd in cmds:
                assert "hermes -p myrole" in cmd
                assert "/blueprint " in cmd
