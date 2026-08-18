"""Profile display_name (#45624): presentation-only label in profile.yaml.

The canonical profile id ("default" for ~/.hermes) is never touched —
resolution, comparison, and spawn paths must be provably unaffected.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hermes_cli.profiles import (
    create_profile,
    format_profile_label,
    get_profile_dir,
    list_profiles,
    profile_exists,
    read_profile_meta,
    rename_profile,
    resolve_profile_env,
    set_profile_display_name,
    write_profile_meta,
)


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    """Isolated environment: Path.home() and HERMES_HOME under tmp_path."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return default_home


class TestMetaAndValidation:
    def test_round_trip_preserves_other_fields(self, profile_env):
        write_profile_meta(profile_env, description="ops agent")
        write_profile_meta(profile_env, display_name="小助手")
        meta = read_profile_meta(profile_env)
        assert meta["display_name"] == "小助手"
        assert meta["description"] == "ops agent"

    def test_missing_file_defaults_empty(self, profile_env):
        assert read_profile_meta(profile_env)["display_name"] == ""

    def test_empty_clears_key_from_file(self, profile_env):
        write_profile_meta(profile_env, display_name="Harumesu")
        write_profile_meta(profile_env, display_name="")
        data = yaml.safe_load((profile_env / "profile.yaml").read_text())
        assert "display_name" not in data

    def test_setter_strips_and_caps_length(self, profile_env):
        assert set_profile_display_name("default", "  Harumesu ") == "Harumesu"
        with pytest.raises(ValueError):
            set_profile_display_name("default", "x" * 65)

    def test_setter_missing_profile_raises(self, profile_env):
        with pytest.raises(FileNotFoundError):
            set_profile_display_name("ghost", "Boo")


class TestFormatProfileLabel:
    def test_shapes(self):
        assert format_profile_label("default", "Harumesu") == "Harumesu (default)"
        # Unset/None/id-equal → byte-for-byte the pre-feature rendering.
        assert format_profile_label("default", "") == "default"
        assert format_profile_label("default", None) == "default"
        assert format_profile_label("worker", "worker") == "worker"


class TestRenameDefault:
    def test_sets_display_name_only(self, profile_env, capsys):
        assert rename_profile("default", "Harumesu") == profile_env
        assert profile_env.is_dir()  # directory untouched
        assert read_profile_meta(profile_env)["display_name"] == "Harumesu"
        assert "canonical id remains 'default'" in capsys.readouterr().out

    def test_reflected_in_list_profiles(self, profile_env):
        rename_profile("default", "Harumesu")
        info = next(p for p in list_profiles() if p.is_default)
        assert info.name == "default"
        assert info.is_default is True
        assert info.display_name == "Harumesu"

    def test_rejects_empty_and_preserves_existing(self, profile_env):
        rename_profile("default", "Harumesu")
        with pytest.raises(ValueError):
            rename_profile("default", "   ")
        # Failed rename must not clear the existing display name.
        assert read_profile_meta(profile_env)["display_name"] == "Harumesu"

    def test_rename_to_default_still_reserved(self, profile_env):
        create_profile("worker", no_alias=True)
        with pytest.raises(ValueError, match="reserved"):
            rename_profile("worker", "default")

    def test_named_rename_still_real_and_keeps_display_name(
        self, profile_env, monkeypatch
    ):
        monkeypatch.setattr(
            "hermes_cli.profiles.check_alias_collision", lambda name: "skip"
        )
        create_profile("oldname", no_alias=True)
        write_profile_meta(get_profile_dir("oldname"), display_name="Old Friend")
        new_dir = rename_profile("oldname", "newname")
        assert not (profile_env / "profiles" / "oldname").is_dir()
        assert new_dir == profile_env / "profiles" / "newname"
        assert read_profile_meta(new_dir)["display_name"] == "Old Friend"


class TestResolutionUnaffected:
    def test_display_name_is_not_a_resolvable_id(self, profile_env):
        rename_profile("default", "Harumesu")
        assert get_profile_dir("default") == profile_env
        assert profile_exists("default") is True
        assert profile_exists("harumesu") is False
        assert resolve_profile_env("default") == str(profile_env)
        with pytest.raises(FileNotFoundError):
            resolve_profile_env("harumesu")
