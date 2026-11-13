"""Profile display_name (#45624): presentation-only label in profile.yaml.

The canonical profile id ("default" for ~/.hermes) is never touched —
resolution, comparison, and spawn paths must be provably unaffected.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import hermes_yaml as yaml

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


def _get_api_profiles(profile_env, monkeypatch) -> list:
    """Rows of a real ``GET /api/profiles`` against the isolated home."""
    from hermes_agent import hermes_state
    from starlette.testclient import TestClient
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", profile_env / "state.db")
    resp = TestClient(app).get("/api/profiles", headers={_SESSION_HEADER_NAME: _SESSION_TOKEN})
    assert resp.status_code == 200, resp.text
    return resp.json()["profiles"]


class TestMetaAndValidation:
    def test_round_trip_preserves_other_fields(self, profile_env):
        write_profile_meta(profile_env, description="ops agent")
        write_profile_meta(profile_env, display_name="小助手")
        meta = read_profile_meta(profile_env)
        assert meta["display_name"] == "小助手"
        assert meta["description"] == "ops agent"

    def test_missing_file_defaults_empty(self, profile_env):
        assert read_profile_meta(profile_env)["display_name"] == ""

    def test_bot_title_read_from_hermes_bots_ui_meta(self, profile_env):
        (profile_env / "profile.yaml").write_text(
            "ui_meta:\n  hermes-bots:\n    title: JordyV\n", encoding="utf-8")
        assert read_profile_meta(profile_env)["bot_title"] == "JordyV"
        # The web/Desktop roster carries it through the same dict the chips read.
        from hermes_cli.web_routers.profiles import _profile_to_dict
        assert _profile_to_dict(list_profiles()[0])["bot_title"] == "JordyV"

    def test_bot_title_absent_when_no_bots_meta(self, profile_env):
        (profile_env / "profile.yaml").write_text(
            "display_name: JordieF\nui_meta:\n  user-sections: {}\n", encoding="utf-8")
        assert read_profile_meta(profile_env)["bot_title"] == ""

    def test_bot_title_never_raises_on_malformed_shapes(self, profile_env):
        (profile_env / "profile.yaml").write_text(
            "ui_meta: hermes-bots\n", encoding="utf-8")  # ui_meta is a scalar
        assert read_profile_meta(profile_env)["bot_title"] == ""
        (profile_env / "profile.yaml").write_text(
            "ui_meta:\n  hermes-bots: 7\n", encoding="utf-8")  # not a mapping
        assert read_profile_meta(profile_env)["bot_title"] == ""

    def test_roster_rows_carry_only_bot_mode_ui_meta(self, profile_env, monkeypatch):
        # A secondary Desktop connection seats bots in sections from /api/profiles (#154536);
        # other plugins' ui_meta (group-chat logs, images) stays off that 5s poll.
        (profile_env / "profile.yaml").write_text(
            "ui_meta:\n  hermes-bots:\n    title: Builder\n    sectionId: jm\n    sectionName: JM\n"
            "    created: 2026-11-12\n  hermes-bots-groups:\n    rooms: {r1: {log: [hi]}}\n", encoding="utf-8")
        from hermes_cli import profiles
        from hermes_cli.web_server_profiles import _fallback_profile_dicts
        expected = {"hermes-bots": {"title": "Builder", "sectionId": "jm", "sectionName": "JM",
                                    "created": "2026-11-12"}}
        import hermes_cli.web_routers.profiles as router  # the normal path, not the fallback
        monkeypatch.setattr(router, "_fallback_profile_dicts", lambda _mod: pytest.fail("fallback served"))
        assert _get_api_profiles(profile_env, monkeypatch)[0]["ui_meta"] == expected
        assert _fallback_profile_dicts(profiles)[0]["ui_meta"] == expected
        # Each caller gets its own copy of the cached value.
        read_profile_meta(profile_env)["ui_meta"]["hermes-bots"]["sectionId"] = "poisoned"
        assert read_profile_meta(profile_env)["ui_meta"] == expected

    def test_non_json_bot_mode_ui_meta_never_fails_the_list(self, profile_env, monkeypatch):
        (profile_env / "profile.yaml").write_text(
            "ui_meta:\n  hermes-bots:\n    title: NaN Bot\n    score: .nan\n", encoding="utf-8")
        row = _get_api_profiles(profile_env, monkeypatch)[0]
        assert row["ui_meta"] == {} and row["bot_title"] == "NaN Bot"

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
