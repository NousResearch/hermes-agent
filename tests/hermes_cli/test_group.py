"""``hermes group`` — Bot Mode group chat CRUD.

Groups are backed by profile metadata (``ui_meta.hermes-bots.groups`` membership
+ ``ui_meta.hermes-bots-groups`` projection), not a database. These tests run
against an isolated HERMES_HOME and never touch the real profile store.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hermes_cli import groups
from hermes_cli.groups import (
    BOT_META_KEY,
    PROJECTION_KEY,
    add_members,
    create_group,
    disband_group,
    group_info,
    list_bots,
    list_groups,
    remove_members,
    rename_group,
)
from hermes_cli.profiles import get_profile_dir, read_profile_ui_meta, set_profile_ui_meta

BOTS = ("w-cto", "designer", "caretaker")


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Isolated Hermes home with a default profile and three bot profiles."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for bot in BOTS:
        (home / "profiles" / bot).mkdir(parents=True)
    return home


class TestUiMeta:
    def test_round_trip_and_revision_bump(self, env):
        set_profile_ui_meta(get_profile_dir("default"), "x", {"a": 1})
        assert read_profile_ui_meta(get_profile_dir("default"), "x") == {"a": 1}
        data = yaml.safe_load((env / "profile.yaml").read_text())
        assert data["_ui_meta_revisions"]["x"] == 1

    def test_none_deletes_key(self, env):
        set_profile_ui_meta(get_profile_dir("default"), "x", {"a": 1})
        set_profile_ui_meta(get_profile_dir("default"), "x", None)
        assert read_profile_ui_meta(get_profile_dir("default"), "x") is None


class TestCreateAndList:
    def test_create_writes_membership_and_projection(self, env):
        result = create_group("Build", ["w-cto", "designer", "caretaker"])
        assert result["roomId"].startswith("r")

        # membership on each bot
        for bot in ("w-cto", "designer", "caretaker"):
            meta = read_profile_ui_meta(get_profile_dir(bot), BOT_META_KEY)
            assert "Build" in meta["groups"]

        # projection on default
        proj = read_profile_ui_meta(get_profile_dir("default"), PROJECTION_KEY)
        room = next(v for v in proj["rooms"].values() if v.get("name") == "Build")
        assert room["roomId"] == result["roomId"]
        assert {m["name"] for m in room["members"]} == {"w-cto", "designer", "caretaker"}

    def test_list_bots_and_groups(self, env):
        create_group("Build", ["w-cto", "designer"])
        assert "default" in list_bots()
        assert "w-cto" in list_bots()
        groups = list_groups()
        assert "Build" in groups
        assert groups["Build"]["members"] == ["designer", "w-cto"]

    def test_create_preserves_bot_appearance(self, env):
        meta = {"shape": "square", "color": "hsl(36 90% 55%)", "title": "w_CTO", "groups": ["Existing"], "group": "Existing"}
        set_profile_ui_meta(get_profile_dir("w-cto"), BOT_META_KEY, meta)

        create_group("Build", ["w-cto", "designer"])

        after = read_profile_ui_meta(get_profile_dir("w-cto"), BOT_META_KEY)
        assert after["shape"] == "square"
        assert after["title"] == "w_CTO"
        assert set(after["groups"]) == {"Existing", "Build"}

    def test_rejects_under_2_and_over_6(self, env):
        with pytest.raises(ValueError, match="at least 2"):
            create_group("Solo", ["w-cto"])
        with pytest.raises(ValueError, match="at most 6"):
            create_group("Big", ["a", "b", "c", "d", "e", "f", "g"])

    def test_collision_auto_suffixes(self, env):
        create_group("Build", ["w-cto", "designer"])
        result = create_group("Build", ["w-cto", "caretaker"])
        assert result["name"] == "Build 2"


class TestMutate:
    def test_add_remove_members(self, env):
        create_group("Build", ["w-cto", "designer"])
        add_members("Build", ["caretaker"])
        assert "caretaker" in list_groups()["Build"]["members"]
        remove_members("Build", ["designer"])
        assert "designer" not in list_groups()["Build"]["members"]
        meta = read_profile_ui_meta(get_profile_dir("designer"), BOT_META_KEY)
        assert "Build" not in meta.get("groups", [])

    def test_rename_preserves_identity(self, env):
        rid = create_group("Build", ["w-cto", "designer"])["roomId"]
        rename_group("Build", "Ship")
        groups = list_groups()
        assert "Build" not in groups
        assert groups["Ship"]["roomId"] == rid

    def test_disband(self, env):
        create_group("Build", ["w-cto", "designer"])
        disband_group("Build")
        assert "Build" not in list_groups()
        proj = read_profile_ui_meta(get_profile_dir("default"), PROJECTION_KEY)
        assert proj["deleted"]  # tombstone recorded

    def test_info_unknown_group(self, env):
        assert group_info("Nope")["exists"] is False


class TestParser:
    def test_build_group_parser_registers(self):
        import argparse

        from hermes_cli.subcommands.group import build_group_parser

        root = argparse.ArgumentParser()
        sub = root.add_subparsers(dest="cmd")
        build_group_parser(sub, cmd_group=lambda args: None)
        ns = root.parse_args(["group", "list"])
        assert ns.cmd == "group"
        assert ns.group_action == "list"
