"""Tests for hermes_cli/skills_groups.py — skill group config, CLI helpers."""

from unittest.mock import patch

from hermes_cli.skills_groups import (
    _validate_group_name,
    add_skills_to_group,
    get_skill_groups,
    remove_skills_from_group,
    save_skill_groups,
)


# ---------------------------------------------------------------------------
# get_skill_groups
# ---------------------------------------------------------------------------

class TestGetSkillGroups:
    def test_empty_config(self):
        assert get_skill_groups({}) == {}

    def test_null_skills_section(self):
        assert get_skill_groups({"skills": None}) == {}

    def test_null_groups_section(self):
        assert get_skill_groups({"skills": {"groups": None}}) == {}

    def test_malformed_groups_section(self):
        assert get_skill_groups({"skills": {"groups": "oops"}}) == {}

    def test_normalizes_scalars_lists_and_dedupes(self):
        config = {
            "skills": {
                "groups": {
                    "security": ["web-pentest", "godmode", "web-pentest"],
                    "writing": "humanizer",
                    "empty": [],
                    "  ": ["spaces"],
                }
            }
        }
        groups = get_skill_groups(config)
        assert groups["security"] == ["godmode", "web-pentest"]
        assert groups["writing"] == ["humanizer"]
        assert "empty" not in groups
        assert "  " not in groups


# ---------------------------------------------------------------------------
# save_skill_groups
# ---------------------------------------------------------------------------

class TestSaveSkillGroups:
    @patch("hermes_cli.skills_groups.save_config")
    def test_writes_sorted_unique_under_skills_groups(self, mock_save):
        config = {}
        save_skill_groups(config, {"security": ["godmode", "web-pentest", "godmode"]})
        assert config["skills"]["groups"] == {
            "security": ["godmode", "web-pentest"]
        }
        mock_save.assert_called_once()

    @patch("hermes_cli.skills_groups.save_config")
    def test_preserves_existing_skills_section(self, mock_save):
        config = {"skills": {"disabled": ["old-skill"]}}
        save_skill_groups(config, {"security": ["godmode"]})
        assert config["skills"]["disabled"] == ["old-skill"]
        assert config["skills"]["groups"] == {"security": ["godmode"]}


# ---------------------------------------------------------------------------
# add_skills_to_group
# ---------------------------------------------------------------------------

class TestAddSkillsToGroup:
    @patch("hermes_cli.skills_groups.save_config")
    def test_creates_group(self, mock_save):
        config = {}
        result = add_skills_to_group(config, "security", ["web-pentest", "godmode"])
        assert result["created"] is True
        assert set(result["added"]) == {"web-pentest", "godmode"}
        assert config["skills"]["groups"]["security"] == ["godmode", "web-pentest"]

    @patch("hermes_cli.skills_groups.save_config")
    def test_appends_and_dedupes(self, mock_save):
        config = {"skills": {"groups": {"security": ["godmode"]}}}
        result = add_skills_to_group(config, "security", ["godmode", "humanizer"])
        assert result["duplicates"] == ["godmode"]
        assert result["added"] == ["humanizer"]
        assert config["skills"]["groups"]["security"] == ["godmode", "humanizer"]


# ---------------------------------------------------------------------------
# remove_skills_from_group
# ---------------------------------------------------------------------------

class TestRemoveSkillsFromGroup:
    @patch("hermes_cli.skills_groups.save_config")
    def test_removes_skills(self, mock_save):
        config = {"skills": {"groups": {"security": ["godmode", "web-pentest"]}}}
        result = remove_skills_from_group(config, "security", ["godmode"])
        assert result["removed"] == ["godmode"]
        assert result["group_deleted"] is False
        assert config["skills"]["groups"]["security"] == ["web-pentest"]

    @patch("hermes_cli.skills_groups.save_config")
    def test_deletes_group_when_empty(self, mock_save):
        config = {"skills": {"groups": {"security": ["godmode"]}}}
        result = remove_skills_from_group(config, "security", ["godmode"])
        assert result["group_deleted"] is True
        assert "security" not in config["skills"]["groups"]

    @patch("hermes_cli.skills_groups.save_config")
    def test_deletes_whole_group_without_skill_args(self, mock_save):
        config = {"skills": {"groups": {"security": ["a", "b"]}}}
        result = remove_skills_from_group(config, "security")
        assert result["group_deleted"] is True
        assert result["removed"] == ["a", "b"]
        assert "security" not in config["skills"]["groups"]

    @patch("hermes_cli.skills_groups.save_config")
    def test_unknown_group_is_noop(self, mock_save):
        config = {}
        result = remove_skills_from_group(config, "nope", ["x"])
        assert result["missing"] == ["x"]
        assert result["group_deleted"] is False
        mock_save.assert_not_called()


# ---------------------------------------------------------------------------
# _validate_group_name
# ---------------------------------------------------------------------------

class TestValidateGroupName:
    def test_rejects_empty(self):
        assert _validate_group_name("") is not None
        assert _validate_group_name("   ") is not None

    def test_rejects_whitespace(self):
        assert _validate_group_name("my group") is not None

    def test_rejects_flag_like(self):
        assert _validate_group_name("-security") is not None

    def test_accepts_valid_names(self):
        assert _validate_group_name("security") is None
        assert _validate_group_name("data-science") is None
        assert _validate_group_name("writing2") is None


# ---------------------------------------------------------------------------
# CLI parser wiring
# ---------------------------------------------------------------------------

def _build_parser():
    import argparse

    from hermes_cli.subcommands.skills import build_skills_parser

    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_skills_parser(subparsers, cmd_skills=lambda args: None)
    return parser


class TestSkillsGroupParser:
    def test_list_group_flag(self):
        args = _build_parser().parse_args(
            ["skills", "list", "--group", "security"]
        )
        assert args.skills_action == "list"
        assert args.group == "security"

    def test_group_add(self):
        args = _build_parser().parse_args(
            ["skills", "group", "add", "security", "web-pentest", "godmode"]
        )
        assert args.skills_action == "group"
        assert args.group_action == "add"
        assert args.group == "security"
        assert args.skills == ["web-pentest", "godmode"]

    def test_group_remove_whole_group(self):
        args = _build_parser().parse_args(
            ["skills", "group", "remove", "security"]
        )
        assert args.group_action == "remove"
        assert args.group == "security"
        assert args.skills == []

    def test_group_remove_skills(self):
        args = _build_parser().parse_args(
            ["skills", "group", "rm", "security", "godmode"]
        )
        assert args.group_action == "rm"
        assert args.skills == ["godmode"]

    def test_group_list_alias(self):
        args = _build_parser().parse_args(["skills", "group", "ls"])
        assert args.group_action == "ls"
