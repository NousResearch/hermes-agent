"""Tests for agent.skill_utils — defensive parse/logging paths and metadata handling."""

import logging
import os
from unittest.mock import patch

from agent.skill_utils import extract_skill_conditions


def test_metadata_as_dict_with_hermes():
    """Normal case: metadata is a dict containing hermes keys."""
    frontmatter = {
        "metadata": {
            "hermes": {
                "fallback_for_toolsets": ["toolset_a"],
                "requires_toolsets": ["toolset_b"],
                "fallback_for_tools": ["tool_x"],
                "requires_tools": ["tool_y"],
            }
        }
    }
    result = extract_skill_conditions(frontmatter)
    assert result["fallback_for_toolsets"] == ["toolset_a"]
    assert result["requires_toolsets"] == ["toolset_b"]
    assert result["fallback_for_tools"] == ["tool_x"]
    assert result["requires_tools"] == ["tool_y"]


def test_metadata_as_string_does_not_crash():
    """Bug case: metadata is a non-dict truthy value (e.g. a YAML string)."""
    frontmatter = {"metadata": "some text"}
    result = extract_skill_conditions(frontmatter)
    assert result == {
        "fallback_for_toolsets": [],
        "requires_toolsets": [],
        "fallback_for_tools": [],
        "requires_tools": [],
    }


def test_metadata_as_none():
    """metadata key is present but set to null/None."""
    frontmatter = {"metadata": None}
    result = extract_skill_conditions(frontmatter)
    assert result == {
        "fallback_for_toolsets": [],
        "requires_toolsets": [],
        "fallback_for_tools": [],
        "requires_tools": [],
    }


def test_metadata_missing_entirely():
    """metadata key is absent from frontmatter."""
    frontmatter = {"name": "my-skill", "description": "Does stuff."}
    result = extract_skill_conditions(frontmatter)
    assert result == {
        "fallback_for_toolsets": [],
        "requires_toolsets": [],
        "fallback_for_tools": [],
        "requires_tools": [],
    }


class TestParseFrontmatter:
    def test_malformed_yaml_falls_back_to_line_parser(self):
        from agent.skill_utils import parse_frontmatter

        content = (
            "---\n"
            "name: broken-skill\n"
            "bad: [ this bracket never closes\n"
            "---\n\n"
            "# Body\n"
        )
        fm, body = parse_frontmatter(content)
        assert "name" in fm
        assert fm["name"] == "broken-skill"
        assert "# Body" in body

    def test_malformed_yaml_emits_debug_log(self, caplog):
        import agent.skill_utils as su

        caplog.set_level(logging.DEBUG, logger=su.__name__)
        content = "---\nx: [\n---\n\nok\n"
        su.parse_frontmatter(content)
        assert any(
            "frontmatter YAML parse failed" in r.message for r in caplog.records
        )


class TestGetExternalSkillsDirsLogging:
    def test_yaml_failure_logs_and_returns_empty(self, tmp_path, caplog):
        home = tmp_path / ".hermes"
        home.mkdir()
        (home / "skills").mkdir()
        (home / "config.yaml").write_text("skills:\n  external_dirs: []\n")

        import agent.skill_utils as su

        caplog.set_level(logging.DEBUG, logger=su.__name__)
        with patch.dict(os.environ, {"HERMES_HOME": str(home)}):
            with patch.object(su, "yaml_load", side_effect=ValueError("forced")):
                result = su.get_external_skills_dirs()
        assert result == []
        assert any("external skills dirs" in r.message for r in caplog.records)


class TestDiscoverAllSkillConfigVarsLogging:
    def test_unreadable_skill_file_skipped_with_log(self, tmp_path, caplog):
        home = tmp_path / ".hermes"
        home.mkdir()
        (home / "skills").mkdir()
        (home / "config.yaml").write_text("skills:\n  external_dirs: []\n")
        skill_dir = home / "skills" / "bad-read"
        skill_dir.mkdir()
        bad = skill_dir / "SKILL.md"
        bad.write_bytes(b"\xff\xfe not utf-8")

        import agent.skill_utils as su

        caplog.set_level(logging.DEBUG, logger=su.__name__)
        with patch.dict(os.environ, {"HERMES_HOME": str(home)}):
            out = su.discover_all_skill_config_vars()
        assert out == []
        assert any("Skipping skill file" in r.message for r in caplog.records)


class TestResolveSkillConfigValuesLogging:
    def test_yaml_failure_logs(self, tmp_path, caplog):
        home = tmp_path / ".hermes"
        home.mkdir()
        (home / "skills").mkdir()
        (home / "config.yaml").write_text("skills: {}\n")

        import agent.skill_utils as su

        caplog.set_level(logging.DEBUG, logger=su.__name__)
        vars_ = [
            {
                "key": "k",
                "description": "d",
                "default": "defval",
                "prompt": "p",
            }
        ]
        with patch.dict(os.environ, {"HERMES_HOME": str(home)}):
            with patch.object(su, "yaml_load", side_effect=RuntimeError("bad yaml")):
                resolved = su.resolve_skill_config_values(vars_)
        assert resolved["k"] == "defval"
        assert any(
            "skill config var resolution" in r.message for r in caplog.records
        )
