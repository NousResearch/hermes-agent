"""Regression tests for agent.skill_utils logging on defensive parse paths."""

import logging
import os
from unittest.mock import patch


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
        content = (
            "---\n"
            "x: [\n"
            "---\n\n"
            "ok\n"
        )
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
        assert any(
            "external skills dirs" in r.message for r in caplog.records
        )


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
