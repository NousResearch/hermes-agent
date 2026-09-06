"""Tests for readiness-annotated skill index (arXiv:2608.01050, Issue #80790).

Covers:
- evaluate_skill_readiness side-effect-free prerequisite evaluation
- _get_required_commands extraction from frontmatter
- tools/skills_tool._skill_readiness with command checks
- build_skills_system_prompt readiness markers and footer note
- Snapshot version 3 invalidation and persistence
- Cache key machine-state sensitivity (env vars and commands)
"""

import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.prompt_builder import (
    _SKILLS_SNAPSHOT_VERSION,
    build_skills_system_prompt,
    clear_skills_system_prompt_cache,
)
from tools.skills_tool_setup import (
    _get_required_commands,
    _get_required_environment_variables,
    evaluate_skill_readiness,
)


class TestEvaluateSkillReadiness:
    def test_ready_skill_with_no_prerequisites(self):
        fm = {"name": "plain-skill", "description": "Does simple things."}
        is_ready, missing = evaluate_skill_readiness(fm)
        assert is_ready is True
        assert missing == []

    def test_missing_environment_variable(self, monkeypatch):
        monkeypatch.delenv("TEST_API_KEY", raising=False)
        fm = {
            "name": "api-skill",
            "required_environment_variables": ["TEST_API_KEY"],
        }
        is_ready, missing = evaluate_skill_readiness(fm)
        assert is_ready is False
        assert missing == ["TEST_API_KEY"]

    def test_present_environment_variable(self, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "secret_value")
        fm = {
            "name": "api-skill",
            "required_environment_variables": ["TEST_API_KEY"],
        }
        is_ready, missing = evaluate_skill_readiness(fm)
        assert is_ready is True
        assert missing == []

    def test_optional_environment_variable_does_not_block_readiness(self, monkeypatch):
        monkeypatch.delenv("OPTIONAL_KEY", raising=False)
        fm = {
            "name": "optional-skill",
            "required_environment_variables": [
                {"name": "OPTIONAL_KEY", "optional": True}
            ],
        }
        is_ready, missing = evaluate_skill_readiness(fm)
        assert is_ready is True
        assert missing == []

    def test_env_snapshot_override(self):
        fm = {
            "name": "snapshot-skill",
            "required_environment_variables": ["CUSTOM_VAR"],
        }
        is_ready, missing = evaluate_skill_readiness(fm, env_snapshot={"CUSTOM_VAR": "val"})
        assert is_ready is True
        assert missing == []

        is_ready, missing = evaluate_skill_readiness(fm, env_snapshot={"CUSTOM_VAR": ""})
        assert is_ready is False
        assert missing == ["CUSTOM_VAR"]

    def test_missing_command_binary(self):
        with patch("shutil.which", return_value=None):
            fm = {
                "name": "cli-skill",
                "required_commands": ["himalaya"],
            }
            is_ready, missing = evaluate_skill_readiness(fm)
            assert is_ready is False
            assert missing == ["himalaya binary"]

    def test_present_command_binary(self):
        with patch("shutil.which", side_effect=lambda c: f"/usr/bin/{c}" if c == "himalaya" else None):
            fm = {
                "name": "cli-skill",
                "required_commands": ["himalaya"],
            }
            is_ready, missing = evaluate_skill_readiness(fm)
            assert is_ready is True
            assert missing == []

    def test_legacy_prerequisites_normalization(self, monkeypatch):
        monkeypatch.delenv("LEGACY_ENV", raising=False)
        with patch("shutil.which", return_value=None):
            fm = {
                "name": "legacy-skill",
                "prerequisites": {
                    "env_vars": ["LEGACY_ENV"],
                    "commands": ["curl", "jq"],
                },
            }
            is_ready, missing = evaluate_skill_readiness(fm)
            assert is_ready is False
            assert missing == ["LEGACY_ENV", "curl binary", "jq binary"]

    def test_setup_collect_secrets_normalization(self, monkeypatch):
        monkeypatch.delenv("COLLECT_SECRET_VAR", raising=False)
        fm = {
            "name": "secret-skill",
            "setup": {
                "collect_secrets": [
                    {"env_var": "COLLECT_SECRET_VAR", "prompt": "Enter token"}
                ]
            },
        }
        is_ready, missing = evaluate_skill_readiness(fm)
        assert is_ready is False
        assert missing == ["COLLECT_SECRET_VAR"]

    def test_snapshot_entry_dict_readiness(self, monkeypatch):
        monkeypatch.delenv("ENTRY_VAR", raising=False)
        with patch("shutil.which", return_value=None):
            entry = {
                "skill_name": "snapshot-entry",
                "required_environment_variables": ["ENTRY_VAR"],
                "required_commands": ["node"],
            }
            is_ready, missing = evaluate_skill_readiness(entry)
            assert is_ready is False
            assert missing == ["ENTRY_VAR", "node binary"]


class TestGetRequiredCommands:
    def test_extracts_from_required_commands_list(self):
        fm = {"required_commands": ["git", "node", "npx"]}
        assert _get_required_commands(fm) == ["git", "node", "npx"]

    def test_extracts_from_legacy_prerequisites(self):
        fm = {"prerequisites": {"commands": ["curl", "jq"]}}
        assert _get_required_commands(fm) == ["curl", "jq"]

    def test_deduplicates_and_ignores_empty(self):
        fm = {
            "required_commands": ["node", " ", "", "node"],
            "prerequisites": {"commands": ["node", "npx"]},
        }
        assert _get_required_commands(fm) == ["node", "npx"]


class TestSkillReadinessTool:
    def test_missing_command_flags_setup_needed(self, monkeypatch, tmp_path):
        from tools.skills_tool import _skill_readiness

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        with patch("shutil.which", return_value=None):
            fm = {
                "name": "cli-tool",
                "description": "CLI tool",
                "required_commands": ["custom-cli"],
            }
            fields, extras = _skill_readiness(fm, "cli-tool")
            assert fields["setup_needed"] is True
            assert fields["readiness_status"] == "setup_needed"
            assert fields["required_commands"] == ["custom-cli"]
            assert fields["missing_required_commands"] == ["custom-cli"]
            assert "command `custom-cli`" in extras.get("setup_note", "")

    def test_legacy_prerequisites_commands_do_not_block_setup_needed_in_skill_view(self, monkeypatch, tmp_path):
        from tools.skills_tool import _skill_readiness

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        with patch("shutil.which", return_value=None):
            fm = {
                "name": "legacy-cli-tool",
                "description": "Legacy CLI tool",
                "prerequisites": {"commands": ["legacy-cmd"]},
            }
            fields, extras = _skill_readiness(fm, "legacy-cli-tool")
            assert fields["setup_needed"] is False
            assert fields["readiness_status"] == "available"
            assert fields["required_commands"] == ["legacy-cmd"]
            assert fields["missing_required_commands"] == ["legacy-cmd"]


class TestSkillsPromptReadinessAnnotation:
    def setup_method(self):
        clear_skills_system_prompt_cache(clear_snapshot=True)

    def teardown_method(self):
        clear_skills_system_prompt_cache(clear_snapshot=True)

    def test_unready_skill_annotated_and_footer_note_present(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("HIMALAYA_SECRET", raising=False)
        skills_root = tmp_path / "skills"

        ready_dir = skills_root / "general" / "ready-skill"
        ready_dir.mkdir(parents=True)
        (ready_dir / "SKILL.md").write_text(
            "---\nname: ready-skill\ndescription: A ready skill.\n---\nBody.\n",
            encoding="utf-8",
        )

        unready_dir = skills_root / "email" / "himalaya"
        unready_dir.mkdir(parents=True)
        (unready_dir / "SKILL.md").write_text(
            "---\nname: himalaya\ndescription: Himalaya CLI: IMAP/SMTP email from terminal.\n"
            "required_commands: [himalaya]\n---\nBody.\n",
            encoding="utf-8",
        )

        with patch("shutil.which", side_effect=lambda c: None if c == "himalaya" else "/usr/bin/" + c):
            prompt = build_skills_system_prompt()

        assert "- ready-skill: A ready skill." in prompt
        assert "[needs setup: himalaya binary]" in prompt
        assert "- himalaya: Himalaya CLI: IMAP/SMTP email from terminal. [needs setup: himalaya binary]" in prompt
        assert "(Skills marked [needs setup] have missing prerequisites — offer setup or an alternative instead of attempting them.)" in prompt

    def test_all_ready_skills_has_no_markers_and_no_footer(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        skills_root = tmp_path / "skills"

        ready_dir = skills_root / "tools" / "my-tool"
        ready_dir.mkdir(parents=True)
        (ready_dir / "SKILL.md").write_text(
            "---\nname: my-tool\ndescription: Fully ready tool.\n---\nBody.\n",
            encoding="utf-8",
        )

        prompt = build_skills_system_prompt()
        assert "- my-tool: Fully ready tool." in prompt
        assert "[needs setup" not in prompt
        assert "(Skills marked [needs setup]" not in prompt

    def test_cache_key_sensitive_to_env_var_changes(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("DYNAMIC_KEY", raising=False)
        skills_root = tmp_path / "skills"

        skill_dir = skills_root / "services" / "dynamic-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: dynamic-skill\ndescription: Requires dynamic key.\n"
            "required_environment_variables: [DYNAMIC_KEY]\n---\nBody.\n",
            encoding="utf-8",
        )

        first = build_skills_system_prompt()
        assert "[needs setup: DYNAMIC_KEY]" in first
        assert "Skills marked [needs setup]" in first

        monkeypatch.setenv("DYNAMIC_KEY", "configured_value")

        second = build_skills_system_prompt()
        assert "[needs setup" not in second
        assert "Skills marked [needs setup]" not in second
        assert "- dynamic-skill: Requires dynamic key." in second

    def test_cache_key_sensitive_to_binary_path_changes(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        skills_root = tmp_path / "skills"

        skill_dir = skills_root / "cli" / "tool-cli"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: tool-cli\ndescription: Requires tool-cli.\n"
            "required_commands: [tool-cli]\n---\nBody.\n",
            encoding="utf-8",
        )

        with patch("shutil.which", return_value=None):
            first = build_skills_system_prompt()
            assert "[needs setup: tool-cli binary]" in first

        with patch("shutil.which", side_effect=lambda c: f"/usr/local/bin/{c}" if c == "tool-cli" else None):
            second = build_skills_system_prompt()
            assert "[needs setup: tool-cli binary]" not in second
            assert "- tool-cli: Requires tool-cli." in second

    def test_snapshot_version_3_invalidation_and_prereqs_persistence(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        skills_root = tmp_path / "skills"

        skill_dir = skills_root / "general" / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: Test skill.\n"
            "required_commands: [git]\n"
            "required_environment_variables: [MY_TOKEN]\n---\nBody.\n",
            encoding="utf-8",
        )

        snapshot_file = tmp_path / ".skills_prompt_snapshot.json"

        snapshot_file.write_text(
            json.dumps({"version": 2, "manifest": {}, "skills": []}),
            encoding="utf-8",
        )

        build_skills_system_prompt()

        assert snapshot_file.exists()
        saved = json.loads(snapshot_file.read_text(encoding="utf-8"))
        assert saved["version"] == _SKILLS_SNAPSHOT_VERSION
        assert saved["version"] == 3

        skill_entry = next(s for s in saved["skills"] if s["frontmatter_name"] == "test-skill")
        assert skill_entry["required_commands"] == ["git"]
        assert skill_entry["required_environment_variables"] == ["MY_TOKEN"]
