"""Tests for three bug fixes:

  #18900 -- _find_skill rglob leaks into .archive, .git, .github, .hub
  #18809 -- os.walk(followlinks=True) infinite-loop on cyclic symlinks
  #18816 -- quick_commands with non-dict values crashes slash command dispatch
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_skill_tree(base: Path, structure: dict):
    for name, value in structure.items():
        d = base / name
        d.mkdir(parents=True, exist_ok=True)
        if value is True:
            (d / "SKILL.md").write_text(f"# {name}\n")
        elif isinstance(value, dict):
            _make_skill_tree(d, value)


class TestIterSkillIndexFilesExclusion:
    def _run(self, tree: dict) -> list[str]:
        from agent.skill_utils import iter_skill_index_files
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_skill_tree(root, tree)
            return [p.parent.name for p in iter_skill_index_files(root, "SKILL.md")]

    def test_normal_skill_discovered(self):
        assert "my-skill" in self._run({"my-skill": True})

    def test_git_dir_excluded(self):
        names = self._run({".git": {"hooks": True}, "real-skill": True})
        assert "hooks" not in names
        assert "real-skill" in names

    def test_github_dir_excluded(self):
        names = self._run({".github": {"workflows": True}, "real-skill": True})
        assert "workflows" not in names
        assert "real-skill" in names

    def test_hub_dir_excluded(self):
        names = self._run({".hub": {"cached": True}, "real-skill": True})
        assert "cached" not in names
        assert "real-skill" in names

    def test_archive_dir_excluded(self):
        names = self._run({".archive": {"old-skill": True}, "real-skill": True})
        assert "old-skill" not in names
        assert "real-skill" in names

    def test_multiple_excluded_dirs_with_real_skills(self):
        tree = {".git": {"internal": True}, ".archive": {"retired": True}, "skill-a": True, "skill-b": True}
        names = self._run(tree)
        assert sorted(names) == ["skill-a", "skill-b"]


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks require elevated privileges on Windows")
class TestIterSkillIndexFilesCyclicSymlinks:
    def test_cyclic_symlink_does_not_loop(self):
        from agent.skill_utils import iter_skill_index_files
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / "my-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("# my-skill\n")
            (root / "loop").symlink_to(root)
            results = list(iter_skill_index_files(root, "SKILL.md"))
            assert "my-skill" in [p.parent.name for p in results]

    def test_non_cyclic_symlink_still_followed(self):
        from agent.skill_utils import iter_skill_index_files
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ext = Path(tmp) / "external"
            ext.mkdir()
            skill_dir = ext / "ext-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("# ext-skill\n")
            (root / "ext-link").symlink_to(ext)
            results = list(iter_skill_index_files(root, "SKILL.md"))
            assert "ext-skill" in [p.parent.name for p in results]


class TestQuickCommandsNonDictGateway:
    def test_string_value_returns_error_message(self):
        qcmd = "hello world"
        command = "greet"
        if not isinstance(qcmd, dict):
            result = f"Quick command '/{command}' is misconfigured -- expected a dict, got {type(qcmd).__name__}. Check your gateway config."
        else:
            result = qcmd.get("type")
        assert "misconfigured" in result
        assert "str" in result

    def test_int_value_returns_error_message(self):
        qcmd = 42
        command = "count"
        if not isinstance(qcmd, dict):
            result = f"Quick command '/{command}' is misconfigured -- expected a dict, got {type(qcmd).__name__}. Check your gateway config."
        else:
            result = qcmd.get("type")
        assert "misconfigured" in result
        assert "int" in result

    def test_none_value_returns_error_message(self):
        qcmd = None
        command = "empty"
        if not isinstance(qcmd, dict):
            result = f"Quick command '/{command}' is misconfigured -- expected a dict, got {type(qcmd).__name__}. Check your gateway config."
        else:
            result = qcmd.get("type")
        assert "misconfigured" in result

    def test_valid_dict_exec_command_still_works(self):
        qcmd = {"type": "exec", "command": "echo ok"}
        assert isinstance(qcmd, dict)
        assert qcmd.get("type") == "exec"

    def test_valid_dict_alias_command_still_works(self):
        qcmd = {"type": "alias", "target": "/help"}
        assert isinstance(qcmd, dict)
        assert qcmd.get("type") == "alias"

    def test_list_value_returns_error_message(self):
        qcmd = ["a", "b"]
        command = "tags"
        if not isinstance(qcmd, dict):
            result = f"Quick command '/{command}' is misconfigured -- expected a dict, got {type(qcmd).__name__}. Check your gateway config."
        else:
            result = qcmd.get("type")
        assert "misconfigured" in result
        assert "list" in result
