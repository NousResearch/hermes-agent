"""Subdirectory context hints: fallback working dirs and credential redaction.

Scenario: a gateway or desktop backend launched with cwd ``/`` resolves ``terminal.cwd: "."`` to
the filesystem root, so every directory on the machine is "inside the working dir" and a ``grep``
that touches a project directory gets that project's whole CLAUDE.md — a credential quoted in
backticks included — spliced onto the terminal result. The home guard (#76902) does not fire
because the working dir is ``/``, not ``~``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import agent.redact as redact_mod
from agent.redact import clear_env_redaction_values, register_env_redaction_values
from agent.subdirectory_hints import SubdirectoryHintTracker, _is_home_like_working_dir

APP_TOKEN = "q7Vm2kPz9RtL4xWc8NbY"


@pytest.fixture
def home_tree(tmp_path, monkeypatch):
    """A fake $HOME (two levels under tmp_path, so tmp_path's parent plays ``/Users``) holding a
    project whose CLAUDE.md quotes a bare token the regex redactor cannot recognize."""
    home = tmp_path / "Users" / "someone"
    home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))  # Path.home() reads os.environ on macOS fallback
    project = home / "projects" / "media-server"
    project.mkdir(parents=True)
    (project / "CLAUDE.md").write_text(
        f"# Media server\n\nService auth token for the API: `{APP_TOKEN}`\n"
        "GitHub: ghp_abcdefghijklmnopqrstuvwxyz0123456789\n", encoding="utf-8")
    return home


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_env_redaction_values()
    yield
    clear_env_redaction_values()


class TestFallbackWorkingDirs:
    def test_filesystem_root_is_not_a_project(self):
        assert _is_home_like_working_dir(Path(Path.cwd().anchor))

    def test_ancestors_of_home_are_not_projects(self, home_tree):
        assert _is_home_like_working_dir(home_tree.parent)          # /Users
        assert _is_home_like_working_dir(home_tree.parent.parent)   # tmp root standing in for /

    def test_hermes_home_is_not_a_project(self, home_tree, monkeypatch):
        from hermes_constants import get_hermes_home
        assert _is_home_like_working_dir(Path(get_hermes_home()).resolve())

    def test_real_project_under_home_still_qualifies(self, home_tree):
        project = home_tree / "projects" / "media-server"
        assert not _is_home_like_working_dir(project)

    @pytest.mark.parametrize("anchor", ["root", "users"])
    def test_root_or_users_cwd_never_injects_a_project_claude_md(self, home_tree, anchor):
        working_dir = home_tree.parent.parent if anchor == "root" else home_tree.parent
        tracker = SubdirectoryHintTracker(working_dir=str(working_dir))
        project = home_tree / "projects" / "media-server"
        assert tracker.check_tool_call("terminal", {"command": f"grep -r -l token {project}/docs"}) is None
        assert tracker.check_tool_call("read_file", {"path": str(project / "README.md")}) is None

    def test_rebind_to_root_is_refused(self, home_tree):
        project = home_tree / "projects" / "media-server"
        tracker = SubdirectoryHintTracker(working_dir=str(project))
        tracker.rebind_working_dir(str(home_tree.parent.parent))
        assert tracker.working_dir == project.resolve()


class TestHintContentRedaction:
    def _project_tracker(self, home_tree):
        # Working dir = the project's parent, so the project itself is a discoverable subdirectory.
        parent = home_tree / "projects"
        return SubdirectoryHintTracker(working_dir=str(parent)), parent / "media-server"

    def test_prefixed_credential_in_hint_file_is_masked_by_pattern(self, home_tree):
        tracker, project = self._project_tracker(home_tree)
        hint = tracker.check_tool_call("read_file", {"path": str(project / "README.md")})
        assert hint and "media-server/CLAUDE.md" in hint
        assert "ghp_abcdefghijklmnopqrstuvwxyz0123456789" not in hint
        assert "«redacted:ghp_" in hint  # non-reusable file_read sentinel, same as read_file

    def test_bare_token_passes_pattern_but_registered_env_value_is_masked(self, home_tree):
        tracker, project = self._project_tracker(home_tree)
        hint = tracker.check_tool_call("read_file", {"path": str(project / "README.md")})
        assert APP_TOKEN in hint  # documents the residual gap the registry closes
        register_env_redaction_values({"APP_TOKEN": APP_TOKEN})
        tracker2, _ = self._project_tracker(home_tree)
        hint2 = tracker2.check_tool_call("read_file", {"path": str(project / "README.md")})
        assert APP_TOKEN not in hint2 and "«redacted-env-secret:APP_TOKEN»" in hint2

    def test_hint_redaction_is_not_switched_off_by_the_preference(self, home_tree, monkeypatch):
        monkeypatch.setattr(redact_mod, "_REDACT_ENABLED", False)
        register_env_redaction_values({"APP_TOKEN": APP_TOKEN})
        tracker, project = self._project_tracker(home_tree)
        hint = tracker.check_tool_call("read_file", {"path": str(project / "README.md")})
        assert APP_TOKEN not in hint


class TestStartupContextFiles:
    def test_cwd_context_file_is_redacted_before_the_system_prompt(self, home_tree):
        from agent.prompt_builder import _read_context_file
        project = home_tree / "projects" / "media-server"
        register_env_redaction_values({"APP_TOKEN": APP_TOKEN})
        text = _read_context_file(project / "CLAUDE.md")
        assert "# Media server" in text
        assert APP_TOKEN not in text and "ghp_abcdefghijklmnopqrstuvwxyz0123456789" not in text

    def test_reader_is_deterministic(self, home_tree):
        from agent.prompt_builder import _read_context_file
        project = home_tree / "projects" / "media-server"
        register_env_redaction_values({"APP_TOKEN": APP_TOKEN})
        assert _read_context_file(project / "CLAUDE.md") == _read_context_file(project / "CLAUDE.md")
