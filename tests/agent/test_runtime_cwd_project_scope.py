"""Tests for runtime_cwd.resolve_project_scope() — project-scoped memory (issue #33638)."""

from pathlib import Path

import pytest

import agent.runtime_cwd as rt


class TestResolveProjectScope:
    """resolve_project_scope() maps the agent's working directory to a project
    scope identifier (the basename of the nearest ancestor project root).

    The walk-up semantics mirror _marker_root in coding_context.py: bounded at
    6 levels past the start, skip $HOME and the shared temp root, and never
    scope into the Hermes install tree.
    """

    # ── (a) Git repo subdirectory → repo-root basename ──────────────────────────

    def test_cwd_in_git_repo_subdirectory(self, monkeypatch, tmp_path):
        """A .git marker at the project root, with cwd in a subdirectory."""
        project = tmp_path / "my-app"
        project.mkdir()
        (project / ".git").mkdir()
        subdir = project / "src" / "sub"
        subdir.mkdir(parents=True)

        monkeypatch.setenv("TERMINAL_CWD", str(subdir))
        assert rt.resolve_project_scope() == "my-app"

    # ── (b) AGENTS.md marker discovered ─────────────────────────────────────────

    def test_agents_md_marker(self, monkeypatch, tmp_path):
        """An AGENTS.md marker at the project root (no .git)."""
        project = tmp_path / "agents-project"
        project.mkdir()
        (project / "AGENTS.md").write_text("agent instructions")

        monkeypatch.setenv("TERMINAL_CWD", str(project))
        assert rt.resolve_project_scope() == "agents-project"

    # ── (c) cwd in $HOME with no markers → "" ───────────────────────────────────

    def test_home_hermetic(self, monkeypatch, tmp_path):
        """Monkeypatch home to a tmp dir with no markers; cwd there -> ""."""
        home = tmp_path / "fakehome"
        home.mkdir()

        monkeypatch.setattr(Path, "home", lambda: home)
        monkeypatch.setenv("TERMINAL_CWD", str(home))

        assert rt.resolve_project_scope() == ""

    # ── (d) cwd inside the package/install tree → "" ────────────────────────────

    def test_inside_install_tree_returns_empty(self, monkeypatch):
        """When cwd is inside the Hermes package root, scope is empty."""
        # The repo checkout itself IS the package root.
        # So pointing cwd to the repo root must return "".
        repo_root = Path(__file__).resolve().parent.parent.parent
        monkeypatch.setenv("TERMINAL_CWD", str(repo_root))

        assert rt.resolve_project_scope() == ""

    def test_inside_install_tree_subdir_returns_empty(self, monkeypatch):
        """Even inside a subdirectory of the install tree, scope is empty."""
        # agent/ subdirectory of the package root
        repo_root = Path(__file__).resolve().parent.parent.parent
        cwd = repo_root / "agent"
        monkeypatch.setenv("TERMINAL_CWD", str(cwd))

        assert rt.resolve_project_scope() == ""

    # ── (e) Deepest marker wins when markers nest ───────────────────────────────

    def test_deepest_marker_wins_when_nested(self, monkeypatch, tmp_path):
        """When markers exist at both parent and child, the deepest one wins."""
        outer = tmp_path / "outer"
        inner = outer / "inner"
        inner.mkdir(parents=True)
        (outer / ".git").mkdir()
        (inner / "AGENTS.md").write_text("in inner")

        monkeypatch.setenv("TERMINAL_CWD", str(inner))
        assert rt.resolve_project_scope() == "inner"

    # ── (f) Unmatched deep path → "" ────────────────────────────────────────────

    def test_unmatched_deep_path_returns_empty(self, monkeypatch, tmp_path):
        """A deep path with no markers in any ancestor returns empty."""
        deep = tmp_path / "a" / "b" / "c" / "d" / "e" / "f" / "g"
        deep.mkdir(parents=True)

        monkeypatch.setenv("TERMINAL_CWD", str(deep))
        assert rt.resolve_project_scope() == ""

    # ── Additional edge cases ───────────────────────────────────────────────────

    def test_hermes_memory_marker(self, monkeypatch, tmp_path):
        """.hermes-memory.md marker discovered."""
        project = tmp_path / "memory-project"
        project.mkdir()
        (project / ".hermes-memory.md").write_text("memory content")

        monkeypatch.setenv("TERMINAL_CWD", str(project / "sub"))
        (project / "sub").mkdir()
        assert rt.resolve_project_scope() == "memory-project"

    def test_marker_outside_home_is_ignored_when_home_is_default(self, monkeypatch, tmp_path):
        """A marker sitting in $HOME is not treated as a project scope."""
        home = tmp_path / "home"
        home.mkdir()
        (home / "AGENTS.md").write_text("global config")
        sub = home / "sub"
        sub.mkdir()

        monkeypatch.setattr(Path, "home", lambda: home)
        monkeypatch.setenv("TERMINAL_CWD", str(sub))

        # The walk starts at sub, goes up to home (which has AGENTS.md but is
        # skipped because it is $HOME), then past home — no markers beyond.
        assert rt.resolve_project_scope() == ""

    def test_marker_in_shared_temp_ignored(self, monkeypatch, tmp_path):
        """A marker in the system temp root does not produce a project scope."""
        # Monkeypatch tempfile.gettempdir() to return a subdir of tmp_path,
        # and place cwd there with a .git marker.
        import tempfile
        temp_root = tmp_path / "temp"
        temp_root.mkdir()
        (temp_root / ".git").mkdir()
        sub = temp_root / "work"
        sub.mkdir()

        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(temp_root))
        monkeypatch.setenv("TERMINAL_CWD", str(sub))

        assert rt.resolve_project_scope() == ""

    def test_permission_error_on_ancestor_does_not_raise(self, monkeypatch, tmp_path):
        """A PermissionError during marker probes on an ancestor is caught and
        does not abort the walk — marker checks above the inaccessible level
        still succeed."""
        project = tmp_path / "permission-project"
        project.mkdir()
        (project / ".git").mkdir()

        # Set cwd deep inside project
        deep = project / "a" / "b"
        deep.mkdir(parents=True)

        # Monkeypatch Path.exists to raise PermissionError when probing any
        # project-scope marker file inside the "b" directory.  The walk starts
        # at permission-project/a/b; all three marker probes at depth=0 will
        # fire the condition (self.name is one of the markers, and the parent
        # is "b"), so PermissionError is actually exercised.
        original_exists = Path.exists

        def fragile_exists(self):
            if (
                self.name in (".git", "AGENTS.md", ".hermes-memory.md")
                and self.parent.name == "b"
            ):
                raise PermissionError("Access denied")
            return original_exists(self)

        monkeypatch.setattr(Path, "exists", fragile_exists)
        monkeypatch.setenv("TERMINAL_CWD", str(deep))

        result = rt.resolve_project_scope()

        # The walk:
        #   depth=0 (permission-project/a/b)  — all markers raise PermissionError → continue
        #   depth=1 (permission-project/a)    — no markers exist
        #   depth=2 (permission-project)      — .git found → returns "permission-project"
        #
        # The PermissionError is caught and the walk still resolves to the
        # marker above the inaccessible directory.
        assert result == "permission-project"

    def test_resolve_context_cwd_fallback_to_getcwd(self, monkeypatch, tmp_path):
        """When resolve_context_cwd returns None, resolve_project_scope falls back
        to os.getcwd()."""
        project = tmp_path / "fallback-project"
        project.mkdir()
        (project / ".git").mkdir()

        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        monkeypatch.chdir(str(project))

        assert rt.resolve_project_scope() == "fallback-project"

    def test_no_terminal_cwd_and_not_in_project_returns_empty(self, monkeypatch, tmp_path):
        """No markers anywhere in the cwd chain -> empty string."""
        empty = tmp_path / "empty-dir"
        empty.mkdir()

        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        monkeypatch.chdir(str(empty))

        assert rt.resolve_project_scope() == ""

    def test_walk_stops_at_home(self, monkeypatch, tmp_path):
        """Marker in a parent of $HOME is NOT found because the walk breaks at $HOME.

        Create a fake home /x/home with a marker at /x/AGENTS.md, and cwd at
        /x/home/proj. The walk reaches /x/home (home) and breaks, never
        reaching /x where the marker sits → scope is empty.
        """
        home = tmp_path / "x" / "home"
        home.mkdir(parents=True)
        # Marker in /x (parent of home)
        (tmp_path / "x" / "AGENTS.md").write_text("marker")
        # cwd inside /x/home
        proj = home / "proj"
        proj.mkdir()

        monkeypatch.setattr(Path, "home", lambda: home)
        monkeypatch.setenv("TERMINAL_CWD", str(proj))

        assert rt.resolve_project_scope() == ""
