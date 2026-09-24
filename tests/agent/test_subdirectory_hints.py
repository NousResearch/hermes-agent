"""Tests for progressive subdirectory hint discovery."""

import time

import pytest
from pathlib import Path
from unittest.mock import patch

from agent.search_policy import SEARCH_PRUNE_DIR_NAMES
from agent.prompt_builder import drain_truncation_warnings
from agent.subdirectory_hints import SubdirectoryHintTracker


@pytest.fixture
def project(tmp_path):
    """Create a mock project tree with hint files in subdirectories."""
    # Root — already loaded at startup
    (tmp_path / "AGENTS.md").write_text("Root project instructions", encoding="utf-8")

    # backend/ — has its own AGENTS.md
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "AGENTS.md").write_text("Backend-specific instructions:\n- Use FastAPI\n- Always add type hints", encoding="utf-8")

    # backend/src/ — no hints
    (backend / "src").mkdir()
    (backend / "src" / "main.py").write_text("print('hello')", encoding="utf-8")

    # frontend/ — has CLAUDE.md
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "CLAUDE.md").write_text("Frontend rules:\n- Use TypeScript\n- No any types", encoding="utf-8")

    # docs/ — no hints
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "README.md").write_text("Documentation", encoding="utf-8")

    # deep/nested/path/ — has .cursorrules
    deep = tmp_path / "deep" / "nested" / "path"
    deep.mkdir(parents=True)
    (deep / ".cursorrules").write_text("Cursor rules for nested path", encoding="utf-8")

    return tmp_path


class TestSubdirectoryHintTracker:
    """Unit tests for SubdirectoryHintTracker."""



    def test_discovers_claude_md(self, project):
        """Frontend CLAUDE.md should be discovered."""
        tracker = SubdirectoryHintTracker(working_dir=str(project))
        result = tracker.check_tool_call(
            "read_file", {"path": str(project / "frontend" / "index.ts")}
        )
        assert result is not None
        assert "Frontend rules" in result

    def test_disabled_tracker_never_injects_hints(self, project):
        """A session that skips context files (cron without a workdir) must not have the same
        files spliced into tool results, where they leak into exact-output deliveries (#9441)."""
        tracker = SubdirectoryHintTracker(working_dir=str(project), enabled=False)
        assert tracker.check_tool_call("read_file", {"path": str(project / "frontend" / "index.ts")}) is None

    def test_no_duplicate_loading(self, project):
        """Same directory should not be loaded twice."""
        tracker = SubdirectoryHintTracker(working_dir=str(project))
        result1 = tracker.check_tool_call(
            "read_file", {"path": str(project / "frontend" / "a.ts")}
        )
        assert result1 is not None

        result2 = tracker.check_tool_call(
            "read_file", {"path": str(project / "frontend" / "b.ts")}
        )
        assert result2 is None  # already loaded




    @pytest.mark.parametrize("command", ["cd backend && ls", "pushd backend", "echo start; cd backend; ls", "cd backend;ls"])
    def test_bare_directory_after_navigation_command_is_a_path(self, project, command):
        """`cd backend` has no `/` or `.` yet names a subdirectory; its AGENTS.md must load (#11032)."""
        tracker = SubdirectoryHintTracker(working_dir=str(project))
        result = tracker.check_tool_call("terminal", {"command": command})
        assert result is not None and "Backend-specific instructions" in result

    @pytest.mark.parametrize("command", ["echo cd backend", "printf '%s %s' cd backend", "cd 'backend;'"])
    def test_cd_as_an_argument_or_a_quoted_other_name_is_not_navigation(self, project, command):
        """Only a `cd` that starts a shell segment navigates, and a quoted `'backend;'` is a
        different directory than `backend` — neither may inject backend/AGENTS.md."""
        tracker = SubdirectoryHintTracker(working_dir=str(project))
        assert tracker.check_tool_call("terminal", {"command": command}) is None

    def test_relative_path(self, project):
        """Relative paths resolved against working_dir."""
        tracker = SubdirectoryHintTracker(working_dir=str(project))
        result = tracker.check_tool_call(
            "read_file", {"path": "frontend/index.ts"}
        )
        assert result is not None
        assert "Frontend rules" in result





    def test_workdir_arg(self, project):
        """The workdir argument from terminal tool is checked."""
        tracker = SubdirectoryHintTracker(working_dir=str(project))
        result = tracker.check_tool_call(
            "terminal", {"command": "ls", "workdir": str(project / "frontend")}
        )
        assert result is not None
        assert "Frontend rules" in result



    def test_truncation_of_large_hints(self, tmp_path, caplog):
        """Over the ceiling: head AND tail survive, the marker names the file to read_file, and it is logged
        (the old silent tail-chop hid a truncated apps/desktop/AGENTS.md for months)."""
        import logging
        from agent import subdirectory_hints as sh
        sub = tmp_path / "bigdir"
        sub.mkdir()
        body = "HEAD-MARKER " + ("x" * (sh._MAX_HINT_CHARS + 5_000)) + " TAIL-MARKER"
        (sub / "AGENTS.md").write_text(body, encoding="utf-8")

        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        drain_truncation_warnings()
        with caplog.at_level(logging.WARNING, logger="agent.prompt_builder"):
            result = tracker.check_tool_call("read_file", {"path": str(sub / "file.py")})
        assert result is not None
        assert "HEAD-MARKER" in result and "TAIL-MARKER" in result
        assert "truncated AGENTS.md" in result and "bigdir/AGENTS.md" in result
        assert len(result) < len(body)
        assert any("TRUNCATED" in r.message and "AGENTS.md" in r.message for r in caplog.records)
        # A preview capped by a constant is not a context_file_max_chars problem: no chat status warning is
        # queued and the log does not send the user to a knob that cannot raise the cap (#111772).
        assert drain_truncation_warnings() == []
        assert "context_file_max_chars" not in caplog.text

    def test_area_file_under_ceiling_is_delivered_whole(self, tmp_path):
        """An area AGENTS.md sized like ours (well under the ceiling) arrives intact — no marker."""
        sub = tmp_path / "gateway"
        sub.mkdir()
        body = "# Gateway rules\n" + ("- rule\n" * 1500)   # ~12k chars: over the OLD 8k cap, under the new one
        (sub / "AGENTS.md").write_text(body, encoding="utf-8")
        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        result = tracker.check_tool_call("read_file", {"path": str(sub / "run.py")})
        assert result is not None and "truncated" not in result.lower()
        assert result.endswith(body.strip())

    def test_empty_args(self, project):
        """Empty args should not crash."""
        tracker = SubdirectoryHintTracker(working_dir=str(project))
        assert tracker.check_tool_call("read_file", {}) is None
        assert tracker.check_tool_call("terminal", {"command": ""}) is None



    def test_timeout_skips_slow_hint_files(self, project, monkeypatch, caplog):
        """Slow hint reads time out instead of blocking the turn."""
        backend = project / "backend"
        (backend / "AGENTS.md").write_text("Backend-specific instructions", encoding="utf-8")
        import sys

        from agent import subdirectory_hints as sh_mod

        # Patch the module object the hint tracker's helper closes over.
        pb_mod = sys.modules[sh_mod._read_text_with_timeout.__module__]
        monkeypatch.setattr(pb_mod, "_get_context_file_read_timeout", lambda: 0.05)

        original_read_text = Path.read_text

        def slow_read_text(self, *args, **kwargs):
            if self.name.lower() == "agents.md" and self.parent == backend:
                time.sleep(0.6)
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", slow_read_text)

        tracker = SubdirectoryHintTracker(working_dir=str(project))
        start = time.monotonic()
        with caplog.at_level("WARNING", logger="agent.prompt_builder"):
            result = tracker.check_tool_call(
                "read_file", {"path": str(project / "backend" / "src" / "main.py")}
            )
        elapsed = time.monotonic() - start

        assert elapsed < 0.4, f"hint load blocked for {elapsed:.2f}s"
        assert result is None
        assert "timed out" in caplog.text.lower()


class TestPermissionErrorHandling:
    """Regression tests for PermissionError in filesystem checks (ref #6214)."""


    def test_load_hints_permission_error_on_is_file(self, tmp_path):
        """_load_hints_for_directory should skip files when is_file() raises PermissionError."""
        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        restricted = tmp_path / "restricted"
        restricted.mkdir()
        original_is_file = Path.is_file
        def patched_is_file(self):
            if "restricted" in str(self):
                raise PermissionError("Permission denied")
            return original_is_file(self)
        with patch.object(Path, "is_file", patched_is_file):
            result = tracker._load_hints_for_directory(restricted)
        assert result is None

    def test_check_tool_call_survives_inaccessible_path(self, project):
        """Full check_tool_call should not crash when a path is inaccessible."""
        tracker = SubdirectoryHintTracker(working_dir=str(project))
        original_is_dir = Path.is_dir
        def patched_is_dir(self):
            if "backend" in str(self) and "src" not in str(self):
                raise PermissionError("Permission denied")
            return original_is_dir(self)
        with patch.object(Path, "is_dir", patched_is_dir):
            # Should not raise — gracefully skip the inaccessible directory
            result = tracker.check_tool_call(
                "read_file", {"path": str(project / "backend" / "src" / "main.py")}
            )
            # Result may be None (backend skipped) — the key point is no crash
            assert result is None or isinstance(result, str)


class TestOutsideWorkspaceRejection:
    """Direct tests for _is_valid_subdir rejecting outside-workspace paths."""




    def test_is_valid_subdir_rejects_sibling_dir(self, tmp_path, project):
        """_is_valid_subdir should reject a sibling directory (simulating ~/.codex)."""
        parent = tmp_path.parent
        outside = parent / ".test-codex"
        outside.mkdir(exist_ok=True)
        tracker = SubdirectoryHintTracker(working_dir=str(project))
        assert tracker._is_valid_subdir(outside) is False


class TestContentDeduplication:
    """The same context content must never be injected twice (ref: symlinked
    shared workspaces, hardlinks, and copied backups all alias one file)."""

    def test_symlinked_duplicate_not_reinjected(self, tmp_path):
        """Two directories whose AGENTS.md is the same file yield one injection."""
        real = tmp_path / "real"
        real.mkdir()
        (real / "AGENTS.md").write_text("Shared workspace instructions", encoding="utf-8")

        mirror = tmp_path / "mirror"
        mirror.mkdir()
        (mirror / "AGENTS.md").symlink_to(real / "AGENTS.md")

        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        first = tracker.check_tool_call("read_file", {"path": str(real / "x.py")})
        second = tracker.check_tool_call("read_file", {"path": str(mirror / "y.py")})

        assert first is not None
        assert "Shared workspace instructions" in first
        assert second is None

    def test_identical_copy_not_reinjected(self, tmp_path):
        """Byte-identical copies in unrelated directories dedupe by digest."""
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        (a / "AGENTS.md").write_text("Same content", encoding="utf-8")
        (b / "AGENTS.md").write_text("Same content", encoding="utf-8")

        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        assert tracker.check_tool_call("read_file", {"path": str(a / "f.py")}) is not None
        assert tracker.check_tool_call("read_file", {"path": str(b / "f.py")}) is None

    def test_differing_content_still_injected(self, tmp_path):
        """Dedupe must not suppress genuinely different context."""
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        (a / "AGENTS.md").write_text("Alpha rules", encoding="utf-8")
        (b / "AGENTS.md").write_text("Beta rules", encoding="utf-8")

        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        first = tracker.check_tool_call("read_file", {"path": str(a / "f.py")})
        second = tracker.check_tool_call("read_file", {"path": str(b / "f.py")})

        assert first is not None and "Alpha rules" in first
        assert second is not None and "Beta rules" in second

    def test_working_dir_content_seeded(self, tmp_path):
        """A copy of the CWD's own context file is not re-injected."""
        (tmp_path / "AGENTS.md").write_text("Root instructions", encoding="utf-8")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "AGENTS.md").write_text("Root instructions", encoding="utf-8")

        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        assert tracker.check_tool_call("read_file", {"path": str(elsewhere / "f.py")}) is None


class TestExcludedDirectories:
    """Backups, vendored deps, and caches hold copies — never context."""

    @pytest.mark.parametrize(
        "excluded",
        sorted(SEARCH_PRUNE_DIR_NAMES),
    )
    def test_excluded_directory_skipped(self, tmp_path, excluded):
        target = tmp_path / excluded / "snapshot"
        target.mkdir(parents=True)
        (target / "AGENTS.md").write_text("Stale archived instructions", encoding="utf-8")

        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        assert tracker.check_tool_call("read_file", {"path": str(target / "f.py")}) is None

    def test_excluded_ancestor_blocks_descendant(self, tmp_path):
        """A hint nested under an excluded ancestor is still skipped."""
        deep = tmp_path / "backups" / "2026" / "proj"
        deep.mkdir(parents=True)
        (deep / "AGENTS.md").write_text("Archived", encoding="utf-8")

        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        assert tracker.check_tool_call("read_file", {"path": str(deep / "f.py")}) is None

    def test_working_dir_inside_excluded_name_still_works(self, tmp_path):
        """If the user works inside e.g. vendor/, its own subdirs stay eligible."""
        root = tmp_path / "vendor" / "myproject"
        root.mkdir(parents=True)
        sub = root / "pkg"
        sub.mkdir()
        (sub / "AGENTS.md").write_text("Package rules", encoding="utf-8")

        tracker = SubdirectoryHintTracker(working_dir=str(root))
        result = tracker.check_tool_call("read_file", {"path": str(sub / "f.py")})
        assert result is not None and "Package rules" in result

    def test_normal_directory_unaffected(self, tmp_path):
        normal = tmp_path / "backend"
        normal.mkdir()
        (normal / "AGENTS.md").write_text("Backend rules", encoding="utf-8")

        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        result = tracker.check_tool_call("read_file", {"path": str(normal / "f.py")})
        assert result is not None and "Backend rules" in result

    def test_agents_override_md_wins_in_subdirectory(self, tmp_path):
        """AGENTS.override.md takes priority over AGENTS.md per directory."""
        sub = tmp_path / "backend"
        sub.mkdir()
        (sub / "AGENTS.md").write_text("Committed backend rules", encoding="utf-8")
        (sub / "AGENTS.override.md").write_text("Personal backend override", encoding="utf-8")

        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        result = tracker.check_tool_call("read_file", {"path": str(sub / "f.py")})
        assert result is not None
        assert "Personal backend override" in result
        assert "Committed backend rules" not in result


class TestSymlinkedHintTargets:
    """A hint file's resolved target must stay inside the working tree and off
    the read deny-list; the link's in-tree name does not sanitize its target (#116429)."""

    def test_escaping_or_denied_symlink_is_never_injected(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.md").write_text("OUTSIDE-MARKER-9f3a", encoding="utf-8")
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".env").write_text("API_KEY=SECRET-VALUE-7b2d", encoding="utf-8")
        # CWD-level escaping link: the __init__ digest seed must skip it too.
        (workspace / "AGENTS.md").symlink_to(outside / "secret.md")
        sub = workspace / "sub"
        sub.mkdir()
        (sub / "AGENTS.md").symlink_to(outside / "secret.md")
        envlink = workspace / "envlink"
        envlink.mkdir()
        (envlink / "AGENTS.md").symlink_to(workspace / ".env")

        tracker = SubdirectoryHintTracker(working_dir=str(workspace))
        assert tracker._loaded_digests == set()
        assert tracker.check_tool_call("read_file", {"path": str(sub / "f.py")}) is None
        assert tracker.check_tool_call("read_file", {"path": str(envlink / "f.py")}) is None
        assert SubdirectoryHintTracker(working_dir=str(workspace)).check_tool_call(
            "terminal", {"command": f"cd {sub}"}) is None

    def test_in_tree_hint_files_still_load(self, tmp_path):
        """A plain hint file and a symlink whose target stays inside the tree keep working."""
        workspace = tmp_path / "workspace"
        docs = workspace / "docs"
        docs.mkdir(parents=True)
        (docs / "AGENTS.md").write_text("Shared in-tree instructions", encoding="utf-8")
        linked = workspace / "linked"
        linked.mkdir()
        (linked / "AGENTS.md").symlink_to(docs / "AGENTS.md")
        plain = workspace / "plain"
        plain.mkdir()
        (plain / "AGENTS.md").write_text("Legit subdirectory rules", encoding="utf-8")

        tracker = SubdirectoryHintTracker(working_dir=str(workspace))
        result = tracker.check_tool_call("read_file", {"path": str(linked / "f.py")})
        assert result is not None and "Shared in-tree instructions" in result
        result = tracker.check_tool_call("read_file", {"path": str(plain / "f.py")})
        assert result is not None and "Legit subdirectory rules" in result
