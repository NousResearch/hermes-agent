"""Tests for progressive subdirectory hint discovery."""

import pytest
from pathlib import Path
from unittest.mock import patch

from agent.subdirectory_hints import SubdirectoryHintTracker


@pytest.fixture
def project(tmp_path):
    """Create a mock project tree with hint files in subdirectories."""
    # Root — already loaded at startup
    (tmp_path / "AGENTS.md").write_text("Root project instructions")

    # backend/ — has its own AGENTS.md
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "AGENTS.md").write_text("Backend-specific instructions:\n- Use FastAPI\n- Always add type hints")

    # backend/src/ — no hints
    (backend / "src").mkdir()
    (backend / "src" / "main.py").write_text("print('hello')")

    # frontend/ — has CLAUDE.md
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "CLAUDE.md").write_text("Frontend rules:\n- Use TypeScript\n- No any types")

    # docs/ — no hints
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "README.md").write_text("Documentation")

    # deep/nested/path/ — has .cursorrules
    deep = tmp_path / "deep" / "nested" / "path"
    deep.mkdir(parents=True)
    (deep / ".cursorrules").write_text("Cursor rules for nested path")

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




    def test_relative_path(self, project):
        """Relative paths resolved against working_dir."""
        tracker = SubdirectoryHintTracker(working_dir=str(project))
        result = tracker.check_tool_call(
            "read_file", {"path": "frontend/index.ts"}
        )
        assert result is not None
        assert "Frontend rules" in result




    def test_denied_subtree_does_not_load_context(self, project):
        """permissions.deny.paths blocks progressive context discovery."""
        private = project / "private"
        private.mkdir()
        (private / "AGENTS.md").write_text("PRIVATE CONTEXT MUST NOT LEAK")
        (private / "data.txt").write_text("private data")

        tracker = SubdirectoryHintTracker(working_dir=str(project))
        with patch(
            "agent.deny_policy.permissions_deny_paths",
            return_value=[str(private / "**")],
        ):
            result = tracker.check_tool_call(
                "read_file", {"path": str(private / "data.txt")}
            )

        assert result is None

    def test_denied_hint_file_is_skipped(self, project):
        """A file-specific deny applies before context hint content is read."""
        mixed = project / "mixed"
        mixed.mkdir()
        agents = mixed / "AGENTS.md"
        agents.write_text("DENIED AGENTS CONTEXT")
        (mixed / "CLAUDE.md").write_text("Allowed sibling context")
        (mixed / "data.txt").write_text("ordinary data")

        tracker = SubdirectoryHintTracker(working_dir=str(project))
        with patch(
            "agent.deny_policy.permissions_deny_paths",
            return_value=[str(agents)],
        ):
            result = tracker.check_tool_call(
                "read_file", {"path": str(mixed / "data.txt")}
            )

        assert result is not None
        assert "Allowed sibling context" in result
        assert "DENIED AGENTS CONTEXT" not in result

    def test_relative_deny_rule_skips_hint_file(self, project):
        """Relative hint-file rules are anchored to the tracker working dir."""
        mixed = project / "mixed-relative"
        mixed.mkdir()
        (mixed / "AGENTS.md").write_text("RELATIVE DENIED HINT")
        (mixed / "CLAUDE.md").write_text("Allowed relative hint sibling")
        (mixed / "data.txt").write_text("ordinary data")

        tracker = SubdirectoryHintTracker(working_dir=str(project))
        with patch(
            "agent.deny_policy.permissions_deny_paths",
            return_value=["mixed-relative/AGENTS.md"],
        ):
            result = tracker.check_tool_call(
                "read_file", {"path": str(mixed / "data.txt")}
            )

        assert result is not None
        assert "Allowed relative hint sibling" in result
        assert "RELATIVE DENIED HINT" not in result

    def test_progressive_context_rejects_symlink_to_sensitive_file(self, tmp_path):
        secret = tmp_path / "secrets" / ".env"
        secret.parent.mkdir()
        secret.write_text("PROGRESSIVE_CONTEXT_SECRET=must-not-load", encoding="utf-8")
        subdir = tmp_path / "pkg"
        subdir.mkdir()
        try:
            (subdir / "AGENTS.md").symlink_to(secret)
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")

        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        result = tracker.check_tool_call(
            "read_file",
            {"path": str(subdir / "module.py")},
        )

        assert result is None

    def test_denied_working_dir_context_is_not_read_during_digest_seed(self, project):
        """Tracker initialization must not read a denied startup context file."""
        agents = project / "AGENTS.md"
        original_read_text = Path.read_text

        def guarded_read_text(path_obj, *args, **kwargs):
            if path_obj == agents:
                raise AssertionError("denied context was read during digest seeding")
            return original_read_text(path_obj, *args, **kwargs)

        with (
            patch(
                "agent.deny_policy.permissions_deny_paths",
                return_value=[str(agents)],
            ),
            patch.object(Path, "read_text", guarded_read_text),
        ):
            SubdirectoryHintTracker(working_dir=str(project))

    def test_digest_seed_checks_lexical_working_dir_before_resolve(self, tmp_path):
        lexical_workdir = tmp_path / "alias-workspace"
        lexical_workdir.mkdir()
        (lexical_workdir / "AGENTS.md").write_text("DENIED LEXICAL HINT")
        original_resolve = Path.resolve

        def guarded_resolve(path_obj, *args, **kwargs):
            if path_obj == lexical_workdir:
                raise AssertionError("working dir resolved before lexical deny")
            return original_resolve(path_obj, *args, **kwargs)

        with (
            patch(
                "agent.deny_policy.permissions_deny_paths",
                return_value=[str(tmp_path / "alias-*" / "AGENTS.md")],
            ),
            patch.object(Path, "resolve", autospec=True, side_effect=guarded_resolve),
        ):
            SubdirectoryHintTracker(working_dir=str(lexical_workdir))

    def test_digest_seed_preflights_fixed_width_denied_ancestor(self, tmp_path):
        denied_ancestor = tmp_path / "private1"
        working_dir = denied_ancestor / "src"
        working_dir.mkdir(parents=True)
        (working_dir / "AGENTS.md").write_text("DENIED DIGEST ANCESTOR")
        original_is_file = Path.is_file
        from agent import deny_policy

        def in_denied_subtree(path_obj):
            return path_obj == denied_ancestor or denied_ancestor in path_obj.parents

        def guarded_is_file(path_obj):
            if in_denied_subtree(path_obj):
                raise AssertionError("digest seed probed inside denied ancestor")
            return original_is_file(path_obj)

        with (
            patch(
                "agent.deny_policy.permissions_deny_paths",
                return_value=[str(tmp_path / "private?")],
            ),
            patch(
                "agent.deny_policy.os.path.realpath",
                wraps=deny_policy.os.path.realpath,
            ) as mock_realpath,
            patch.object(Path, "is_file", guarded_is_file),
        ):
            SubdirectoryHintTracker(working_dir=str(working_dir))

        assert not any(
            in_denied_subtree(Path(call.args[0]))
            for call in mock_realpath.call_args_list
        )

    def test_progressive_discovery_rechecks_ancestor_above_working_dir(
        self, tmp_path
    ):
        """A denied parent of working_dir blocks every later hint discovery."""
        denied_ancestor = tmp_path / "private1"
        working_dir = denied_ancestor / "src"
        subdir = working_dir / "pkg"
        subdir.mkdir(parents=True)
        hint = subdir / "AGENTS.md"
        hint.write_text("SHOULD_NOT_LOAD")
        target = subdir / "module.py"
        target.write_text("data")
        original_resolve = Path.resolve
        original_read_text = Path.read_text
        from agent import deny_policy

        def in_denied_subtree(path_obj):
            return path_obj == denied_ancestor or denied_ancestor in path_obj.parents

        def guarded_resolve(path_obj, *args, **kwargs):
            if in_denied_subtree(path_obj):
                raise AssertionError("progressive discovery resolved inside denied ancestor")
            return original_resolve(path_obj, *args, **kwargs)

        def guarded_read_text(path_obj, *args, **kwargs):
            if in_denied_subtree(path_obj):
                raise AssertionError("progressive discovery read inside denied ancestor")
            return original_read_text(path_obj, *args, **kwargs)

        with (
            patch(
                "agent.deny_policy.permissions_deny_paths",
                return_value=[str(tmp_path / "private?")],
            ),
            patch(
                "agent.deny_policy.os.path.realpath",
                wraps=deny_policy.os.path.realpath,
            ) as mock_realpath,
            patch.object(Path, "resolve", autospec=True, side_effect=guarded_resolve),
            patch.object(Path, "read_text", autospec=True, side_effect=guarded_read_text),
        ):
            tracker = SubdirectoryHintTracker(working_dir=str(working_dir))
            result = tracker.check_tool_call(
                "read_file",
                {"path": str(target)},
            )

        assert result is None
        assert not any(
            in_denied_subtree(Path(call.args[0]))
            for call in mock_realpath.call_args_list
        )

    def test_progressive_discovery_preflights_denied_ancestor_before_metadata(
        self,
        tmp_path,
    ):
        denied_ancestor = tmp_path / "private1"
        nested = denied_ancestor / "src"
        nested.mkdir(parents=True)
        hint = nested / "AGENTS.md"
        hint.write_text("DENIED ANCESTOR CONTEXT")
        target = nested / "file.py"
        target.write_text("data")
        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        original_is_dir = Path.is_dir
        original_read_text = Path.read_text

        def guarded_is_dir(path_obj):
            if denied_ancestor == path_obj or denied_ancestor in path_obj.parents:
                raise AssertionError("denied ancestor subtree was probed")
            return original_is_dir(path_obj)

        def guarded_read_text(path_obj, *args, **kwargs):
            if denied_ancestor == path_obj or denied_ancestor in path_obj.parents:
                raise AssertionError("denied ancestor context was read")
            return original_read_text(path_obj, *args, **kwargs)

        from agent import deny_policy

        with (
            patch(
                "agent.deny_policy.permissions_deny_paths",
                return_value=[str(tmp_path / "private?")],
            ),
            patch(
                "agent.deny_policy.os.path.realpath",
                wraps=deny_policy.os.path.realpath,
            ) as mock_realpath,
            patch.object(Path, "is_dir", guarded_is_dir),
            patch.object(Path, "read_text", guarded_read_text),
        ):
            result = tracker.check_tool_call(
                "read_file",
                {"path": str(target)},
            )

        assert result is None
        mock_realpath.assert_not_called()

    def test_workdir_arg(self, project):
        """The workdir argument from terminal tool is checked."""
        tracker = SubdirectoryHintTracker(working_dir=str(project))
        result = tracker.check_tool_call(
            "terminal", {"command": "ls", "workdir": str(project / "frontend")}
        )
        assert result is not None
        assert "Frontend rules" in result



    def test_truncation_of_large_hints(self, tmp_path):
        """Hint files over the limit are truncated."""
        sub = tmp_path / "bigdir"
        sub.mkdir()
        (sub / "AGENTS.md").write_text("x" * 20_000)

        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        result = tracker.check_tool_call(
            "read_file", {"path": str(sub / "file.py")}
        )
        assert result is not None
        assert "truncated" in result.lower()
        # Should be capped
        assert len(result) < 20_000

    def test_empty_args(self, project):
        """Empty args should not crash."""
        tracker = SubdirectoryHintTracker(working_dir=str(project))
        assert tracker.check_tool_call("read_file", {}) is None
        assert tracker.check_tool_call("terminal", {"command": ""}) is None



class TestPermissionErrorHandling:
    """Regression tests for PermissionError in filesystem checks (ref #6214)."""

    def test_is_valid_subdir_permission_error(self, tmp_path):
        """_is_valid_subdir should return False when is_dir() raises PermissionError."""
        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        restricted = tmp_path / "restricted"
        restricted.mkdir()
        with patch.object(Path, "is_dir", side_effect=PermissionError("Permission denied")):
            assert tracker._is_valid_subdir(restricted) is False

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


    def test_is_valid_subdir_allows_inside_path(self, project):
        """_is_valid_subdir should return True for paths inside working_dir."""
        tracker = SubdirectoryHintTracker(working_dir=str(project))
        backend = project / "backend"
        assert tracker._is_valid_subdir(backend) is True


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
        (real / "AGENTS.md").write_text("Shared workspace instructions")

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
        (a / "AGENTS.md").write_text("Same content")
        (b / "AGENTS.md").write_text("Same content")

        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        assert tracker.check_tool_call("read_file", {"path": str(a / "f.py")}) is not None
        assert tracker.check_tool_call("read_file", {"path": str(b / "f.py")}) is None

    def test_differing_content_still_injected(self, tmp_path):
        """Dedupe must not suppress genuinely different context."""
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        (a / "AGENTS.md").write_text("Alpha rules")
        (b / "AGENTS.md").write_text("Beta rules")

        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        first = tracker.check_tool_call("read_file", {"path": str(a / "f.py")})
        second = tracker.check_tool_call("read_file", {"path": str(b / "f.py")})

        assert first is not None and "Alpha rules" in first
        assert second is not None and "Beta rules" in second

    def test_working_dir_content_seeded(self, tmp_path):
        """A copy of the CWD's own context file is not re-injected."""
        (tmp_path / "AGENTS.md").write_text("Root instructions")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "AGENTS.md").write_text("Root instructions")

        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        assert tracker.check_tool_call("read_file", {"path": str(elsewhere / "f.py")}) is None


class TestExcludedDirectories:
    """Backups, vendored deps, and caches hold copies — never context."""

    @pytest.mark.parametrize(
        "excluded",
        ["backups", "node_modules", ".git", "venv", "site-packages", ".Trash", "vendor"],
    )
    def test_excluded_directory_skipped(self, tmp_path, excluded):
        target = tmp_path / excluded / "snapshot"
        target.mkdir(parents=True)
        (target / "AGENTS.md").write_text("Stale archived instructions")

        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        assert tracker.check_tool_call("read_file", {"path": str(target / "f.py")}) is None

    def test_excluded_ancestor_blocks_descendant(self, tmp_path):
        """A hint nested under an excluded ancestor is still skipped."""
        deep = tmp_path / "backups" / "2026" / "proj"
        deep.mkdir(parents=True)
        (deep / "AGENTS.md").write_text("Archived")

        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        assert tracker.check_tool_call("read_file", {"path": str(deep / "f.py")}) is None

    def test_working_dir_inside_excluded_name_still_works(self, tmp_path):
        """If the user works inside e.g. vendor/, its own subdirs stay eligible."""
        root = tmp_path / "vendor" / "myproject"
        root.mkdir(parents=True)
        sub = root / "pkg"
        sub.mkdir()
        (sub / "AGENTS.md").write_text("Package rules")

        tracker = SubdirectoryHintTracker(working_dir=str(root))
        result = tracker.check_tool_call("read_file", {"path": str(sub / "f.py")})
        assert result is not None and "Package rules" in result

    def test_normal_directory_unaffected(self, tmp_path):
        normal = tmp_path / "backend"
        normal.mkdir()
        (normal / "AGENTS.md").write_text("Backend rules")

        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        result = tracker.check_tool_call("read_file", {"path": str(normal / "f.py")})
        assert result is not None and "Backend rules" in result

    def test_agents_override_md_wins_in_subdirectory(self, tmp_path):
        """AGENTS.override.md takes priority over AGENTS.md per directory."""
        sub = tmp_path / "backend"
        sub.mkdir()
        (sub / "AGENTS.md").write_text("Committed backend rules")
        (sub / "AGENTS.override.md").write_text("Personal backend override")

        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        result = tracker.check_tool_call("read_file", {"path": str(sub / "f.py")})
        assert result is not None
        assert "Personal backend override" in result
        assert "Committed backend rules" not in result
