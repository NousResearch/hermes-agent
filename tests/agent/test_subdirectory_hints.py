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


class TestCloudBackedPathRefuse:
    """iCloud / FileProvider paths must never be opened (gateway wedge 2026-08-18)."""

    def test_is_cloud_backed_mobile_documents(self):
        from agent.subdirectory_hints import _is_cloud_backed_path

        p = Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "AGENTS.md"
        assert _is_cloud_backed_path(p) is True

    def test_is_cloud_backed_cloudstorage(self):
        from agent.subdirectory_hints import _is_cloud_backed_path

        p = Path.home() / "Library" / "CloudStorage" / "Dropbox" / "AGENTS.md"
        assert _is_cloud_backed_path(p) is True

    def test_is_cloud_backed_substring_icloud(self):
        from agent.subdirectory_hints import _is_cloud_backed_path

        assert _is_cloud_backed_path(Path("/Users/x/iCloudDrive/proj/AGENTS.md")) is True
        assert _is_cloud_backed_path(Path("/tmp/foo.icloud")) is True
        assert _is_cloud_backed_path(Path("/tmp/com.apple.fileprovider/x")) is True

    def test_local_path_not_refused(self, tmp_path):
        from agent.subdirectory_hints import _is_cloud_backed_path

        assert _is_cloud_backed_path(tmp_path / "AGENTS.md") is False

    def test_load_skips_icloud_named_path_without_open(self, tmp_path):
        """Even inside working_dir, cloud-shaped names are refused preflight."""
        from agent.subdirectory_hints import _read_hint_text

        # Synthetic path string containing refuse markers — never open.
        cloudish = Path("/Users/test/Library/Mobile Documents/com~apple~CloudDocs/AGENTS.md")
        assert _read_hint_text(cloudish) is None

    def test_seed_skips_cloud_cwd_hint(self):
        """Startup digest seed must not open a cloud-backed CWD hint file."""
        from agent import subdirectory_hints as mod

        calls = {"open": 0}
        real_open = open

        def counting_open(*args, **kwargs):
            calls["open"] += 1
            return real_open(*args, **kwargs)

        cloud_root = Path(
            "/Users/x/Library/Mobile Documents/com~apple~CloudDocs/proj"
        )
        tracker = object.__new__(mod.SubdirectoryHintTracker)
        tracker.working_dir = cloud_root
        tracker._loaded_dirs = {cloud_root}
        tracker._loaded_digests = set()
        with patch("builtins.open", side_effect=counting_open):
            tracker._seed_working_dir_digest()
        assert calls["open"] == 0
        assert tracker._loaded_digests == set()


class TestBoundedHintRead:
    """Slow/hanging hint files must not wedge the tool path."""

    def test_normal_local_file_still_loads(self, tmp_path):
        sub = tmp_path / "backend"
        sub.mkdir()
        (sub / "AGENTS.md").write_text("Backend rules safe")

        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        result = tracker.check_tool_call(
            "read_file", {"path": str(sub / "f.py")}
        )
        assert result is not None
        assert "Backend rules safe" in result

    def test_fifo_read_times_out_fast(self, tmp_path):
        """A FIFO with no writer hangs bare read_text forever — must return fast."""
        import os
        import time

        from agent.subdirectory_hints import (
            _HINT_READ_TIMEOUT_SECONDS,
            _read_hint_text,
        )

        fifo = tmp_path / "AGENTS.md"
        os.mkfifo(fifo)

        started = time.monotonic()
        content = _read_hint_text(fifo)
        elapsed = time.monotonic() - started

        assert content is None
        # Allow generous slack over the 2s cap, but nowhere near a wedge.
        assert elapsed < _HINT_READ_TIMEOUT_SECONDS + 2.0

    def test_load_hints_survives_slow_file(self, tmp_path):
        """_load_hints_for_directory must hit bounded read and return fast.

        FIFOs make Path.is_file() False, so they never reach _read_hint_text.
        Hang open() on a real regular file to exercise the load-path timeout.

        Important: pass a *resolved* directory. On macOS tmp paths often live
        under /var -> /private/var; SubdirectoryHintTracker resolves working_dir,
        and an unresolved subdir fails is_relative_to and returns before open.
        """
        import time

        from agent.subdirectory_hints import _HINT_READ_TIMEOUT_SECONDS

        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        # Use a path under the tracker's resolved working_dir, not raw tmp_path.
        sub = tracker.working_dir / "slowdir"
        sub.mkdir()
        hint = sub / "AGENTS.md"
        # Real regular file so path construction is normal; open is mocked hang.
        hint.write_text("should never be returned if open hangs")

        open_calls = {"n": 0}

        def hanging_open(*_args, **_kwargs):
            open_calls["n"] += 1
            time.sleep(30)
            raise AssertionError("open should have been abandoned on timeout")

        started = time.monotonic()
        with patch("builtins.open", side_effect=hanging_open):
            result = tracker._load_hints_for_directory(sub)
        elapsed = time.monotonic() - started

        assert result is None
        assert open_calls["n"] >= 1  # load path actually reached bounded open/read
        # Must actually wait near the timeout (proves we reached _read_hint_text),
        # but nowhere near a multi-minute wedge.
        # macOS default APFS is case-insensitive, so AGENTS.md and agents.md are
        # the same inode and the load loop may pay the timeout twice (~4s).
        assert elapsed >= _HINT_READ_TIMEOUT_SECONDS - 0.5
        assert elapsed < (_HINT_READ_TIMEOUT_SECONDS * 2) + 2.0
