"""Tests for hermes_cli.workspace_guard — workspace identity validation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    """Create a minimal git repo at tmp_path and return its root."""
    (tmp_path / "README.md").write_text("# test\n")
    os.chdir(str(tmp_path))
    # Use subprocess to init git without importing it at module level
    import subprocess

    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    return tmp_path


class TestResolveGitRepoRoot:
    """Test _resolve_git_repo_root helper."""

    def test_returns_empty_for_none(self):
        from hermes_cli.workspace_guard import _resolve_git_repo_root

        assert _resolve_git_repo_root(None) == ""
        assert _resolve_git_repo_root("") == ""

    def test_returns_repo_root_for_git_dir(self, git_repo: Path):
        from hermes_cli.workspace_guard import _resolve_git_repo_root

        result = _resolve_git_repo_root(str(git_repo))
        assert result == str(git_repo.resolve())

    def test_returns_empty_for_non_git_dir(self, tmp_path: Path):
        from hermes_cli.workspace_guard import _resolve_git_repo_root

        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "file.txt").write_text("x")
        result = _resolve_git_repo_root(str(sub))
        assert result == ""


class TestResolveWorkspaceIdentity:
    """Test _resolve_workspace_identity helper."""

    def test_prefers_git_repo_root_over_cwd(self):
        from hermes_cli.workspace_guard import _resolve_workspace_identity

        row = {
            "git_repo_root": "/a/b/c",
            "cwd": "/x/y/z",
        }
        assert _resolve_workspace_identity(row) == "/a/b/c"

    def test_falls_back_to_cwd_when_no_git_repo_root(self):
        from hermes_cli.workspace_guard import _resolve_workspace_identity

        row = {
            "git_repo_root": "",
            "cwd": "/some/path",
        }
        # Should resolve cwd to git root if possible, else return cwd
        result = _resolve_workspace_identity(row)
        assert result == "/some/path"  # not a git repo

    def test_returns_empty_for_missing_fields(self):
        from hermes_cli.workspace_guard import _resolve_workspace_identity

        row: dict[str, str] = {}
        assert _resolve_workspace_identity(row) == ""


class TestCurrentWorkspaceIdentity:
    """Test _current_workspace_identity helper."""

    def test_returns_empty_for_none(self):
        from hermes_cli.workspace_guard import _current_workspace_identity

        assert _current_workspace_identity(None) == ""
        assert _current_workspace_identity("") == ""

    def test_resolves_git_repo_root(self, git_repo: Path):
        from hermes_cli.workspace_guard import _current_workspace_identity

        result = _current_workspace_identity(str(git_repo))
        assert result == str(git_repo.resolve())


class TestValidateSessionWorkspace:
    """Test validate_session_workspace main validation function."""

    def test_ok_when_both_match(self):
        from hermes_cli.workspace_guard import (
            WorkspaceGuardResult,
            _resolve_git_repo_root,
            validate_session_workspace,
        )

        with tempfile.TemporaryDirectory() as tmp:
            # Create a git repo
            import subprocess

            (Path(tmp) / "README.md").write_text("# test\n")
            subprocess.run(["git", "init"], cwd=tmp, check=True, capture_output=True)
            root = str(Path(tmp).resolve())

            row = {"git_repo_root": root}
            result = validate_session_workspace(row, current_cwd=root)

            assert result.ok is True
            assert result.blocked is False
            assert result.reason is None
            assert result.stored_workspace == root
            assert result.current_workspace == root

    def test_blocked_on_mismatch(self):
        from hermes_cli.workspace_guard import validate_session_workspace

        row = {"git_repo_root": "/repo/scout"}
        result = validate_session_workspace(row, current_cwd="/repo/hermes-agent")

        assert result.ok is False
        assert result.blocked is True
        assert result.reason == "workspace_mismatch"
        assert result.stored_workspace == "/repo/scout"
        assert result.current_workspace == "/repo/hermes-agent"

    def test_warns_on_legacy_session(self):
        from hermes_cli.workspace_guard import validate_session_workspace

        row: dict[str, str] = {}  # no git_repo_root or cwd
        result = validate_session_workspace(row, current_cwd="/some/path")

        assert result.ok is True
        assert result.blocked is False
        assert result.reason == "legacy_session"
        assert result.warning is not None

    def test_warns_on_no_current_identity(self):
        from hermes_cli.workspace_guard import validate_session_workspace

        row = {"git_repo_root": "/repo/scout"}
        result = validate_session_workspace(row, current_cwd=None)

        assert result.ok is True
        assert result.blocked is False
        assert result.reason == "no_current_identity"


class TestFilterSessionsByWorkspace:
    """Test filter_sessions_by_workspace helper."""

    def test_separates_compatible_from_incompatible(self):
        from hermes_cli.workspace_guard import filter_sessions_by_workspace

        sessions = [
            {"git_repo_root": "/repo/scout"},  # matches current
            {"git_repo_root": "/repo/hermes-agent"},  # doesn't match
            {},  # legacy — compatible but lower priority
        ]
        compatible, incompatible = filter_sessions_by_workspace(
            sessions, current_cwd="/repo/scout"
        )

        assert len(compatible) == 2
        assert len(incompatible) == 1
        assert incompatible[0]["git_repo_root"] == "/repo/hermes-agent"


class TestStampCompactionMetadata:
    """Test stamp_compaction_metadata helper."""

    def test_stamps_when_git_repo_root_present(self):
        from hermes_cli.workspace_guard import stamp_compaction_metadata

        row = {"git_repo_root": "/repo/scout"}
        summary = "some compaction text"
        result = stamp_compaction_metadata(summary, row)

        assert "<!-- HERMES_WORKSPACE:/repo/scout -->" in result
        assert result.startswith("some compaction text")

    def test_returns_unchanged_when_no_git_repo_root(self):
        from hermes_cli.workspace_guard import stamp_compaction_metadata

        row: dict[str, str] = {}
        summary = "some compaction text"
        result = stamp_compaction_metadata(summary, row)

        assert result == summary


class TestExtractWorkspaceFromCompaction:
    """Test extract_workspace_from_compaction helper."""

    def test_parses_stamped_marker(self):
        from hermes_cli.workspace_guard import extract_workspace_from_compaction

        text = "some text\n<!-- HERMES_WORKSPACE:/repo/scout -->\nmore text"
        result = extract_workspace_from_compaction(text)
        assert result == "/repo/scout"

    def test_returns_empty_when_no_marker(self):
        from hermes_cli.workspace_guard import extract_workspace_from_compaction

        result = extract_workspace_from_compaction("no marker here")
        assert result == ""


class TestAugmentSessionRowFromCompaction:
    """Test stamped compaction metadata fallback for legacy session rows."""

    def test_fills_missing_workspace_from_stamped_summary(self):
        from hermes_cli.workspace_guard import augment_session_row_from_compaction

        row: dict[str, str] = {"id": "session-1", "git_repo_root": "", "cwd": ""}
        messages = [
            {
                "role": "user",
                "content": "summary\n<!-- HERMES_WORKSPACE:/repo/from-summary -->",
            }
        ]

        augmented = augment_session_row_from_compaction(row, messages)

        assert augmented["git_repo_root"] == "/repo/from-summary"
        assert row["git_repo_root"] == ""

    def test_existing_workspace_identity_wins_over_summary_marker(self):
        from hermes_cli.workspace_guard import augment_session_row_from_compaction

        row = {"id": "session-1", "git_repo_root": "/repo/db"}
        messages = [
            {
                "role": "user",
                "content": "summary\n<!-- HERMES_WORKSPACE:/repo/from-summary -->",
            }
        ]

        assert augment_session_row_from_compaction(row, messages) is row

    def test_extracts_marker_from_multimodal_content(self):
        from hermes_cli.workspace_guard import augment_session_row_from_compaction

        row: dict[str, str] = {}
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "summary"},
                    {"type": "text", "text": "<!-- HERMES_WORKSPACE:/repo/mm -->"},
                ],
            }
        ]

        augmented = augment_session_row_from_compaction(row, messages)

        assert augmented["git_repo_root"] == "/repo/mm"


class TestFormatWorkspaceMismatchError:
    """Test format_workspace_mismatch_error helper."""

    def test_formats_user_message(self):
        from hermes_cli.workspace_guard import (
            WorkspaceGuardResult,
            format_workspace_mismatch_error,
        )

        result = WorkspaceGuardResult(
            ok=False,
            reason="workspace_mismatch",
            stored_workspace="/repo/scout",
            current_workspace="/repo/hermes-agent",
        )
        msg = format_workspace_mismatch_error(result)

        assert "Cross-workspace resume blocked" in msg
        assert "/repo/scout" in msg
        assert "/repo/hermes-agent" in msg


class TestFormatLegacyWarning:
    """Test format_legacy_warning helper."""

    def test_wraps_in_dim_markup(self):
        from hermes_cli.workspace_guard import format_legacy_warning

        result = format_legacy_warning("test message")
        assert "[dim]" in result
        assert "test message" in result
        assert "[/dim]" in result
