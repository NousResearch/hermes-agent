"""Tests for agent.subdirectory_hints RuntimeError guards.

Python 3.11+ raises RuntimeError (not OSError/ValueError) from
Path.expanduser() and Path.home() when $HOME is unset.  These tests
verify that the hint tracker handles the missing-$HOME case gracefully
instead of crashing the agent conversation loop.

Regression test for https://github.com/NousResearch/hermes-agent/issues/45401
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.subdirectory_hints import SubdirectoryHintTracker


@pytest.fixture()
def tracker(tmp_path: Path) -> SubdirectoryHintTracker:
    """Create a tracker rooted in a temporary directory."""
    return SubdirectoryHintTracker(working_dir=tmp_path)


# ---------------------------------------------------------------------------
# _add_path_candidate — expanduser() RuntimeError
# ---------------------------------------------------------------------------

class TestAddPathCandidateRuntimeError:
    """expanduser() raises RuntimeError when $HOME is unset (Python 3.11+)."""

    def test_expanduser_runtime_error_gracefully_skipped(
        self, tracker: SubdirectoryHintTracker, tmp_path: Path
    ) -> None:
        """_add_path_candidate must not crash when expanduser() raises RuntimeError."""
        candidates: set[Path] = set()

        with patch("pathlib.Path.expanduser", side_effect=RuntimeError("Could not determine home directory.")):
            # Should not raise — RuntimeError is caught
            tracker._add_path_candidate("~/some/path", candidates)

        # Candidate set should be empty (path was skipped)
        assert candidates == set()


# ---------------------------------------------------------------------------
# _is_valid_subdir — is_relative_to() RuntimeError
# ---------------------------------------------------------------------------

class TestIsValidSubdirRuntimeError:
    """is_relative_to() may raise RuntimeError on some Python builds."""

    def test_is_relative_to_runtime_error_returns_false(
        self, tracker: SubdirectoryHintTracker, tmp_path: Path
    ) -> None:
        """_is_valid_subdir must return False (not crash) when is_relative_to raises RuntimeError."""
        # Use a path outside the working directory so _is_ancestor_or_same
        # also returns False after the RuntimeError fallback.
        outside = Path("/tmp/outside-workdir-subdir")
        outside.mkdir(exist_ok=True)
        try:
            with patch.object(Path, "is_relative_to", side_effect=RuntimeError("Could not determine home directory.")):
                result = tracker._is_valid_subdir(outside)
        finally:
            outside.rmdir()

        assert result is False


# ---------------------------------------------------------------------------
# _load_hints_for_directory — is_relative_to() + Path.home() RuntimeError
# ---------------------------------------------------------------------------

class TestLoadHintsForDirectoryRuntimeError:
    """_load_hints_for_directory must not crash when $HOME is unset."""

    def test_is_relative_to_runtime_error_returns_none(
        self, tracker: SubdirectoryHintTracker, tmp_path: Path
    ) -> None:
        """When is_relative_to raises RuntimeError, returns None (skip directory)."""
        subdir = tmp_path / "project"
        subdir.mkdir()

        with patch.object(Path, "is_relative_to", side_effect=RuntimeError("Could not determine home directory.")):
            result = tracker._load_hints_for_directory(subdir)

        assert result is None

    def test_path_home_runtime_error_in_relative_path_fallback(
        self, tracker: SubdirectoryHintTracker, tmp_path: Path
    ) -> None:
        """When Path.home() raises RuntimeError in the relative-path fallback, hint is still returned."""
        subdir = tmp_path / "project"
        subdir.mkdir()
        hint_file = subdir / "AGENTS.md"
        hint_file.write_text("# Test hint\n")

        original_home = Path.home

        def _no_home():
            raise RuntimeError("Could not determine home directory.")

        with patch.object(Path, "home", side_effect=_no_home):
            result = tracker._load_hints_for_directory(subdir)

        # Hint should still be loaded — the RuntimeError only affects the
        # display path fallback, not the actual content.
        assert result is not None
        assert "Test hint" in result
