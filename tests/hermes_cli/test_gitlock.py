"""Tests for hermes_cli/gitlock.py — stale lock detection and recovery."""

from pathlib import Path
import tempfile
from unittest.mock import patch


def test_clear_stale_locks_returns_list():
    from hermes_cli.gitlock import clear_stale_git_locks
    with tempfile.TemporaryDirectory() as d:
        result = clear_stale_git_locks(Path(d))
        assert isinstance(result, list)


def test_clear_stale_locks_discovers_shallow_lock():
    from hermes_cli.gitlock import clear_stale_git_locks
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        shallow = root / ".git" / "shallow.lock"
        shallow.parent.mkdir(parents=True)
        shallow.write_text("")
        with patch("hermes_cli.gitlock._git_proc_running", return_value=False):
            result = clear_stale_git_locks(root)
        assert len(result) > 0


def test_is_ancestor_of_head_nonexistent_repo():
    from hermes_cli.gitlock import is_ancestor_of_head
    with tempfile.TemporaryDirectory() as d:
        assert is_ancestor_of_head(Path(d), "HEAD") is False


def test_is_ancestor_of_head_empty_rev():
    from hermes_cli.gitlock import is_ancestor_of_head
    with tempfile.TemporaryDirectory() as d:
        assert is_ancestor_of_head(Path(d), "") is False
