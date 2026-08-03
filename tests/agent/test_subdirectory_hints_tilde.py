"""Regression tests for the home-directory RuntimeError bug.

Without the fix to ``agent/subdirectory_hints.py`` (add ``RuntimeError`` to
the three ``except`` clauses around ``Path.expanduser()`` /
``Path.home()``), the first two tests raise ``RuntimeError`` from inside
the hint walker on POSIX systems.

These tests use pytest's built-in ``tmp_path`` fixture and intentionally
do not depend on the richer ``project`` fixture from
``test_subdirectory_hints.py`` so the file is runnable standalone.
"""

from agent.subdirectory_hints import SubdirectoryHintTracker


class TestSubdirectoryHintTrackerTildeRobustness:
    """Regression: literal ``~`` in tool-call args must not crash the walker."""

    def test_tilde_approximately_in_command_does_not_crash(self, tmp_path):
        """LLMs use ``~`` for "approximately" (e.g. ``~500 agencies``).

        ``pathlib.Path('~500-700').expanduser()`` raises ``RuntimeError`` —
        the walker must catch this, not propagate it as a tool failure.
        """
        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        # Heredoc-style terminal command body containing "~500-700"
        # used as "approximately 500-700"
        cmd = (
            "cat > out.md <<EOF\n"
            "Segment size signal: ~500-700 agencies in DACH region.\n"
            "CVE volume: ~45,000 disclosed in 2025.\n"
            "Founder blended rate: ~80/hr.\n"
            "EOF"
        )
        # Must not raise — return value can be None / empty
        tracker.check_tool_call("terminal", {"command": cmd})

    def test_tilde_with_unknown_user_does_not_crash(self, tmp_path):
        """``~unknown_user`` similarly raises RuntimeError on POSIX systems
        whose /etc/passwd does not contain that user.  Walker must absorb it."""
        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        cmd = "echo path: ~nonexistent_user_xyzzy_12345/some/file"
        # Must not raise
        tracker.check_tool_call("terminal", {"command": cmd})

    def test_valid_tilde_user_still_works(self, tmp_path):
        """The fix must not regress the legitimate-tilde-user path.

        ``~`` alone resolves to ``Path.home()`` and should still be
        recognised as a candidate path (no exception either way).
        """
        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        tracker.check_tool_call("terminal", {"command": "ls ~/Documents"})
        # No exception, no assertion required
