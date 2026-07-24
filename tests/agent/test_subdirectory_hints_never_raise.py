"""Regression tests: ``check_tool_call`` must never raise.

Subdirectory hint discovery is a best-effort enrichment. Both call sites in
``agent/tool_executor.py`` (sequential and parallel tool paths) invoke it
*after* the tool result has already been computed, and neither wraps the
call — so any exception escaping ``check_tool_call`` aborts the entire
agent turn and discards a perfectly good tool result.

That is exactly how the ``Path.expanduser()`` RuntimeError class of crashes
(fixed in ``test_subdirectory_hints_tilde.py``'s target change) took down
live agent turns: the specific ``except`` tuples inside the walker missed
one exception type. Rather than keep chasing individual exception types at
each internal site, this change guards the public entrypoint: the internal
logic moves to ``_check_tool_call`` and ``check_tool_call`` logs and
swallows anything that escapes, returning ``None``.

These tests intentionally use ``tmp_path`` only, so the file is runnable
standalone (same convention as ``test_subdirectory_hints_tilde.py``).
"""

import pytest

from agent.subdirectory_hints import SubdirectoryHintTracker


class TestCheckToolCallNeverRaises:
    """The public entrypoint absorbs arbitrary internal failures."""

    @pytest.mark.parametrize(
        "exc",
        [RuntimeError("Could not determine home directory."),
         KeyError("user"),
         TypeError("unexpected argument shape"),
         Exception("anything else")],
        ids=["runtime-error", "key-error", "type-error", "bare-exception"],
    )
    def test_internal_failure_returns_none(self, tmp_path, monkeypatch, exc):
        """Any exception from the internal walker is logged and swallowed."""
        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))

        def boom(*args, **kwargs):
            raise exc

        monkeypatch.setattr(tracker, "_extract_directories", boom)
        result = tracker.check_tool_call("terminal", {"command": "ls ."})
        assert result is None

    def test_happy_path_still_returns_hints(self, tmp_path):
        """The guard must not regress normal hint discovery."""
        sub = tmp_path / "backend"
        sub.mkdir()
        (sub / "AGENTS.md").write_text("Backend instructions")
        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))

        result = tracker.check_tool_call(
            "read_file", {"path": str(sub / "app.py")}
        )
        assert result is not None
        assert "Backend instructions" in result
