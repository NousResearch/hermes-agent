"""Read-only terminal commands may join parallel tool batches (single, non-chained,
non-redirecting reads only). Everything else stays a sequential barrier — fail-closed.

Regression guard for the 9router speed optimization (2026-09-09): parallelizing the
common [git status, read_file, search_files] reconnaissance pattern.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from agent.tool_dispatch_helpers import (
    _is_read_only_terminal_command,
    _plan_tool_batch_segments,
)


def _tc(name, args):
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def _terminal(cmd, **extra):
    return _tc("terminal", {"command": cmd, **extra})


class TestReadOnlyTerminalClassification:
    def test_plain_reads_are_parallel_safe(self):
        for cmd in ("ls -la", "git status", "git log --oneline -5", "grep -rn foo .",
                    "cat /etc/hosts", "df -h", "ps aux", "which hermes", "date"):
            assert _is_read_only_terminal_command({"command": cmd}), cmd

    def test_chained_commands_are_barriers(self):
        for cmd in ("ls && ls", "cat x; cat y", "ps aux | grep python",
                    "echo `date`", "echo $(date)", "git status || true"):
            assert not _is_read_only_terminal_command({"command": cmd}), cmd

    def test_redirects_and_writers_are_barriers(self):
        for cmd in ("ls > out.txt", "rm -rf x", "mv a b", "cp a b", "git checkout main",
                    "sed -i s/a/b/ f", "echo hi >> log"):
            assert not _is_read_only_terminal_command({"command": cmd}), cmd

    def test_background_and_pty_are_barriers(self):
        assert not _is_read_only_terminal_command({"command": "ls", "background": True})
        assert not _is_read_only_terminal_command({"command": "ls", "pty": True})

    def test_unknown_and_empty_commands_are_barriers(self):
        assert not _is_read_only_terminal_command({"command": "brew install go"})
        assert not _is_read_only_terminal_command({"command": ""})
        assert not _is_read_only_terminal_command({})


class TestPlannerIntegration:
    def test_readonly_terminal_joins_parallel_run(self):
        batch = [
            _tc("read_file", {"path": "/etc/hosts"}),
            _terminal("git status"),
            _tc("search_files", {"pattern": "x", "path": "/etc"}),
        ]
        segments = _plan_tool_batch_segments(batch, execution_cwd=None)
        assert segments == [
            ("parallel", [
                batch[0],
                batch[1],
                batch[2],
            ])
        ], segments

    def test_writing_terminal_stays_sequential(self):
        batch = [
            _tc("read_file", {"path": "/etc/hosts"}),
            _terminal("git checkout main"),
        ]
        segments = _plan_tool_batch_segments(batch, execution_cwd=None)
        kinds = [kind for kind, _ in segments]
        assert "parallel" not in kinds
        # Call order preserved
        flat = [c for _, calls in segments for c in calls]
        assert flat == batch

    def test_mixed_batch_order_preserved(self):
        batch = [
            _terminal("ls -la"),
            _tc("read_file", {"path": "/etc/hosts"}),
            _terminal("brew install go"),      # barrier
            _terminal("date"),                  # read-only again
            _tc("read_file", {"path": "/etc/passwd"}),
        ]
        segments = _plan_tool_batch_segments(batch, execution_cwd=None)
        flat = [c for _, calls in segments for c in calls]
        assert flat == batch  # order never crossed
        # The first two form a parallel run; the barrier splits the rest.
        assert segments[0][0] == "parallel"
        assert [c.function.name for c in segments[0][1]] == ["terminal", "read_file"]
