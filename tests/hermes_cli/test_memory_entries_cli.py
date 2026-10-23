"""Tests for hermes_cli/memory_entries_cli.py — ``hermes memory show`` / ``forget``.

Both commands run with no agent, against a temp HERMES_HOME holding real
MEMORY.md / USER.md files. They must read the same on-disk entries the agent
sees, remove exactly one entry by an unambiguous substring, and never touch the
file on an ambiguous or failed lookup.
"""
from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import pytest

from hermes_cli import memory_entries_cli


def _write_memory(home: Path, entries: list[str], filename: str = "MEMORY.md") -> None:
    mem = home / "memories"
    mem.mkdir(parents=True, exist_ok=True)
    # The store's real entry delimiter, so the file parses exactly as a written one would.
    from tools.memory_tool_store import ENTRY_DELIMITER
    (mem / filename).write_text(ENTRY_DELIMITER.join(entries), encoding="utf-8")


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An isolated Hermes home; the store resolves per call, so one env var is enough."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


class TestShow:
    def test_show_lists_entries_and_usage(self, home, capsys):
        _write_memory(home, ["alpha fact one", "beta fact two"])
        _write_memory(home, ["user profile line"], "USER.md")

        memory_entries_cli.cmd_memory_show(Namespace(target=None))

        out = capsys.readouterr().out
        assert "MEMORY.md" in out and "2 entries" in out
        assert "alpha fact one" in out and "beta fact two" in out
        assert "USER.md" in out and "user profile line" in out

    def test_show_reports_empty_store(self, home, capsys):
        _write_memory(home, [], "MEMORY.md")

        memory_entries_cli.cmd_memory_show(Namespace(target=None))

        assert "(empty)" in capsys.readouterr().out

    def test_show_target_restricts_to_one_store(self, home, capsys):
        _write_memory(home, ["alpha fact"])
        _write_memory(home, ["secret user line"], "USER.md")

        memory_entries_cli.cmd_memory_show(Namespace(target="user"))

        out = capsys.readouterr().out
        assert "secret user line" in out
        assert "alpha fact" not in out


class TestForget:
    def test_forget_removes_only_the_matching_entry(self, home, capsys):
        _write_memory(home, ["keep this fact", "drop that fact"])

        memory_entries_cli.cmd_memory_forget(Namespace(entry="drop that", target="memory"))

        out = capsys.readouterr().out
        assert "✓ Removed" in out
        assert "1 entry left" in out
        # The store re-reads from disk, so the file itself must have lost one entry.
        from tools.memory_tool import load_on_disk_store
        store = load_on_disk_store()
        assert store._entries_for("memory") == ["keep this fact"]

    def test_forget_requires_text(self, home, capsys):
        memory_entries_cli.cmd_memory_forget(Namespace(entry="   ", target="memory"))

        err = capsys.readouterr().err
        assert "Provide the text" in err

    def test_forget_ambiguous_match_refuses_without_writing(self, home, capsys):
        _write_memory(home, ["shared prefix one", "shared prefix two"])

        memory_entries_cli.cmd_memory_forget(Namespace(entry="shared prefix", target="memory"))

        out = capsys.readouterr().out + capsys.readouterr().err
        assert "Matching entries" in out
        from tools.memory_tool import load_on_disk_store
        store = load_on_disk_store()
        assert len(store._entries_for("memory")) == 2  # nothing removed

    def test_forget_no_match_refuses_without_writing(self, home, capsys):
        _write_memory(home, ["only fact"])

        memory_entries_cli.cmd_memory_forget(Namespace(entry="nothing like this", target="memory"))

        assert "✗" in capsys.readouterr().err
        from tools.memory_tool import load_on_disk_store
        assert load_on_disk_store()._entries_for("memory") == ["only fact"]

    def test_forget_empty_store(self, home, capsys):
        _write_memory(home, [])

        memory_entries_cli.cmd_memory_forget(Namespace(entry="anything", target="memory"))

        assert "empty" in capsys.readouterr().err


class TestDispatcher:
    def test_dispatch_routes_show_and_forget(self):
        from hermes_cli.main_agent_cmds import cmd_memory

        # show is pure output; assert it routes (not the interactive setup path).
        cmd_memory(Namespace(memory_command="show", target=None))
        cmd_memory(Namespace(memory_command="forget", entry="", target="memory"))
