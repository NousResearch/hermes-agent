"""Tests for the ``hermes memory`` CLI subcommand."""

import json
import sys

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_memory_cli(monkeypatch, argv, memory_entries=None, user_entries=None):
    """Run ``hermes memory ...`` with a fake MemoryStore and return stdout."""
    import hermes_cli.main as main_mod
    import tools.memory_tool as mem_mod

    memory_entries = list(memory_entries or [])
    user_entries = list(user_entries or [])

    class FakeStore:
        def __init__(self, **_kw):
            self.memory_entries = list(memory_entries)
            self.user_entries = list(user_entries)
            self.memory_char_limit = 2200
            self.user_char_limit = 1375
            self._removed = []

        def load_from_disk(self):
            pass

        def _entries_for(self, target):
            return self.user_entries if target == "user" else self.memory_entries

        def _char_count(self, target):
            entries = self._entries_for(target)
            if not entries:
                return 0
            return len("\n§\n".join(entries))

        def _char_limit(self, target):
            return self.user_char_limit if target == "user" else self.memory_char_limit

        def remove(self, target, old_text):
            entries = self._entries_for(target)
            matches = [(i, e) for i, e in enumerate(entries) if old_text in e]
            if not matches:
                return json.dumps({"success": False, "error": f"No entry matched '{old_text}'."})
            if len(matches) > 1:
                unique = set(e for _, e in matches)
                if len(unique) > 1:
                    return json.dumps({"success": False, "error": f"Multiple entries matched."})
            idx = matches[0][0]
            entries.pop(idx)
            self._removed.append(old_text)
            return json.dumps({"success": True, "usage": "10% — 100/2,200 chars"})

    monkeypatch.setattr(mem_mod, "MemoryStore", FakeStore)
    monkeypatch.setattr(sys, "argv", ["hermes"] + argv)
    main_mod.main()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_memory_list_empty(monkeypatch, capsys):
    """Both stores empty shows (empty) for each."""
    _run_memory_cli(monkeypatch, ["memory", "list"])
    out = capsys.readouterr().out
    assert out.count("(empty)") == 2
    assert "MEMORY (agent notes)" in out
    assert "USER (user profile)" in out


def test_memory_list_populated(monkeypatch, capsys):
    """Populated stores show numbered entries with usage stats."""
    _run_memory_cli(
        monkeypatch,
        ["memory", "list"],
        memory_entries=["Docker needs --platform=linux/amd64", "Project uses pytest"],
        user_entries=["Prefers dark themes"],
    )
    out = capsys.readouterr().out
    assert "Docker needs" in out
    assert "Project uses pytest" in out
    assert "Prefers dark themes" in out
    assert "2 entries" in out
    assert "1 entries" in out


def test_memory_list_is_default_action(monkeypatch, capsys):
    """Running ``hermes memory`` with no subcommand defaults to list."""
    _run_memory_cli(monkeypatch, ["memory"], memory_entries=["test entry"])
    out = capsys.readouterr().out
    assert "test entry" in out
    assert "MEMORY (agent notes)" in out


def test_memory_show_memory(monkeypatch, capsys):
    """``hermes memory show memory`` displays full entries."""
    _run_memory_cli(
        monkeypatch,
        ["memory", "show", "memory"],
        memory_entries=["Line one\nLine two"],
    )
    out = capsys.readouterr().out
    assert "Line one" in out
    assert "Line two" in out
    assert "[1]" in out


def test_memory_show_user(monkeypatch, capsys):
    """``hermes memory show user`` displays user profile entries."""
    _run_memory_cli(
        monkeypatch,
        ["memory", "show", "user"],
        user_entries=["Senior engineer, 8 years Python"],
    )
    out = capsys.readouterr().out
    assert "Senior engineer" in out
    assert "USER (user profile)" in out


def test_memory_delete_success(monkeypatch, capsys):
    """``hermes memory delete memory "Docker"`` removes matching entry."""
    monkeypatch.setattr("builtins.input", lambda _: "y")
    _run_memory_cli(
        monkeypatch,
        ["memory", "delete", "memory", "Docker"],
        memory_entries=["Docker needs --platform=linux/amd64", "Project uses pytest"],
    )
    out = capsys.readouterr().out
    assert "Deleted" in out


def test_memory_delete_with_yes_flag(monkeypatch, capsys):
    """``--yes`` skips the confirmation prompt."""
    _run_memory_cli(
        monkeypatch,
        ["memory", "delete", "memory", "Docker", "--yes"],
        memory_entries=["Docker needs --platform=linux/amd64"],
    )
    out = capsys.readouterr().out
    assert "Deleted" in out


def test_memory_delete_no_match(monkeypatch, capsys):
    """Non-matching substring exits with error."""
    with pytest.raises(SystemExit) as exc_info:
        _run_memory_cli(
            monkeypatch,
            ["memory", "delete", "memory", "nonexistent", "--yes"],
            memory_entries=["Docker needs --platform=linux/amd64"],
        )
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "No entry" in out


def test_memory_delete_cancelled(monkeypatch, capsys):
    """User declining confirmation cancels the delete."""
    monkeypatch.setattr("builtins.input", lambda _: "n")
    _run_memory_cli(
        monkeypatch,
        ["memory", "delete", "memory", "Docker"],
        memory_entries=["Docker needs --platform=linux/amd64"],
    )
    out = capsys.readouterr().out
    assert "Cancelled" in out
