"""Regression tests for built-in-memory CRUD CLI commands."""

import argparse
from types import SimpleNamespace

import pytest


@pytest.fixture
def memory_home(tmp_path, monkeypatch):
    """Create a profile-scoped built-in memory store for CLI mutations."""
    hermes_home = tmp_path / ".hermes"
    memories = hermes_home / "memories"
    memories.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (memories / "MEMORY.md").write_text("First fact", encoding="utf-8")
    (memories / "USER.md").write_text("User likes tea", encoding="utf-8")
    return memories


def _run_memory_command(action, *, target="memory", content=None, old_text=None):
    from hermes_cli.main import cmd_memory

    cmd_memory(SimpleNamespace(
        memory_command=action,
        target=target,
        content=content,
        old_text=old_text,
    ))


def test_memory_parser_accepts_crud_commands():
    """CLI syntax exposes the same targeted fields as the memory tool."""
    from hermes_cli.subcommands.memory import build_memory_parser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_memory_parser(subparsers, cmd_memory=lambda args: None)

    add = parser.parse_args(["memory", "add", "New fact", "--target", "user"])
    replace = parser.parse_args(["memory", "replace", "old", "new"])
    remove = parser.parse_args(["memory", "remove", "stale", "--target", "user"])

    assert (add.memory_command, add.content, add.target) == ("add", "New fact", "user")
    assert (replace.memory_command, replace.old_text, replace.content) == ("replace", "old", "new")
    assert (remove.memory_command, remove.old_text, remove.target) == ("remove", "stale", "user")


def test_memory_add_writes_to_selected_builtin_store(memory_home, monkeypatch, capsys):
    """`memory add` persists a new profile-scoped entry instead of only printing it."""
    monkeypatch.setattr("tools.memory_tool._apply_write_gate", lambda *args: None)

    _run_memory_command("add", target="user", content="User prefers concise replies")

    assert (memory_home / "USER.md").read_text(encoding="utf-8") == (
        "User likes tea\n§\nUser prefers concise replies"
    )
    assert "Memory add complete" in capsys.readouterr().out


def test_memory_replace_and_remove_use_unique_entry_text(memory_home, monkeypatch, capsys):
    """`replace` and `remove` delegate matching and persistence to MemoryStore."""
    monkeypatch.setattr("tools.memory_tool._apply_write_gate", lambda *args: None)

    _run_memory_command("replace", old_text="First", content="Current fact")
    assert (memory_home / "MEMORY.md").read_text(encoding="utf-8") == "Current fact"

    _run_memory_command("remove", old_text="Current")
    # MemoryStore keeps an empty file after the last entry is removed.
    assert (memory_home / "MEMORY.md").read_text(encoding="utf-8") == ""
    output = capsys.readouterr().out
    assert "Memory replace complete" in output
    assert "Memory remove complete" in output


def test_memory_add_reports_staged_write_gate(memory_home, monkeypatch, capsys):
    """CLI surfaces write-gate staging instead of claiming a completed write."""

    def _stage(*_args, **_kwargs):
        return {
            "success": True,
            "staged": True,
            "message": "write staged pending approval",
        }

    # Bypass the real gate by short-circuiting memory_tool result envelope.
    import json

    def fake_memory_tool(**kwargs):
        assert kwargs["action"] == "add"
        return json.dumps(_stage())

    monkeypatch.setattr("tools.memory_tool.memory_tool", fake_memory_tool)
    monkeypatch.setattr("tools.memory_tool.load_on_disk_store", lambda: object())

    # cmd_memory imports symbols inside the branch — patch module attributes used after import.
    import tools.memory_tool as mt

    monkeypatch.setattr(mt, "memory_tool", fake_memory_tool)
    monkeypatch.setattr(mt, "load_on_disk_store", lambda: object())

    _run_memory_command("add", content="Staged fact")
    out = capsys.readouterr().out
    assert "Memory add staged" in out
    assert "write staged pending approval" in out
    # Disk untouched because the tool was faked before persistence.
    assert (memory_home / "MEMORY.md").read_text(encoding="utf-8") == "First fact"
