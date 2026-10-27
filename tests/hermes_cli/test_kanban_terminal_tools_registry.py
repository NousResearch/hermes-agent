"""Pluggable kanban terminal-tools registry (:mod:`agent.kanban_stop`).

Covers the PR-3 contract:
- the default sanctioned set is unchanged (the built-in frozenset);
- a registered extra tool name is sanctioned by the stop guard;
- wrong-type / empty names are warned about and ignored;
- ``PluginContext.register_kanban_terminal_tool`` registers and unload revokes.
"""

from __future__ import annotations

import logging

import pytest

import agent.kanban_stop as kanban_stop
from agent.kanban_stop import (
    _TERMINAL_KANBAN_TOOLS,
    register_terminal_tool,
    registered_terminal_tools,
    session_called_kanban_terminal,
    unregister_terminal_tool,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Isolate the process-global extra-terminal-tool registry per test."""
    saved = set(kanban_stop._EXTRA_TERMINAL_KANBAN_TOOLS)
    yield
    kanban_stop._EXTRA_TERMINAL_KANBAN_TOOLS.clear()
    kanban_stop._EXTRA_TERMINAL_KANBAN_TOOLS.update(saved)


def _assistant_call(name):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": name, "arguments": "{}"}},
        ],
    }


# ── Default set ──────────────────────────────────────────────────────


def test_default_set_unchanged():
    assert _TERMINAL_KANBAN_TOOLS == frozenset({
        "kanban_complete", "kanban_block", "kanban_schedule",
        "kanban_request_review", "kanban_request_changes",
    })
    assert registered_terminal_tools() == frozenset()
    assert session_called_kanban_terminal([_assistant_call("kanban_comment")]) is False


# ── Registry primitives ──────────────────────────────────────────────


def test_registered_extra_name_is_sanctioned():
    assert register_terminal_tool("board_handoff") is True
    assert registered_terminal_tools() == {"board_handoff"}
    assert session_called_kanban_terminal([_assistant_call("board_handoff")]) is True
    # The tool-role message path too.
    assert session_called_kanban_terminal(
        [{"role": "tool", "name": "board_handoff", "content": "ok"}]) is True
    # Built-ins stay sanctioned and non-terminal names stay non-terminal.
    assert session_called_kanban_terminal([_assistant_call("kanban_complete")]) is True
    assert session_called_kanban_terminal([_assistant_call("kanban_heartbeat")]) is False


def test_invalid_names_warned_and_ignored(caplog):
    with caplog.at_level(logging.WARNING):
        assert register_terminal_tool(42) is False
        assert register_terminal_tool("") is False
        assert register_terminal_tool("   ") is False
    assert registered_terminal_tools() == frozenset()
    assert len(caplog.records) == 3
    assert session_called_kanban_terminal([_assistant_call("kanban_comment")]) is False


def test_duplicate_registration_is_idempotent_and_unregister_revokes():
    register_terminal_tool("board_handoff")
    assert register_terminal_tool("board_handoff") is True
    assert registered_terminal_tools() == {"board_handoff"}
    assert unregister_terminal_tool("board_handoff") is True
    assert unregister_terminal_tool("board_handoff") is False
    assert registered_terminal_tools() == frozenset()
    assert session_called_kanban_terminal([_assistant_call("board_handoff")]) is False


# ── PluginContext seam ───────────────────────────────────────────────


def _plugin_ctx(tmp_path, monkeypatch):
    """A real PluginContext for a single enabled plugin (kanbanplug)."""
    import hermes_yaml as yaml

    from hermes_cli.plugins import PluginManager

    home = tmp_path / ".hermes"
    plugin_dir = home / "plugins" / "kanbanplug"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": "kanbanplug", "version": "1.0"}), encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n"
        "    ctx.register_kanban_terminal_tool('board_handoff')\n",
        encoding="utf-8")
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["kanbanplug"]}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    manager.discover_and_load()
    return manager


def test_plugin_context_registration_and_unload_removal(tmp_path, monkeypatch):
    manager = _plugin_ctx(tmp_path, monkeypatch)
    try:
        assert any(k == "kanbanplug" for k in manager._plugins), dict(manager._plugins)
        assert registered_terminal_tools() == {"board_handoff"}
        assert session_called_kanban_terminal([_assistant_call("board_handoff")]) is True
    finally:
        manager.unload()
    assert registered_terminal_tools() == frozenset()
    assert session_called_kanban_terminal([_assistant_call("board_handoff")]) is False


def test_plugin_context_ignores_wrong_type_and_empty(tmp_path, monkeypatch):
    import hermes_yaml as yaml

    from hermes_cli.plugins import PluginManager

    home = tmp_path / ".hermes"
    plugin_dir = home / "plugins" / "badkanbanplug"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": "badkanbanplug", "version": "1.0"}), encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n"
        "    ctx.register_kanban_terminal_tool(123)\n"
        "    ctx.register_kanban_terminal_tool('   ')\n",
        encoding="utf-8")
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["badkanbanplug"]}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    manager = PluginManager()
    try:
        manager.discover_and_load()
        assert registered_terminal_tools() == frozenset()
    finally:
        manager.unload()
    assert registered_terminal_tools() == frozenset()
