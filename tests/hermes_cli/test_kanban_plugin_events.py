"""``ctx.kanban_events``: a plugin appends and reads Kanban task events
through the real discovery path, without importing ``kanban_db`` internals."""

from __future__ import annotations

import pytest
import yaml

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.plugins import PluginManager

PLUGIN = '''
CTX = []
def register(ctx):
    CTX.append(ctx)
'''


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    plugin = home / "plugins" / "board-writer"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: board-writer\nversion: '0.1.0'\n")
    (plugin / "__init__.py").write_text(PLUGIN)
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": ["board-writer"]}}))
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_HOME", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(var, raising=False)
    kb._INITIALIZED_PATHS.clear()
    manager = PluginManager()
    manager.discover_and_load()
    loaded = manager._plugins["board-writer"]
    assert loaded.enabled and loaded.error is None
    return loaded.module.CTX[0]


def test_append_and_cursor_read(ctx):
    kb.create_board("xg")
    with kbc.connect_closing(board="xg") as conn:
        tid = kb.create_task(conn, title="t", assignee="dev")
    events = ctx.kanban_events

    first = events.append(tid, "verdict_v1", {"ok": True}, board="xg")
    second = events.append(tid, "failure_v1", None, board="xg")
    assert second > first

    mine = events.read(task_id=tid, board="xg",
                       kinds=[events.kind("verdict_v1"), events.kind("failure_v1")])
    assert [(e["id"], e["kind"], e["payload"]) for e in mine] == [
        (first, "board-writer:verdict_v1", {"ok": True}),
        (second, "board-writer:failure_v1", None),
    ]
    # Cursor: only what came after.
    assert [e["id"] for e in events.read(task_id=tid, board="xg", since_id=first)] == [second]
    # Core events are readable by their plain kind ("created" is written by create_task).
    assert events.read(task_id=tid, board="xg", kinds=["created"])
    # Board-scoped: nothing leaks into the default board.
    assert events.read(task_id=tid) == []


def test_append_rejects_bad_input(ctx):
    kb.create_board("xg")
    with kbc.connect_closing(board="xg") as conn:
        tid = kb.create_task(conn, title="t", assignee="dev")
    events = ctx.kanban_events
    with pytest.raises(KeyError):
        events.append("t_missing", "x", board="xg")
    for bad in ("completed", "Upper", "a:b", "", "x" * 65):
        if bad == "completed":
            # A plugin cannot emit a core lifecycle kind: it is always namespaced.
            assert events.kind(bad) == "board-writer:completed"
            continue
        with pytest.raises(ValueError):
            events.append(tid, bad, board="xg")
    with pytest.raises(ValueError):
        events.append(tid, "big", {"x": "y" * 70_000}, board="xg")
    with pytest.raises(ValueError):
        events.append(tid, "list", ["not", "a", "dict"], board="xg")
    assert events.read(task_id=tid, board="xg", kinds=[events.kind("big")]) == []
