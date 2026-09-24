"""Operator-facing regression for coalesced dispatch guards (#121651)."""

import argparse
import json

from hermes_cli import kanban as cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_ops


def test_coalesced_guard_is_visible_on_operator_surfaces(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda _: True)
    kbc.init_db()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="held", assignee="default", workspace_kind="scratch")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET last_failure_error='401 auth failed' WHERE id=?", (tid,))

        def guarded_tick():
            kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: None, max_spawn=1, reconcile_orphans=False)

        guarded_tick()
        guarded_tick()
        assert len([e for e in kb.list_events(conn, tid) if e.kind == "respawn_guarded"]) == 1

        args = argparse.Namespace(task_id=tid, json=True)
        assert cli._cmd_show(args) == 0
        shown = json.loads(capsys.readouterr().out)["task"]
        assert shown["guard_reason"] == "blocker_auth"
        assert shown["guard_count"] == 2
        assert shown["guard_last_seen_at"]
        args.json = False
        assert cli._cmd_show(args) == 0
        assert "guard:     blocker_auth (count=2" in capsys.readouterr().out
        assert cli._cmd_diagnostics(argparse.Namespace(task=tid, severity=None, json=True)) == 0
        diagnostics = json.loads(capsys.readouterr().out)[0]["diagnostics"]
        assert any(d["kind"] == "respawn_guard" and d["count"] == 2 for d in diagnostics)

        def poll(interval, tick):
            tick()
            guarded_tick()
            tick()
            with kb.write_txn(conn):
                conn.execute("UPDATE tasks SET guard_reason=NULL, guard_count=0, "
                             "guard_last_seen_at=NULL WHERE id=?", (tid,))
            tick()
            return 0

        monkeypatch.setattr(kanban_ops, "_poll_loop", poll)
        assert kanban_ops._cmd_tail(argparse.Namespace(task_id=tid, interval=0.1)) == 0
        tail = capsys.readouterr().out
        assert "current guard: blocker_auth (count=2)" in tail
        assert "current guard: blocker_auth (count=3)" in tail
        assert "Current guard cleared" in tail
