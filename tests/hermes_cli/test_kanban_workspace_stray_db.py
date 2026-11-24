"""Workspace cleanup must not be paralysed by a board DB that is not a live board,
and must actually delete what it decides to delete.

Regression for a fleet incident: a 0-byte ``kanban/boards/<slug>/kanban.db`` —
created by a bare ``sqlite3.connect()`` on a guessed path — made every board's
sibling scan raise ``no such table: tasks``, so every completion cleanup since
was deferred with ``reason=unknown`` and nothing was ever retried. A second,
silent failure: ``shutil.rmtree(..., ignore_errors=True)`` gives up on the
first read-only file (Windows), leaving half-deleted trees with no event.
A third: the dispatcher spawns every worker with ``cwd=workspace`` and the
worker runs completion cleanup itself; Windows refuses ``rmdir`` on a process's
own cwd, so each finished scratch card left an empty shell behind, silently.
"""
import argparse
import json
import os
import shutil
import sqlite3
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw
from hermes_cli import kanban_ops


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    return tmp_path


def _ws_args(*, workspaces=True, apply=False, as_json=False):
    return argparse.Namespace(event_retention_days=30, log_retention_days=30,
                              workspaces=workspaces, apply=apply, json=as_json)


def _terminal_scratch(conn, status="done", path=None) -> tuple[str, Path]:
    tid = kb.create_task(conn, title=f"{status} scratch")
    ws = path or (kb.workspaces_root() / tid)
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "scratch.txt").write_text("work", encoding="utf-8")
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status=?, workspace_kind='scratch', workspace_path=? WHERE id=?",
            (status, str(ws), tid),
        )
    return tid, ws


def _deferred_reasons(conn, task_id: str) -> list:
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind=? ORDER BY id",
        (task_id, "workspace_cleanup_deferred_shared"),
    ).fetchall()
    return [json.loads(r["payload"])["reason"] for r in rows]


def _stray_board_dir(name="stray", *, archived=False, with_meta=True) -> Path:
    d = kb.boards_root() / name
    d.mkdir(parents=True, exist_ok=True)
    if with_meta:
        (d / "board.json").write_text(json.dumps({"slug": name, "archived": archived}), encoding="utf-8")
    return d


def _done_task_with_old_event(conn):
    tid = kb.create_task(conn, title="finished long ago")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (tid,))
        conn.execute("UPDATE task_events SET created_at=0 WHERE task_id=?", (tid,))
    return tid


def _event_rows(conn, tid):
    return conn.execute("SELECT count(*) FROM task_events WHERE task_id=?", (tid,)).fetchone()[0]


def _old_log_file() -> Path:
    log_dir = kb.worker_logs_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    p = log_dir / "worker-1.log"
    p.write_text("log line")
    os.utime(p, (0, 0))
    return p


def _write_foo_table(db: Path) -> None:
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE foo(x)")
    conn.commit()
    conn.close()


def _write_tasks_table(db: Path) -> None:
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE tasks(id TEXT PRIMARY KEY, status TEXT, workspace_path TEXT, workspace_kind TEXT)"
    )
    conn.commit()
    conn.close()


# kind -> how to plant a boards/<slug>/kanban.db that can never hold a live task
_STRAY_BUILDERS = {
    "zero-byte": lambda db: db.write_bytes(b""),
    "garbage": lambda db: db.write_bytes(b"this is not a database\n"),
    "no-tasks-table": _write_foo_table,  # a real SQLite file, never initialized as a board
    "archived": _write_tasks_table,  # a real, initialized board whose tombstone says archived
}


def _plant_stray(kind: str) -> Path:
    d = _stray_board_dir(f"stray-{kind}", archived=(kind == "archived"))
    db = d / "kanban.db"
    _STRAY_BUILDERS[kind](db)
    return db


@pytest.mark.parametrize(
    "kind",
    ["zero-byte", "garbage", "no-tasks-table", "archived"],
)
def test_stray_sibling_board_db_does_not_defer_cleanup(board, kind):
    """The incident: ``boards/<slug>/kanban.db`` is not a live board, so the
    shared-workspace scan skips it and the done task's scratch dir goes."""
    _plant_stray(kind)
    with kbc.connect_closing() as conn:
        tid, ws = _terminal_scratch(conn)
        kbw._cleanup_workspace(conn, tid)
        assert not ws.exists()
        assert _deferred_reasons(conn, tid) == []


def test_sibling_db_state_classes_without_opening(board):
    assert kbw._sibling_db_state(_plant_stray("zero-byte")) == "empty"
    assert kbw._sibling_db_state(_plant_stray("garbage")) == "not-sqlite"
    assert kbw._sibling_db_state(_plant_stray("archived")) == "archived"
    live = _stray_board_dir("live")
    _write_tasks_table(live / "kanban.db")
    assert kbw._sibling_db_state(live / "kanban.db") == "ready"
    # The header is read directly: no -wal/-shm sidecar is created by the probe.
    assert sorted(p.name for p in live.iterdir()) == ["board.json", "kanban.db"]


def test_initialized_sibling_board_that_cannot_be_read_defers_cleanup(board, monkeypatch):
    """Only a board DB with a schema that still cannot be read (locked, corrupt)
    keeps the fail-closed ``unknown`` deferral."""
    live = _stray_board_dir("live")
    _write_tasks_table(live / "kanban.db")

    def locked(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(kbw, "_other_board_uses_path", locked)
    with kbc.connect_closing() as conn:
        tid, ws = _terminal_scratch(conn)
        kbw._cleanup_workspace(conn, tid)
        assert ws.exists()
        assert _deferred_reasons(conn, tid) == ["unknown"]


def test_live_sibling_board_still_shares(board):
    """Sanity: a real sibling board's live task still holds the directory."""
    kb.create_board("other")
    shared = kb.workspaces_root() / "shared-scratch"
    shared.mkdir(parents=True)
    (shared / "note.txt").write_text("still needed", encoding="utf-8")
    with kbc.connect_closing() as conn:
        archived, _ = _terminal_scratch(conn, status="archived", path=shared)
    with kbc.connect_closing(board="other") as other:
        live = kb.create_task(other, title="live on other")
        with kb.write_txn(other):
            other.execute(
                "UPDATE tasks SET status='ready', workspace_kind='scratch', workspace_path=? WHERE id=?",
                (str(shared), live),
            )
    with kbc.connect_closing() as conn:
        kbw._cleanup_workspace(conn, archived)
        assert (shared / "note.txt").exists()
        assert _deferred_reasons(conn, archived) == ["shared"]


# --- A2: read-only entries must not silently abort a removal -----------------

def _readonly_tree(ws: Path) -> Path:
    nested = ws / "plugins" / "fixture"
    nested.mkdir(parents=True)
    (ws / "plugins" / "README.md").write_text("x", encoding="utf-8")
    f = nested / "gate.py"
    f.write_text("print('fixture')", encoding="utf-8")
    os.chmod(f, stat.S_IREAD)
    if os.name == "nt":
        os.chmod(nested, stat.S_IREAD)  # read-only directory: rmdir refuses too
    return f


def test_force_rmtree_clears_readonly_entries(board):
    ws = kb.workspaces_root() / "ro-tree"
    f = _readonly_tree(ws)
    if os.name == "nt":
        # Reproduce the silent failure of the old call first.
        shutil.rmtree(ws, ignore_errors=True)
        assert f.exists() and ws.exists()
    assert kbw._force_rmtree(ws) is None
    assert not ws.exists()


def test_cleanup_workspace_removes_scratch_with_readonly_files(board):
    with kbc.connect_closing() as conn:
        tid, ws = _terminal_scratch(conn)
    f = _readonly_tree(ws)
    with kbc.connect_closing() as conn:
        kbw._cleanup_workspace(conn, tid)
    assert not ws.exists(), f"read-only {f} survived completion cleanup"
    with kbc.connect_closing() as conn:
        assert _deferred_reasons(conn, tid) == []


def test_try_cleanup_parent_workspaces_removes_readonly_parent_scratch(board):
    parent_ws = kb.workspaces_root() / "parent-ws"
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child")  # todo: active
        kb.link_tasks(conn, parent, child)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='done', workspace_kind='scratch', workspace_path=? WHERE id=?",
                (str(parent_ws), parent),
            )
    _readonly_tree(parent_ws)
    with kbc.connect_closing() as conn:
        kbw._try_cleanup_parent_workspaces(conn, child)  # child still active: keep
        assert parent_ws.exists()
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='done' WHERE id=?", (child,))
        kbw._try_cleanup_parent_workspaces(conn, child)
    assert not parent_ws.exists()


def test_cmd_gc_retention_sweep_also_clears_readonly_scratch(board):
    with kbc.connect_closing() as conn:
        _, ws = _terminal_scratch(conn, status="archived")
    _readonly_tree(ws)
    assert kanban_ops._cmd_gc(argparse.Namespace(event_retention_days=0, log_retention_days=0)) == 0
    assert not ws.exists()


# --- A3: ``kanban gc --workspaces`` is the retry path -------------------------

def _gc_fixture():
    """Six terminal tasks + one live one, every keep reason represented once."""
    with kbc.connect_closing() as conn:
        done, done_ws = _terminal_scratch(conn, "done")
        archived, archived_ws = _terminal_scratch(conn, "archived")
        _, gone_ws = _terminal_scratch(conn, "done")
        shutil.rmtree(gone_ws)  # already cleaned at completion
        _, unmanaged_ws = _terminal_scratch(conn, "done", path=Path.home() / "outside" / "ws")
        shared_done, shared = _terminal_scratch(conn, "done")
        live, _ = _terminal_scratch(conn, "ready", path=shared)
        plain = Path.home() / "plain-wt"
        plain.mkdir()
        wt, wt_old = _terminal_scratch(conn, "done")
        shutil.rmtree(wt_old)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET workspace_kind='worktree', workspace_path=? WHERE id=?",
                (str(plain), wt),
            )
        # A plain create_task row: terminal, workspace_kind='scratch', no path.
        old_event_task = _done_task_with_old_event(conn)
    return {
        "removable": [done_ws, archived_ws],
        "gone": gone_ws,
        "unmanaged": unmanaged_ws,
        "shared": shared,
        "shared_done": shared_done,
        "live": live,
        "plain": plain,
        "ids": (done, archived),
        "old_event_task": old_event_task,
    }


def test_gc_workspaces_dry_run_lists_and_touches_nothing(board, capsys):
    fx = _gc_fixture()
    stub = _plant_stray("zero-byte")
    log = _old_log_file()
    assert kanban_ops._cmd_gc(_ws_args(as_json=True)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "dry-run"
    assert report["counts"]["removable"] == 2
    assert {e["path"] for e in report["removable"]} == {str(p) for p in fx["removable"]}
    assert {e["status"] for e in report["removable"]} == {"done", "archived"}
    assert report["bytes"] > 0
    reasons = {(e["id"], e["reason"]) for e in report["kept"]}
    assert {r for _id, r in reasons} == {
        "missing", "unmanaged", "shared", "not_linked_worktree", "not_removable",
    }
    assert (fx["shared_done"], "shared") in reasons
    assert (fx["old_event_task"], "not_removable") in reasons
    assert report["counts"]["kept_by_reason"]["shared"] == 1
    assert [s["path"] for s in report["stray_board_db_stubs"]] == [str(stub.resolve())]
    assert "removed" not in report  # dry run: no apply counters
    for p in fx["removable"] + [fx["shared"], fx["unmanaged"], fx["plain"]]:
        assert p.exists(), p
    with kbc.connect_closing() as conn:
        assert all(_deferred_reasons(conn, tid) == [] for tid in fx["ids"])
        assert _deferred_reasons(conn, fx["shared_done"]) == []
        assert _event_rows(conn, fx["old_event_task"]) > 0
    assert log.exists()


def test_gc_workspaces_text_report_groups_by_status_and_reason(board, capsys):
    _gc_fixture()
    assert kanban_ops._cmd_gc(_ws_args()) == 0
    out = capsys.readouterr().out
    lines = [line.split() for line in out.splitlines()]
    assert "Workspace GC — dry run" in out
    assert "removable: 2 task workspace(s)" in out
    assert ["done", "scratch", "1"] in lines
    assert ["archived", "scratch", "1"] in lines
    assert "kept: 5" in out
    assert "missing" in out and "directory already gone" in out
    assert "shared" in out and "still used by another non-terminal task" in out
    assert "stray board db stubs: 0" in out


def test_gc_workspaces_apply_removes_only_what_the_dry_run_listed(board, capsys):
    fx = _gc_fixture()
    log = _old_log_file()
    assert kanban_ops._cmd_gc(_ws_args(apply=True, as_json=True)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "apply"
    assert report["counts"]["removable"] == 2
    assert report["removed"] == 2 and report["failed"] == 0
    for p in fx["removable"]:
        assert not p.exists(), p
    for p in (fx["shared"], fx["unmanaged"], fx["plain"]):
        assert p.exists(), p
    # A workspace sweep is not a retention sweep: old events and logs survive.
    with kbc.connect_closing() as conn:
        assert _event_rows(conn, fx["old_event_task"]) > 0
    assert log.exists()


def test_gc_workspaces_apply_leaves_shared_alone_and_records_it(board):
    fx = _gc_fixture()
    assert kanban_ops._cmd_gc(_ws_args(apply=True)) == 0
    assert (fx["shared"] / "scratch.txt").exists()
    assert not fx["removable"][0].exists()
    with kbc.connect_closing() as conn:
        # A sweep is not a completion: kept workspaces are only reported, never
        # given a deferral event (a scheduled gc would pile one up per run).
        assert _deferred_reasons(conn, fx["live"]) == []
        assert _deferred_reasons(conn, fx["shared_done"]) == []


def test_slash_kanban_gc_workspaces_is_a_dry_run(board):
    from hermes_cli import kanban

    fx = _gc_fixture()
    out = kanban.run_slash("gc --workspaces --json")
    report = json.loads(out)
    assert report["mode"] == "dry-run" and report["counts"]["removable"] == 2
    assert all(p.exists() for p in fx["removable"])


# --- C: the worker's own cwd must not keep its workspace alive -----------------
#
# ``_default_spawn`` starts every worker with ``cwd=<workspace>`` and
# ``complete_task`` runs the cleanup inside that worker. Windows keeps a
# handle on the directory a process was started in until that process exits,
# so the worker's ``rmdir`` on its own workspace always fails: contents gone,
# shell kept, nothing to retry from inside. Two layers fix it —
# :func:`_force_rmtree` moves a process whose cwd sits inside the tree out
# first (covers in-process callers, and never strands anyone on a deleted
# cwd), and the dispatcher runs the cleanup once more after the worker itself
# has exited (``reap_terminal_workers``), from a process that holds nothing.


def test_force_rmtree_removes_the_tree_the_process_is_sitting_in(board, monkeypatch):
    ws = kb.workspaces_root() / "cwd-tree"
    nested = ws / "sub"
    nested.mkdir(parents=True)
    (nested / "scratch.txt").write_text("work", encoding="utf-8")
    monkeypatch.chdir(nested)

    assert kbw._force_rmtree(ws) is None

    assert not ws.exists()
    # Parked next to the tree — not left on a directory that no longer exists.
    assert Path.cwd().resolve() == ws.parent.resolve()
    assert Path.cwd().is_dir()


def test_complete_task_removes_the_scratch_workspace_that_is_the_cwd(board, monkeypatch):
    with kbc.connect_closing() as conn:
        tid, ws = _terminal_scratch(conn, status="ready")
    monkeypatch.chdir(ws)

    with kbc.connect_closing() as conn:
        assert kb.complete_task(conn, tid, summary="finished") is True

    assert not ws.exists(), "completion left the workspace shell behind"
    with kbc.connect_closing() as conn:
        assert _deferred_reasons(conn, tid) == []


_WORKER_SCRIPT = """
import json, os, sys
from pathlib import Path
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
db_path, task_id, run_id = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
workspace = kb.workspaces_root() / task_id
with kbc.connect_closing(db_path) as conn:
    report = {
        "pid": os.getpid(),
        "cwd_inside": Path.cwd().resolve() == workspace.resolve(),
        "done": bool(kb.complete_task(conn, task_id, result="finished", expected_run_id=run_id)),
        "exists_after": workspace.exists(),
        "cwd_after": os.getcwd(),
    }
print(json.dumps(report))
"""


def test_worker_spawned_inside_its_workspace_leaves_no_shell_behind(board):
    """The dispatcher's exact lifecycle: a real worker started with
    ``cwd=workspace`` completes its own card, exits, and the next dispatch
    sweep removes the directory the worker could not delete from inside."""
    from hermes_cli import kanban_db_dispatch as kbd

    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="spawned in its own workspace")
        ws = kb.workspaces_root() / tid
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "scratch.txt").write_text("work", encoding="utf-8")
        conn.execute(
            "UPDATE tasks SET workspace_kind='scratch', workspace_path=? WHERE id=?",
            (str(ws), tid),
        )
        kb.claim_task(conn, tid, claimer=kb._claimer_id())
        run_id = kb._current_run_id(conn, tid)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])  # the checkout under test
    proc = subprocess.Popen(
        [sys.executable, "-c", _WORKER_SCRIPT, str(kb.kanban_db_path()), tid, str(run_id)],
        cwd=ws,  # what _default_spawn does
        env=env,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )
    with kbc.connect_closing() as conn:
        kbd._set_worker_pid(conn, tid, proc.pid)
    out, err = proc.communicate(timeout=180)
    assert proc.returncode == 0, err
    report = json.loads(out.strip().splitlines()[-1])
    assert report["cwd_inside"] is True and report["done"] is True
    # The worker lives on for its final turn: it must still have a real cwd.
    assert Path(report["cwd_after"]).is_dir()

    # Worker exited; the next tick cleans up what it could not.
    assert proc.wait(timeout=10) == 0
    grace = kbd.TERMINAL_WORKER_REAP_GRACE_SECONDS + 10
    with kbc.connect_closing() as conn:
        conn.execute("UPDATE task_runs SET ended_at = ended_at - ? WHERE id=?", (grace, run_id))
        assert kbd.reap_terminal_workers(conn) == []  # nothing to terminate, pid simply gone
    assert not ws.exists(), "the sweep left the shell the worker could not delete"
