"""Tests for the kanban CLI surface (hermes_cli.kanban)."""

from __future__ import annotations

import argparse
import json
import os
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Workspace flag parsing
# ---------------------------------------------------------------------------







# ---------------------------------------------------------------------------
# run_slash smoke tests (end-to-end via the same entry both CLI and gateway use)
# ---------------------------------------------------------------------------



def test_kanban_list_json_includes_session_id(kanban_home):
    """JSON output exposes `session_id` so external clients (Scarf, web
    dashboards) don't need a side query to filter by chat session."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect() as conn:
        kb.create_task(
            conn, title="acp task", assignee="alice", session_id="acp-x"
        )
    raw = kc.run_slash("list --json")
    payload = json.loads(raw)
    assert any(
        row.get("title") == "acp task"
        and row.get("session_id") == "acp-x"
        for row in payload
    )


def test_kanban_show_text_renders_graph_with_open_connection(kanban_home):
    with kbc.connect_closing() as conn:
        parent_id = kb.create_task(conn, title="parent task")
        child_id = kb.create_task(conn, title="child task")
        kb.link_tasks(conn, parent_id=parent_id, child_id=child_id)

    output = kc.run_slash(f"show {child_id}")

    assert f"Task {child_id}: child task" in output
    assert f"parents:   {parent_id}" in output
    assert "Cannot operate on a closed database" not in output


def test_worker_link_preserves_foreign_child_rules(kanban_home, monkeypatch):
    with kbc.connect_closing() as conn:
        worker = kb.create_task(conn, title="worker")
        assert kb.claim_task(conn, worker, claimer="worker") is not None
        worker_run_id = kb.get_task(conn, worker).current_run_id
        parent = kb.create_task(conn, title="unfinished parent")
        ready_child = kb.create_task(conn, title="foreign ready child")
        running_child = kb.create_task(conn, title="foreign running child")
        assert kb.claim_task(conn, running_child, claimer="other") is not None

    monkeypatch.setenv("HERMES_KANBAN_TASK", worker)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(worker_run_id))

    assert kc._cmd_link(argparse.Namespace(
        parent_id=parent, child_id=ready_child,
    )) == 0
    with pytest.raises(ValueError, match="child is already running"):
        kc._cmd_link(argparse.Namespace(
            parent_id=parent, child_id=running_child,
        ))
    # Owner handoff: the worker links its own running card, proving ownership
    # with HERMES_KANBAN_RUN_ID — the one path _cmd_link forwards a run id for.
    assert kc._cmd_link(argparse.Namespace(
        parent_id=parent, child_id=worker,
    )) == 0

    with kbc.connect_closing() as conn:
        assert kb.parent_ids(conn, ready_child) == [parent]
        assert kb.parent_ids(conn, running_child) == []
        assert kb.parent_ids(conn, worker) == [parent]


def test_board_override_is_isolated_per_concurrent_call(kanban_home, monkeypatch):
    kb.create_board("alpha")
    kb.create_board("beta")

    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)

    barrier = threading.Barrier(2)
    original_init_db = kb.init_db

    def slow_init_db(*args, **kwargs):
        try:
            barrier.wait(timeout=5)
        except threading.BrokenBarrierError:
            pass
        return original_init_db(*args, **kwargs)

    monkeypatch.setattr(kb, "init_db", slow_init_db)

    failures: list[str] = []

    def worker(board: str, title: str) -> None:
        args = parser.parse_args(["kanban", "--board", board, "create", title])
        rc = kc.kanban_command(args)
        if rc != 0:
            failures.append(f"{board}:{rc}")

    t1 = threading.Thread(target=worker, args=("alpha", "alpha-task"))
    t2 = threading.Thread(target=worker, args=("beta", "beta-task"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert failures == []

    with kbc.connect_closing(board="alpha") as conn:
        alpha_titles = [row.title for row in kb.list_tasks(conn, limit=100)]
    with kbc.connect_closing(board="beta") as conn:
        beta_titles = [row.title for row in kb.list_tasks(conn, limit=100)]

    assert alpha_titles == ["alpha-task"]
    assert beta_titles == ["beta-task"]


# ---------------------------------------------------------------------------
# Integration with the COMMAND_REGISTRY
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# reclaim + reassign CLI smoke tests
# ---------------------------------------------------------------------------

def test_run_slash_reclaim_running_task(kanban_home):
    import re
    import time
    import secrets
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    out1 = kc.run_slash("create 'stuck worker task' --assignee broken-model")
    m = re.search(r"(t_[a-f0-9]+)", out1)
    assert m
    tid = m.group(1)

    # Simulate a running claim outside TTL.
    conn = kbc.connect()
    try:
        lock = secrets.token_hex(4)
        conn.execute(
            "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, "
            "worker_pid=? WHERE id=?",
            (lock, int(time.time()) + 3600, 4242, tid),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, status, claim_lock, claim_expires, "
            "worker_pid, started_at) VALUES (?, 'running', ?, ?, ?, ?)",
            (tid, lock, int(time.time()) + 3600, 4242, int(time.time())),
        )
        rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (rid, tid))
        conn.commit()
    finally:
        conn.close()

    out = kc.run_slash(f"reclaim {tid} --reason 'test'")
    assert "Reclaimed" in out, out
    # Status back to ready.
    out2 = kc.run_slash(f"show {tid}")
    assert "ready" in out2.lower()




# ---------------------------------------------------------------------------
# /kanban specify — slash surface (same entry point CLI + gateway use)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# /kanban help / no-args / unknown-action UX (issue #21794)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# kanban gc — honest workspace removal accounting (read-only trees on Windows)
# ---------------------------------------------------------------------------


def _archived_scratch_task(kbc, kb, title):
    """Create a task and flip it to archived WITHOUT the archive-hook cleanup,
    simulating the leftover gc is the backstop for."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title=title, assignee="alice",
                             workspace_kind="scratch")
    with kbc.connect() as conn:
        conn.execute("UPDATE tasks SET status='archived' WHERE id=?", (tid,))
    return tid


def test_gc_removes_readonly_workspace_and_counts_it(kanban_home, capsys):
    """A leftover scratch workspace containing read-only files (git keeps
    .git/objects 0444; Windows refuses to unlink read-only files) must be
    fully removed by gc — the old rmtree(ignore_errors=True) silently left
    it on disk while still counting it as removed."""
    import os
    import stat

    tid = _archived_scratch_task(kbc, kb, "gc readonly leftover")
    ws = kb.workspaces_root() / tid
    (ws / "repo" / ".git" / "objects").mkdir(parents=True)
    ro_file = ws / "repo" / ".git" / "objects" / "packfile"
    ro_file.write_text("object data\n")
    os.chmod(ro_file, stat.S_IREAD)

    out = kc.run_slash("gc")
    try:
        assert not ws.exists(), f"read-only workspace survived gc: {ws}"
        assert "1 workspace(s)" in out, out
    finally:
        if ro_file.exists():  # only if gc left it (test failure path) — restore +w for tmp cleanup
            os.chmod(ro_file, stat.S_IWRITE | stat.S_IREAD)


def test_gc_reports_leftover_instead_of_counting_it(kanban_home, monkeypatch, capsys):
    """When removal genuinely fails (locked files, non-permission I/O errors),
    gc must NOT count the workspace as removed — the old code incremented the
    counter unconditionally after rmtree(ignore_errors=True)."""
    import shutil as _shutil

    tid = _archived_scratch_task(kbc, kb, "gc unremovable leftover")
    ws = kb.workspaces_root() / tid
    ws.mkdir(parents=True)
    (ws / "locked.db").write_text("held open by a process\n")

    def _no_removal(*args, **kwargs):
        raise PermissionError(13, "simulated lock: even +w retry cannot remove")

    monkeypatch.setattr(_shutil, "rmtree", _no_removal)

    out = kc.run_slash("gc")
    assert ws.exists(), "workspace should still be there (removal was simulated as impossible)"
    assert "0 workspace(s)" in out, out
    assert "1 left on disk" in out, out


def test_gc_refuses_scratch_root_even_if_archived_task_points_at_it(kanban_home, capsys):
    """Review on #109586: an archived malformed/imported task whose
    workspace_path points AT the scratch root itself must not make gc delete
    the root (which holds every task's workspace). The old ``relative_to``
    containment accepted equality; the workspace module's strict-descendant
    managed-scratch predicate refuses roots."""
    root = kb.workspaces_root()
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "survivor.txt"
    marker.write_text("must outlive gc\n")

    tid = _archived_scratch_task(kbc, kb, "gc root pointer")
    with kbc.connect() as conn:
        conn.execute("UPDATE tasks SET workspace_path=? WHERE id=?", (str(root), tid))

    out = kc.run_slash("gc")
    assert root.exists(), "scratch root itself was deleted by gc!"
    assert marker.exists(), "content inside the scratch root was deleted by gc!"
    assert "0 workspace(s)" in out, out


def test_gc_readonly_removal_goes_through_repair_callback(kanban_home, monkeypatch, capsys):
    """Review on #109586: the read-only test must prove the permission-repair
    callback runs, not just that unlink eventually worked. The first deletion
    attempt on the read-only directory fails deterministically (PermissionError),
    the onerror/onexc handler clears +w and retries; the workspace must be gone,
    the callback invoked, and the workspace still counted as removed."""
    import os
    import stat

    tid = _archived_scratch_task(kbc, kb, "gc readonly repair callback")
    ws = kb.workspaces_root() / tid
    ro_dir = ws / "data"
    ro_dir.mkdir(parents=True)
    ro_file = ro_dir / "readonly.bin"
    ro_file.write_text("payload\n")
    # Read-only DIRECTORY: on POSIX unlink inside it fails until the handler
    # chmods the parent; on Windows the read-only file itself refuses unlink.
    os.chmod(ro_file, stat.S_IREAD)
    os.chmod(ro_dir, stat.S_IREAD | stat.S_IEXEC)

    calls = {"repair": 0}
    from hermes_cli import kanban_ops as kops
    real_handler = kops._rmtree_onerror_make_writable

    def spy_handler(func, path, exc):
        calls["repair"] += 1
        real_handler(func, path, exc)

    # Patch where _rmtree_force resolves it (kanban_ops), not the kanban.py re-import.
    monkeypatch.setattr(kops, "_rmtree_onerror_make_writable", spy_handler)

    out = kc.run_slash("gc")
    try:
        assert not ws.exists(), f"read-only workspace survived gc: {ws}"
        assert calls["repair"] >= 1, "permission-repair callback was never invoked"
        assert "1 workspace(s)" in out, out
    finally:
        # Restore +w so pytest tmp cleanup can remove any failure leftovers.
        for target in (ro_file, ro_dir):
            if target.exists():
                with __import__("contextlib").suppress(OSError):
                    os.chmod(target, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)


