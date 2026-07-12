"""Tests for two related dispatcher resilience fixes.

1. Opt-in per-worker systemd scope isolation
   (``kanban.worker_slice``): the ``_maybe_wrap_worker_scope`` helper wraps
   a worker argv in ``systemd-run --scope`` when the feature is enabled AND
   the host is Linux AND ``systemd-run`` is on PATH, and otherwise returns
   the plain argv unchanged. This decouples each worker's cgroup /
   resource limits from the dispatcher's unit.

2. Fleet-wide (cross-board) concurrency cap
   (``kanban.max_in_progress_global``): the dispatcher ticks every board, so
   a per-board ``max_in_progress`` cannot bound total host concurrency. The
   global cap counts in-flight workers ACROSS boards and stops dispatch once
   the fleet-wide limit is reached.
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest


# ---------------------------------------------------------------------------
# Change 1 — per-worker systemd scope wrapper
# ---------------------------------------------------------------------------

@pytest.fixture()
def kb_module():
    """Import kanban_db under an isolated HERMES_HOME (no real fleet config)."""
    test_home = tempfile.mkdtemp(prefix="kanban_worker_slice_test_")
    for prof in ("alpha", "beta", "default"):
        os.makedirs(os.path.join(test_home, "profiles", prof), exist_ok=True)
    os.environ["HERMES_HOME"] = test_home
    for mod in list(sys.modules.keys()):
        if (
            mod.startswith("hermes_cli")
            or mod.startswith("hermes_state")
            or mod == "hermes_constants"
        ):
            del sys.modules[mod]
    from hermes_cli import kanban_db
    yield kanban_db


BASE_CMD = ["/opt/hermes/bin/hermes", "-p", "alpha", "--cli", "chat", "-q", "work"]


def test_wrap_disabled_by_default_returns_plain(kb_module, monkeypatch):
    """No config flag → plain argv, even if systemd-run exists."""
    kb = kb_module
    monkeypatch.setattr(kb.sys, "platform", "linux")
    monkeypatch.setattr(kb.shutil, "which", lambda _n: "/usr/bin/systemd-run")
    out = kb._maybe_wrap_worker_scope(BASE_CMD, kanban_cfg={})
    assert out == BASE_CMD


def test_wrap_enabled_linux_with_systemd_run_wraps(kb_module, monkeypatch):
    """Enabled + Linux + systemd-run present → systemd-run --scope wrapper."""
    kb = kb_module
    monkeypatch.setattr(kb.sys, "platform", "linux")
    monkeypatch.setattr(kb.shutil, "which", lambda _n: "/usr/bin/systemd-run")
    cfg = {
        "worker_slice": True,
        "worker_slice_name": "hermes-workers.slice",
        "worker_slice_properties": {"MemoryHigh": "900M", "CPUQuota": "50%"},
    }
    out = kb._maybe_wrap_worker_scope(BASE_CMD, kanban_cfg=cfg)
    assert out[0] == "/usr/bin/systemd-run"
    assert "--scope" in out
    assert "--slice=hermes-workers.slice" in out
    assert "--user" in out  # default manager
    assert "--property=MemoryHigh=900M" in out
    assert "--property=CPUQuota=50%" in out
    # `--` must separate systemd-run's options from the worker argv, and the
    # worker argv must follow intact so the captured PID is the worker's.
    assert "--" in out
    assert out[out.index("--") + 1:] == BASE_CMD


def test_wrap_enabled_but_systemd_run_absent_falls_back(kb_module, monkeypatch):
    """Enabled + Linux but systemd-run not on PATH → plain argv."""
    kb = kb_module
    monkeypatch.setattr(kb.sys, "platform", "linux")
    monkeypatch.setattr(kb.shutil, "which", lambda _n: None)
    out = kb._maybe_wrap_worker_scope(BASE_CMD, kanban_cfg={"worker_slice": True})
    assert out == BASE_CMD


def test_wrap_enabled_but_not_linux_falls_back(kb_module, monkeypatch):
    """Enabled + systemd-run present but non-Linux host → plain argv."""
    kb = kb_module
    monkeypatch.setattr(kb.sys, "platform", "darwin")
    monkeypatch.setattr(kb.shutil, "which", lambda _n: "/usr/bin/systemd-run")
    out = kb._maybe_wrap_worker_scope(BASE_CMD, kanban_cfg={"worker_slice": True})
    assert out == BASE_CMD


def test_wrap_invalid_slice_name_fails_closed(kb_module, monkeypatch):
    """A slice name that isn't a valid unit name disables the wrapper."""
    kb = kb_module
    monkeypatch.setattr(kb.sys, "platform", "linux")
    monkeypatch.setattr(kb.shutil, "which", lambda _n: "/usr/bin/systemd-run")
    cfg = {"worker_slice": True, "worker_slice_name": "not-a-slice"}
    out = kb._maybe_wrap_worker_scope(BASE_CMD, kanban_cfg=cfg)
    assert out == BASE_CMD


def test_wrap_system_manager_when_user_false(kb_module, monkeypatch):
    """worker_slice_user: false → no --user flag (system scope)."""
    kb = kb_module
    monkeypatch.setattr(kb.sys, "platform", "linux")
    monkeypatch.setattr(kb.shutil, "which", lambda _n: "/usr/bin/systemd-run")
    cfg = {"worker_slice": True, "worker_slice_user": False}
    out = kb._maybe_wrap_worker_scope(BASE_CMD, kanban_cfg=cfg)
    assert out[0] == "/usr/bin/systemd-run"
    assert "--user" not in out


# ---------------------------------------------------------------------------
# Change 2 — fleet-wide (cross-board) concurrency cap
# ---------------------------------------------------------------------------

@pytest.fixture()
def kb_two_boards():
    """Fresh HERMES_HOME with two kanban boards + an 'alpha' profile."""
    test_home = tempfile.mkdtemp(prefix="kanban_global_cap_test_")
    for prof in ("alpha", "default"):
        os.makedirs(os.path.join(test_home, "profiles", prof), exist_ok=True)
    os.environ["HERMES_HOME"] = test_home
    for mod in list(sys.modules.keys()):
        if (
            mod.startswith("hermes_cli")
            or mod.startswith("hermes_state")
            or mod == "hermes_constants"
        ):
            del sys.modules[mod]
    from hermes_cli import kanban_db as kb
    kb.create_board(slug="boardA", name="Board A")
    kb.create_board(slug="boardB", name="Board B")
    yield kb


def _fake_spawn(*args, **kwargs):
    return 4242


def _mark_running(kb, slug, n):
    """Force ``n`` tasks into 'running' on a board (simulates in-flight)."""
    with kb.connect_closing(board=slug) as conn:
        for i in range(n):
            tid = kb.create_task(conn, title=f"run{i}", assignee="alpha")
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status='running', claim_lock=? WHERE id=?",
                    (f"test:{i}", tid),
                )


def _add_ready(kb, slug, n):
    with kb.connect_closing(board=slug) as conn:
        for i in range(n):
            kb.create_task(conn, title=f"ready{i}", assignee="alpha")


def test_no_global_cap_dispatches_all(kb_two_boards):
    """Baseline: without the global cap, all ready tasks dispatch."""
    kb = kb_two_boards
    _add_ready(kb, "boardA", 4)
    with kb.connect_closing(board="boardA") as conn:
        res = kb.dispatch_once(
            conn, board="boardA", spawn_fn=_fake_spawn, dry_run=True,
        )
    assert len(res.spawned) == 4


def test_global_cap_accounts_for_other_boards(kb_two_boards):
    """boardB already runs 3; global cap=4 → boardA may spawn only 1."""
    kb = kb_two_boards
    _mark_running(kb, "boardB", 3)
    _add_ready(kb, "boardA", 5)
    with kb.connect_closing(board="boardA") as conn:
        res = kb.dispatch_once(
            conn, board="boardA", spawn_fn=_fake_spawn, dry_run=False,
            max_in_progress_global=4,
            global_in_progress=3,  # boardB's live workers, per the watcher
        )
    assert len(res.spawned) == 1


def test_global_cap_already_reached_spawns_nothing(kb_two_boards):
    """Other boards already at the cap → boardA spawns nothing this tick."""
    kb = kb_two_boards
    _add_ready(kb, "boardA", 5)
    with kb.connect_closing(board="boardA") as conn:
        res = kb.dispatch_once(
            conn, board="boardA", spawn_fn=_fake_spawn, dry_run=True,
            max_in_progress_global=4,
            global_in_progress=4,
        )
    assert len(res.spawned) == 0


def test_global_cap_counts_this_boards_own_running(kb_two_boards):
    """This board's own running workers count toward the global total."""
    kb = kb_two_boards
    _mark_running(kb, "boardA", 2)  # boardA already has 2 in flight
    _add_ready(kb, "boardA", 5)
    with kb.connect_closing(board="boardA") as conn:
        # cap=3, other boards contribute 0 → 2 already local, 1 slot left.
        res = kb.dispatch_once(
            conn, board="boardA", spawn_fn=_fake_spawn, dry_run=False,
            max_in_progress_global=3,
            global_in_progress=0,
        )
    assert len(res.spawned) == 1


def test_global_cap_stops_dispatch_across_two_boards(kb_two_boards):
    """End-to-end of the watcher's cross-board accumulation.

    Replicates the sequential per-board tick the gateway watcher performs:
    pre-scan running per board, then dispatch each board passing the count
    of workers already running on the OTHER boards plus spawns made earlier
    this tick. With cap=3 and both boards holding ready work, exactly 3
    workers should be dispatched across the two boards combined.
    """
    kb = kb_two_boards
    _add_ready(kb, "boardA", 5)
    _add_ready(kb, "boardB", 5)
    cap = 3

    slugs = ["boardA", "boardB"]
    # Pre-scan (mirrors _tick_once): running per board, total across boards.
    running_by_board = {}
    total_running = 0
    for slug in slugs:
        with kb.connect_closing(board=slug) as conn:
            c = int(
                conn.execute(
                    "SELECT COUNT(*) FROM tasks WHERE status='running'"
                ).fetchone()[0]
            )
        running_by_board[slug] = c
        total_running += c

    spawned_global = 0
    per_board_spawned = {}
    for slug in slugs:
        global_other = total_running - running_by_board[slug] + spawned_global
        with kb.connect_closing(board=slug) as conn:
            res = kb.dispatch_once(
                conn, board=slug, spawn_fn=_fake_spawn, dry_run=False,
                max_in_progress_global=cap,
                global_in_progress=global_other,
            )
        n = len(res.spawned)
        per_board_spawned[slug] = n
        spawned_global += n

    assert spawned_global == cap, per_board_spawned
    # boardA is dispatched first and fills the cap; boardB gets the remainder.
    assert per_board_spawned["boardA"] == 3
    assert per_board_spawned["boardB"] == 0


def test_global_and_per_board_caps_compose(kb_two_boards):
    """The tighter of (per-board max_in_progress, global cap) wins."""
    kb = kb_two_boards
    _add_ready(kb, "boardA", 10)
    with kb.connect_closing(board="boardA") as conn:
        # per-board cap 2 is tighter than the global headroom of 5 → spawn 2.
        res = kb.dispatch_once(
            conn, board="boardA", spawn_fn=_fake_spawn, dry_run=False,
            max_in_progress=2,
            max_in_progress_global=8,
            global_in_progress=3,
        )
    assert len(res.spawned) == 2
