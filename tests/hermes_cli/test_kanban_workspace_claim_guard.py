"""Invariant: two workers never hold the same workspace (issue #128259).

``_claim_and_open_run`` performed the claim CAS with no workspace-busy check,
so two workers could claim different cards that resolve to the same
``workspace_path`` and clobber each other's worktree. The claim guard refuses
the second claim while another *live* running task holds the same normalized
``workspace_path``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _running_holder(conn, workspace_path, *, live_worker):
    """A claimed (running) task holding ``workspace_path``."""
    tid = kb.create_task(
        conn, title="holder", assignee="coder",
        workspace_kind="dir", workspace_path=workspace_path,
    )
    assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
    if live_worker:
        # This process stands in for the spawned worker: alive, fingerprinted.
        kbd._set_worker_pid(conn, tid, os.getpid())
    return tid


def _ready_contender(conn, workspace_path):
    return kb.create_task(
        conn, title="contender", assignee="coder",
        workspace_kind="dir", workspace_path=workspace_path,
    )


def test_second_claim_refused_while_workspace_busy(conn, tmp_path):
    ws = str(tmp_path / "ws")
    _running_holder(conn, ws, live_worker=True)

    contender = _ready_contender(conn, ws)
    assert kb.claim_task(conn, contender, claimer=kb._claimer_id()) is None
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (contender,)).fetchone()["status"] == "ready"
    event = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        (contender,),
    ).fetchone()
    assert event["kind"] == "claim_rejected"
    assert "workspace_busy" in (event["payload"] or "")


def test_workspace_busy_normalizes_trailing_slash(conn, tmp_path):
    _running_holder(conn, str(tmp_path / "ws") + "/", live_worker=True)

    contender = _ready_contender(conn, str(tmp_path / "ws"))
    assert kb.claim_task(conn, contender, claimer=kb._claimer_id()) is None


def test_dead_holder_does_not_block_claim(conn, tmp_path):
    ws = str(tmp_path / "ws")
    _running_holder(conn, ws, live_worker=False)
    # No worker PID: stale claim, protects no live run.

    contender = _ready_contender(conn, ws)
    assert kb.claim_task(conn, contender, claimer=kb._claimer_id()) is not None


def test_distinct_workspaces_claim_fine(conn, tmp_path):
    _running_holder(conn, str(tmp_path / "ws-a"), live_worker=True)

    contender = _ready_contender(conn, str(tmp_path / "ws-b"))
    assert kb.claim_task(conn, contender, claimer=kb._claimer_id()) is not None
