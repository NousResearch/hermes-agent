"""Invariant: two workers never write the same workspace directory (#128259).

``_claim_and_open_run`` performed the claim CAS with no workspace-busy check,
so two cards on one ``dir`` workspace (decomposed siblings inherit the root's)
ran concurrently in one directory. The claim guard refuses the second claim
while another live running task holds the same directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import subprocess
import sys

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw


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


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"], stdin=subprocess.DEVNULL)
    proc.wait()
    return proc.pid


def _running_holder(conn, workspace_path, *, worker="live", ttl_seconds=None):
    """A claimed (running) task holding ``workspace_path``. ``worker``: ``live``
    (this process stands in for the spawned worker), ``dead`` (the recorded
    worker exited) or ``none`` (a CLI claim: no worker recorded)."""
    tid = kb.create_task(
        conn, title="holder", assignee="coder",
        workspace_kind="dir", workspace_path=workspace_path,
    )
    assert kb.claim_task(conn, tid, claimer=kb._claimer_id(), ttl_seconds=ttl_seconds) is not None
    if worker != "none":
        kbd._set_worker_pid(conn, tid, os.getpid() if worker == "live" else _dead_pid())
    return tid


def _ready_contender(conn, workspace_path, kind="dir"):
    return kb.create_task(
        conn, title="contender", assignee="coder",
        workspace_kind=kind, workspace_path=workspace_path,
    )


def test_second_claim_refused_while_workspace_busy(conn, tmp_path):
    ws = str(tmp_path / "ws")
    _running_holder(conn, ws, worker="live")

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
    _running_holder(conn, str(tmp_path / "ws") + "/", worker="live")

    contender = _ready_contender(conn, str(tmp_path / "ws"))
    assert kb.claim_task(conn, contender, claimer=kb._claimer_id()) is None


def test_dead_holder_does_not_block_claim(conn, tmp_path):
    ws = str(tmp_path / "ws")
    _running_holder(conn, ws, worker="dead")

    contender = _ready_contender(conn, ws)
    assert kb.claim_task(conn, contender, claimer=kb._claimer_id()) is not None


def test_distinct_workspaces_claim_fine(conn, tmp_path):
    _running_holder(conn, str(tmp_path / "ws-a"), worker="live")

    contender = _ready_contender(conn, str(tmp_path / "ws-b"))
    assert kb.claim_task(conn, contender, claimer=kb._claimer_id()) is not None


def test_cli_claim_without_worker_still_holds_workspace(conn, tmp_path):
    """``hermes kanban claim`` (and the dispatcher between claim and spawn)
    records no worker PID; the unexpired claim still owns the directory."""
    ws = str(tmp_path / "ws")
    _running_holder(conn, ws, worker="none")
    assert kb.claim_task(conn, _ready_contender(conn, ws), claimer=kb._claimer_id()) is None


def test_expired_workerless_claim_does_not_block(conn, tmp_path, monkeypatch):
    ws = str(tmp_path / "ws")
    _running_holder(conn, ws, worker="none", ttl_seconds=1)
    real_time = kb.time.time
    monkeypatch.setattr(kb.time, "time", lambda: real_time() + 5)
    assert kb.claim_task(conn, _ready_contender(conn, ws), claimer=kb._claimer_id()) is not None


def _git_repo(path):
    def git(*args):
        subprocess.run(["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                       check=True, capture_output=True)
    path.mkdir()
    git("init", "-q")
    git("commit", "-q", "--allow-empty", "-m", "init")
    return git


def test_worktree_contender_on_a_busy_repo_root_still_claims(conn, tmp_path):
    """A ``worktree`` card on a repo root runs in its own ``.worktrees/<id>``."""
    repo = tmp_path / "repo"
    _git_repo(repo)
    _running_holder(conn, str(repo), worker="live")
    contender = _ready_contender(conn, str(repo), kind="worktree")
    assert kb.claim_task(conn, contender, claimer=kb._claimer_id()) is not None


def test_retried_worktree_card_refused_while_its_checkout_is_held(conn, tmp_path):
    """After a first run the stored path IS the card's checkout, which a retry
    reuses; a live ``dir`` card writing there must hold the retry back."""
    repo = tmp_path / "repo"
    git = _git_repo(repo)
    contender = _ready_contender(conn, str(repo), kind="worktree")
    checkout = repo / ".worktrees" / contender
    git("worktree", "add", "-q", "-b", f"wt/{contender}", str(checkout))
    kbw.set_workspace_path(conn, contender, checkout)
    _running_holder(conn, str(checkout), worker="live")
    assert kb.claim_task(conn, contender, claimer=kb._claimer_id()) is None


def test_repeated_refusals_record_one_event(conn, tmp_path):
    ws = str(tmp_path / "ws")
    _running_holder(conn, ws, worker="live")
    contender = _ready_contender(conn, ws)
    for _ in range(3):
        assert kb.claim_task(conn, contender, claimer=kb._claimer_id()) is None
    kinds = [r["kind"] for r in conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ?", (contender,))]
    assert kinds.count("claim_rejected") == 1


def test_review_on_a_held_workspace_leaves_the_last_slot_to_ready_work(
    conn, tmp_path, monkeypatch, all_assignees_spawnable,
):
    """The review claim would be refused, so the review card must not reserve
    the only free slot and starve an unrelated ready card."""
    import hermes_cli.config as cfgmod
    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: {"kanban": {"review_dispatch": True}})
    ws = str(tmp_path / "ws")
    _running_holder(conn, ws, worker="live")
    review = _ready_contender(conn, ws)
    conn.execute("UPDATE tasks SET status = 'review', assignee = 'reviewer' WHERE id = ?", (review,))
    ready = kb.create_task(conn, title="unrelated", assignee="alice")
    res = kbd.dispatch_once(conn, spawn_fn=lambda task, ws, board=None: 42, max_in_progress=2)
    assert [task_id for task_id, *_ in res.spawned] == [ready]
