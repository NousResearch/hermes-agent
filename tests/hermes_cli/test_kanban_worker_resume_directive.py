"""Tests: a respawned kanban worker gets a resume directive in its kickoff prompt.

When a worker run is reclaimed/killed (restart wave, timeout, stale-claim
reclaim) and the dispatcher re-dispatches the task, the new worker used to
start from the bare ``work kanban task <id>`` prompt and re-discover everything
from scratch — re-cloning worktrees, re-running env setup — burning runs/hours.

``_worker_kickoff_prompt`` now appends a resume directive ONLY on a respawn
(the task has >=1 prior run that did not complete). The directive names the
existing worktree/branch to reuse and points at the prior-run evidence. A
task's first dispatch (no prior closed runs) keeps the bare kickoff prompt.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kb.connect() as c:
        yield c


def test_first_run_prompt_is_bare(conn):
    """A task on its first dispatch (no prior closed runs) gets the bare
    ``work kanban task <id>`` prompt — first-run behaviour is unchanged."""
    tid = kb.create_task(
        conn, title="fresh", assignee="w",
        workspace_kind="worktree", branch_name="wt/fresh",
    )
    # Claim opens the first (active) run — no prior CLOSED runs exist yet.
    kb.claim_task(conn, tid, claimer=f"{kb._claimer_id().split(':', 1)[0]}:A")
    task = kb.get_task(conn, tid)

    prompt = kb._worker_kickoff_prompt(conn, task, "/tmp/ws")

    assert prompt == f"work kanban task {tid}"
    assert "RESUMING" not in prompt


def test_respawn_prompt_has_resume_directive_and_worktree(conn):
    """After a prior run is reclaimed, the respawn kickoff prompt carries the
    resume directive, the concrete worktree path, and the branch."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(
        conn, title="resume me", assignee="w",
        workspace_kind="worktree", branch_name="wt/resume",
    )

    # Attempt 1: claim then reclaim (simulates a restart-wave kill). This
    # closes the run with outcome='reclaimed' and resets the task to ready.
    kb.claim_task(conn, tid, claimer=f"{host}:A")
    dead = subprocess.Popen(["true"])
    dead.wait()
    kb._set_worker_pid(conn, tid, dead.pid)
    assert kb.reclaim_task(conn, tid, reason="restart wave") is True

    # Dispatcher re-dispatches: new claim (active run) + persisted workspace.
    kb.claim_task(conn, tid, claimer=f"{host}:B")
    worktree = "/repos/app/.worktrees/" + tid
    kb.set_workspace_path(conn, tid, worktree)
    kb.set_branch_name(conn, tid, "wt/resume")
    task = kb.get_task(conn, tid)

    prompt = kb._worker_kickoff_prompt(conn, task, worktree)

    # Bare kickoff still leads.
    assert prompt.startswith(f"work kanban task {tid}")
    # Resume directive present.
    assert "RESUMING A PRIOR RUN" in prompt
    assert "1 prior run" in prompt
    # Concrete worktree + branch are named so the worker reuses them.
    assert worktree in prompt
    assert "wt/resume" in prompt
    assert "do NOT create a new worktree" in prompt
    # Points at the prior-run evidence.
    assert f"hermes kanban context {tid}" in prompt


def test_completed_prior_run_does_not_trigger_directive(conn):
    """A prior run that COMPLETED is not a failed respawn — no directive.

    Only runs that did not complete (reclaimed/crashed/timed_out) count."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="stepwise", assignee="w")

    kb.claim_task(conn, tid, claimer=f"{host}:A")
    # Close the first run as a success (the only prior run completed).
    kb._end_run(conn, tid, outcome="completed", status="done", summary="ok")
    task = kb.get_task(conn, tid)

    prompt = kb._worker_kickoff_prompt(conn, task, "/tmp/ws")

    assert prompt == f"work kanban task {tid}"
    assert "RESUMING" not in prompt
