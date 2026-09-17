"""Regression tests for issue #113004: kanban terminal fence on pid-less claims
plus ``show`` ownership fields and ``closed_by`` audit trail.

The issue:
- ``_claim_is_live`` was gated on ``worker_pid`` liveness, so a control-plane /
  CLI / library claim (no worker spawned) had ``worker_pid IS NULL`` and the
  fence returned False — a non-claimant's ``complete_task`` silently closed
  the lane's run row and overwrote its summary/metadata.
- Same shape when a worker pid is dead (crashed / recycled): the dead pid
  also flipped the fence off.
- ``show`` printed no claim owner / pid / heartbeat / run id, and
  ``show --json``'s task dict omitted those fields.
- A foreign close left no durable audit trace — the run's
  ``profile`` (the original claimant) survived but the closer was not
  recorded anywhere on the closed row.

The fix:
- ``_claim_is_live`` returns True when EITHER the worker is alive (the
  existing #111764 contract) OR the claim record is in force AND the
  caller is NOT the claimant. The same-caller library/CLI flow
  (``claim_task`` then ``complete_task`` from the same process, no
  spawned worker) still works because the caller's host:pid matches the
  claim's ``claim_lock``.
- ``show`` prints a dedicated ``Ownership:`` section on running tasks and
  exposes ``claim_lock``, ``claim_expires``, ``worker_pid``,
  ``current_run_id``, ``last_heartbeat_at`` in the JSON task dict.
- ``task_runs.closed_by`` records the ``"<host>:<pid>"`` of the caller
  that wrote the terminal row; ``show`` prints it with a ``foreign close``
  marker when it differs from ``claim_lock``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban as kcli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


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


def _claim_under(
    conn, claimer: str, *, live_worker_pid: int | None = None
) -> tuple[str, int]:
    """Claim a card under ``claimer``; optionally spawn a worker on it."""
    tid = kb.create_task(conn, title="pid-less", assignee="coder")
    assert kb.claim_task(conn, tid, claimer=claimer) is not None
    if live_worker_pid is not None:
        kbd._set_worker_pid(conn, tid, live_worker_pid)
    return tid, kb._current_run_id(conn, tid)


class _Capturing:
    """Minimal write-buffering stdout stand-in for ``redirect_stdout``."""

    def __init__(self, sink: list[str]) -> None:
        self._sink = sink

    def write(self, s: str) -> int:
        if s:
            self._sink.append(s)
        return len(s)

    def flush(self) -> None:
        return None


def _capture_cmd_show(args) -> str:
    buf: list[str] = []
    real = sys.stdout
    try:
        sys.stdout = _Capturing(buf)
        kcli._cmd_show(args)
    finally:
        sys.stdout = real
    return "".join(buf)


# --- Fence on the claim, not the process (#113004 cases A/C) ---


def test_pid_less_claim_refuses_non_claimant_close(conn):
    """Case A: lane K claims (no worker); lane D tries to complete."""
    tid, run_id = _claim_under(conn, claimer="lane-k-host:111")

    # Caller is the test process — NOT lane-k — so the fence MUST hold.
    with pytest.raises(kb.LiveClaimError) as exc:
        kb.complete_task(
            conn, tid, result="LANE-D blind completion", expected_run_id=None
        )

    # The error names the live lock so the operator knows who's holding it.
    assert "lane-k-host:111" in str(exc.value)
    assert exc.value.claim_lock == "lane-k-host:111"
    # The card is still running, the run is still active, the worker_pid is
    # NULL, and the original claim lock survives.
    trow = conn.execute(
        "SELECT status, claim_lock, worker_pid, current_run_id FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    assert trow["status"] == "running"
    assert trow["claim_lock"] == "lane-k-host:111"
    assert trow["worker_pid"] is None
    assert trow["current_run_id"] == run_id
    run = conn.execute(
        "SELECT ended_at, outcome, summary FROM task_runs WHERE id = ?", (run_id,)
    ).fetchone()
    assert run["ended_at"] is None and run["outcome"] is None and run["summary"] is None


def test_pid_less_claim_owner_still_can_close_without_expected_run_id(conn):
    """Same-caller library/CLI flow: claim_task + complete_task from this process
    with no spawned worker must still close (the legitimate control-plane path).
    """
    # Claim under our own _claimer_id() — same host:pid as the closing call.
    tid, run_id = _claim_under(conn, claimer=kb._claimer_id(), live_worker_pid=None)

    assert (
        kb.complete_task(conn, tid, result="library did it", expected_run_id=None)
        is True
    )

    run = conn.execute(
        "SELECT ended_at, outcome, summary FROM task_runs WHERE id = ?", (run_id,)
    ).fetchone()
    assert run["ended_at"] is not None and run["outcome"] == "completed"
    assert run["summary"] == "library did it"


def test_dead_worker_pid_still_protects_claim(conn):
    """Case C: lane K claims and the worker pid is dead. The dead pid used
    to short-circuit the fence off; now the live claim keeps it on for a
    non-claimant."""
    tid, _run_id = _claim_under(
        conn, claimer="lane-k-host:333", live_worker_pid=999_999
    )

    with pytest.raises(kb.LiveClaimError):
        kb.complete_task(
            conn, tid, result="LANE-D blind completion", expected_run_id=None
        )

    trow = conn.execute(
        "SELECT status, claim_lock, worker_pid, current_run_id FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    assert trow["status"] == "running"
    assert trow["claim_lock"] == "lane-k-host:333"


def test_force_true_still_closes_a_live_claim(conn):
    """``force=True`` is the explicit operator override and must still close,
    even when the caller is NOT the claimant."""
    tid, run_id = _claim_under(conn, claimer="lane-k-host:444")

    assert kb.complete_task(conn, tid, result="operator override", force=True) is True

    run = conn.execute(
        "SELECT ended_at, outcome, summary, closed_by FROM task_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    assert run["ended_at"] is not None
    assert run["outcome"] == "completed"
    assert run["summary"] == "operator override"
    # Operator override leaves a durable audit trace.
    assert run["closed_by"] == kb._claimer_id()


def test_live_worker_fence_still_holds_for_same_caller(conn):
    """#111764 contract: a live worker is the actor of the run, not the
    claimant. The same process that claimed cannot close the worker's run
    without ``expected_run_id``."""
    tid, run_id = _claim_under(
        conn, claimer=kb._claimer_id(), live_worker_pid=os.getpid()
    )

    with pytest.raises(kb.LiveClaimError):
        kb.complete_task(conn, tid, result="worker still working", expected_run_id=None)

    # Owner's correct close with expected_run_id still works.
    assert (
        kb.complete_task(conn, tid, result="worker done", expected_run_id=run_id)
        is True
    )


def test_expired_claim_does_not_protect(conn):
    """A claim whose ``claim_expires`` has passed does not protect from a
    non-claimant close — the existing reclaim_stale_tasks sweeps these
    and they are not 'live' anymore."""
    tid, _run_id = _claim_under(conn, claimer="lane-k-host:expired")
    # Force-expire the claim.
    conn.execute(
        "UPDATE tasks SET claim_expires = ? WHERE id = ?",
        (int(os.environ.get("FAKE_NOW", "1000000000")) - 1, tid),
    )

    # Now a non-claimant may close — the claim is no longer in force.
    assert (
        kb.complete_task(conn, tid, result="stale closed", expected_run_id=None) is True
    )


# --- request_review shares the fence ---


def test_request_review_refuses_non_claimant_under_pid_less_claim(conn):
    """``request_review`` keys on the same ``_claim_is_live`` as
    ``complete_task``; a non-claimant cannot review a live pid-less claim."""
    tid, _run_id = _claim_under(conn, claimer="lane-k-host:rev")

    ok, reason = kb.request_review(
        conn, tid, summary="foreign review", with_reason=True
    )
    assert ok is False
    assert "live claim" in reason
    assert "lane-k-host:rev" in reason
    assert kb.get_task(conn, tid).status == "running"


# --- closed_by audit trail ---


def test_closed_by_records_self_on_default_close(conn):
    """Without an explicit closer, ``closed_by`` is the closing caller's
    ``_claimer_id()`` (host:pid)."""
    tid, run_id = _claim_under(conn, claimer=kb._claimer_id())
    kb.complete_task(conn, tid, result="own", expected_run_id=run_id)
    run = conn.execute(
        "SELECT closed_by FROM task_runs WHERE id = ?", (run_id,)
    ).fetchone()
    assert run["closed_by"] == kb._claimer_id()


def test_closed_by_records_operator_on_force(conn):
    """Force-override records the operator's id, not the claim's."""
    tid, run_id = _claim_under(conn, claimer="lane-k-host:force")
    kb.complete_task(conn, tid, result="force", force=True)
    run = conn.execute(
        "SELECT closed_by, profile FROM task_runs WHERE id = ?", (run_id,)
    ).fetchone()
    assert run["closed_by"] == kb._claimer_id()  # not lane-k-host:force
    assert run["profile"] == "coder"  # the original claimant survives


# --- show text + JSON output ---


def test_show_json_exposes_ownership_fields(conn):
    """``show --json`` carries ``claim_lock``, ``claim_expires``,
    ``worker_pid``, ``current_run_id``, ``last_heartbeat_at`` on the task
    dict (#113004 acceptance test)."""
    tid, run_id = _claim_under(
        conn, claimer="lane-k-host:show", live_worker_pid=os.getpid()
    )

    out = _capture_cmd_show(kcli.argparse.Namespace(task_id=tid, json=True))
    task = json.loads(out)["task"]
    assert task["claim_lock"] == "lane-k-host:show"
    assert task["worker_pid"] == os.getpid()
    assert task["current_run_id"] == run_id
    assert task["last_heartbeat_at"] is None  # never heartbeated
    # runs[] exposes closed_by (None on a still-active run, but the key is
    # present so dashboards don't have to special-case the field).
    runs = json.loads(out)["runs"]
    assert all("closed_by" in r for r in runs)


def test_show_text_prints_ownership_section_for_running_card(conn):
    """``show`` text output adds an ``Ownership:`` block on ``status=running``
    tasks showing the claim, the worker, and the run id."""
    tid, run_id = _claim_under(
        conn, claimer="lane-k-host:text", live_worker_pid=os.getpid()
    )

    out = _capture_cmd_show(kcli.argparse.Namespace(task_id=tid, json=False))
    assert "Ownership:" in out
    assert "lane-k-host:text" in out
    assert f"pid {os.getpid()}" in out
    assert f"#{run_id}" in out


def test_show_text_marks_pid_less_claim_lane(conn):
    """A pid-less claim (no worker spawned) prints the explanatory lane label."""
    tid, _run_id = _claim_under(conn, claimer="lane-k-host:lane", live_worker_pid=None)

    out = _capture_cmd_show(kcli.argparse.Namespace(task_id=tid, json=False))
    assert "Ownership:" in out
    assert "pid-less claim" in out
    assert "lane-k-host:lane" in out


def test_show_text_omits_ownership_section_for_terminal_card(conn):
    """Done / ready / todo cards do not show an Ownership section — the
    fields would be NULL/0 and the block is for live claims only."""
    tid = kb.create_task(conn, title="ready", assignee="coder")
    out = _capture_cmd_show(kcli.argparse.Namespace(task_id=tid, json=False))
    assert "Ownership:" not in out
