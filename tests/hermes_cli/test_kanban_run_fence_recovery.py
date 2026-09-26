# The fence half of the witness: what a caller may do to a card it does NOT own.
# The run fence (superseded vs un-owned), the recorded recovery, and the refusal text that
# must come from the live row. The witness half lives in test_kanban_liveness_witness.py.
"""One liveness witness, three answers.

The 2026-09-26 false-reclaim tick declared six live workers dead inside ONE dispatch tick and spawned
a duplicate beside each (``EVIDENCE-blast-radius.txt``). Two probes answered "dead" where the honest
answer was "cannot tell": a secondary ``ps`` that exited non-zero, and a start-time read that returned
nothing (``EVIDENCE-reproduce-defect.txt``). The recorded start also moved by ~9 s for a live process
because the macOS boot reference it was built from was adjusted mid-flight
(``EVIDENCE-reference-change.txt``).

These tests pin the contract that follows from that: a liveness read is ``alive`` / ``dead`` /
``unknown``, only PROOF of death releases a claim or authorizes a signal, and an unprovable read holds
the claim. They also pin the run fence that rides on the verdict, and the recovery that records a
reclaim honestly instead of leaving a crash as the attempt history's last word.
"""

import json
import os
import signal
import subprocess
import time

import pytest

from gateway import status as gw_status
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()


def _claimed_running(conn, *, pid=None, started_at, max_runtime=None) -> str:
    """A ``running`` card claimed by a host-local worker whose pid/fingerprint we control."""
    tid = kb.create_task(conn, title="job", assignee="worker", max_runtime_seconds=max_runtime)
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, pid if pid is not None else os.getpid())
    old = int(time.time()) - 3600
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET worker_started_at = ?, started_at = ?, claim_expires = ? WHERE id = ?",
            (started_at, old, old, tid),
        )
        conn.execute(
            "UPDATE task_runs SET started_at = ? WHERE id = "
            "(SELECT current_run_id FROM tasks WHERE id = ?)",
            (old, tid),
        )
    return tid


# The two fingerprints a deployed board carries that can no longer be compared: the pre-witness shape
# (a bare start tick) and the macOS shape whose boot-witness component is empty. Neither can be
# salvaged, because for both of them "the value differs" is indistinguishable from "the boot reference
# was adjusted" — so both must read UNKNOWN, never DEAD.
UNCOMPARABLE_FINGERPRINTS = ["179041285600", "|1790412856", 179041285600]


def test_a_named_run_that_is_not_current_is_refused_on_every_path(board):
    """Direction one: a caller that names a run it does not hold is refused, and the refusal is
    a refused WRITE — the transition must not land on whatever run happens to be current."""
    conn = board
    tid = _claimed_running(conn, started_at=kbd._process_fingerprint(os.getpid()))
    current = conn.execute("SELECT current_run_id FROM tasks WHERE id = ?", (tid,)).fetchone()[0]
    stale = int(current) + 999

    # A named run that is not the current one is refused as a WRITE on every path: the transition
    # cannot land on whatever run happens to be current.
    assert kb.complete_task(conn, tid, result="done", expected_run_id=stale) is False
    assert kb.get_task(conn, tid).status == "running"

    assert kb.request_review(conn, tid, reviewer="reviewer", expected_run_id=stale) is False
    assert kb.get_task(conn, tid).status == "running"

    assert kb.block_task(conn, tid, reason="nope", expected_run_id=stale) is False
    assert kb.get_task(conn, tid).status == "running"


def test_the_fence_does_not_turn_away_an_unprovable_owner(board):
    """Direction two: a claim that cannot be certified does not refuse the caller that holds
    it — the run guard is what keeps the write from landing on a run it never read."""
    conn = board
    unprovable = _claimed_running(conn, started_at="|1790412856")
    assert kb.complete_task(conn, unprovable, result="done") is True
    assert kb.get_task(conn, unprovable).status == "done"

    proven_live = _claimed_running(conn, started_at=kbd._process_fingerprint(os.getpid()))
    with pytest.raises(kb.LiveClaimError):
        kb.complete_task(conn, proven_live, result="done")
    assert kb.get_task(conn, proven_live).status == "running"


def test_unknown_claim_refusal_never_recommends_force():
    """Refusing on an uncertifiable claim must not send the operator through the live run."""
    alive = kb.live_claim_refusal("t_x", verdict=kb.WORKER_ALIVE, run_id=42)
    assert "force=True" in alive and "expected_run_id" in alive

    unknown = kb.live_claim_refusal("t_x", verdict=kb.WORKER_UNKNOWN, run_id=42)
    assert "expected_run_id" in unknown
    # No force ESCAPE is offered — only the prohibition. ``--force``/``force=True`` are the strings an
    # operator acts on, so those are what must be absent.
    assert "--force" not in unknown and "force=True" not in unknown
    assert "may be live" in unknown

    err = kb.LiveClaimError("t_x", verdict=kb.WORKER_UNKNOWN, run_id=42)
    assert "expected_run_id=42" in str(err)


def test_a_reclaimed_worker_still_closes_its_own_card(board):
    """The incident's end state: the reclaim declared a LIVE worker dead, NULLed the card's
    ``current_run_id`` and blocked it — and that worker, the only legitimate owner, must still be
    able to close its own card (the un-owned branch) with the recovery recorded honestly
    rather than a crash standing as the attempt history's last word."""
    conn = board
    tid = _claimed_running(conn, started_at=kbd._process_fingerprint(os.getpid()))
    run_id = conn.execute("SELECT current_run_id FROM tasks WHERE id = ?", (tid,)).fetchone()[0]
    # Exactly what `_reclaim_dead_workers` left behind: the run closed as a crash, the card un-owned
    # and blocked, the failure counter charged to a worker that never failed.
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_runs SET outcome = 'crashed', status = 'crashed', ended_at = ?, metadata = ? "
            "WHERE id = ?",
            (int(time.time()), json.dumps({"reason": "crashed_worker"}), run_id),
        )
        conn.execute(
            "UPDATE tasks SET current_run_id = NULL, status = 'blocked', consecutive_failures = 2 "
            "WHERE id = ?",
            (tid,),
        )

    assert kb.complete_task(conn, tid, result="done", expected_run_id=run_id) is True
    assert kb.get_task(conn, tid).status == "done"
    assert conn.execute(
        "SELECT consecutive_failures FROM tasks WHERE id = ?", (tid,)).fetchone()[0] == 0, \
        "the run did not fail; the infrastructure did"

    recovery = conn.execute(
        "SELECT outcome, metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1", (tid,)
    ).fetchone()
    assert recovery["outcome"] == "completed"
    meta = json.loads(recovery["metadata"] or "{}")
    assert meta.get("infra_reclaimed") is True
    assert meta.get("recovered_from_run_id") == run_id


def test_a_superseded_caller_is_refused_even_while_the_card_is_un_owned(board):
    """The other half: an un-owned card admits the run that is still the card's NEWEST — a caller
    whose run was superseded is refused by the same clause, so the recovery cannot be hijacked."""
    conn = board
    tid = _claimed_running(conn, started_at=kbd._process_fingerprint(os.getpid()))
    superseded = conn.execute("SELECT current_run_id FROM tasks WHERE id = ?", (tid,)).fetchone()[0]
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET current_run_id = NULL WHERE id = ?", (tid,))
        conn.execute(
            "INSERT INTO task_runs (task_id, status, started_at) VALUES (?, 'running', ?)",
            (tid, int(time.time())),
        )

    assert kb.complete_task(conn, tid, result="done", expected_run_id=superseded) is False
    assert kb.get_task(conn, tid).status == "running"


def test_a_refusal_reads_the_live_row_not_a_stale_crash_string(board):
    """A refusal answers from the card's LIVE state. ``last_failure_error`` is durable and
    describes a run that is OVER; presenting it as the current reason is how a two-day-old crash
    string answered a live call and sent the operator chasing a stale run (#123811)."""
    conn = board
    planted = "worker crashed 2 days ago: stale-run-text-must-not-be-the-answer"
    tid = _claimed_running(conn, started_at=kbd._process_fingerprint(os.getpid()))
    owner = conn.execute("SELECT current_run_id FROM tasks WHERE id = ?", (tid,)).fetchone()[0]
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET last_failure_error = ? WHERE id = ?", (planted, tid))

    text = kb.live_row_refusal(conn, tid, caller_run_id=987654, verb="complete")
    lead = text.split("history only")[0]
    assert planted not in lead, "the durable crash string answered a live refusal"
    assert "status=running" in lead, lead
    assert f"run {owner} owns it" in lead and "SUPERSEDED" in lead, lead
    # The crash text is not hidden — it is quoted, and labelled as what it is.
    assert "history only" in text and planted in text, text


def test_an_un_owned_card_names_which_run_the_infrastructure_closed(board):
    """The other half: when the card's run was closed by the infrastructure, the refusal
    names that run and the way back, instead of a generic "(stale run)"."""
    conn = board
    tid = _claimed_running(conn, started_at=kbd._process_fingerprint(os.getpid()))
    run_id = conn.execute("SELECT current_run_id FROM tasks WHERE id = ?", (tid,)).fetchone()[0]
    # The false reclaim's end state, as `_reclaim_dead_workers` left it on the lost card.
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_runs SET outcome = 'crashed', status = 'crashed', metadata = ? WHERE id = ?",
            ('{"reason": "crashed_worker"}', run_id),
        )
        conn.execute(
            "UPDATE tasks SET current_run_id = NULL, status = 'blocked' WHERE id = ?", (tid,))

    text = kb.live_row_refusal(conn, tid, caller_run_id=None, verb="complete")
    assert "no run owns it" in text, text
    assert f"run {run_id} was closed by the INFRASTRUCTURE" in text, text
    assert f"expected_run_id={run_id}" in text and "recorded as a recovery" in text, text


