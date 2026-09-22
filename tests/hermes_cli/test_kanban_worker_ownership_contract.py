"""Kanban worker ownership contract (design §12 "Claim/start/crash sequence"):

  1. A stale/reclaimed attempt must be rejected BEFORE first inference, not
     merely have its receipt silently ignored.
  2. ``set_routing_receipt`` has no unguarded fallback: a receipt can only
     ever be attached under a real, live ``expected_run_id`` — never blind.
  3. A prior attempt's failure/reclaim path must never release, end, or
     block a NEWER (successor) run that already superseded it.

Real sqlite DB (separate connections, simulating separate processes for the
reclaim/replace race), real ``_enforce_kanban_routing_receipt`` worker-side
gate, real ``_record_task_failure``/``_end_run`` — nothing here mocks away
the enforcement path itself.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def _kanban_conn():
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return kbc.connect()


# ---------------------------------------------------------------------------
# 1. set_routing_receipt: no unguarded fallback.
# ---------------------------------------------------------------------------

def test_set_routing_receipt_refuses_none_expected_run_id():
    """There is no more "blind write" fallback: expected_run_id=None (or any
    falsy value) must refuse, never attach the receipt unconditionally."""
    from hermes_cli import kanban_db as kb

    conn = _kanban_conn()
    try:
        tid = kb.create_task(conn, title="t", assignee="alice", routing_role="builder")
        kb.claim_task(conn, tid, claimer="worker-a")
        assert kb.set_routing_receipt(conn, tid, "rr1", expected_run_id=None) is False
        assert kb.set_routing_receipt(conn, tid, "rr1", expected_run_id=0) is False
        reloaded = kb.get_task(conn, tid)
        assert reloaded.routing_receipt_id is None
    finally:
        conn.close()


def test_set_routing_receipt_links_under_genuine_live_claim():
    from hermes_cli import kanban_db as kb

    conn = _kanban_conn()
    try:
        tid = kb.create_task(conn, title="t", assignee="alice", routing_role="builder")
        claimed = kb.claim_task(conn, tid, claimer="worker-a")
        assert kb.set_routing_receipt(conn, tid, "rr1", expected_run_id=claimed.current_run_id)
        reloaded = kb.get_task(conn, tid)
        assert reloaded.routing_receipt_id == "rr1"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 2. Worker-side rejection of a stale/reclaimed attempt BEFORE inference.
# ---------------------------------------------------------------------------

def test_worker_rejects_stale_claim_before_receipt_check(monkeypatch, tmp_path):
    """Separate connections simulate separate processes: process A claims and
    is spawned with HERMES_KANBAN_RUN_ID=run_a; before A calls the
    enforcement gate, a second connection (a fresh dispatcher tick /
    process B) reclaims + replaces the run. A's enforcement call must fail
    closed with ZERO calls into enforce_worker_route — the receipt/model
    check never even runs for a superseded claim."""
    import cli as cli_mod
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    conn_a = _kanban_conn()
    try:
        tid = kb.create_task(conn_a, title="t", assignee="alice", routing_role="builder")
        claimed = kb.claim_task(conn_a, tid, claimer="worker-a")
        run_a = claimed.current_run_id
        conn_a.commit()
    finally:
        conn_a.close()

    # Process B: separate connection reclaims (crash/timeout releases the
    # claim) and re-claims — a genuinely different, newer run_id.
    conn_b = kbc.connect()
    try:
        conn_b.execute(
            "UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL, "
            "worker_pid=NULL WHERE id=?", (tid,),
        )
        conn_b.commit()
        reclaimed = kb.claim_task(conn_b, tid, claimer="worker-b")
        run_b = reclaimed.current_run_id
        conn_b.commit()
    finally:
        conn_b.close()
    assert run_b != run_a

    # Process A (the stale worker) now reaches the first-inference gate,
    # still carrying its ORIGINAL (now-superseded) run id.
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_a))
    monkeypatch.setenv("HERMES_KANBAN_ROUTING_RECEIPT", "rr_never_should_be_loaded")

    calls = []
    monkeypatch.setattr(
        "hermes_cli.kanban_model_routing.enforce_worker_route",
        lambda *a, **kw: calls.append((a, kw)),
    )

    cli = SimpleNamespace(agent=SimpleNamespace(
        provider="openai", requested_provider="openai", model="gpt-5", base_url=None,
    ))
    assert cli_mod._enforce_kanban_routing_receipt(cli) is False
    assert calls == [], "stale claim must be rejected before any receipt/model check runs"


def test_live_claim_without_linked_receipt_cannot_authorize_worker(monkeypatch, tmp_path):
    """A live run alone is insufficient without its canonical linked receipt."""
    import cli as cli_mod
    from hermes_cli import kanban_db as kb

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    conn = _kanban_conn()
    try:
        tid = kb.create_task(conn, title="t", assignee="alice", routing_role="builder")
        claimed = kb.claim_task(conn, tid, claimer="worker-a")
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_ROUTING_RECEIPT", "rr_live")

    calls = []
    monkeypatch.setattr(
        "hermes_cli.kanban_model_routing.enforce_worker_route",
        lambda *a, **kw: calls.append((a, kw)),
    )
    cli = SimpleNamespace(agent=SimpleNamespace(
        provider="openai", requested_provider="openai", model="gpt-5", base_url=None,
    ))
    assert cli_mod._enforce_kanban_routing_receipt(cli) is False
    assert calls == [], "an unlinked receipt must not reach the route check"


# ---------------------------------------------------------------------------
# 3. Successor runs survive a stale attempt's failure path.
# ---------------------------------------------------------------------------

def test_stale_spawn_failure_never_touches_successor_run(tmp_path):
    """A's spawn attempt fails AFTER B has already reclaimed+replaced the run
    (real race, two separate connections). _record_task_failure must be a
    complete no-op against B's live run/claim/status — B's claim, run row,
    and status must be bit-for-bit unchanged."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kdd

    conn_a = _kanban_conn()
    try:
        tid = kb.create_task(conn_a, title="t", assignee="alice")
        claimed = kb.claim_task(conn_a, tid, claimer="worker-a")
        run_a = claimed.current_run_id
        conn_a.commit()
    finally:
        conn_a.close()

    conn_b = kb  # reuse module, new connection below
    from hermes_cli import kanban_db_connect as kbc
    conn_b = kbc.connect()
    try:
        conn_b.execute(
            "UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL, "
            "worker_pid=NULL WHERE id=?", (tid,),
        )
        conn_b.commit()
        reclaimed = kb.claim_task(conn_b, tid, claimer="worker-b")
        run_b = reclaimed.current_run_id
        conn_b.commit()
        before = conn_b.execute(
            "SELECT status, claim_lock, current_run_id FROM tasks WHERE id=?", (tid,),
        ).fetchone()
        before_run_row = conn_b.execute(
            "SELECT status, ended_at FROM task_runs WHERE id=?", (run_b,),
        ).fetchone()
    finally:
        conn_b.close()
    assert run_b != run_a

    # Process A's spawn now fails, still reporting on run_a.
    conn_a2 = kbc.connect()
    try:
        tripped = kdd._record_task_failure(
            conn_a2, tid, "spawn exploded",
            outcome="spawn_failed", release_claim=True, end_run=True,
            expected_run_id=run_a,
        )
        assert tripped is False

        after = conn_a2.execute(
            "SELECT status, claim_lock, current_run_id FROM tasks WHERE id=?", (tid,),
        ).fetchone()
        after_run_row = conn_a2.execute(
            "SELECT status, ended_at FROM task_runs WHERE id=?", (run_b,),
        ).fetchone()
    finally:
        conn_a2.close()

    assert tuple(after) == tuple(before), "successor's task row must be untouched"
    assert tuple(after_run_row) == tuple(before_run_row), "successor's run row must be untouched"
    assert after["current_run_id"] == run_b


def test_stale_spawn_failure_breaker_trip_also_spares_successor(tmp_path):
    """Same race, but this time the failure trips the circuit breaker
    (force_trip=True) — even a breaker trip must not blocked/clobber the
    successor run."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kdd
    from hermes_cli import kanban_db_connect as kbc

    conn_a = _kanban_conn()
    try:
        tid = kb.create_task(conn_a, title="t", assignee="alice")
        claimed = kb.claim_task(conn_a, tid, claimer="worker-a")
        run_a = claimed.current_run_id
        conn_a.commit()
    finally:
        conn_a.close()

    conn_b = kbc.connect()
    try:
        conn_b.execute(
            "UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL, "
            "worker_pid=NULL WHERE id=?", (tid,),
        )
        conn_b.commit()
        reclaimed = kb.claim_task(conn_b, tid, claimer="worker-b")
        run_b = reclaimed.current_run_id
        conn_b.commit()
    finally:
        conn_b.close()

    conn_a2 = kbc.connect()
    try:
        tripped = kdd._record_task_failure(
            conn_a2, tid, "spawn exploded again",
            outcome="spawn_failed", release_claim=True, end_run=True,
            force_trip=True, expected_run_id=run_a,
        )
        assert tripped is False, "a stale attempt must not be able to trip the breaker on B's behalf"
        row = conn_a2.execute(
            "SELECT status, current_run_id FROM tasks WHERE id=?", (tid,),
        ).fetchone()
    finally:
        conn_a2.close()
    assert row["status"] == "running"
    assert row["current_run_id"] == run_b


def test_managed_pid_registration_rejects_stale_and_conflicting_owners():
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as dispatch

    conn = _kanban_conn()
    try:
        tid = kb.create_task(conn, title="managed ownership", assignee="alice", routing_role="builder")
        first = kb.claim_task(conn, tid, claimer="first")
        conn.execute("UPDATE tasks SET status='ready', claim_lock=NULL WHERE id=?", (tid,))
        second = kb.claim_task(conn, tid, claimer="second")
        assert dispatch._set_worker_pid(conn, tid, os.getpid(), expected_run_id=first.current_run_id) is False
        assert kb.get_task(conn, tid).worker_pid is None
        assert dispatch._set_worker_pid(conn, tid, os.getpid(), expected_run_id=second.current_run_id) is True
        assert dispatch._set_worker_pid(conn, tid, os.getpid(), expected_run_id=second.current_run_id) is True
        assert dispatch._set_worker_pid(conn, tid, os.getpid() + 1, expected_run_id=second.current_run_id) is False
        assert kb.get_task(conn, tid).worker_pid == os.getpid()
        run = conn.execute("SELECT worker_pid FROM task_runs WHERE id=?", (second.current_run_id,)).fetchone()
        assert run["worker_pid"] == os.getpid()
    finally:
        conn.close()


def test_end_run_cas_guard_is_noop_for_superseded_run(tmp_path):
    """Direct unit check on _end_run's expected_run_id CAS (the primitive the
    above race relies on)."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    conn = _kanban_conn()
    try:
        tid = kb.create_task(conn, title="t", assignee="alice")
        claimed = kb.claim_task(conn, tid, claimer="worker-a")
        run_a = claimed.current_run_id
        conn.execute(
            "UPDATE tasks SET status='ready', claim_lock=NULL WHERE id=?", (tid,),
        )
        reclaimed = kb.claim_task(conn, tid, claimer="worker-b")
        run_b = reclaimed.current_run_id
        conn.commit()
        assert run_b != run_a

        result = kb._end_run(conn, tid, outcome="crashed", expected_run_id=run_a)
        assert result is None

        reloaded = kb.get_task(conn, tid)
        assert reloaded.current_run_id == run_b
        run_b_row = conn.execute(
            "SELECT ended_at FROM task_runs WHERE id=?", (run_b,),
        ).fetchone()
        assert run_b_row["ended_at"] is None, "successor run must still be open"
    finally:
        conn.close()
