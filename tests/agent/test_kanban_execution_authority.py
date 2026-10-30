"""Regression suite: trusted Kanban execution identity and lifecycle authority.

Every case runs against a REAL temporary board (schema, claims, run rows, events created
by the kernel) and through the REAL public entrypoints — ``kanban_db``'s lifecycle
mutators, the worker bootstrap, the stop guard and the identity snapshot. No mocked
kernel and no synthesised rows beyond what the dispatcher itself writes (claims and the
worker pid/fingerprint registration).

The coverage categories follow the design's test plan; each test names the category it
covers.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time

import pytest

from agent import kanban_execution_identity as ident
from agent import kanban_stop as stop_guard
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


# --------------------------------------------------------------------------- fixtures

@pytest.fixture()
def board(tmp_path, monkeypatch):
    """A real board in its own HERMES_HOME, with the surrounding worker/env fences cleared."""
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in ("HERMES_KANBAN_DB", "HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID",
                "HERMES_KANBAN_CLAIM_LOCK", "HERMES_KANBAN_BOARD", "HERMES_DELEGATED_CHILD_CONTEXT"):
        monkeypatch.delenv(key, raising=False)
    ident.bind_execution_context(None)
    ident.install_process_identity(None)
    kbc.init_db()
    conn = kbc.connect()
    try:
        yield conn
    finally:
        conn.close()
        ident.bind_execution_context(None)
        ident.install_process_identity(None)


def new_task(conn, title: str = "card") -> str:
    created = kb.create_task(conn, title=title, assignee="coder", created_by="test")
    tid = created.id if hasattr(created, "id") else str(created)
    conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
    return tid


def spawn_worker(conn, task_id: str, *, claimer: str | None = None, review: bool = False) -> dict:
    """Emulate the dispatcher: claim the card, register this process, bootstrap identity."""
    if review:
        claimed = kb.claim_review_task(conn, task_id, ttl_seconds=600, claimer=claimer)
    else:
        claimed = kb.claim_task(conn, task_id, ttl_seconds=600, claimer=claimer)
    assert claimed is not None, "claim failed"
    kbd._set_worker_pid(conn, task_id, os.getpid())
    row = conn.execute(
        "SELECT current_run_id, claim_lock FROM tasks WHERE id = ?", (task_id,)).fetchone()
    env = {
        "HERMES_KANBAN_TASK": task_id,
        "HERMES_KANBAN_RUN_ID": str(row["current_run_id"]),
        "HERMES_KANBAN_CLAIM_LOCK": str(row["claim_lock"]),
        "HERMES_KANBAN_DB": str(kb.kanban_db_path()),
        "HERMES_KANBAN_BOARD": "default",
    }
    ident.bind_execution_context(None)
    ctx = ident.bootstrap_worker_identity(env=env, wait_seconds=0.0)
    return {"env": env, "run_id": int(row["current_run_id"]), "lock": str(row["claim_lock"]), "ctx": ctx}


def snapshot(conn) -> dict:
    """Every durable row a rejected operation must leave untouched."""
    return {
        "tasks": [tuple(r) for r in conn.execute(
            "SELECT id, status, claim_lock, claim_expires, current_run_id, worker_pid, "
            "worker_started_at, assignee, block_kind, block_recurrences FROM tasks ORDER BY id")],
        "runs": [tuple(r) for r in conn.execute(
            "SELECT id, task_id, status, outcome, claim_lock, ended_at FROM task_runs ORDER BY id")],
        "events": [tuple(r) for r in conn.execute(
            "SELECT id, task_id, kind, run_id FROM task_events ORDER BY id")],
    }


def denial(callable_):
    with pytest.raises(ident.KanbanAuthorizationError) as excinfo:
        callable_()
    return excinfo.value


# ----------------------------------------------------------------- 1. OWNER vs child

def test_01_owner_bootstrap_and_scrubbed_child(board, monkeypatch):
    """Category 1: parent stays OWNER, the terminal child stays fenced, context is truthful."""
    tid = new_task(board)
    worker = spawn_worker(board, tid)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)

    assert worker["ctx"].role == ident.ROLE_OWNER
    snap = ident.identity_snapshot(board, target_task_id=tid)
    assert snap["role"] == ident.ROLE_OWNER
    assert snap["claim_match"] is True
    assert snap["owned_task_id"] == tid
    assert "claim_lock" not in snap  # no credentials in model-visible output

    # A terminal descendant carries no ownership pins and stays fenced.
    from agent.delegation_context import delegated_child_subprocess_env

    child_env = delegated_child_subprocess_env(dict(os.environ))
    assert "HERMES_KANBAN_TASK" not in child_env
    assert "HERMES_KANBAN_RUN_ID" not in child_env
    assert "HERMES_KANBAN_CLAIM_LOCK" not in child_env
    assert child_env.get("HERMES_DELEGATED_CHILD_CONTEXT")

    # The in-process child scope reduces authority and restores the parent afterwards.
    with ident.delegate_child_scope():
        assert denial(lambda: kb.block_task(board, tid, reason="child",
                                            expected_run_id=worker["run_id"])).reason == "delegate_child"
    assert ident.current_execution_context().role == ident.ROLE_OWNER


# ------------------------------------------- 2. foreign claim refusal, no side effects

def test_02_foreign_claim_refusal_is_detached_and_side_effect_free(board):
    """Category 2: live owner A + foreign B -> refusal, no nudge, zero mutation."""
    tid = new_task(board)
    worker = spawn_worker(board, tid)
    before = snapshot(board)

    with ident.foreign_inspector_scope():
        err = denial(lambda: kb.block_task(board, tid, reason="foreign",
                                           expected_run_id=worker["run_id"]))
        # The exit guard must not tell a non-owner to hand off.
        assert stop_guard.build_kanban_stop_nudge(messages=[], attempts=0, task_id=tid) is None

    assert err.code == ident.CODE_AUTHORIZATION_DENIED
    assert snapshot(board) == before
    # The foreign scope is gone; the process-wide worker identity is what remains.
    assert ident.current_execution_context().role == ident.ROLE_OWNER


# ------------------------------------- 3-6. foreign lifecycle mutations are rejected

@pytest.mark.parametrize("operation", ["block", "schedule", "request_changes", "request_review"])
def test_03_to_06_foreign_lifecycle_operations_rejected(board, operation):
    """Categories 3-6: ordinary unbound/foreign callers cannot mutate a live run."""
    tid = new_task(board)
    worker = spawn_worker(board, tid)
    before = snapshot(board)

    def attempt():
        if operation == "block":
            kb.block_task(board, tid, reason="foreign", expected_run_id=worker["run_id"])
        elif operation == "schedule":
            kb.schedule_task(board, tid, reason="foreign", expected_run_id=worker["run_id"])
        elif operation == "request_changes":
            kb.request_changes(board, tid, reason="foreign", expected_run_id=worker["run_id"])
        else:
            kb.request_review(board, tid, summary="foreign", expected_run_id=worker["run_id"])

    # A foreign execution owns nothing, so it has no lifecycle authority of its own.
    with ident.foreign_inspector_scope(reason="foreign_inspector"):
        err = denial(attempt)
    assert err.code == ident.CODE_AUTHORIZATION_DENIED
    assert snapshot(board) == before


def test_03b_unbound_caller_is_denied_not_treated_as_admin(board):
    """No identity is NOT administrative authority (no missing-run fallback)."""
    tid = new_task(board)
    worker = spawn_worker(board, tid)
    ident.bind_execution_context(None)
    ident.install_process_identity(None)
    before = snapshot(board)
    err = denial(lambda: kb.block_task(board, tid, reason="unbound", expected_run_id=worker["run_id"]))
    assert err.code == ident.CODE_AUTHORIZATION_DENIED
    assert snapshot(board) == before


# ------------------------------------------ 7. legitimate owner lifecycle operations

def test_07_legitimate_owner_operations_succeed(board):
    """Category 7: the owner may complete/review/block/schedule its own run."""
    tid_block = new_task(board, "block card")
    worker = spawn_worker(board, tid_block)
    assert kb.block_task(board, tid_block, reason="owner block", kind="transient",
                         expected_run_id=worker["run_id"]) is True
    assert board.execute("SELECT status FROM tasks WHERE id = ?", (tid_block,)).fetchone()["status"] == "blocked"

    tid_sched = new_task(board, "schedule card")
    w2 = spawn_worker(board, tid_sched)
    assert kb.schedule_task(board, tid_sched, reason="owner schedule",
                           expected_run_id=w2["run_id"]) is True
    assert board.execute("SELECT status FROM tasks WHERE id = ?", (tid_sched,)).fetchone()["status"] == "scheduled"

    tid_review = new_task(board, "review card")
    w3 = spawn_worker(board, tid_review)
    assert kb.request_review(board, tid_review, summary="owner review", reviewer="reviewer",
                             expected_run_id=w3["run_id"]) is True
    assert board.execute("SELECT status FROM tasks WHERE id = ?", (tid_review,)).fetchone()["status"] == "review"

    tid_done = new_task(board, "done card")
    w4 = spawn_worker(board, tid_done)
    assert kb.complete_task(board, tid_done, summary="owner complete",
                            expected_run_id=w4["run_id"]) is True
    assert board.execute("SELECT status FROM tasks WHERE id = ?", (tid_done,)).fetchone()["status"] == "done"


# ------------------------------------------- 8. same-card reviewer lifecycle succeeds

def test_08_same_card_reviewer_can_request_changes(board):
    """Category 8: a run claimed FROM review is the reviewer and may send the card back."""
    tid = new_task(board)
    implementer = spawn_worker(board, tid, claimer="implementer")
    assert kb.request_review(board, tid, summary="handoff", reviewer="reviewer",
                             expected_run_id=implementer["run_id"]) is True

    reviewer = spawn_worker(board, tid, claimer="reviewer", review=True)
    assert ident.reconcile_owner(board, reviewer["ctx"]).review_source_phase is True
    ok, routed_to = kb.request_changes(board, tid, reason="tighten the probe",
                                       expected_run_id=reviewer["run_id"])
    assert ok is True and routed_to
    handoff = board.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'review_requested' "
        "ORDER BY id DESC LIMIT 1", (tid,)).fetchone()
    assert json.loads(handoff["payload"])["implementer"] == routed_to

    kinds = [r["kind"] for r in board.execute(
        "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (tid,))]
    assert "review_requested" in kinds and "changes_requested" in kinds


def test_08b_non_review_run_cannot_request_changes(board):
    """request_changes from a run NOT claimed from review is refused (phase required)."""
    tid = new_task(board)
    worker = spawn_worker(board, tid)
    err = denial(lambda: kb.request_changes(board, tid, reason="wrong phase",
                                            expected_run_id=worker["run_id"]))
    assert err.code == ident.CODE_REVIEW_PHASE_REQUIRED
    assert board.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] == "running"


# ----------------------------------------------- 9-10. recovery paths and audit trail

def test_09_dead_stale_recovery_still_works_for_authenticated_admin(board):
    """Category 9: an expired/dead claim is recoverable through the admin path."""
    tid = new_task(board)
    spawn_worker(board, tid)
    board.execute("UPDATE tasks SET claim_expires = ? WHERE id = ?", (int(time.time()) - 60, tid))

    with ident.admin_recovery_scope(principal="operator@example", transport="dashboard_session",
                                    action="reclaim", reason="dead worker", board="default"):
        assert kb.reclaim_task(board, tid, reason="dead worker",
                               signal_fn=lambda *a, **k: None) is True

    audit = board.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'admin_recovery' "
        "ORDER BY id DESC LIMIT 1", (tid,)).fetchone()
    payload = json.loads(audit["payload"])
    assert payload["principal"] == "operator@example"
    assert payload["transport"] == "dashboard_session"
    assert payload["operation"] == "reclaim"
    assert payload["task_id"] == tid


def test_10_admin_only_through_authenticated_scope(board):
    """Category 10: reclaim/reassign/unblock need ADMIN_RECOVERY; forged attempts fail."""
    tid = new_task(board)
    worker = spawn_worker(board, tid)
    kb.block_task(board, tid, reason="parked", kind="needs_input", expected_run_id=worker["run_id"])
    before = snapshot(board)

    # A chat/tool caller with no administrative transport cannot unblock it.
    ident.bind_execution_context(None)
    ident.install_process_identity(None)
    assert denial(lambda: kb.unblock_task(board, tid)).code == ident.CODE_AUTHORIZATION_DENIED
    # A delegate child cannot either.
    with ident.delegate_child_scope():
        assert denial(lambda: kb.unblock_task(board, tid)).code == ident.CODE_AUTHORIZATION_DENIED
    assert snapshot(board) == before

    with ident.admin_recovery_scope(principal="operator@example", transport="dashboard_session",
                                    action="unblock", reason="resume"):
        assert kb.unblock_task(board, tid) is True
    assert board.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] in (
        "ready", "todo", "review")


def test_10b_admin_mint_requires_verified_provenance(board):
    """An administrative scope cannot be minted without a principal and transport."""
    with pytest.raises(ident.KanbanAuthorizationError):
        with ident.admin_recovery_scope(principal="", transport="dashboard_session"):
            pass


def test_10c_admin_scope_is_not_inherited_across_threads(board):
    """A request-scoped grant belongs to its caller, never to the process.

    The dashboard process also runs ordinary chat turns on its own threads
    (``tui-rpc`` -> ``run_conversation``).  Nothing binds a Kanban identity for
    those, so they must stay detached (a) while an authenticated administrative
    request is in flight and (b) after two overlapping requests exit out of
    order.  Every refused write leaves the board untouched; the same write from
    an authenticated administrative scope still succeeds, so the denials prove
    the scope is the discriminator and not a broken mutator.
    """
    tid = new_task(board)
    ident.bind_execution_context(None)
    ident.install_process_identity(None)
    before = snapshot(board)

    def unbound_block(label: str) -> object:
        """Attempt a lifecycle write from a caller that bound no identity."""
        conn = kbc.connect()
        try:
            return kb.block_task(conn, tid, reason=label)
        except ident.KanbanAuthorizationError as exc:
            return exc
        finally:
            conn.close()

    def run_in_thread(target) -> tuple[threading.Thread, dict]:
        box: dict[str, object] = {}

        def wrapper():
            box["result"] = target()
        thread = threading.Thread(target=wrapper)
        thread.start()
        return thread, box

    # (a) another thread writes while one authenticated request holds the scope.
    in_flight = threading.Event()
    release = threading.Event()

    def admin_request_in_flight():
        with ident.admin_recovery_scope(principal="alice@example", transport="dashboard_session"):
            in_flight.set()
            assert release.wait(20), "probe never finished"

    holder = threading.Thread(target=admin_request_in_flight)
    holder.start()
    assert in_flight.wait(20), "administrative scope never entered"
    probe, probe_box = run_in_thread(lambda: unbound_block("chat turn while admin in flight"))
    probe.join(20)
    release.set()
    holder.join(20)
    assert not probe.is_alive() and not holder.is_alive()
    assert isinstance(probe_box["result"], ident.KanbanAuthorizationError)
    assert probe_box["result"].code == ident.CODE_AUTHORIZATION_DENIED
    assert snapshot(board) == before

    # (b) two overlapping requests, A entering before B and A exiting before B.
    a_in, b_in, a_out = threading.Event(), threading.Event(), threading.Event()
    exits: list[str] = []

    def request_a():
        with ident.admin_recovery_scope(principal="alice@example", transport="dashboard_session"):
            a_in.set()
            assert b_in.wait(20), "request B never entered"
        exits.append("a")
        a_out.set()  # A has left its scope while B is still inside its own

    def request_b():
        assert a_in.wait(20), "request A never entered"
        with ident.admin_recovery_scope(principal="bob@example", transport="dashboard_token"):
            b_in.set()
            assert a_out.wait(20), "request A never exited"
        exits.append("b")

    thread_a = threading.Thread(target=request_a)
    thread_b = threading.Thread(target=request_b)
    thread_a.start()
    thread_b.start()
    thread_a.join(20)
    thread_b.join(20)
    assert not thread_a.is_alive() and not thread_b.is_alive()
    assert exits == ["a", "b"]

    # Nothing was left behind for a caller that binds nothing.
    assert ident.process_identity() is None
    after, after_box = run_in_thread(lambda: unbound_block("chat turn after overlapping requests"))
    after.join(20)
    assert not after.is_alive()
    assert isinstance(after_box["result"], ident.KanbanAuthorizationError)
    assert after_box["result"].code == ident.CODE_AUTHORIZATION_DENIED
    assert snapshot(board) == before

    # The same write IS authorized when an authenticated request makes it.
    with ident.admin_recovery_scope(principal="alice@example", transport="dashboard_session",
                                    action="block", reason="operator park"):
        assert kb.block_task(board, tid, reason="operator park") is True
    assert board.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] == "blocked"


# -------------------------------------------------- 11. race orderings, no sleeps

@pytest.mark.parametrize("owner_first", [True, False])
def test_11_concurrent_completion_and_foreign_write(board, owner_first):
    """Category 11: barriers + separate connections; the foreign writer never wins."""
    tid = new_task(board)
    worker = spawn_worker(board, tid)
    barrier = threading.Barrier(2, timeout=10)
    outcome: dict[str, object] = {}

    def owner_side():
        conn = kbc.connect()
        ident.bind_execution_context(worker["ctx"])
        barrier.wait()
        try:
            outcome["owner"] = kb.complete_task(
                conn, tid, summary="race completion", expected_run_id=worker["run_id"])
        except Exception as exc:
            outcome["owner"] = exc
        finally:
            conn.close()

    def foreign_side():
        conn = kbc.connect()
        with ident.foreign_inspector_scope(reason="race foreign"):
            barrier.wait()
            try:
                outcome["foreign"] = kb.block_task(
                    conn, tid, reason="race foreign", expected_run_id=worker["run_id"])
            except Exception as exc:
                outcome["foreign"] = exc
        conn.close()

    first, second = (owner_side, foreign_side) if owner_first else (foreign_side, owner_side)
    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)

    assert isinstance(outcome["foreign"], ident.KanbanAuthorizationError)
    assert outcome["owner"] is True
    ident.bind_execution_context(None)
    assert board.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] == "done"
    blocked = board.execute(
        "SELECT COUNT(*) c FROM task_events WHERE task_id = ? AND kind LIKE '%block%'", (tid,)).fetchone()["c"]
    assert blocked == 0


# --------------------------------- 12. successor runs, recycled pids, expired leases

def test_12_stale_owner_cannot_mutate_successor(board):
    """Category 12: old task/run/lock pins never touch the successor's claim."""
    tid = new_task(board)
    old = spawn_worker(board, tid)
    with ident.admin_recovery_scope(principal="operator@example", transport="dashboard_session",
                                    action="reclaim", reason="stale"):
        assert kb.reclaim_task(board, tid, reason="stale", signal_fn=lambda *a, **k: None) is True
    successor = spawn_worker(board, tid)
    assert successor["run_id"] != old["run_id"]

    ident.bind_execution_context(old["ctx"])
    before = snapshot(board)
    err = denial(lambda: kb.request_review(board, tid, summary="stale", expected_run_id=old["run_id"]))
    assert err.code == ident.CODE_OWNERSHIP_LOST
    assert snapshot(board) == before


def test_12b_recycled_pid_fingerprint_is_refused(board):
    """A matching PID with a changed boot/start fingerprint is not the worker."""
    tid = new_task(board)
    worker = spawn_worker(board, tid)
    board.execute("UPDATE tasks SET worker_started_at = '0|deadbeef' WHERE id = ?", (tid,))
    ident.bind_execution_context(worker["ctx"])
    err = denial(lambda: kb.block_task(board, tid, reason="recycled", expected_run_id=worker["run_id"]))
    assert err.code == ident.CODE_OWNERSHIP_LOST


def test_12c_expired_lease_refuses_writes(board):
    """A lapsed lease means detached, not permission (uniform ``expiry < now`` boundary)."""
    tid = new_task(board)
    worker = spawn_worker(board, tid)
    board.execute("UPDATE tasks SET claim_expires = ? WHERE id = ?", (int(time.time()) - 1, tid))
    ident.bind_execution_context(worker["ctx"])
    before = snapshot(board)
    err = denial(lambda: kb.schedule_task(board, tid, reason="expired", expected_run_id=worker["run_id"]))
    assert err.code in (ident.CODE_OWNERSHIP_LOST, ident.CODE_IDENTITY_UNAVAILABLE)
    assert snapshot(board) == before


# ------------------------------------------------ 13. incident-shaped transcript

def test_13_incident_shape_keeps_parent_owner_and_only_nudges_a_live_owner(board, monkeypatch):
    """Category 13: scrubbed shell output still resolves the parent as OWNER."""
    tid = new_task(board)
    worker = spawn_worker(board, tid)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)

    scrubbed = {k: v for k, v in os.environ.items()
                if k not in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_CLAIM_LOCK")}
    assert "HERMES_KANBAN_TASK" not in scrubbed

    snap = ident.identity_snapshot(board, target_task_id=tid)
    assert snap["role"] == ident.ROLE_OWNER and snap["claim_match"] is True

    failed = [
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "kanban_complete"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "kanban_complete",
         "content": "kanban_complete refused: this worker no longer owns the live run"},
    ]
    assert stop_guard.session_called_kanban_terminal(failed) is False
    # A lifecycle call refused through `tool_error` JSON is a refusal too, and a real
    # success envelope is a handoff: the classifier must separate the two shapes.
    denied_json = [
        {"role": "assistant", "tool_calls": [{"id": "c2", "function": {"name": "kanban_block"}}]},
        {"role": "tool", "tool_call_id": "c2", "name": "kanban_block",
         "content": json.dumps({"error": "kanban_block: authorization_denied: no trusted Kanban "
                                         "execution identity for 'block'"})},
    ]
    assert stop_guard.session_called_kanban_terminal(denied_json) is False
    succeeded = [
        {"role": "assistant", "tool_calls": [{"id": "c3", "function": {"name": "kanban_block"}}]},
        {"role": "tool", "tool_call_id": "c3", "name": "kanban_block",
         "content": json.dumps({"ok": True, "task_id": tid, "run_id": 1, "status": "blocked"})},
    ]
    assert stop_guard.session_called_kanban_terminal(succeeded) is True
    nudge = stop_guard.build_kanban_stop_nudge(messages=failed, attempts=0, task_id=tid)
    assert nudge and "does NOT mean you lost ownership" in nudge

    with ident.admin_recovery_scope(principal="operator@example", transport="dashboard_session",
                                    action="reclaim", reason="detach"):
        kb.reclaim_task(board, tid, reason="detach", signal_fn=lambda *a, **k: None)
    ident.bind_execution_context(worker["ctx"])
    assert stop_guard.build_kanban_stop_nudge(messages=failed, attempts=0, task_id=tid) is None


# ----------------------------------------- read-only inspection and board isolation

def test_observational_reads_do_not_create_write_or_promote(tmp_path, monkeypatch):
    """list/show open an existing board read-only: no create, no readiness/event writes."""
    home = tmp_path / "empty-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)

    with pytest.raises(kbc.KanbanDbUnavailable):
        kbc.connect_readonly()
    assert not kb.kanban_db_path().exists()  # a read must not mint a board

    kbc.init_db()
    conn = kbc.connect()
    try:
        tid = new_task(conn, "pending card")
        conn.execute(
            "INSERT INTO task_links (parent_id, child_id) SELECT ?, ? WHERE 0", (tid, tid))
        conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (tid,))
        before = snapshot(conn)
        with kbc.connect_readonly_closing() as ro:
            assert kb.list_tasks(ro) is not None
        assert snapshot(conn) == before            # no readiness promotion
        assert conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] == "todo"
    finally:
        conn.close()


def test_board_isolation_board_grant_cannot_write_another_board(tmp_path, monkeypatch):
    """A worker's grant is scoped to its own canonical board DB."""
    home = tmp_path / "iso-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in ("HERMES_KANBAN_DB", "HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID",
                "HERMES_KANBAN_CLAIM_LOCK", "HERMES_DELEGATED_CHILD_CONTEXT"):
        monkeypatch.delenv(key, raising=False)

    kbc.init_db()
    board_conn = kbc.connect()
    try:
        tid = new_task(board_conn, "board A card")
        worker = spawn_worker(board_conn, tid)
    finally:
        board_conn.close()

    # A second, independent board DB created through the same real kernel.
    other_root = tmp_path / "other-root"
    other_root.mkdir()
    other_db = other_root / "kanban.db"
    kbc.init_db(other_db)
    other = kbc.connect(other_db)
    try:
        created = kb.create_task(other, title="board B card", assignee="coder", created_by="test")
        other_tid = created.id if hasattr(created, "id") else str(created)
        ident.bind_execution_context(worker["ctx"])
        before = snapshot(other)
        err = denial(lambda: kb.block_task(other, other_tid, reason="cross board"))
        assert err.reason == "db_mismatch"
        assert snapshot(other) == before
    finally:
        other.close()


# ------------------------ 14. a refused CLI write leaves no reason comment behind

def _comment_rows(conn) -> list[tuple]:
    return [tuple(r) for r in conn.execute(
        "SELECT id, task_id, author, body FROM task_comments ORDER BY id")]


def _event_rows(conn) -> list[tuple]:
    return [tuple(r) for r in conn.execute(
        "SELECT id, task_id, kind, run_id FROM task_events ORDER BY id")]


def run_kanban_cli(argv: list[str]) -> tuple[int, str, str]:
    """Drive the real ``hermes kanban`` entry: its argparse tree plus ``kanban_command``.

    The same pair the shell command uses, so the path exercised here is the production one
    and the exit code / stderr are exactly what an unbound CLI caller would see.
    """
    import argparse
    import contextlib
    import io

    from hermes_cli import kanban as kc

    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    kanban_parser = kc.build_parser(parser.add_subparsers(dest="_top"))
    args = kanban_parser.parse_args(argv)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = kc.kanban_command(args)
    return int(rc or 0), out.getvalue(), err.getvalue()


def test_14_refused_cli_lifecycle_write_records_no_reason_comment(board):
    """Category 14: a denial is total — no status/claim/run change, and no reason comment.

    A refused ``hermes kanban block <tid> 'probe'`` used to commit ``BLOCKED: probe`` as a
    comment plus a ``commented`` event: the CLI wrapper recorded the reason *before* the
    authorized mutation. With the ordering fixed, a refused call that carries a reason must be
    indistinguishable from one that carries none.
    """
    tid = new_task(board)
    ident.bind_execution_context(None)
    ident.install_process_identity(None)
    tasks_before = snapshot(board)
    events_before = _event_rows(board)
    comments_before = _comment_rows(board)

    for argv in (["block", tid, "foreign", "residue", "probe"],
                 ["schedule", tid, "foreign", "residue", "probe"],
                 ["request-changes", tid, "foreign", "residue", "probe"],
                 ["unblock", "--reason", "foreign residue probe", tid]):
        rc, out, err = run_kanban_cli(argv)
        assert rc == 1, (argv, out, err)
        assert "no trusted Kanban execution identity" in err, (argv, err)
        assert _comment_rows(board) == comments_before, argv
        assert _event_rows(board) == events_before, argv
        assert snapshot(board) == tasks_before, argv

    assert board.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] == "ready"


@pytest.mark.parametrize("verb", ["block", "schedule"])
def test_14b_owner_cli_write_still_records_its_reason_comment(board, monkeypatch, verb):
    """Positive control: through the same CLI path, the authorizing owner still records it.

    Every ``_commented`` caller is exercised, so the refusal above cannot be satisfied by
    dropping reason comments altogether instead of ordering them after the authorized
    mutation.
    """
    tid = new_task(board)
    worker = spawn_worker(board, tid)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(worker["run_id"]))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", worker["lock"])
    ident.bind_execution_context(worker["ctx"])
    comments_before = len(_comment_rows(board))

    rc, out, err = run_kanban_cli([verb, tid, "owner", "parked", "here"])
    assert rc == 0, (out, err)
    landed = board.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"]
    assert landed == ("blocked" if verb == "block" else "scheduled")
    prefix = "BLOCKED" if verb == "block" else "SCHEDULED"
    assert [row[3] for row in _comment_rows(board)[comments_before:]] == [f"{prefix}: owner parked here"]
