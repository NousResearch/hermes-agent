"""Managed Kanban execution: the adopted exec's scoped ``OWNER`` and its enforcement.

The upstream dispatcher no longer runs a worker in the spawned process: it submits the
claim to the profile owner, which adopts a private ``agent.managed_worker`` interpreter
to execute the turn.  The task/run process fields are the spawned *submitter*'s there,
so the executing process is proven by the ``worker_bound`` event the adopted exec
appends transactionally, and the grant is scoped inside that interpreter only.

These cases cover the issuer and its enforcement.  The real end-to-end proof (actual
dispatcher argv -> owner -> native ingress -> private reservation/adoption -> real exec,
owning and handing off its card, with a scrubbed terminal child) is
``tests/gateway/test_session_kanban.py::test_real_dispatcher_lifecycle``.
"""

from __future__ import annotations

import contextvars
import json
import os
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from agent import kanban_execution_identity as ident
from agent import kanban_stop as stop_guard
from gateway import session_kanban
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


# --------------------------------------------------------------------------- fixtures

@pytest.fixture()
def board(tmp_path, monkeypatch):
    """A real board in its own HERMES_HOME; the submitter watcher and env are fenced."""
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in ("HERMES_KANBAN_DB", "HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID",
                "HERMES_KANBAN_CLAIM_LOCK", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_WORKSPACE",
                "HERMES_DELEGATED_CHILD_CONTEXT"):
        monkeypatch.delenv(key, raising=False)
    # ``os.environ``/cwd are written by the production bootstrap; restore them wholesale so
    # one case cannot leak a board pin (or a workspace cwd) into the next.
    saved_env, saved_cwd = dict(os.environ), os.getcwd()
    monkeypatch.setattr(session_kanban, "_watch_submitter", lambda *a, **k: None)
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
        os.environ.clear()
        os.environ.update(saved_env)
        os.chdir(saved_cwd)


def new_task(conn, assignee: str = "default", title: str = "card") -> str:
    created = kb.create_task(conn, title=title, assignee=assignee, created_by="test")
    tid = created.id if hasattr(created, "id") else str(created)
    conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
    return tid


def spawn_submitter(conn, task_id: str, *, review: bool = False, pid: int = 424242) -> dict:
    """Emulate the dispatcher: claim the card, then record its spawned submitter.

    ``_set_worker_pid`` is the production write (task and open run row, one transaction):
    a dead pid here stores the ``unverified`` fingerprint, which is exactly what the
    dispatcher records when capture fails, and it keeps the submitter distinct from the
    executing interpreter this suite runs in.
    """
    claimed = kb.claim_review_task(conn, task_id, ttl_seconds=600) if review else kb.claim_task(conn, task_id, ttl_seconds=600)
    assert claimed is not None, "claim failed"
    kbd._set_worker_pid(conn, task_id, pid)
    row = conn.execute("SELECT current_run_id, claim_lock FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return {"run_id": int(row["current_run_id"]), "lock": str(row["claim_lock"]), "submitter_pid": pid}


def _context(conn, task_id: str, home: Path, **over) -> dict:
    row = conn.execute("SELECT current_run_id, claim_lock FROM tasks WHERE id = ?", (task_id,)).fetchone()
    workspace = Path(home) / "workspace"
    workspace.mkdir(exist_ok=True)
    context = {
        "board": "default", "task_id": task_id,
        "run_id": int(row["current_run_id"]) if row["current_run_id"] is not None else -1,
        "claim_lock": str(row["claim_lock"] or ""), "db": str(kb.kanban_db_path()),
        "workspace": str(workspace), "workspaces_root": str(Path(home) / "workspaces"),
        "profile": "default", "branch": None, "tenant": None, "goal_mode": False,
        "max_runtime_seconds": None,
    }
    context.update(over)
    return context


def managed_frame(home: Path, context: dict) -> dict:
    """The private bootstrap frame the owner sends to an adopted exec."""
    import psutil
    from hermes_cli.process_identity import worker_birth
    return {
        "home": str(home),
        "policy": {"source": "kanban", "kanban_json": json.dumps(context)},
        "scope": {"profile_id": str(home), "session_id": "session-under-test",
                  "execution_id": "admission-worker:test", "generation": 2,
                  "pid": os.getpid(), "birth": worker_birth(psutil.Process()),
                  "secret": "s" * 32, "epoch": 7},
    }


def adoption(frame: dict) -> dict:
    """What ``worker.adopt`` returns for this frame (the issuer's precondition)."""
    scope = frame["scope"]
    return {"execution_id": scope["execution_id"], "session_id": scope["session_id"],
            "kind": "compute", "owner_epoch": scope["epoch"], "generation": scope["generation"],
            "status": "running"}


def bind(conn, task_id: str, home: Path, **context_over):
    """Bind a managed grant through the production issuer against a real board."""
    context = _context(conn, task_id, home, **context_over)
    frame = managed_frame(home, context)
    return session_kanban.bind_worker_context(frame, adoption(frame), wait_seconds=0.0), frame, context


def snapshot(conn) -> dict:
    return {
        "tasks": [tuple(r) for r in conn.execute(
            "SELECT id, status, claim_lock, claim_expires, current_run_id, worker_pid, "
            "worker_started_at, assignee, block_kind FROM tasks ORDER BY id")],
        "runs": [tuple(r) for r in conn.execute(
            "SELECT id, task_id, status, outcome, claim_lock, ended_at FROM task_runs ORDER BY id")],
        "events": [tuple(r) for r in conn.execute(
            "SELECT id, task_id, kind, run_id FROM task_events ORDER BY id")],
    }


def denial(callable_):
    with pytest.raises(ident.KanbanAuthorizationError) as excinfo:
        callable_()
    return excinfo.value


def home_of(board) -> Path:
    return Path(os.environ["HERMES_HOME"])


# ------------------------------------------- 2. no managed turn acquires OWNER by accident

def test_ordinary_managed_turn_scopes_a_detached_role(board):
    """A non-Kanban managed turn scopes FOREIGN_INSPECTOR and can never mutate a card."""
    tid = new_task(board)
    spawn_submitter(board, tid)
    before = snapshot(board)
    with session_kanban.managed_execution_scope(None) as scoped:
        assert scoped.role == ident.ROLE_FOREIGN_INSPECTOR
        err = denial(lambda: kb.block_task(board, tid, reason="ordinary turn"))
    assert err.code == ident.CODE_AUTHORIZATION_DENIED
    assert snapshot(board) == before
    # Scoping is context-local: nothing was published process-wide, and it did not outlive
    # the ``with`` (an unrelated thread of the same process would inherit otherwise).
    assert ident.process_identity() is None
    assert ident.current_execution_context() is None


@pytest.mark.parametrize("over", [
    {"status": "registered"},
    {"status": "terminal"},
    {"owner_epoch": 6},
    {"execution_id": "admission-worker:other"},
    {"generation": 1},
])
def test_unverified_adoption_never_mints_owner(board, monkeypatch, over):
    """Only a successful adoption of THIS scope mints a grant; anything else is refused."""
    home = home_of(board)
    tid = new_task(board)
    spawn_submitter(board, tid)
    context = _context(board, tid, home)
    frame = managed_frame(home, context)
    adopted = {**adoption(frame), **over}
    before = snapshot(board)
    with pytest.raises(session_kanban.RuntimeStoreError):
        session_kanban.bind_worker_context(frame, adopted, wait_seconds=0.0)
    assert snapshot(board) == before            # no worker_bound, no claim change
    assert ident.process_identity() is None     # nothing bound, nothing process-wide


def test_no_adoption_result_and_recycled_birth_are_refused(board):
    """A hand-built frame cannot stand in for the owner's verified adoption."""
    home = home_of(board)
    tid = new_task(board)
    spawn_submitter(board, tid)
    context = _context(board, tid, home)
    frame = managed_frame(home, context)
    before = snapshot(board)
    with pytest.raises(session_kanban.RuntimeStoreError):
        session_kanban.bind_worker_context(frame, None, wait_seconds=0.0)
    with pytest.raises(session_kanban.RuntimeStoreError):
        session_kanban.bind_worker_context(
            {**frame, "scope": {**frame["scope"], "birth": frame["scope"]["birth"] + 1}},
            adoption(frame), wait_seconds=0.0)
    with pytest.raises(session_kanban.RuntimeStoreError):
        session_kanban.bind_worker_context(
            {**frame, "scope": {**frame["scope"], "pid": os.getpid() + 1}},
            adoption(frame), wait_seconds=0.0)
    assert snapshot(board) == before


def test_non_kanban_frame_returns_no_grant_and_foreign_policy_is_refused(board):
    home = home_of(board)
    frame = managed_frame(home, _context(board, new_task(board), home))
    assert session_kanban.bind_worker_context(
        {**frame, "policy": {"source": "cli", "kanban_json": "null"}}, adoption(frame)) is None
    with pytest.raises(ValueError):
        session_kanban.bind_worker_context(
            {**frame, "policy": {**frame["policy"], "source": "cli"}}, adoption(frame))


@pytest.mark.parametrize("over", [
    {"claim_lock": "someone-elses-lock"},
    {"board": ""},
    {"profile": "somebody-else"},
])
def test_binding_coordinates_deny_before_the_binding_event(board, over):
    """Wrong board/lock/profile: refused, and not one ``worker_bound`` was appended."""
    home = home_of(board)
    tid = new_task(board)
    spawn_submitter(board, tid)
    before = snapshot(board)
    with pytest.raises(session_kanban.RuntimeStoreError):
        bind(board, tid, home, **over)
    assert snapshot(board) == before


def test_another_boards_file_is_refused_at_binding(board, tmp_path):
    """A frame naming a different board DB binds nothing (no cross-board grant)."""
    home = home_of(board)
    tid = new_task(board)
    spawn_submitter(board, tid)
    other_db = tmp_path / "other-board" / "kanban.db"
    kbc.init_db(other_db)
    before = snapshot(board)
    with pytest.raises(session_kanban.RuntimeStoreError):
        bind(board, tid, home, db=str(other_db))
    assert snapshot(board) == before


def test_wrong_run_and_expired_leases_deny_before_the_binding_event(board):
    home = home_of(board)
    tid = new_task(board)
    other = new_task(board, title="other")
    worker = spawn_submitter(board, tid)
    spawn_submitter(board, other)
    stale_run = _context(board, other, home)["run_id"]
    before = snapshot(board)
    with pytest.raises(session_kanban.RuntimeStoreError):
        bind(board, tid, home, run_id=stale_run)
    assert snapshot(board) == before

    board.execute("UPDATE tasks SET claim_expires = ? WHERE id = ?", (int(time.time()) - 1, tid))
    board.commit()
    before = snapshot(board)
    with pytest.raises(session_kanban.RuntimeStoreError):
        bind(board, tid, home)
    assert snapshot(board) == before
    assert board.execute("SELECT claim_lock FROM tasks WHERE id = ?", (tid,)).fetchone()["claim_lock"] == worker["lock"]


def test_a_live_conflicting_binding_is_never_overwritten(board):
    """A run bound to a still-live interpreter refuses a second, caller-coordinated one."""
    home = home_of(board)
    tid = new_task(board)
    worker = spawn_submitter(board, tid)
    from hermes_cli.kanban_db_dispatch import _process_fingerprint
    # A live holder that is not this claim: this process, this fingerprint, a foreign lock.
    kb._append_event(board, tid, "worker_bound",
                     {"pid": os.getpid(), "claim_lock": "foreign-lock",
                      "started_at": _process_fingerprint(os.getpid())},
                     run_id=worker["run_id"])
    board.commit()
    before = snapshot(board)
    with pytest.raises(session_kanban.RuntimeStoreError):
        bind(board, tid, home)
    assert snapshot(board) == before


def test_a_gone_interpreter_binding_is_superseded_by_the_fresh_exec(board):
    """A crash/restart leaves the previous attempt's binding: the fresh interpreter binds."""
    home = home_of(board)
    tid = new_task(board)
    worker = spawn_submitter(board, tid)
    kb._append_event(board, tid, "worker_bound",
                     {"pid": 999999, "claim_lock": worker["lock"], "started_at": "boot|other"},
                     run_id=worker["run_id"])
    board.commit()
    candidate, _frame, _context_map = bind(board, tid, home)
    bindings = [json.loads(r["payload"]) for r in board.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'worker_bound' ORDER BY id", (tid,))]
    assert [b["pid"] for b in bindings] == [999999, os.getpid()]     # superseded, not overwritten
    with session_kanban.managed_execution_scope(candidate):
        assert ident.owner_is_current() is True


def test_unregistered_submitter_fails_closed_after_the_bounded_wait(board, monkeypatch):
    """A claim whose spawned submitter was never recorded grants nothing (no relaxed proof)."""
    home = home_of(board)
    tid = new_task(board)
    claimed = kb.claim_task(board, tid, ttl_seconds=600)
    assert claimed is not None    # claimed, but ``_set_worker_pid`` never ran
    before = snapshot(board)
    started = time.monotonic()
    with pytest.raises(session_kanban.RuntimeStoreError):
        bind(board, tid, home)
    assert time.monotonic() - started < 5.0     # bounded: wait_seconds=0 in this suite
    assert snapshot(board) == before


# ------------------------------- 3. lifecycle enforcement against managed evidence

def test_live_managed_owner_can_hand_off_its_own_card(board):
    """Positive control: the bound grant authorizes the owner's own lifecycle write."""
    home = home_of(board)
    tid = new_task(board)
    worker = spawn_submitter(board, tid)
    candidate, _frame, _context_map = bind(board, tid, home)
    assert candidate.role == ident.ROLE_OWNER and candidate.provenance == ident.PROVENANCE_MANAGED_BOUND
    assert candidate.submitter_pid == worker["submitter_pid"] != os.getpid()
    with session_kanban.managed_execution_scope(candidate):
        assert kb.block_task(board, tid, reason="managed owner handoff", kind="transient") is True
    assert board.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] == "blocked"


@pytest.mark.parametrize("corruption", [
    "worker_bound_missing",
    "worker_bound_malformed",
    "worker_bound_conflicting",
    "submitter_pid_changed",
    "submitter_fingerprint_changed",
    "executing_fingerprint_changed",
    "lease_expired",
    "run_closed",
])
def test_stale_managed_evidence_denies_with_zero_residue(board, corruption):
    """Every way the managed proof can go stale is a denial, never a fallback."""
    home = home_of(board)
    tid = new_task(board)
    worker = spawn_submitter(board, tid)
    candidate, _frame, _context_map = bind(board, tid, home)

    def rewrite_binding(payload):
        board.execute("DELETE FROM task_events WHERE task_id = ? AND kind = 'worker_bound'", (tid,))
        kb._append_event(board, tid, "worker_bound", payload, run_id=worker["run_id"])

    corruptions = {
        "worker_bound_missing": lambda: board.execute(
            "DELETE FROM task_events WHERE task_id = ? AND kind = 'worker_bound'", (tid,)),
        "worker_bound_malformed": lambda: rewrite_binding(
            {"pid": str(os.getpid()), "claim_lock": worker["lock"], "started_at": candidate.worker_started_at}),
        "worker_bound_conflicting": lambda: rewrite_binding(
            {"pid": os.getpid() + 1, "claim_lock": worker["lock"], "started_at": candidate.worker_started_at}),
        "submitter_pid_changed": lambda: board.execute(
            "UPDATE tasks SET worker_pid = ? WHERE id = ?", (worker["submitter_pid"] + 1, tid)),
        "submitter_fingerprint_changed": lambda: board.execute(
            "UPDATE tasks SET worker_started_at = 'boot|other' WHERE id = ?", (tid,)),
        "executing_fingerprint_changed": lambda: ident.bind_execution_context(
            replace(candidate, worker_started_at="boot|other")),
        "lease_expired": lambda: board.execute(
            "UPDATE task_runs SET claim_expires = ? WHERE id = ?", (int(time.time()) - 1, worker["run_id"])),
        "run_closed": lambda: board.execute(
            "UPDATE task_runs SET ended_at = ? WHERE id = ?", (int(time.time()), worker["run_id"])),
    }
    corruptions[corruption]()
    board.commit()
    candidate = ident.current_execution_context() if corruption == "executing_fingerprint_changed" else candidate
    before = snapshot(board)
    # A lapsed lease is a write refusal, not a lost identity: the exit guard answers only
    # "does this execution still own the claim", so every identity loss must detach it.
    detached = corruption != "lease_expired"
    with session_kanban.managed_execution_scope(candidate):
        err = denial(lambda: kb.block_task(board, tid, reason="stale managed write"))
        if detached:
            assert stop_guard.build_kanban_stop_nudge(messages=[], attempts=0, task_id=tid) is None
    assert err.code in (ident.CODE_OWNERSHIP_LOST, ident.CODE_IDENTITY_UNAVAILABLE)
    assert snapshot(board) == before
    assert board.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] == "running"


def test_successor_run_refuses_the_previous_managed_grant(board):
    """After reclaim the copied coordinates of the old attempt cannot touch the successor."""
    home = home_of(board)
    tid = new_task(board)
    first = spawn_submitter(board, tid)
    candidate, _frame, _context_map = bind(board, tid, home)
    with ident.admin_recovery_scope(principal="operator@example", transport="dashboard_session",
                                    action="reclaim", reason="superseded"):
        assert kb.reclaim_task(board, tid, reason="superseded", signal_fn=lambda *a, **k: None) is True
    successor = spawn_submitter(board, tid)
    assert successor["run_id"] != first["run_id"]
    before = snapshot(board)
    with session_kanban.managed_execution_scope(candidate):
        err = denial(lambda: kb.request_review(board, tid, summary="stale", expected_run_id=first["run_id"]))
    assert err.code == ident.CODE_IDENTITY_UNAVAILABLE or err.code == ident.CODE_OWNERSHIP_LOST
    assert snapshot(board) == before


# ------------------------------------- 4. deterministic orderings, separate connections

def test_binding_then_reclaim_detaches_the_grant(board):
    home = home_of(board)
    tid = new_task(board)
    spawn_submitter(board, tid)
    candidate, _frame, context = bind(board, tid, home)
    with ident.admin_recovery_scope(principal="operator@example", transport="dashboard_session",
                                    action="reclaim", reason="after binding"):
        assert kb.reclaim_task(board, tid, reason="after binding", signal_fn=lambda *a, **k: None) is True
    with session_kanban.managed_execution_scope(candidate):
        assert session_kanban._bound_claim_is_live(context) is False
        err = denial(lambda: kb.complete_task(board, tid, summary="too late"))
    assert err.code in (ident.CODE_OWNERSHIP_LOST, ident.CODE_IDENTITY_UNAVAILABLE)
    assert board.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] != "done"


def test_reclaim_then_binding_raises_without_residue(board):
    home = home_of(board)
    tid = new_task(board)
    spawn_submitter(board, tid)
    with ident.admin_recovery_scope(principal="operator@example", transport="dashboard_session",
                                    action="reclaim", reason="before binding"):
        assert kb.reclaim_task(board, tid, reason="before binding", signal_fn=lambda *a, **k: None) is True
    before = snapshot(board)
    with pytest.raises(session_kanban.RuntimeStoreError):
        bind(board, tid, home)
    assert snapshot(board) == before


def test_concurrent_binding_and_reclaim_leave_a_consistent_board(board):
    """Barrier + separate connections: whichever transaction wins, the board stays coherent."""
    home = home_of(board)
    tid = new_task(board)
    spawn_submitter(board, tid)
    context = _context(board, tid, home)
    frame = managed_frame(home, context)
    barrier = threading.Barrier(2, timeout=20)
    outcome: dict[str, object] = {}

    def binder():
        barrier.wait()
        try:
            outcome["bind"] = session_kanban.bind_worker_context(frame, adoption(frame), wait_seconds=0.0)
        except Exception as exc:
            outcome["bind"] = exc

    def reclaimer():
        from contextlib import closing
        conn = kbc.connect()
        try:
            with ident.admin_recovery_scope(principal="operator@example", transport="dashboard_session",
                                            action="reclaim", reason="race"):
                barrier.wait()
                try:
                    outcome["reclaim"] = kb.reclaim_task(conn, tid, reason="race", signal_fn=lambda *a, **k: None)
                except Exception as exc:
                    outcome["reclaim"] = exc
        finally:
            conn.close()

    threads = [threading.Thread(target=binder), threading.Thread(target=reclaimer)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert all(not thread.is_alive() for thread in threads)

    task = board.execute("SELECT status, current_run_id, claim_lock FROM tasks WHERE id = ?", (tid,)).fetchone()
    bound = board.execute("SELECT COUNT(*) c FROM task_events WHERE task_id = ? AND kind = 'worker_bound'",
                          (tid,)).fetchone()["c"]
    if isinstance(outcome["bind"], Exception):
        assert bound == 0                       # refused binding appended nothing
        assert task["status"] != "running"
    else:
        assert bound == 1                       # bound first; the reclaim then closed the run
        assert task["status"] != "running"
        with session_kanban.managed_execution_scope(outcome["bind"]):
            assert session_kanban._bound_claim_is_live(context) is False


# ------------------------------------- 5. scope propagation, shadowing and isolation

def test_scope_reaches_copied_tool_threads_and_no_unrelated_thread(board):
    home = home_of(board)
    tid = new_task(board)
    spawn_submitter(board, tid)
    candidate, _frame, _context_map = bind(board, tid, home)
    seen: dict[str, object] = {}

    with session_kanban.managed_execution_scope(candidate):
        # The tool executor submits through ``contextvars.copy_context()``: that thread
        # must be the owner, and a lifecycle write it makes must land.
        def copied():
            from contextlib import closing
            conn = kbc.connect()
            try:
                seen["role"] = ident.current_execution_context().role
                seen["block"] = kb.block_task(conn, tid, reason="from the tool thread")
            finally:
                conn.close()

        contextvars.copy_context().run(copied)
        thread_box: dict[str, object] = {}

        def plain():
            thread_box["ctx"] = ident.current_execution_context()
            thread_box["owner"] = ident.owner_is_current()

        thread = threading.Thread(target=plain)
        thread.start()
        thread.join(20)
        assert ident.process_identity() is None      # never installed process-wide
        # A delegate child inside the same scope is shadowed, and the parent is restored.
        with ident.delegate_child_scope():
            assert ident.current_execution_context().role == ident.ROLE_DELEGATE_CHILD
            err = denial(lambda: kb.schedule_task(board, tid, reason="child"))
            assert err.reason == "delegate_child"
        assert ident.current_execution_context().role == ident.ROLE_OWNER

    assert seen["role"] == ident.ROLE_OWNER and seen["block"] is True
    assert thread_box["ctx"] is None and thread_box["owner"] is False
    assert ident.current_execution_context() is None

    # A terminal/delegate subprocess carries no ownership pins.
    from agent.delegation_context import delegated_child_subprocess_env
    child_env = delegated_child_subprocess_env(dict(os.environ))
    for pin in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_CLAIM_LOCK"):
        assert pin not in child_env


def test_scope_resets_on_exception_and_keeps_the_process_slot_untouched(board):
    home = home_of(board)
    tid = new_task(board)
    spawn_submitter(board, tid)
    candidate, _frame, _context_map = bind(board, tid, home)
    with pytest.raises(RuntimeError):
        with session_kanban.managed_execution_scope(candidate):
            assert ident.current_execution_context().role == ident.ROLE_OWNER
            raise RuntimeError("constructor failed")
    assert ident.current_execution_context() is None
    assert ident.process_identity() is None


# ------------------------------------------------ 6. stop / identity read path

def test_live_managed_owner_is_recognized_despite_receipt_pid_and_scrubbed_shell(board, monkeypatch):
    home = home_of(board)
    tid = new_task(board)
    worker = spawn_submitter(board, tid)
    candidate, _frame, _context_map = bind(board, tid, home)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    try:
        with session_kanban.managed_execution_scope(candidate):
            # The recipe: the run's pid is the spawned submitter's, not this interpreter's.
            recorded = board.execute("SELECT worker_pid FROM task_runs WHERE id = ?",
                                     (worker["run_id"],)).fetchone()["worker_pid"]
            assert recorded == worker["submitter_pid"] != os.getpid()
            snap = ident.identity_snapshot(board, target_task_id=tid)
            assert snap["role"] == ident.ROLE_OWNER and snap["claim_match"] is True
            assert "claim_lock" not in snap
            nudge = stop_guard.build_kanban_stop_nudge(messages=[], attempts=0, task_id=tid)
            assert nudge and "does NOT mean you lost ownership" in nudge
    finally:
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)


def test_detached_managed_turn_gets_no_nudge_and_writes_nothing(board):
    home = home_of(board)
    tid = new_task(board)
    spawn_submitter(board, tid)
    candidate, _frame, _context_map = bind(board, tid, home)
    with ident.admin_recovery_scope(principal="operator@example", transport="dashboard_session",
                                    action="reclaim", reason="detach"):
        kb.reclaim_task(board, tid, reason="detach", signal_fn=lambda *a, **k: None)
    before = snapshot(board)
    with session_kanban.managed_execution_scope(candidate):
        assert stop_guard.build_kanban_stop_nudge(messages=[], attempts=0, task_id=tid) is None
        assert ident.identity_snapshot(board, target_task_id=tid)["claim_match"] is False
    assert snapshot(board) == before


def test_a_settled_managed_run_keeps_its_result_bookkeeping(board):
    """A completed own run is booked once; losing OWNER afterwards is not a crash."""
    home = home_of(board)
    tid = new_task(board)
    worker = spawn_submitter(board, tid)
    candidate, _frame, context = bind(board, tid, home)
    with session_kanban.managed_execution_scope(candidate):
        assert kb.complete_task(board, tid, summary="managed completion") is True
        # The worker's own result record still lands after its claim closed (the receipt the
        # dispatcher books), and the grant is simply no longer current.
        session_kanban._record_worker_result(context, 0, "done")
        assert session_kanban._bound_claim_is_live(context) is False
    result = board.execute("SELECT payload FROM task_events WHERE task_id = ? AND kind = 'worker_result' "
                           "ORDER BY id DESC LIMIT 1", (tid,)).fetchone()
    assert result is not None and json.loads(result["payload"])["exit_code"] == 0
    assert board.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] == "done"
    assert board.execute("SELECT outcome FROM task_runs WHERE id = ?",
                         (worker["run_id"],)).fetchone()["outcome"] == "completed"


def test_recovery_retry_proof_requires_a_live_lease(board):
    """In-place recovery consumes the same reconciliation, so an expired lease stops it."""
    home = home_of(board)
    tid = new_task(board)
    worker = spawn_submitter(board, tid)
    candidate, _frame, context = bind(board, tid, home)
    with session_kanban.managed_execution_scope(candidate):
        assert session_kanban._bound_claim_is_live(context) is True
        board.execute("UPDATE tasks SET claim_expires = ? WHERE id = ?", (int(time.time()) - 1, tid))
        board.commit()
        assert session_kanban._bound_claim_is_live(context) is False
    assert board.execute("SELECT current_run_id FROM tasks WHERE id = ?",
                         (tid,)).fetchone()["current_run_id"] == worker["run_id"]
