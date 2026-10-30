"""Explicit Kanban execution identities for kernel-level lifecycle tests.

``hermes_cli/kanban_db.py`` authorizes every lifecycle write inside its own
``write_txn`` against the caller's trusted execution identity
(``agent/kanban_execution_identity.py``).  A suite that drives those mutators
directly has no dispatcher and no dashboard in the loop, so it has to state
which production actor it is standing in for:

* :func:`owner_authority` — the spawned worker of one card.  It performs the
  registration the dispatcher performs for a real worker (pid, boot/start
  fingerprint and lease on both the task row and its open run row) and then
  binds ``OWNER`` through the production issuer
  ``bootstrap_worker_identity``.  The reconciliation a real worker gets is the
  reconciliation the test gets; nothing is relaxed and no role is invented.
* :func:`admin_authority` — the authenticated dashboard administration
  boundary, for the recovery operations (``reclaim``/``reassign``/``unblock``)
  that only an administrator may perform.  It mints through
  ``admin_recovery_scope`` with an explicit principal and transport, so those
  mutations land with the same audit event production writes.

Both are explicit per-test context managers.  There is deliberately no autouse
fixture and no "identity missing, therefore allow" switch: a suite that forgets
to ask for an identity still fails closed with
``KanbanAuthorizationError``.
"""

from __future__ import annotations

import contextlib
import os
import time
from typing import Any, Iterator, Optional

from agent import kanban_execution_identity as identity
from agent.kanban_execution_identity import connection_db_path

#: Long enough that a lease never expires mid-test, matching a worker that is
#: heartbeating normally.
DEFAULT_LEASE_SECONDS = 3600


class KanbanTestIdentityError(AssertionError):
    """The fixture's board state is not one a real worker could own."""


def register_worker(
    conn: Any, task_id: str, *, lease_seconds: int = DEFAULT_LEASE_SECONDS,
) -> dict[str, str]:
    """Bind the card's open run to THIS process the way the dispatcher does.

    Writes exactly the coordinates a launcher records after ``Popen``: the pid
    and boot/start fingerprint on the task row, the pid on the run row, and a
    lease on both.  Returns the lookup coordinates
    :func:`bootstrap_worker_identity` reads.
    """
    db_path = connection_db_path(conn)
    if not db_path:
        raise KanbanTestIdentityError("test connection is not a Kanban board file")
    row = conn.execute(
        "SELECT status, current_run_id, claim_lock FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    if row is None:
        raise KanbanTestIdentityError(f"task {task_id} is not on this board")
    status = row["status"]
    run_id = row["current_run_id"]
    claim_lock = row["claim_lock"]
    if status != "running":
        raise KanbanTestIdentityError(
            f"task {task_id} is {status!r}; only a running card has an owner to register"
        )
    if run_id is None or not claim_lock:
        raise KanbanTestIdentityError(
            f"task {task_id} has no open claimed run (run_id={run_id!r}); claim it first"
        )

    pid = os.getpid()
    fingerprint = identity._process_identity(pid)
    if fingerprint is None:
        raise KanbanTestIdentityError(f"no boot/start fingerprint available for pid {pid}")
    expires = int(time.time()) + int(lease_seconds)
    conn.execute(
        "UPDATE tasks SET worker_pid = ?, worker_started_at = ?, claim_expires = ? WHERE id = ?",
        (pid, fingerprint, expires, task_id),
    )
    conn.execute(
        "UPDATE task_runs SET worker_pid = ?, claim_expires = ? WHERE id = ?",
        (pid, expires, int(run_id)),
    )
    conn.commit()
    return {
        "HERMES_KANBAN_TASK": str(task_id),
        "HERMES_KANBAN_RUN_ID": str(int(run_id)),
        "HERMES_KANBAN_CLAIM_LOCK": str(claim_lock),
        "HERMES_KANBAN_DB": str(db_path),
    }


@contextlib.contextmanager
def owner_authority(
    conn: Any, task_id: str, *, lease_seconds: int = DEFAULT_LEASE_SECONDS,
) -> Iterator[identity.KanbanExecutionContext]:
    """Bind ``OWNER`` of *task_id* for the duration of the block."""
    previous_context = identity.current_execution_context()
    previous_process = identity.process_identity()
    coordinates = register_worker(conn, task_id, lease_seconds=lease_seconds)
    ctx = identity.bootstrap_worker_identity(env=coordinates, wait_seconds=0.0)
    try:
        if ctx.role != identity.ROLE_OWNER:
            raise KanbanTestIdentityError(
                f"bootstrap refused OWNER for {task_id}: role={ctx.role} reason={ctx.reason!r}"
            )
        yield ctx
    finally:
        identity.install_process_identity(previous_process)
        identity.bind_execution_context(previous_context)


@contextlib.contextmanager
def admin_authority(
    *,
    principal: str = "pytest-operator",
    transport: str = "dashboard:kanban-admin",
    action: Optional[str] = None,
    reason: str = "test administrative lifecycle write",
    db_path: Optional[str] = None,
    board: Optional[str] = None,
) -> Iterator[identity.KanbanExecutionContext]:
    """Mint ``ADMIN_RECOVERY`` from an explicit, verified test principal."""
    with identity.admin_recovery_scope(
        principal=principal,
        transport=transport,
        action=action,
        reason=reason,
        db_path=db_path,
        board=board,
    ) as ctx:
        yield ctx


#: A managed attempt's spawned submitter: some other process that waits for this
#: execution's receipt.  Dead on purpose — the managed grant never consults its liveness.
MANAGED_SUBMITTER_PID = 424242


def bind_managed_owner(conn: Any, task_id: str, *, submitter_pid: int = MANAGED_SUBMITTER_PID):
    """The adopted exec's ``OWNER`` grant for one claimed card, as production issues it.

    ``gateway.session_kanban.bind_worker_context`` grants a managed run only after the
    owner's verified adoption, and records the executing interpreter in the run's
    ``worker_bound`` event while the task/run process fields stay the spawned
    submitter's.  A suite standing in for that exec makes the same two writes — the
    dispatcher's ``_set_worker_pid`` registration and the binding event — and builds the
    same frozen candidate.

    The issuer's own checks (adoption/epoch/birth, coordinates, profile, leases,
    conflicting bindings) are covered by ``tests/gateway/test_kanban_managed_owner.py``
    and end to end by ``test_session_kanban.py::test_real_dispatcher_lifecycle``; do not
    use this helper as proof that the issuer is safe.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as dispatch

    db_path = connection_db_path(conn)
    if not db_path:
        raise KanbanTestIdentityError("test connection is not a Kanban board file")
    dispatch._set_worker_pid(conn, task_id, int(submitter_pid))
    row = conn.execute(
        "SELECT status, current_run_id, claim_lock, worker_pid, worker_started_at FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if row is None or row["status"] != "running" or row["current_run_id"] is None or not row["claim_lock"]:
        raise KanbanTestIdentityError(f"task {task_id} has no open claimed run to bind")
    fingerprint = dispatch._process_fingerprint(os.getpid())
    if not fingerprint or fingerprint == dispatch.UNVERIFIED_WORKER_FINGERPRINT:
        raise KanbanTestIdentityError(f"no boot/start fingerprint available for pid {os.getpid()}")
    with kb.write_txn(conn):
        kb._append_event(conn, task_id, "worker_bound",
                         {"pid": os.getpid(), "claim_lock": row["claim_lock"], "started_at": fingerprint},
                         run_id=int(row["current_run_id"]))
    ctx = identity.KanbanExecutionContext(
        role=identity.ROLE_OWNER,
        provenance=identity.PROVENANCE_MANAGED_BOUND,
        db_path=db_path,
        task_id=str(task_id),
        run_id=int(row["current_run_id"]),
        claim_lock=str(row["claim_lock"]),
        worker_pid=os.getpid(),
        worker_started_at=fingerprint,
        submitter_pid=int(row["worker_pid"]),
        submitter_started_at=str(row["worker_started_at"]),
    )
    return ctx, {"run_id": int(row["current_run_id"]), "claim_lock": str(row["claim_lock"])}
