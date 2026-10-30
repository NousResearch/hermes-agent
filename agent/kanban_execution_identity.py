"""Trusted Kanban execution identity and lifecycle authorization.

The board's lifecycle mutators used to trust either nothing (an *unbound* caller
could ``block``/``schedule``/``request_changes`` a live task) or a caller-supplied
``expected_run_id``.  Both are provenance a model, a prompt-injected tool call or
a stale shell can produce.  This module is the one place where a *trusted*
execution role is established and re-checked against the database under the
caller's write transaction.

Roles (never inferred from a profile name, task text, workspace path or by
echoing a caller-supplied string):

``OWNER``
    The dispatcher-launched worker for the task it was spawned on.  Granted only
    after the issuer for its provenance reconciles the claim against the live
    database *and* the recorded process fingerprint of **this** process.  Two
    provenances exist and never substitute for each other: ``direct_cli``
    (:func:`bootstrap_worker_identity`, whose recorded process fields are the
    task/run ``worker_pid``/``worker_started_at``) and ``managed_bound`` (the
    adopted private exec, ``gateway.session_kanban.bind_worker_context``, whose
    recorded comparator is the run's latest ``worker_bound`` event because the
    task/run process fields name the spawned *submitter* there).
``FOREIGN_INSPECTOR``
    Any other runtime caller (ordinary chat, an orchestrator session, a timer,
    a bare CLI invocation).  It may read the board; it may not mutate a live
    run's lifecycle.
``DELEGATE_CHILD``
    An in-process ``delegate_task`` child or a spawned descendant of a worker.
    It inherits neither the parent's ownership nor its authority.
``ADMIN_RECOVERY``
    Minted only by :func:`admin_recovery_scope`, which the *authenticated*
    dashboard administration boundary calls after its own middleware verified
    the request.  An administrative transport is the only admin ingress.
``DISPATCHER_MAINTENANCE``
    The dispatcher's own internal maintenance service (stale release, crash
    accounting).  Private to the dispatcher process; never exposed as a model
    tool and never reached by falling back from a failed owner operation.

Threat model, stated honestly: this is cooperative-runtime scoping.  It closes
authority that arrives through the model tool surface, the CLI, the HTTP
surface and shell environment variables.  It does **not** confine a hostile
same-UID process that opens the SQLite file directly; that limitation already
exists for every Kanban table and is unchanged here.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

logger = logging.getLogger(__name__)

ROLE_OWNER = "OWNER"
ROLE_FOREIGN_INSPECTOR = "FOREIGN_INSPECTOR"
ROLE_DELEGATE_CHILD = "DELEGATE_CHILD"
ROLE_ADMIN_RECOVERY = "ADMIN_RECOVERY"
ROLE_DISPATCHER_MAINTENANCE = "DISPATCHER_MAINTENANCE"

#: How an ``OWNER`` grant was issued.  A closed internal discriminator: it is
#: never accepted from a caller, a frame or a tool argument, and the two proofs
#: never fall back to each other (unknown values fail closed).
PROVENANCE_DIRECT_CLI = "direct_cli"
PROVENANCE_MANAGED_BOUND = "managed_bound"
PROVENANCES: frozenset[str] = frozenset({PROVENANCE_DIRECT_CLI, PROVENANCE_MANAGED_BOUND})

#: Lifecycle operations an owning worker may perform on ITS OWN active run.
OWNER_OPERATIONS: frozenset[str] = frozenset({
    "complete", "block", "schedule", "request_review", "request_changes",
})

#: Operations an authenticated administrative transport may perform on any task.
ADMIN_OPERATIONS: frozenset[str] = frozenset({
    "complete", "block", "schedule", "request_review", "request_changes",
    "reclaim", "reassign", "unblock",
})

#: Operations the dispatcher's own maintenance service may perform.
MAINTENANCE_OPERATIONS: frozenset[str] = frozenset({
    "complete", "block", "schedule", "reclaim", "unblock", "request_review", "request_changes",
})

#: Bound on the spawn/registration startup race: the dispatcher writes the
#: worker pid + fingerprint right after ``Popen``, so a worker that starts
#: faster than that waits (read-only) instead of relaxing its proof.
REGISTRATION_WAIT_SECONDS = 5.0
REGISTRATION_POLL_SECONDS = 0.05

# Stable denial codes.  Callers and tests match on these, not on prose.
CODE_AUTHORIZATION_DENIED = "authorization_denied"
CODE_OWNERSHIP_LOST = "ownership_lost"
CODE_REVIEW_PHASE_REQUIRED = "review_phase_required"
CODE_IDENTITY_UNAVAILABLE = "identity_unavailable"

_ADMIN_INSTRUCTION = (
    "Kanban lifecycle writes require either the dispatcher-owned worker of the "
    "exact task/run or an authenticated Kanban administration request. Perform "
    "recovery through the Kanban administration dashboard; a terminal, chat turn "
    "or CLI invocation cannot mint administrative authority."
)


class KanbanAuthorizationError(ValueError):
    """A lifecycle write was refused by the trusted authority boundary.

    A ``ValueError`` so the tool/CLI error paths treat it as recoverable (the
    caller may hand off or retry legitimately) instead of crashing the process.
    ``code`` is one of the stable ``CODE_*`` values.
    """

    def __init__(self, message: str, *, code: str = CODE_AUTHORIZATION_DENIED, reason: str = ""):
        self.code = code
        self.reason = reason
        super().__init__(message)


@dataclass(frozen=True)
class KanbanExecutionContext:
    """The trusted execution identity of the current scope.

    Credentials are intentionally absent: ``claim_lock`` is an identifier the
    dispatcher also records in the DB, never a secret this record exposes back
    to a model.

    ``worker_pid``/``worker_started_at`` always mean the process EXECUTING the
    turn.  For ``managed_bound`` the task/run process fields belong to the
    spawned *submitter* instead, so the grant also carries
    ``submitter_pid``/``submitter_started_at``.  Both are internal proof detail:
    no model-visible surface renders them.
    """

    role: str
    db_path: Optional[str] = None
    board: Optional[str] = None
    task_id: Optional[str] = None
    run_id: Optional[int] = None
    claim_lock: Optional[str] = None
    worker_pid: Optional[int] = None
    worker_started_at: Optional[str] = None
    reason: str = ""
    provenance: str = PROVENANCE_DIRECT_CLI
    submitter_pid: Optional[int] = None
    submitter_started_at: Optional[str] = None
    # Administrative provenance (ADMIN_RECOVERY only).
    admin_principal: Optional[str] = None
    admin_transport: Optional[str] = None
    admin_action: Optional[str] = None
    admin_reason: Optional[str] = None


@dataclass(frozen=True)
class Reconciliation:
    """Outcome of re-checking a context against the current database state."""

    ok: bool
    reason: str = ""
    current_db_run_id: Optional[int] = None
    claim_match: bool = False
    review_source_phase: bool = False


_EXECUTION_CONTEXT: ContextVar[Any] = ContextVar(
    "hermes_kanban_execution_context", default=object(),
)
#: Process-wide identity, written by :func:`bootstrap_worker_identity` ONLY.
#:
#: A dispatcher-launched worker has exactly one identity for the whole process:
#: its own claimed run.  Tool handlers can run on another thread (or outside any
#: copied context), and a worker that lost its execution identity mid-turn would
#: misdiagnose itself again — so that one identity is process-wide.
#:
#: Request- and maintenance-scoped identities (``ADMIN_RECOVERY``,
#: ``DISPATCHER_MAINTENANCE``, ``FOREIGN_INSPECTOR``, ``DELEGATE_CHILD``) are
#: NEVER published here: they belong to one caller.  Publishing them would let
#: any unbound thread in the same process (a ``tui-rpc`` chat turn in the
#: dashboard process, for instance) inherit administrative authority, and two
#: overlapping requests could leave the slot stuck on an administrative scope
#: after they exit out of order.  A caller on another thread receives such a
#: scope by explicit propagation (``contextvars.copy_context().run`` or a bind
#: inside that thread), never by inheritance.
_PROCESS_CONTEXT: list[Optional[KanbanExecutionContext]] = [None]


def install_process_identity(ctx: Optional[KanbanExecutionContext]) -> None:
    """Record the resolved identity for the whole process (bootstrap only)."""
    _PROCESS_CONTEXT[0] = ctx


def process_identity() -> Optional[KanbanExecutionContext]:
    return _PROCESS_CONTEXT[0]


# ---------------------------------------------------------------------------
# Scope primitives
# ---------------------------------------------------------------------------

def bind_execution_context(ctx: Optional[KanbanExecutionContext]) -> Token:
    """Install *ctx* for the current context; returns the reset token.

    ``None`` is an EXPLICIT "no identity in this scope" (it shadows the process-wide
    worker identity), not "inherit".
    """
    return _EXECUTION_CONTEXT.set(ctx)


def reset_execution_context(token: Token) -> None:
    _EXECUTION_CONTEXT.reset(token)


def current_execution_context() -> Optional[KanbanExecutionContext]:
    """The context bound by a trusted issuer, else the process-wide worker identity.

    An explicitly bound ``None`` stays ``None`` — the scope says "no identity", it does
    not silently inherit one.
    """
    value = _EXECUTION_CONTEXT.get()
    if isinstance(value, KanbanExecutionContext):
        return value
    if value is None:
        return None
    return _PROCESS_CONTEXT[0]


@contextmanager
def scoped_execution_context(ctx: Optional[KanbanExecutionContext]) -> Iterator[KanbanExecutionContext]:
    """Enter *ctx* for the caller that requested it — and nobody else.

    The scope is context-local by design.  It binds the ContextVar, so work the
    caller runs in this context (including anything it dispatches through the
    existing ``contextvars`` propagation: copied contexts, executor threads that
    copy the caller's context, awaited coroutines) sees it.  It deliberately
    does NOT touch the process-wide worker identity: a request-scoped grant must
    never be inheritable by an unrelated thread of the same process.

    A caller that runs part of the scope on a thread it created itself propagates
    explicitly (``contextvars.copy_context().run``) or binds inside that thread.
    """
    token = bind_execution_context(ctx)
    try:
        yield ctx  # type: ignore[misc]
    finally:
        reset_execution_context(token)


@contextmanager
def foreign_inspector_scope(reason: str = "foreign_inspector") -> Iterator[KanbanExecutionContext]:
    """Explicit read-only scope: an ordinary runtime call that owns nothing."""
    ctx = KanbanExecutionContext(role=ROLE_FOREIGN_INSPECTOR, reason=reason)
    with scoped_execution_context(ctx) as installed:
        yield installed


@contextmanager
def delegate_child_scope(reason: str = "delegate_child") -> Iterator[KanbanExecutionContext]:
    """Delegate/descendant scope: no ownership, no administrative authority."""
    ctx = KanbanExecutionContext(role=ROLE_DELEGATE_CHILD, reason=reason)
    with scoped_execution_context(ctx) as installed:
        yield installed


@contextmanager
def admin_recovery_scope(
    *, principal: str, transport: str, action: Optional[str] = None,
    reason: Optional[str] = None, db_path: Optional[str] = None, board: Optional[str] = None,
) -> Iterator[KanbanExecutionContext]:
    """Mint ``ADMIN_RECOVERY`` for one authenticated administrative request.

    ONLY the authenticated dashboard administration boundary may call this after
    its own middleware has verified the request (see
    ``plugins/kanban/dashboard/plugin_api.py``).  ``principal``/``transport`` are
    the verified provenance recorded in the audit event; they are never passed
    through to a model or a worker.
    """
    if not (principal or "").strip() or not (transport or "").strip():
        raise KanbanAuthorizationError(
            f"administrative Kanban authority requires an authenticated principal and transport ({_ADMIN_INSTRUCTION})",
            code=CODE_AUTHORIZATION_DENIED,
            reason="missing_admin_provenance",
        )
    ctx = KanbanExecutionContext(
        role=ROLE_ADMIN_RECOVERY,
        db_path=_canonical(db_path) if db_path else None,
        board=(board or None),
        reason=reason or "",
        admin_principal=principal.strip(),
        admin_transport=transport.strip(),
        admin_action=(action or None),
        admin_reason=reason,
    )
    with scoped_execution_context(ctx) as installed:
        yield installed


@contextmanager
def dispatcher_maintenance_scope(
    *, db_path: Optional[str] = None, board: Optional[str] = None, reason: str = "dispatcher_maintenance",
) -> Iterator[KanbanExecutionContext]:
    """The dispatcher's own maintenance service scope (private, never a tool)."""
    ctx = KanbanExecutionContext(
        role=ROLE_DISPATCHER_MAINTENANCE,
        db_path=_canonical(db_path) if db_path else None,
        board=(board or None),
        reason=reason,
    )
    with scoped_execution_context(ctx) as installed:
        yield installed


# ---------------------------------------------------------------------------
# DB helpers (read-only: identity resolution never creates or migrates a board)
# ---------------------------------------------------------------------------

def _canonical(path: Any) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(Path(str(path)).expanduser().resolve())
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def connection_db_path(conn: sqlite3.Connection) -> Optional[str]:
    """Filesystem path of *conn*'s main database (``None`` for in-memory)."""
    try:
        for _seq, name, file in conn.execute("PRAGMA database_list") or ():
            if name == "main":
                return file or None
    except (sqlite3.Error, TypeError, ValueError):
        pass
    return None


def _open_readonly(db_path: str) -> Optional[sqlite3.Connection]:
    """Open an existing board read-only; ``None`` when it does not exist."""
    p = Path(db_path).expanduser()
    if not p.exists():
        return None
    conn = sqlite3.connect(p.resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _process_identity(pid: Optional[int]) -> Optional[str]:
    """Restart-stable fingerprint of *pid* (boot epoch + process start time)."""
    if pid is None:
        return None
    try:
        from hermes_cli.kanban_db_dispatch import _process_fingerprint

        return _process_fingerprint(int(pid))
    except Exception:
        logger.debug("kanban identity: process fingerprint unavailable for %s", pid, exc_info=True)
        return None


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    try:
        value = row[key]
    except (KeyError, IndexError, TypeError):
        return default
    return default if value is None else value


def _claimed_source_status(conn: sqlite3.Connection, task_id: str, run_id: int) -> Optional[str]:
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND run_id = ? AND kind = 'claimed' "
        "ORDER BY id DESC LIMIT 1",
        (task_id, int(run_id)),
    ).fetchone()
    if row is None:
        return None
    try:
        payload = json.loads(row["payload"]) if row["payload"] else {}
    except (TypeError, ValueError):
        return None
    if isinstance(payload, dict):
        value = payload.get("source_status")
        return str(value) if value else None
    return None


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

def reconcile_owner(
    conn: sqlite3.Connection,
    ctx: KanbanExecutionContext,
    *,
    now: Optional[int] = None,
    require_lease: bool = True,
    pid: Optional[int] = None,
) -> Reconciliation:
    """Re-check an ``OWNER`` context against *conn*'s current task/run rows.

    Every coordinate must match exactly: canonical DB, task id, current run id,
    claim lock, ``running`` status, an OPEN run row, and — for writes —
    unexpired task and run leases read with ONE transaction timestamp.  A
    missing coordinate, a closed run, a successor run or an expired lease all
    mean ``ownership_lost``; none of them mean "fall back to administrative".

    The recorded-process half depends on the grant's provenance only:

    * ``direct_cli`` — the task AND run process fields must be THIS interpreter
      (pid + boot/start fingerprint), exactly as the launcher recorded them.
    * ``managed_bound`` — those fields name the spawned submitter, so they must
      match the GRANT's captured submitter pid/fingerprint, and the run's latest
      ``worker_bound`` event must name THIS interpreter and this claim lock.
      Both provenances additionally require the grant's executing pid and
      fingerprint to describe the live process.
    """
    early = _reconciliation_premise(conn, ctx)
    if early is not None:
        return early

    now = int(time.time()) if now is None else int(now)
    try:
        trow = conn.execute(
            "SELECT status, claim_lock, claim_expires, current_run_id, worker_pid, worker_started_at "
            "FROM tasks WHERE id = ?",
            (ctx.task_id,),
        ).fetchone()
        if trow is None:
            return Reconciliation(ok=False, reason="task_missing", claim_match=False)
        current_run_id = _row_value(trow, "current_run_id")
        claim_match = _claim_matches(trow, ctx, current_run_id)
        refusal = _task_row_refusal(trow, ctx, current_run_id, claim_match, now, require_lease)
        if refusal is not None:
            return refusal

        rrow = conn.execute(
            "SELECT task_id, ended_at, claim_lock, claim_expires, worker_pid FROM task_runs WHERE id = ?",
            (int(ctx.run_id),),
        ).fetchone()
        if rrow is None:
            return Reconciliation(False, "run_missing", current_run_id, claim_match)
        refusal = _run_row_refusal(rrow, ctx, current_run_id, claim_match, now, require_lease)
        if refusal is not None:
            return refusal

        if ctx.provenance == PROVENANCE_MANAGED_BOUND:
            refusal = _managed_binding_refusal(conn, ctx, current_run_id, claim_match)
            if refusal is not None:
                return refusal

        # The recorded fingerprint must still describe THIS process: a recycled
        # PID or a reconstructed env in another process can never match it.
        live_pid = os.getpid() if pid is None else int(pid)
        if int(ctx.worker_pid or -1) != live_pid or ctx.worker_started_at != _process_identity(live_pid):
            return Reconciliation(False, CODE_OWNERSHIP_LOST, current_run_id, claim_match)

        review_phase = _claimed_source_status(conn, ctx.task_id, int(ctx.run_id)) == "review"
        return Reconciliation(True, "", current_run_id, True, review_phase)
    except sqlite3.Error:
        return Reconciliation(False, CODE_IDENTITY_UNAVAILABLE, claim_match=False)


def _reconciliation_premise(conn: sqlite3.Connection, ctx: KanbanExecutionContext) -> Optional[Reconciliation]:
    """The refusals that need no row read: wrong role, no run, no readable board, other DB."""
    if ctx.role != ROLE_OWNER:
        return Reconciliation(ok=False, reason="not_owner", claim_match=False)
    if ctx.provenance not in PROVENANCES:
        return Reconciliation(ok=False, reason=CODE_IDENTITY_UNAVAILABLE, claim_match=False)
    if not ctx.task_id or ctx.run_id is None:
        return Reconciliation(ok=False, reason="no_owned_run", claim_match=False)
    db_path = connection_db_path(conn)
    if not db_path:
        return Reconciliation(ok=False, reason=CODE_IDENTITY_UNAVAILABLE, claim_match=False)
    if ctx.db_path and _canonical(ctx.db_path) != _canonical(db_path):
        return Reconciliation(ok=False, reason="db_mismatch", claim_match=False)
    return None


def _recorded_owner(ctx: KanbanExecutionContext) -> tuple[Any, Any]:
    """The (pid, started_at) the task/run process fields must carry for this grant.

    A direct CLI worker IS the recorded worker; a managed grant's recorded worker
    is the dispatcher-spawned submitter that waits for this execution's receipt.
    """
    if ctx.provenance == PROVENANCE_MANAGED_BOUND:
        return ctx.submitter_pid, ctx.submitter_started_at
    return ctx.worker_pid, ctx.worker_started_at


def _claim_matches(trow, ctx: KanbanExecutionContext, current_run_id: Any) -> bool:
    return bool(
        current_run_id is not None
        and int(current_run_id) == int(ctx.run_id)
        and str(_row_value(trow, "claim_lock", "")) == str(ctx.claim_lock or "")
    )


def _task_row_refusal(
    trow, ctx: KanbanExecutionContext, current_run_id: Any, claim_match: bool,
    now: int, require_lease: bool,
) -> Optional[Reconciliation]:
    """Refusal for a task row that is not this run's live claim (``None`` when it is)."""
    lock = str(ctx.claim_lock or "")
    expected_pid, expected_started_at = _recorded_owner(ctx)
    checks = (
        (str(_row_value(trow, "status", "")) != "running", "not_running"),
        (current_run_id is None or int(current_run_id) != int(ctx.run_id), CODE_OWNERSHIP_LOST),
        (str(_row_value(trow, "claim_lock", "")) != lock, CODE_OWNERSHIP_LOST),
        (expected_pid is None or _row_value(trow, "worker_pid") is None
         or int(_row_value(trow, "worker_pid")) != int(expected_pid), CODE_OWNERSHIP_LOST),
        (str(_row_value(trow, "worker_started_at", "")) != str(expected_started_at or ""),
         CODE_OWNERSHIP_LOST),
    )
    for failed, reason in checks:
        if failed:
            return Reconciliation(False, reason, current_run_id, claim_match)
    # Boundary instant is uniform with existing recovery semantics: a lease is
    # expired when ``expiry < transaction time``.
    if require_lease:
        task_expires = _row_value(trow, "claim_expires")
        if task_expires is None or int(task_expires) < now:
            return Reconciliation(False, "lease_expired", current_run_id, claim_match)
    return None


def _run_row_refusal(
    rrow, ctx: KanbanExecutionContext, current_run_id: Any, claim_match: bool,
    now: int, require_lease: bool,
) -> Optional[Reconciliation]:
    """Refusal for a run row that is not this claim's open run (``None`` when it is)."""
    lock = str(ctx.claim_lock or "")
    expected_pid = _recorded_owner(ctx)[0]
    run_pid = _row_value(rrow, "worker_pid")
    checks = (
        (str(_row_value(rrow, "task_id", "")) != str(ctx.task_id), CODE_OWNERSHIP_LOST),
        (_row_value(rrow, "ended_at") is not None, CODE_OWNERSHIP_LOST),
        (str(_row_value(rrow, "claim_lock", "")) != lock, CODE_OWNERSHIP_LOST),
        (run_pid is not None and int(run_pid) != int(expected_pid or -1), CODE_OWNERSHIP_LOST),
    )
    for failed, reason in checks:
        if failed:
            return Reconciliation(False, reason, current_run_id, claim_match)
    if require_lease:
        run_expires = _row_value(rrow, "claim_expires")
        if run_expires is None or int(run_expires) < now:
            return Reconciliation(False, "lease_expired", current_run_id, claim_match)
    return None


def _latest_worker_binding(
    conn: sqlite3.Connection, task_id: str, run_id: int,
) -> tuple[Optional[dict[str, Any]], bool]:
    """The newest ``worker_bound`` payload for the exact run: ``(payload, parsed_ok)``.

    ``(None, True)`` means the run never bound an interpreter.  Any payload that
    does not carry the three evidence fields with their expected types is
    ``(None, False)`` — malformed evidence fails closed instead of reading as a
    partial match.
    """
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND run_id = ? AND kind = 'worker_bound' "
        "ORDER BY id DESC LIMIT 1",
        (task_id, int(run_id)),
    ).fetchone()
    if row is None:
        return None, True
    try:
        payload = json.loads(row["payload"]) if row["payload"] else None
    except (TypeError, ValueError):
        return None, False
    if not isinstance(payload, dict):
        return None, False
    if (type(payload.get("pid")) is not int
            or not isinstance(payload.get("claim_lock"), str) or not payload["claim_lock"]
            or not isinstance(payload.get("started_at"), str) or not payload["started_at"]):
        return None, False
    return payload, True


def _managed_binding_refusal(
    conn: sqlite3.Connection, ctx: KanbanExecutionContext, current_run_id: Any, claim_match: bool,
) -> Optional[Reconciliation]:
    """The run's own binding must still name THIS interpreter, lock and fingerprint.

    This is the managed grant's recorded-process comparator: the task/run process
    fields are the submitter's, so the executing process is proven by the
    ``worker_bound`` event its bootstrap appended transactionally.  A superseded,
    missing, malformed or conflicting binding grants nothing.
    """
    bound, parsed_ok = _latest_worker_binding(conn, str(ctx.task_id), int(ctx.run_id))
    if not parsed_ok or bound is None:
        return Reconciliation(False, CODE_OWNERSHIP_LOST, current_run_id, claim_match)
    if (int(bound["pid"]) != int(ctx.worker_pid or -1)
            or str(bound["claim_lock"]) != str(ctx.claim_lock or "")
            or str(bound["started_at"]) != str(ctx.worker_started_at or "")):
        return Reconciliation(False, CODE_OWNERSHIP_LOST, current_run_id, claim_match)
    return None


# ---------------------------------------------------------------------------
# Worker bootstrap (the only OWNER issuer)
# ---------------------------------------------------------------------------

def env_coordinates(env: Optional[Mapping[str, str]] = None) -> dict[str, Any]:
    """Read the launcher's lookup coordinates.  These are coordinates, not authority."""
    e = os.environ if env is None else env
    raw_run = str(e.get("HERMES_KANBAN_RUN_ID") or "").strip()
    try:
        run_id = int(raw_run) if raw_run else None
    except ValueError:
        run_id = None
    return {
        "task_id": (str(e.get("HERMES_KANBAN_TASK") or "").strip() or None),
        "run_id": run_id,
        "claim_lock": (str(e.get("HERMES_KANBAN_CLAIM_LOCK") or "").strip() or None),
        "db_path": (str(e.get("HERMES_KANBAN_DB") or "").strip() or None),
        "board": (str(e.get("HERMES_KANBAN_BOARD") or "").strip() or None),
    }


def bootstrap_worker_identity(
    *,
    env: Optional[Mapping[str, str]] = None,
    wait_seconds: float = REGISTRATION_WAIT_SECONDS,
    sleep_fn=time.sleep,
    pid: Optional[int] = None,
) -> KanbanExecutionContext:
    """Resolve and bind this process's execution identity before tool use.

    Env is used ONLY as lookup coordinates.  ``OWNER`` is granted only when the
    task and its open run row agree on task id, run id, claim lock, ``running``
    status, the recorded worker pid and the recorded boot/start fingerprint of
    *this* process.  A launcher that has not registered the pid yet is waited for
    (bounded, read-only, no relaxed checks).  Everything else — a delegate
    descendant, missing coordinates, an unreadable board, a successor run, a
    recycled pid — binds a detached role and is never auto-upgraded later.

    This is the ``direct_cli`` issuer only: it grants nothing to a managed run,
    where the recorded process fields belong to the spawned submitter (that
    bootstrap lives in ``gateway.session_kanban.bind_worker_context``).
    """
    live_pid = os.getpid() if pid is None else int(pid)
    coords = env_coordinates(env)

    from agent.delegation_context import is_delegated_child_process_context

    if is_delegated_child_process_context():
        ctx = KanbanExecutionContext(role=ROLE_DELEGATE_CHILD, reason="delegate_descendant")
        bind_execution_context(ctx)
        install_process_identity(ctx)
        return ctx
    if not (coords["task_id"] and coords["run_id"] is not None and coords["claim_lock"] and coords["db_path"]):
        ctx = KanbanExecutionContext(role=ROLE_FOREIGN_INSPECTOR, reason="no_worker_coordinates")
        bind_execution_context(ctx)
        install_process_identity(ctx)
        return ctx

    deadline = time.monotonic() + max(0.0, float(wait_seconds))
    fingerprint = _process_identity(live_pid)
    if fingerprint is None:
        ctx = KanbanExecutionContext(
            role=ROLE_FOREIGN_INSPECTOR, db_path=coords["db_path"], board=coords["board"],
            task_id=coords["task_id"], run_id=coords["run_id"], reason=CODE_IDENTITY_UNAVAILABLE,
        )
        bind_execution_context(ctx)
        install_process_identity(ctx)
        return ctx

    ordered = KanbanExecutionContext(
        role=ROLE_OWNER,
        db_path=coords["db_path"],
        board=coords["board"],
        task_id=coords["task_id"],
        run_id=coords["run_id"],
        claim_lock=coords["claim_lock"],
        worker_pid=live_pid,
        worker_started_at=fingerprint,
    )
    reason = CODE_IDENTITY_UNAVAILABLE
    while True:
        conn = _open_readonly(coords["db_path"])
        if conn is None:
            reason = CODE_IDENTITY_UNAVAILABLE
        else:
            try:
                rec = reconcile_owner(conn, ordered, require_lease=False, pid=live_pid)
            finally:
                conn.close()
            if rec.ok:
                bind_execution_context(ordered)
                install_process_identity(ordered)
                return ordered
            reason = rec.reason
            # Only the spawn/registration window is a wait: the row exists and is
            # ours but the launcher has not written the pid/fingerprint yet.
            trow = _registration_pending(ordered)
            if not trow:
                break
        if time.monotonic() >= deadline:
            break
        sleep_fn(REGISTRATION_POLL_SECONDS)

    ctx = replace(ordered, role=ROLE_FOREIGN_INSPECTOR, reason=reason or CODE_OWNERSHIP_LOST)
    bind_execution_context(ctx)
    install_process_identity(ctx)
    return ctx


def _registration_pending(ctx: KanbanExecutionContext) -> bool:
    """True while the launcher has not yet recorded this process's pid."""
    conn = _open_readonly(ctx.db_path) if ctx.db_path else None
    if conn is None:
        return False
    try:
        row = conn.execute(
            "SELECT status, current_run_id, claim_lock, worker_pid FROM tasks WHERE id = ?",
            (ctx.task_id,),
        ).fetchone()
        if row is None:
            return False
        if str(_row_value(row, "status", "")) != "running":
            return False
        if _row_value(row, "worker_pid") is not None:
            return False
        if _row_value(row, "current_run_id") is None or int(_row_value(row, "current_run_id")) != int(ctx.run_id or -1):
            return False
        return str(_row_value(row, "claim_lock", "")) == str(ctx.claim_lock or "")
    except sqlite3.Error:
        return False
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Model-facing snapshot (informational only — never a write grant)
# ---------------------------------------------------------------------------

def identity_snapshot(
    conn: Optional[sqlite3.Connection] = None,
    *,
    target_task_id: Optional[str] = None,
) -> dict[str, Any]:
    """Describe the caller's trusted identity for ``kanban_show`` / startup context.

    ``target_task_id`` is reported as a *target*, never relabelled as owned.
    ``claim_match`` is a reconciliation result, so it is ``False`` for every
    non-owner role and for an owner whose claim was superseded.
    """
    ctx = current_execution_context()
    if ctx is None:
        return {
            "role": ROLE_FOREIGN_INSPECTOR,
            "owned_task_id": None,
            "owned_run_id": None,
            "current_db_run_id": None,
            "claim_match": False,
            "reason": "no_execution_context",
            "target_task_id": target_task_id,
        }

    own_conn_closed = False
    rec: Optional[Reconciliation] = None
    if conn is not None and ctx.task_id and connection_db_path(conn) and (
        not ctx.db_path or _canonical(ctx.db_path) == _canonical(connection_db_path(conn))
    ):
        probe = conn
    elif ctx.task_id and ctx.db_path:
        probe = _open_readonly(ctx.db_path)
        own_conn_closed = probe is not None
    else:
        probe = None

    if probe is None:
        rec = None
    else:
        try:
            if ctx.role == ROLE_OWNER:
                rec = reconcile_owner(probe, ctx, require_lease=False)
            else:
                row = probe.execute(
                    "SELECT current_run_id FROM tasks WHERE id = ?", (ctx.task_id,),
                ).fetchone()
                rec = Reconciliation(
                    ok=False,
                    reason=ctx.reason,
                    current_db_run_id=_row_value(row, "current_run_id") if row is not None else None,
                    claim_match=False,
                )
        except sqlite3.Error:
            rec = Reconciliation(ok=False, reason=CODE_IDENTITY_UNAVAILABLE, claim_match=False)
        finally:
            if own_conn_closed and probe is not None:
                probe.close()

    snapshot: dict[str, Any] = {
        "role": ctx.role,
        "owned_task_id": ctx.task_id,
        "owned_run_id": ctx.run_id,
        "current_db_run_id": rec.current_db_run_id if rec else None,
        "claim_match": bool(rec and rec.ok),
        "reason": (rec.reason if rec else ctx.reason) or "",
        "target_task_id": target_task_id,
    }
    return snapshot


# ---------------------------------------------------------------------------
# The single lifecycle authorizer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LifecycleAuthorization:
    """Result of authorizing one lifecycle write (inside its write txn)."""

    role: str
    audit: Optional[dict[str, Any]] = None


def authorization_denied(message: str, *, code: str = CODE_AUTHORIZATION_DENIED, reason: str = "") -> KanbanAuthorizationError:
    return KanbanAuthorizationError(message, code=code, reason=reason)


def authorize_lifecycle_write(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    operation: str,
    now: Optional[int] = None,
) -> LifecycleAuthorization:
    """Authorize *operation* on *task_id* against *conn*'s live board state.

    Runs INSIDE the caller's ``write_txn`` so the reconciliation and the mutation
    serialize: whichever write transaction wins, the loser re-reads the resulting
    run before writing.  Denials raise :class:`KanbanAuthorizationError` before
    any task/run/event/notification/termination effect.
    """
    ctx = current_execution_context()
    if ctx is None:
        raise authorization_denied(f"no trusted Kanban execution identity for {operation!r}. {_ADMIN_INSTRUCTION}")

    if ctx.role == ROLE_DELEGATE_CHILD:
        raise authorization_denied(
            f"{operation} refused: delegate_task children are not Kanban run owners. Return findings to "
            "the parent agent; the owning worker or an authenticated administrator performs the handoff.",
            reason="delegate_child",
        )
    if ctx.role == ROLE_FOREIGN_INSPECTOR:
        raise authorization_denied(
            f"{operation} refused: this execution owns no live Kanban run for {task_id}, so it cannot "
            f"mutate that task's lifecycle. {_ADMIN_INSTRUCTION}"
            + (f" (reason: {ctx.reason})" if ctx.reason else ""),
            reason=ctx.reason or "foreign_inspector",
        )

    if ctx.role == ROLE_ADMIN_RECOVERY:
        if ctx.admin_action and ctx.admin_action != operation:
            raise authorization_denied(
                f"{operation} refused: this administrative request is scoped to {ctx.admin_action!r}.",
                reason="admin_action_scope",
            )
        if operation not in ADMIN_OPERATIONS:
            raise authorization_denied(
                f"{operation} refused: not an administrative recovery operation.", reason="admin_operation",
            )
        return LifecycleAuthorization(role=ctx.role, audit=_admin_audit(conn, ctx, task_id, operation))

    if ctx.role == ROLE_DISPATCHER_MAINTENANCE:
        if operation not in MAINTENANCE_OPERATIONS:
            raise authorization_denied(
                f"{operation} refused: not a dispatcher maintenance operation.", reason="maintenance_operation",
            )
        return LifecycleAuthorization(role=ctx.role)

    if ctx.role != ROLE_OWNER:
        raise authorization_denied(
            f"{operation} refused: unrecognized execution role {ctx.role!r}.", reason="unknown_role",
        )

    if operation not in OWNER_OPERATIONS:
        raise authorization_denied(
            f"{operation} refused: an owning worker may only complete/block/schedule/request-review/"
            "request-changes its own task. "
            f"{_ADMIN_INSTRUCTION}",
            reason="owner_operation",
        )
    if ctx.db_path and connection_db_path(conn) and _canonical(ctx.db_path) != _canonical(connection_db_path(conn)):
        raise authorization_denied(
            f"{operation} refused: this worker's authority is scoped to its own board.",
            reason="db_mismatch",
        )
    if not ctx.task_id or str(ctx.task_id) != str(task_id):
        raise authorization_denied(
            f"{operation} refused: this worker owns {ctx.task_id or 'no task'}, not {task_id}. "
            "Use kanban_comment to hand off information to other tasks.",
            reason="foreign_task",
        )
    rec = reconcile_owner(conn, ctx, now=now)
    if not rec.ok:
        raise authorization_denied(
            f"{operation} refused: this worker no longer owns the live run of {task_id} "
            f"({rec.reason}). The claim was completed, reclaimed or superseded; a stale worker must not "
            "mutate the successor's task.",
            code=CODE_OWNERSHIP_LOST if rec.reason.startswith(CODE_OWNERSHIP_LOST) or rec.reason == "not_running"
            else (rec.reason if rec.reason in {CODE_IDENTITY_UNAVAILABLE} else CODE_OWNERSHIP_LOST),
            reason=rec.reason,
        )
    if operation == "request_changes" and not rec.review_source_phase:
        raise authorization_denied(
            f"request_changes refused: the active run of {task_id} was not claimed from review.",
            code=CODE_REVIEW_PHASE_REQUIRED,
            reason="review_phase_required",
        )
    return LifecycleAuthorization(role=ctx.role)


def _admin_audit(
    conn: sqlite3.Connection, ctx: KanbanExecutionContext, task_id: str, operation: str,
) -> dict[str, Any]:
    """Audit payload for one administrative mutation (no tokens, no credentials)."""
    try:
        row = conn.execute(
            "SELECT status, current_run_id, claim_lock FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
    except sqlite3.Error:
        row = None
    old = {} if row is None else {
        "status": _row_value(row, "status"),
        "current_run_id": _row_value(row, "current_run_id"),
        "claim_lock_present": _row_value(row, "claim_lock") is not None,
    }
    return {
        "operation": operation,
        "principal": ctx.admin_principal,
        "transport": ctx.admin_transport,
        "reason": ctx.admin_reason,
        "board": ctx.board,
        "task_id": task_id,
        "previous": old,
    }


def is_admin_context() -> bool:
    ctx = current_execution_context()
    return bool(ctx and ctx.role == ROLE_ADMIN_RECOVERY)


def owner_is_current(*, conn: Optional[sqlite3.Connection] = None) -> bool:
    """Cheap liveness answer for guards: is this process still the live owner?"""
    ctx = current_execution_context()
    if ctx is None or ctx.role != ROLE_OWNER or not ctx.task_id:
        return False
    probe = conn
    opened = False
    if probe is None:
        probe = _open_readonly(ctx.db_path) if ctx.db_path else None
        opened = probe is not None
    if probe is None:
        return False
    try:
        return reconcile_owner(probe, ctx, require_lease=False).ok
    except sqlite3.Error:
        return False
    finally:
        if opened:
            probe.close()


__all__ = [
    "ADMIN_OPERATIONS",
    "CODE_AUTHORIZATION_DENIED",
    "CODE_IDENTITY_UNAVAILABLE",
    "CODE_OWNERSHIP_LOST",
    "CODE_REVIEW_PHASE_REQUIRED",
    "MAINTENANCE_OPERATIONS",
    "OWNER_OPERATIONS",
    "PROVENANCES",
    "PROVENANCE_DIRECT_CLI",
    "PROVENANCE_MANAGED_BOUND",
    "REGISTRATION_WAIT_SECONDS",
    "ROLE_ADMIN_RECOVERY",
    "ROLE_DELEGATE_CHILD",
    "ROLE_DISPATCHER_MAINTENANCE",
    "ROLE_FOREIGN_INSPECTOR",
    "ROLE_OWNER",
    "KanbanAuthorizationError",
    "KanbanExecutionContext",
    "LifecycleAuthorization",
    "Reconciliation",
    "admin_recovery_scope",
    "authorization_denied",
    "authorize_lifecycle_write",
    "bind_execution_context",
    "bootstrap_worker_identity",
    "connection_db_path",
    "current_execution_context",
    "delegate_child_scope",
    "dispatcher_maintenance_scope",
    "env_coordinates",
    "foreign_inspector_scope",
    "identity_snapshot",
    "is_admin_context",
    "owner_is_current",
    "reconcile_owner",
    "reset_execution_context",
    "scoped_execution_context",
]
