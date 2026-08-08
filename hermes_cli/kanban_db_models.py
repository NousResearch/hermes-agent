"""SQLite-backed Kanban data models and schema (extracted from kanban_db.py)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Optional

@dataclass
class Task:
    """In-memory view of a row from the ``tasks`` table."""

    id: str
    title: str
    body: Optional[str]
    assignee: Optional[str]
    status: str
    priority: int
    created_by: Optional[str]
    created_at: int
    started_at: Optional[int]
    completed_at: Optional[int]
    workspace_kind: str
    workspace_path: Optional[str]
    claim_lock: Optional[str]
    claim_expires: Optional[int]
    tenant: Optional[str]
    branch_name: Optional[str] = None
    project_id: Optional[str] = None
    result: Optional[str] = None
    idempotency_key: Optional[str] = None
    # Unified non-success counter. Incremented on any of:
    #   * spawn failure (dispatcher couldn't launch the worker)
    #   * timed_out outcome (worker exceeded max_runtime_seconds)
    #   * crashed outcome (worker PID vanished)
    # Reset to 0 only on a successful completion. See
    # ``_record_task_failure`` for the circuit-breaker trip rule.
    # (Pre-rename column: ``spawn_failures``.)
    consecutive_failures: int = 0
    worker_pid: Optional[int] = None
    # Short excerpt of the last failure's error text (any outcome, not
    # just spawn). Pre-rename column: ``last_spawn_error``.
    last_failure_error: Optional[str] = None
    max_runtime_seconds: Optional[int] = None
    last_heartbeat_at: Optional[int] = None
    current_run_id: Optional[int] = None
    workflow_template_id: Optional[str] = None
    current_step_key: Optional[str] = None
    # Force-loaded skills for the worker on this task (passed via
    # --skills). Stored as a JSON array of skill names. None = use only
    # the defaults; empty list = explicitly no extra skills.
    skills: Optional[list] = None
    model_override: Optional[str] = None
    # Provider that ``model_override`` belongs to. When set, the dispatcher
    # passes ``--provider <name>`` alongside ``-m <model>`` so the worker
    # resolves the model against the right backend instead of the profile's
    # configured provider. NULL = worker profile's provider resolves the
    # model (pre-existing behaviour). Solves the "model from provider A,
    # profile configured for provider B" mismatch class.
    provider_override: Optional[str] = None
    # Per-task reasoning effort for the worker (one of
    # ``hermes_constants.VALID_REASONING_EFFORTS``, or ``"none"`` for thinking
    # off). When set, the dispatcher passes ``--reasoning <level>`` so the
    # worker runs at that depth regardless of the profile's
    # ``agent.reasoning_effort``. NULL = the worker profile's own setting.
    reasoning_effort: Optional[str] = None
    # Per-task override for the consecutive-failure circuit breaker.
    # The value is the failure count at which the breaker trips — e.g.
    # ``max_retries=1`` blocks on the first failure (zero retries),
    # ``max_retries=3`` blocks on the third (two retries allowed).
    # ``None`` (the common case) falls through to the dispatcher-level
    # ``kanban.failure_limit`` config, and then to ``DEFAULT_FAILURE_LIMIT``.
    # Name matches the ``--max-retries`` CLI flag on ``kanban create``.
    max_retries: Optional[int] = None
    # When True, the dispatched worker runs in a Ralph-style goal loop
    # (the same engine behind the ``/goal`` slash command): after each
    # turn an auxiliary judge model evaluates the worker's response
    # against this card's title/body (treated as the goal). If the judge
    # says "not done" and budget remains, the worker is fed a
    # continuation prompt IN THE SAME SESSION and keeps working until the
    # judge agrees, the goal-turn budget is exhausted (→ kanban_block),
    # or the worker explicitly blocks/completes. ``False`` (default) =
    # the classic single-shot worker. ``goal_max_turns`` bounds the loop.
    goal_mode: bool = False
    # Goal-loop turn budget for ``goal_mode`` workers. ``None`` falls
    # through to the goals engine default (``goals.DEFAULT_MAX_TURNS``).
    goal_max_turns: Optional[int] = None
    # Originating chat/agent session id, when the task was created from
    # within an agent loop that propagated ``HERMES_SESSION_ID``. NULL for
    # tasks created from the CLI, the dashboard, or any path that doesn't
    # set the env var. Lets clients render a per-session board without
    # relying on tenant + time-window heuristics.
    session_id: Optional[str] = None
    # Typed block reason (one of VALID_BLOCK_KINDS) or None for legacy/un-typed
    # blocks. Set by ``block_task``; preserved across unblock so a re-block for
    # the same kind is recognisable as an unblock↔re-block loop.
    block_kind: Optional[str] = None
    # Unblock-loop counter. See the column comment in SCHEMA_SQL and
    # ``BLOCK_RECURRENCE_LIMIT``. Reset only on successful completion.
    block_recurrences: int = 0

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Task":
        keys = set(row.keys())
        # Parse skills JSON blob if present
        skills_value: Optional[list] = None
        if "skills" in keys and row["skills"]:
            try:
                parsed = json.loads(row["skills"])
                if isinstance(parsed, list):
                    skills_value = [str(s) for s in parsed if s]
            except Exception:
                skills_value = None
        return cls(
            id=row["id"],
            title=row["title"],
            body=row["body"],
            assignee=row["assignee"],
            status=row["status"],
            priority=row["priority"],
            created_by=row["created_by"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            workspace_kind=row["workspace_kind"],
            workspace_path=row["workspace_path"],
            branch_name=row["branch_name"] if "branch_name" in keys else None,
            project_id=row["project_id"] if "project_id" in keys else None,
            claim_lock=row["claim_lock"],
            claim_expires=row["claim_expires"],
            tenant=row["tenant"] if "tenant" in keys else None,
            result=row["result"] if "result" in keys else None,
            idempotency_key=row["idempotency_key"] if "idempotency_key" in keys else None,
            consecutive_failures=(
                row["consecutive_failures"] if "consecutive_failures" in keys
                # Pre-migration fallback: ``_migrate_add_optional_columns`` always
                # adds ``consecutive_failures`` now, so this branch is only reachable
                # on a DB that was never opened since pre-#20410 code ran. Keep for
                # belt-and-suspenders safety; in practice it is dead code post-migration.
                else (row["spawn_failures"] if "spawn_failures" in keys else 0)
            ),
            worker_pid=row["worker_pid"] if "worker_pid" in keys else None,
            last_failure_error=(
                row["last_failure_error"] if "last_failure_error" in keys
                # Same belt-and-suspenders fallback as consecutive_failures above.
                else (row["last_spawn_error"] if "last_spawn_error" in keys else None)
            ),
            max_runtime_seconds=(
                row["max_runtime_seconds"] if "max_runtime_seconds" in keys else None
            ),
            last_heartbeat_at=(
                row["last_heartbeat_at"] if "last_heartbeat_at" in keys else None
            ),
            current_run_id=(
                row["current_run_id"] if "current_run_id" in keys else None
            ),
            workflow_template_id=(
                row["workflow_template_id"] if "workflow_template_id" in keys else None
            ),
            current_step_key=(
                row["current_step_key"] if "current_step_key" in keys else None
            ),
            skills=skills_value,
            model_override=row["model_override"] if "model_override" in keys and row["model_override"] else None,
            provider_override=(
                row["provider_override"]
                if "provider_override" in keys and row["provider_override"]
                else None
            ),
            reasoning_effort=(
                row["reasoning_effort"]
                if "reasoning_effort" in keys and row["reasoning_effort"]
                else None
            ),
            max_retries=(
                row["max_retries"] if "max_retries" in keys else None
            ),
            goal_mode=(
                bool(row["goal_mode"]) if "goal_mode" in keys and row["goal_mode"] else False
            ),
            goal_max_turns=(
                row["goal_max_turns"] if "goal_max_turns" in keys and row["goal_max_turns"] else None
            ),
            session_id=(
                row["session_id"] if "session_id" in keys else None
            ),
            block_kind=(
                row["block_kind"] if "block_kind" in keys and row["block_kind"] else None
            ),
            block_recurrences=(
                int(row["block_recurrences"])
                if "block_recurrences" in keys and row["block_recurrences"] is not None
                else 0
            ),
        )


@dataclass
class Run:
    """In-memory view of a ``task_runs`` row.

    A run is one attempt to execute a task — created on claim, closed
    on complete/block/crash/timeout/spawn_failure/reclaim. Multiple runs
    per task when retries happen. Carries the claim machinery, PID,
    heartbeat, and the structured handoff summary that downstream workers
    read via ``build_worker_context``.
    """

    id: int
    task_id: str
    profile: Optional[str]
    step_key: Optional[str]
    status: str
    claim_lock: Optional[str]
    claim_expires: Optional[int]
    worker_pid: Optional[int]
    max_runtime_seconds: Optional[int]
    last_heartbeat_at: Optional[int]
    started_at: int
    ended_at: Optional[int]
    outcome: Optional[str]
    summary: Optional[str]
    metadata: Optional[dict]
    error: Optional[str]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Run":
        try:
            meta = json.loads(row["metadata"]) if row["metadata"] else None
        except Exception:
            meta = None
        return cls(
            id=int(row["id"]),
            task_id=row["task_id"],
            profile=row["profile"],
            step_key=row["step_key"],
            status=row["status"],
            claim_lock=row["claim_lock"],
            claim_expires=row["claim_expires"],
            worker_pid=row["worker_pid"],
            max_runtime_seconds=row["max_runtime_seconds"],
            last_heartbeat_at=row["last_heartbeat_at"],
            started_at=int(row["started_at"]),
            ended_at=(int(row["ended_at"]) if row["ended_at"] is not None else None),
            outcome=row["outcome"],
            summary=row["summary"],
            metadata=meta,
            error=row["error"],
        )


@dataclass
class Comment:
    id: int
    task_id: str
    author: str
    body: str
    created_at: int


@dataclass
class Attachment:
    """In-memory view of a row from the ``task_attachments`` table."""

    id: int
    task_id: str
    filename: str
    stored_path: str
    content_type: Optional[str]
    size: int
    uploaded_by: Optional[str]
    created_at: int


@dataclass
class Event:
    id: int
    task_id: str
    kind: str
    payload: Optional[dict]
    created_at: int
    run_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    id                   TEXT PRIMARY KEY,
    title                TEXT NOT NULL,
    body                 TEXT,
    assignee             TEXT,
    status               TEXT NOT NULL,
    priority             INTEGER DEFAULT 0,
    created_by           TEXT,
    created_at           INTEGER NOT NULL,
    started_at           INTEGER,
    completed_at         INTEGER,
    workspace_kind       TEXT NOT NULL DEFAULT 'scratch',
    workspace_path       TEXT,
    branch_name          TEXT,
    -- Optional link to a first-class Project (hermes_cli/projects_db). When set,
    -- the task's worktree is anchored under the project's primary repo with a
    -- deterministic branch name instead of a random wt/<task-id> fallback.
    project_id           TEXT,
    claim_lock           TEXT,
    claim_expires        INTEGER,
    tenant               TEXT,
    result               TEXT,
    idempotency_key      TEXT,
    -- Unified consecutive-failure counter. Incremented on spawn
    -- failure, timeout, or crash; reset only on successful completion.
    -- The circuit breaker in _record_task_failure trips when this
    -- exceeds DEFAULT_FAILURE_LIMIT consecutive non-successes.
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    worker_pid           INTEGER,
    -- Short excerpt of the most recent failure's error text.
    last_failure_error   TEXT,
    max_runtime_seconds  INTEGER,
    last_heartbeat_at    INTEGER,
    -- Pointer into task_runs for the currently-active run (NULL if no
    -- run is in-flight). Denormalised for cheap reads.
    current_run_id       INTEGER,
    -- Forward-compat for v2 workflow routing. In v1 the kernel writes
    -- these when the task is opted into a template but otherwise ignores
    -- them; the dispatcher doesn't consult them for routing yet.
    workflow_template_id TEXT,
    current_step_key     TEXT,
    -- Force-loaded skills for the worker on this task, stored as JSON.
    -- Passed to the worker via `--skills`. NULL or empty array = no extras.
    skills               TEXT,
    -- Per-task model override. When set, the dispatcher passes -m <model>
    -- to the worker, overriding the profile's default model. NULL = use
    -- the profile default.
    model_override       TEXT,
    -- Provider the model override belongs to. When set (alongside
    -- model_override), the dispatcher passes --provider <name> so the
    -- worker resolves the model against the right backend instead of the
    -- profile's configured provider. NULL = profile provider.
    provider_override    TEXT,
    -- Per-task reasoning effort for the worker (minimal|low|medium|high|
    -- xhigh|max|ultra, or 'none' for thinking off). When set, the dispatcher
    -- passes --reasoning <level> so the worker runs at that depth regardless
    -- of the profile's agent.reasoning_effort. NULL = profile setting.
    reasoning_effort     TEXT,
    -- Per-task override for the consecutive-failure circuit breaker.
    -- The value is the failure count at which the breaker trips — e.g.
    -- ``max_retries=1`` blocks on the first failure. NULL (the common
    -- case) falls through to the dispatcher-level ``kanban.failure_limit``
    -- config and then ``DEFAULT_FAILURE_LIMIT``.
    max_retries          INTEGER,
    -- When 1, the dispatched worker runs in a Ralph-style goal loop: an
    -- auxiliary judge re-evaluates the worker's response against the
    -- card title/body after each turn and feeds a continuation prompt
    -- back into the SAME session until the judge agrees the work is done
    -- or ``goal_max_turns`` is exhausted. NULL/0 = classic single-shot
    -- worker (the default).
    goal_mode            INTEGER NOT NULL DEFAULT 0,
    -- Goal-loop turn budget for ``goal_mode`` workers. NULL = use the
    -- goals-engine default.
    goal_max_turns       INTEGER,
    -- Originating chat/agent session id when the task was created from
    -- inside an agent loop that propagated ``HERMES_SESSION_ID``. NULL
    -- for tasks created from the CLI, dashboard, or any path that doesn't
    -- set the env var. Indexed so per-session list queries stay cheap on
    -- larger boards.
    session_id           TEXT,
    -- Typed block reason set by ``block_task`` (one of VALID_BLOCK_KINDS, or
    -- NULL for legacy/un-typed blocks). Drives routing: ``dependency`` never
    -- sits in ``blocked`` (goes to ``todo`` for parent-gating); the others go
    -- to ``blocked`` for a human. Preserved across unblock so a re-block for
    -- the SAME kind can be recognised as a loop.
    block_kind           TEXT,
    -- Unblock-loop counter. Incremented each time a task is re-blocked for the
    -- same truly-blocked reason after having been unblocked. When it reaches
    -- BLOCK_RECURRENCE_LIMIT the task is routed to ``triage`` instead of
    -- ``blocked`` so a cron can't spin it forever. Reset to 0 only on a
    -- successful completion — NOT on unblock (resetting on unblock is exactly
    -- the amnesia that let the loop run unbounded).
    block_recurrences    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS task_links (
    parent_id  TEXT NOT NULL,
    child_id   TEXT NOT NULL,
    PRIMARY KEY (parent_id, child_id)
);

CREATE TABLE IF NOT EXISTS task_comments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL,
    author     TEXT NOT NULL,
    body       TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS task_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL,
    run_id     INTEGER,
    kind       TEXT NOT NULL,
    payload    TEXT,
    created_at INTEGER NOT NULL
);

-- Historical attempt record. Each time the dispatcher claims a task, a
-- new row is created here; claim state, PID, heartbeat, runtime cap,
-- and structured summary all live on the run, not the task. Multiple
-- rows per task id when the task was retried after crash/timeout/block.
-- v2 of the kanban schema will use ``step_key`` to drive per-stage
-- workflow routing; in v1 the column is nullable and unused (kernel
-- ignores it).
CREATE TABLE IF NOT EXISTS task_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id             TEXT NOT NULL,
    profile             TEXT,
    step_key            TEXT,
    status              TEXT NOT NULL,
    -- status: running | done | blocked | crashed | timed_out | failed | released
    claim_lock          TEXT,
    claim_expires       INTEGER,
    worker_pid          INTEGER,
    max_runtime_seconds INTEGER,
    last_heartbeat_at   INTEGER,
    started_at          INTEGER NOT NULL,
    ended_at            INTEGER,
    outcome             TEXT,
    -- outcome: completed | blocked | crashed | timed_out | spawn_failed |
    --          gave_up | reclaimed | (null while still running)
    summary             TEXT,
    metadata            TEXT,
    error               TEXT
);

-- Files attached to a task (PDFs, images, source documents). The blob
-- lives on disk under ``attachments_root(board)/<task_id>/<stored_name>``;
-- this row carries metadata + the absolute ``stored_path`` so the
-- dashboard can list/download and ``build_worker_context`` can surface
-- the absolute path to the worker (which has full file-tool access). See
-- #35338.
CREATE TABLE IF NOT EXISTS task_attachments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id      TEXT NOT NULL,
    filename     TEXT NOT NULL,
    stored_path  TEXT NOT NULL,
    content_type TEXT,
    size         INTEGER NOT NULL DEFAULT 0,
    uploaded_by  TEXT,
    created_at   INTEGER NOT NULL
);

-- Subscription from a gateway source (platform + chat + thread) to a
-- task. The gateway's kanban-notifier watcher tails task_events and
-- pushes ``completed`` / ``blocked`` / ``spawn_auto_blocked`` events to
-- the original requester so human-in-the-loop workflows close the loop.
CREATE TABLE IF NOT EXISTS kanban_notify_subs (
    task_id       TEXT NOT NULL,
    platform      TEXT NOT NULL,
    chat_id       TEXT NOT NULL,
    chat_type     TEXT,
    thread_id     TEXT NOT NULL DEFAULT '',
    user_id       TEXT,
    notifier_profile TEXT,
    delivery_metadata TEXT,
    created_at    INTEGER NOT NULL,
    last_event_id INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (task_id, platform, chat_id, thread_id)
);

CREATE INDEX IF NOT EXISTS idx_tasks_assignee_status ON tasks(assignee, status);
CREATE INDEX IF NOT EXISTS idx_tasks_status          ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_links_child           ON task_links(child_id);
CREATE INDEX IF NOT EXISTS idx_links_parent          ON task_links(parent_id);
CREATE INDEX IF NOT EXISTS idx_comments_task         ON task_comments(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_events_task           ON task_events(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_runs_task             ON task_runs(task_id, started_at);
CREATE INDEX IF NOT EXISTS idx_runs_status           ON task_runs(status);
CREATE INDEX IF NOT EXISTS idx_attachments_task      ON task_attachments(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_notify_task           ON kanban_notify_subs(task_id);
"""
