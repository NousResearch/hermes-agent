"""Atomic runtime registration for project repair and checker tasks.

The pure Wave 2B routers return mutation intents.  This module is the durable
SQLite adapter that applies one intent as one fenced aggregate transaction.
It deliberately performs no worker execution, notification sending, or gateway
lifecycle work.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import time
from dataclasses import dataclass, field as dataclass_field
from typing import Callable, Iterable, Mapping

from hermes_cli import kanban_db as kb
from hermes_cli.project_repair_router import (
    REGISTRATION_ALREADY_EXISTS,
    REGISTRATION_CREATED,
    REGISTRATION_STALE_SNAPSHOT,
    AtomicRepairRegistration,
    ProjectIdentity,
    ProjectVersionToken,
    RepairAction,
)
from hermes_cli.sqlite_util import write_txn

FailureInjector = Callable[[str], None]

DESTINATION_FOUND = "found"
DESTINATION_MISSING = "missing"


@dataclass(frozen=True)
class CheckerRegistrationAction:
    """One deterministic checker-task registration intent."""

    project: ProjectIdentity
    checker_identity: str
    idempotency_key: str
    candidate_snapshot_version: str
    candidate_id: str
    worker_profile: str
    task_contract: Mapping[str, object]
    notification_route_identities: tuple[str, ...]


@dataclass(frozen=True)
class AtomicCheckerRegistration:
    """Stable result from the checker aggregate transaction."""

    disposition: str
    checker_task_id: str | None = None
    checker_identity: str | None = None


@dataclass(frozen=True)
class ProjectTelegramDestination:
    """Gateway-consumable destination with privacy-safe representation."""

    status: str
    platform: str | None = None
    chat_id: str | None = dataclass_field(default=None, repr=False)
    thread_id: str | None = dataclass_field(default=None, repr=False)
    route_identity: str | None = None
    reason: str | None = None


def notification_route_identity(platform: str, chat_id: str, thread_id: str | None = None) -> str:
    """Return a stable non-reversible identity for one durable notify route."""

    values = (platform, chat_id, thread_id or "")
    if any(not isinstance(value, str) or not value for value in values[:2]):
        raise ValueError("platform and chat_id are required")
    canonical = json.dumps(values, ensure_ascii=True, separators=(",", ":"))
    return "subscription:sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def checker_registration_identity(
    project: ProjectIdentity,
    *,
    candidate_snapshot_version: str,
    candidate_id: str,
) -> str:
    """Derive the logical checker identity for one generation/candidate."""

    if not isinstance(project, ProjectIdentity):
        raise TypeError("project must be a ProjectIdentity")
    if not candidate_snapshot_version or not candidate_id:
        raise ValueError("candidate snapshot version and candidate id are required")
    payload = {
        "project_id": project.project_id,
        "board_id": project.board_id,
        "root_task_id": project.root_task_id,
        "generation": project.generation,
        "candidate_snapshot_version": candidate_snapshot_version,
        "candidate_id": candidate_id,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return "checker:sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolve_project_telegram_destination(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
) -> ProjectTelegramDestination:
    """Resolve one terminal Telegram route from generation-scoped durable state.

    Root-task subscriptions take precedence over inherited member routes.  Each
    tier is sorted by route fields, so the same snapshot always resolves the
    same destination.  No send is attempted and subscriber ownership metadata
    is neither selected nor returned.
    """

    if _project_row(conn, board_id, root_task_id, generation) is None:
        return ProjectTelegramDestination(
            status=DESTINATION_MISSING,
            reason="project_generation_missing",
        )
    task_ids = _project_task_ids(
        conn,
        board_id=board_id,
        root_task_id=root_task_id,
        generation=generation,
    )
    placeholders = ",".join("?" for _ in task_ids)
    rows = conn.execute(
        f"""
        SELECT task_id, platform, chat_id, thread_id
        FROM kanban_notify_subs
        WHERE task_id IN ({placeholders}) AND platform = 'telegram'
        ORDER BY CASE WHEN task_id = ? THEN 0 ELSE 1 END,
                 chat_id, thread_id, task_id
        """,
        (*task_ids, root_task_id),
    ).fetchall()
    if not rows:
        return ProjectTelegramDestination(
            status=DESTINATION_MISSING,
            reason="telegram_destination_missing",
        )
    row = rows[0]
    thread_id = row["thread_id"] or None
    return ProjectTelegramDestination(
        status=DESTINATION_FOUND,
        platform="telegram",
        chat_id=row["chat_id"],
        thread_id=thread_id,
        route_identity=notification_route_identity("telegram", row["chat_id"], thread_id),
    )


def register_project_repair(
    conn: sqlite3.Connection,
    action: RepairAction,
    expected_token: ProjectVersionToken,
    *,
    now: int | None = None,
    inject_failure: FailureInjector | None = None,
) -> AtomicRepairRegistration:
    """Create or resolve one repair task and persist its project aggregate.

    The existing identity fast path is checked *inside* the write transaction
    and only succeeds when every aggregate component is already present.  This
    makes restart replay idempotent even though the original version token is
    necessarily stale after the first successful registration.
    """

    if not isinstance(action, RepairAction):
        raise TypeError("action must be a RepairAction")
    if not isinstance(expected_token, ProjectVersionToken):
        raise TypeError("expected_token must be a ProjectVersionToken")
    if action.membership_kind != "repair" or action.required is not True:
        raise ValueError("repair registration requires required repair membership")
    if action.idempotency_key != action.repair_identity:
        raise ValueError("repair idempotency key must equal repair identity")
    contract_blob = kb.serialize_task_contract(dict(action.task_contract))
    current_time = _validated_now(now)

    with write_txn(conn):
        project_row = _project_row(conn, action.project.board_id, action.project.root_task_id, action.project.generation)
        if project_row is None:
            return AtomicRepairRegistration(REGISTRATION_STALE_SNAPSHOT)

        existing = _task_for_identity(conn, action.idempotency_key)
        if existing is not None:
            _verify_existing_repair(conn, action, existing["id"], project_row)
            return AtomicRepairRegistration(
                REGISTRATION_ALREADY_EXISTS,
                repair_task_id=existing["id"],
            )

        if not _token_matches(project_row, expected_token, now=current_time):
            return AtomicRepairRegistration(REGISTRATION_STALE_SNAPSHOT)
        if int(project_row["repair_generation"]) + 1 != action.repair_index:
            return AtomicRepairRegistration(REGISTRATION_STALE_SNAPSHOT)

        route_rows = _resolve_route_rows(
            conn,
            board_id=action.project.board_id,
            root_task_id=action.project.root_task_id,
            generation=action.project.generation,
            extra_task_ids=(action.failed_task_id,),
            identities=action.notification_route_identities,
        )
        source_task = conn.execute("SELECT * FROM tasks WHERE id = ?", (action.failed_task_id,)).fetchone()
        if source_task is None:
            raise ValueError("failed task does not exist")

        task_id = _insert_runtime_task(
            conn,
            title=f"Repair project task {action.failed_task_id}",
            body=_repair_binding(action),
            assignee=action.worker_profile,
            idempotency_key=action.idempotency_key,
            project_id=action.project.project_id,
            contract_blob=contract_blob,
            source_task=source_task,
            now=current_time,
        )
        _copy_routes(conn, task_id=task_id, rows=route_rows, now=current_time)
        _inject(inject_failure, "after_task")

        conn.execute(
            """
            INSERT INTO project_finalization_members
                (board_id, root_task_id, generation, task_id, membership_kind, required, created_at)
            VALUES (?, ?, ?, ?, 'repair', 1, ?)
            """,
            (
                action.project.board_id,
                action.project.root_task_id,
                action.project.generation,
                task_id,
                current_time,
            ),
        )
        binding = _repair_binding(action)
        kb._append_event(conn, task_id, "project_repair_registered", binding)
        _inject(inject_failure, "after_membership")

        updated = conn.execute(
            """
            UPDATE project_finalizations
               SET repair_generation = ?,
                   checker_verdict = NULL,
                   state = 'repairing',
                   updated_at = ?,
                   version = version + 1
             WHERE board_id = ? AND root_task_id = ? AND generation = ?
               AND version = ? AND lock_owner = ?
               AND COALESCE(lock_expires_at, 0) >= ?
            """,
            (
                action.repair_index,
                current_time,
                action.project.board_id,
                action.project.root_task_id,
                action.project.generation,
                expected_token.project_version,
                expected_token.lock_token,
                current_time,
            ),
        )
        if updated.rowcount != 1:
            raise RuntimeError("project repair aggregate compare-and-set failed")
        _inject(inject_failure, "after_project_update")
        return AtomicRepairRegistration(REGISTRATION_CREATED, repair_task_id=task_id)


def register_project_checker(
    conn: sqlite3.Connection,
    action: CheckerRegistrationAction,
    expected_token: ProjectVersionToken,
    *,
    now: int | None = None,
    inject_failure: FailureInjector | None = None,
) -> AtomicCheckerRegistration:
    """Create or resolve one current checker and persist its authority atomically."""

    if not isinstance(action, CheckerRegistrationAction):
        raise TypeError("action must be a CheckerRegistrationAction")
    if not isinstance(action.project, ProjectIdentity):
        raise TypeError("action project must be a ProjectIdentity")
    if not isinstance(expected_token, ProjectVersionToken):
        raise TypeError("expected_token must be a ProjectVersionToken")
    expected_identity = checker_registration_identity(
        action.project,
        candidate_snapshot_version=action.candidate_snapshot_version,
        candidate_id=action.candidate_id,
    )
    if action.checker_identity != expected_identity or action.idempotency_key != expected_identity:
        raise ValueError("checker identity must match the project candidate binding")
    contract_blob = kb.serialize_task_contract(dict(action.task_contract))
    current_time = _validated_now(now)

    with write_txn(conn):
        project_row = _project_row(
            conn,
            action.project.board_id,
            action.project.root_task_id,
            action.project.generation,
        )
        if project_row is None:
            return AtomicCheckerRegistration(REGISTRATION_STALE_SNAPSHOT)

        existing = _task_for_identity(conn, action.idempotency_key)
        if existing is not None:
            if _existing_checker_is_current(conn, action, existing["id"], project_row):
                return AtomicCheckerRegistration(
                    REGISTRATION_ALREADY_EXISTS,
                    checker_task_id=existing["id"],
                    checker_identity=action.checker_identity,
                )
            return AtomicCheckerRegistration(REGISTRATION_STALE_SNAPSHOT)

        if not _token_matches(project_row, expected_token, now=current_time):
            return AtomicCheckerRegistration(REGISTRATION_STALE_SNAPSHOT)

        route_rows = _resolve_route_rows(
            conn,
            board_id=action.project.board_id,
            root_task_id=action.project.root_task_id,
            generation=action.project.generation,
            identities=action.notification_route_identities,
        )
        source_task = conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (action.project.root_task_id,)
        ).fetchone()
        if source_task is None:
            raise ValueError("project root task does not exist")

        task_id = _insert_runtime_task(
            conn,
            title=f"Check project generation {action.project.generation}",
            body=_checker_binding(action),
            assignee=action.worker_profile,
            idempotency_key=action.idempotency_key,
            project_id=action.project.project_id,
            contract_blob=contract_blob,
            source_task=source_task,
            now=current_time,
        )
        _copy_routes(conn, task_id=task_id, rows=route_rows, now=current_time)
        _inject(inject_failure, "after_task")

        # Preserve old checker task history as optional support while ensuring
        # the frozen evaluator sees exactly one authoritative checker member.
        conn.execute(
            """
            UPDATE project_finalization_members
               SET membership_kind = 'support', required = 0
             WHERE board_id = ? AND root_task_id = ? AND generation = ?
               AND membership_kind = 'checker'
            """,
            (
                action.project.board_id,
                action.project.root_task_id,
                action.project.generation,
            ),
        )
        conn.execute(
            """
            INSERT INTO project_finalization_members
                (board_id, root_task_id, generation, task_id, membership_kind, required, created_at)
            VALUES (?, ?, ?, ?, 'checker', 1, ?)
            """,
            (
                action.project.board_id,
                action.project.root_task_id,
                action.project.generation,
                task_id,
                current_time,
            ),
        )
        kb._append_event(
            conn,
            task_id,
            "project_checker_registered",
            _checker_binding(action),
        )
        _inject(inject_failure, "after_membership")

        updated = conn.execute(
            """
            UPDATE project_finalizations
               SET final_checker_task_id = ?,
                   checker_verdict = NULL,
                   state = 'evaluating',
                   updated_at = ?,
                   version = version + 1
             WHERE board_id = ? AND root_task_id = ? AND generation = ?
               AND version = ? AND lock_owner = ?
               AND COALESCE(lock_expires_at, 0) >= ?
            """,
            (
                task_id,
                current_time,
                action.project.board_id,
                action.project.root_task_id,
                action.project.generation,
                expected_token.project_version,
                expected_token.lock_token,
                current_time,
            ),
        )
        if updated.rowcount != 1:
            raise RuntimeError("project checker aggregate compare-and-set failed")
        _inject(inject_failure, "after_project_update")
        return AtomicCheckerRegistration(
            REGISTRATION_CREATED,
            checker_task_id=task_id,
            checker_identity=action.checker_identity,
        )


def _project_row(
    conn: sqlite3.Connection,
    board_id: str,
    root_task_id: str,
    generation: int,
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM project_finalizations WHERE board_id = ? AND root_task_id = ? AND generation = ?",
        (board_id, root_task_id, generation),
    ).fetchone()


def _task_for_identity(conn: sqlite3.Connection, idempotency_key: str) -> sqlite3.Row | None:
    rows = conn.execute(
        "SELECT * FROM tasks WHERE idempotency_key = ? ORDER BY created_at, id",
        (idempotency_key,),
    ).fetchall()
    if len(rows) > 1:
        raise RuntimeError("duplicate durable task identity")
    return rows[0] if rows else None


def _token_matches(row: sqlite3.Row, token: ProjectVersionToken, *, now: int) -> bool:
    return (
        int(row["version"]) == token.project_version
        and row["lock_owner"] == token.lock_token
        and row["lock_owner"] is not None
        and row["lock_expires_at"] is not None
        and int(row["lock_expires_at"]) >= now
    )


def _verify_existing_repair(
    conn: sqlite3.Connection,
    action: RepairAction,
    task_id: str,
    project_row: sqlite3.Row,
) -> None:
    member = conn.execute(
        """
        SELECT membership_kind, required FROM project_finalization_members
        WHERE board_id = ? AND root_task_id = ? AND generation = ? AND task_id = ?
        """,
        (
            action.project.board_id,
            action.project.root_task_id,
            action.project.generation,
            task_id,
        ),
    ).fetchone()
    event = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'project_repair_registered' ORDER BY id",
        (task_id,),
    ).fetchall()
    expected_binding = _repair_binding(action)
    valid_event = len(event) == 1 and json.loads(event[0]["payload"]) == expected_binding
    if (
        member is None
        or member["membership_kind"] != "repair"
        or int(member["required"]) != 1
        or not valid_event
        or int(project_row["repair_generation"]) < action.repair_index
    ):
        raise RuntimeError("existing repair identity has split aggregate state")


def _existing_checker_is_current(
    conn: sqlite3.Connection,
    action: CheckerRegistrationAction,
    task_id: str,
    project_row: sqlite3.Row,
) -> bool:
    member = conn.execute(
        """
        SELECT membership_kind, required FROM project_finalization_members
        WHERE board_id = ? AND root_task_id = ? AND generation = ? AND task_id = ?
        """,
        (
            action.project.board_id,
            action.project.root_task_id,
            action.project.generation,
            task_id,
        ),
    ).fetchone()
    events = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'project_checker_registered' ORDER BY id",
        (task_id,),
    ).fetchall()
    return (
        project_row["final_checker_task_id"] == task_id
        and member is not None
        and member["membership_kind"] == "checker"
        and int(member["required"]) == 1
        and len(events) == 1
        and json.loads(events[0]["payload"]) == _checker_binding(action)
    )


def _checker_binding(action: CheckerRegistrationAction) -> dict[str, object]:
    return {
        "checker_identity": action.checker_identity,
        "candidate_snapshot_version": action.candidate_snapshot_version,
        "candidate_id": action.candidate_id,
        "checker_profile": action.worker_profile,
    }


def _repair_binding(action: RepairAction) -> dict[str, object]:
    return {
        "repair_identity": action.repair_identity,
        "failed_task_id": action.failed_task_id,
        "failed_run_id": action.failed_run_id,
        "failure_fingerprint": action.failure_fingerprint,
        "repair_index": action.repair_index,
        "task_retry_index": action.task_retry_index,
    }


def _project_task_ids(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    extra_task_ids: Iterable[str] = (),
) -> tuple[str, ...]:
    ids = {root_task_id, *extra_task_ids}
    ids.update(
        row["task_id"]
        for row in conn.execute(
            """
            SELECT task_id FROM project_finalization_members
            WHERE board_id = ? AND root_task_id = ? AND generation = ?
            """,
            (board_id, root_task_id, generation),
        )
    )
    return tuple(sorted(ids))


def _resolve_route_rows(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    identities: Iterable[str],
    extra_task_ids: Iterable[str] = (),
) -> tuple[sqlite3.Row, ...]:
    requested = set(identities)
    if not requested:
        raise ValueError("at least one notification route identity is required")
    task_ids = _project_task_ids(
        conn,
        board_id=board_id,
        root_task_id=root_task_id,
        generation=generation,
        extra_task_ids=extra_task_ids,
    )
    placeholders = ",".join("?" for _ in task_ids)
    rows = conn.execute(
        f"SELECT * FROM kanban_notify_subs WHERE task_id IN ({placeholders}) "
        "ORDER BY platform, chat_id, thread_id, task_id",
        task_ids,
    ).fetchall()
    matched: dict[tuple[str, str, str], sqlite3.Row] = {}
    for row in rows:
        identity = notification_route_identity(row["platform"], row["chat_id"], row["thread_id"])
        if identity in requested:
            matched.setdefault((row["platform"], row["chat_id"], row["thread_id"]), row)
    if set(
        notification_route_identity(row["platform"], row["chat_id"], row["thread_id"])
        for row in matched.values()
    ) != requested:
        raise ValueError("notification route identity is not durable project state")
    return tuple(matched[key] for key in sorted(matched))


def _insert_runtime_task(
    conn: sqlite3.Connection,
    *,
    title: str,
    body: dict[str, object],
    assignee: str,
    idempotency_key: str,
    project_id: str,
    contract_blob: str,
    source_task: sqlite3.Row,
    now: int,
) -> str:
    contract = kb.deserialize_task_contract(contract_blob)
    admission = kb.evaluate_admission(
        contract=contract,
        workspace_kind=source_task["workspace_kind"],
        workspace_path=source_task["workspace_path"],
        has_notification_subscription=True,
        contract_present=True,
    )
    status = "ready" if admission.admitted else "todo"
    for _ in range(2):
        task_id = "t_" + secrets.token_hex(4)
        try:
            conn.execute(
                """
                INSERT INTO tasks (
                    id, title, body, assignee, status, priority, created_by, created_at,
                    workspace_kind, workspace_path, branch_name, project_id, tenant,
                    idempotency_key, contract
                ) VALUES (?, ?, ?, ?, ?, 100, 'project-runtime', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    title,
                    json.dumps(body, sort_keys=True, separators=(",", ":")),
                    assignee,
                    status,
                    now,
                    source_task["workspace_kind"],
                    source_task["workspace_path"],
                    source_task["branch_name"],
                    project_id,
                    source_task["tenant"],
                    idempotency_key,
                    contract_blob,
                ),
            )
            kb._append_event(
                conn,
                task_id,
                "created",
                {
                    "assignee": assignee,
                    "status": status,
                    "parents": [],
                    "tenant": source_task["tenant"],
                    "branch_name": source_task["branch_name"],
                    "skills": None,
                    "goal_mode": None,
                },
            )
            if not admission.admitted:
                kb._append_admission_rejection_event(
                    conn,
                    task_id,
                    admission,
                    phase="create",
                    event_kind="admission_rejected",
                )
            return task_id
        except sqlite3.IntegrityError:
            continue
    raise RuntimeError("could not allocate runtime task identity")


def _copy_routes(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    rows: Iterable[sqlite3.Row],
    now: int,
) -> None:
    for row in rows:
        kb._insert_notify_sub_unlocked(
            conn,
            task_id=task_id,
            platform=row["platform"],
            chat_id=row["chat_id"],
            thread_id=row["thread_id"],
            user_id=row["user_id"],
            notifier_profile=row["notifier_profile"],
            created_at=now,
            last_event_id=0,
        )


def _validated_now(now: int | None) -> int:
    value = int(time.time()) if now is None else now
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("now must be a non-negative integer")
    return value


def _inject(injector: FailureInjector | None, point: str) -> None:
    if injector is not None:
        injector(point)


__all__ = [
    "DESTINATION_FOUND",
    "DESTINATION_MISSING",
    "AtomicCheckerRegistration",
    "CheckerRegistrationAction",
    "ProjectTelegramDestination",
    "checker_registration_identity",
    "notification_route_identity",
    "register_project_checker",
    "register_project_repair",
    "resolve_project_telegram_destination",
]
