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
from typing import Any, Callable, Iterable, Mapping, Sequence

from hermes_cli import kanban_db as kb
from hermes_cli.profiles import normalize_profile_name, profile_exists, validate_profile_name
from hermes_cli.project_finalization_contract import (
    CHECKER_VERDICTS,
    ProjectFinalization,
    ensure_project_finalization_schema,
    get_project_finalization,
    validate_notification_policy,
    validate_repair_budget,
    validate_retention_days,
)
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
ADMISSION_CREATED = "created"
ADMISSION_ALREADY_ADMITTED = "already_admitted"

CHECKER_EVIDENCE_KINDS = frozenset({"command", "file", "test", "review", "other"})


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
class AtomicProjectAdmission:
    """Stable, privacy-safe result of admitting an existing task set."""

    disposition: str
    finalization: ProjectFinalization
    admission_key: str


@dataclass(frozen=True)
class CheckerVerdictSubmission:
    """Result of the structured checker-verdict protocol."""

    disposition: str
    checker_task_id: str
    verdict: str
    completed: bool


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


def _canonical_digest(prefix: str, payload: Mapping[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return prefix + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _pending_checker_identity(admission_key: str, *, generation: int, repair_index: int) -> str:
    return _canonical_digest(
        "pending-checker:sha256:",
        {
            "admission_key": admission_key,
            "generation": generation,
            "repair_index": repair_index,
        },
    )


def _admission_identity(
    *,
    board_id: str,
    root_task_id: str,
    required_task_ids: Sequence[str],
    checker_profile: str,
    notification_route_identity_value: str,
    notification_policy: str,
    retention_days: int,
    repair_budget: int,
) -> str:
    return _canonical_digest(
        "admission:sha256:",
        {
            "board_id": board_id,
            "root_task_id": root_task_id,
            "required_task_ids": list(required_task_ids),
            "checker_profile": checker_profile,
            "notification_route_identity": notification_route_identity_value,
            "notification_policy": notification_policy,
            "retention_days": retention_days,
            "repair_budget": repair_budget,
        },
    )


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

    project_row = _project_row(conn, board_id, root_task_id, generation)
    if project_row is None:
        return ProjectTelegramDestination(
            status=DESTINATION_MISSING,
            reason="project_generation_missing",
        )
    if project_row["admission_key"] is not None:
        rows = conn.execute(
            """
            SELECT platform, chat_id, thread_id
              FROM kanban_notify_subs
             WHERE task_id = ? AND platform = 'telegram'
             ORDER BY chat_id, thread_id
            """,
            (root_task_id,),
        ).fetchall()
        if len(rows) != 1:
            return ProjectTelegramDestination(
                status=DESTINATION_MISSING,
                reason="admitted_telegram_destination_ambiguous",
            )
        row = rows[0]
        thread_id = row["thread_id"] or None
        identity = notification_route_identity("telegram", row["chat_id"], thread_id)
        if identity != project_row["notification_route_identity"]:
            return ProjectTelegramDestination(
                status=DESTINATION_MISSING,
                reason="admitted_telegram_destination_changed",
            )
        return ProjectTelegramDestination(
            status=DESTINATION_FOUND,
            platform="telegram",
            chat_id=row["chat_id"],
            thread_id=thread_id,
            route_identity=identity,
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


def admit_existing_project(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    required_task_ids: Sequence[str],
    checker_profile: str,
    notification_policy: str = "project_summary",
    retention_days: int = 3,
    repair_budget: int = 1,
    now: int | None = None,
    inject_failure: FailureInjector | None = None,
) -> AtomicProjectAdmission:
    """Atomically admit an explicit set of existing tasks as one project.

    Admission never creates or decomposes implementation work.  It validates
    the existing task contracts, one root-owned Telegram route, and an
    independent checker profile before persisting any project authority.
    """

    if not isinstance(board_id, str) or not board_id.strip():
        raise ValueError("board_id is required")
    if not isinstance(root_task_id, str) or not root_task_id.strip():
        raise ValueError("root_task_id is required")
    if isinstance(required_task_ids, (str, bytes)) or not required_task_ids:
        raise ValueError("at least one required task is required")
    supplied_ids = tuple(required_task_ids)
    if any(not isinstance(task_id, str) or not task_id.strip() for task_id in supplied_ids):
        raise ValueError("required task ids must be non-empty strings")
    required_ids = tuple(sorted({root_task_id, *supplied_ids}))

    canonical_profile = normalize_profile_name(checker_profile)
    validate_profile_name(canonical_profile)
    if not profile_exists(canonical_profile):
        raise ValueError(f"checker profile does not exist: {canonical_profile}")
    validate_notification_policy(notification_policy)
    if notification_policy != "project_summary":
        raise ValueError("production admission requires notification_policy='project_summary'")
    validate_retention_days(retention_days)
    validate_repair_budget(repair_budget)
    current_time = _validated_now(now)

    ensure_project_finalization_schema(conn)
    with write_txn(conn):
        placeholders = ",".join("?" for _ in required_ids)
        rows = conn.execute(
            f"SELECT * FROM tasks WHERE id IN ({placeholders}) ORDER BY id",
            required_ids,
        ).fetchall()
        rows_by_id = {row["id"]: row for row in rows}
        missing = [task_id for task_id in required_ids if task_id not in rows_by_id]
        if missing:
            raise ValueError("required tasks do not exist: " + ", ".join(missing))

        for task_id in required_ids:
            row = rows_by_id[task_id]
            try:
                contract = kb.deserialize_task_contract(row["contract"])
                contract_error = None
            except Exception as exc:
                contract = None
                contract_error = str(exc)
            decision = kb.evaluate_admission(
                contract=contract,
                workspace_kind=row["workspace_kind"],
                workspace_path=row["workspace_path"],
                has_notification_subscription=True,
                enforce_mode=kb.ADMISSION_ENFORCE_ALL,
                contract_present=row["contract"] is not None,
                contract_error=contract_error,
            )
            if not decision.admitted:
                raise ValueError(f"task {task_id} is not admissible: {decision.summary()}")
            assignee = row["assignee"]
            if isinstance(assignee, str) and assignee.strip():
                try:
                    assigned_profile = normalize_profile_name(assignee)
                except ValueError:
                    assigned_profile = assignee.strip().lower()
                if assigned_profile == canonical_profile:
                    raise ValueError(
                        f"checker profile must be independent from required task {task_id}"
                    )

        route_rows = conn.execute(
            """
            SELECT platform, chat_id, thread_id
              FROM kanban_notify_subs
             WHERE task_id = ? AND platform = 'telegram'
             ORDER BY chat_id, thread_id
            """,
            (root_task_id,),
        ).fetchall()
        if len(route_rows) != 1:
            raise ValueError("project root must have exactly one Telegram notification route")
        route = route_rows[0]
        route_identity = notification_route_identity(
            route["platform"], route["chat_id"], route["thread_id"] or None
        )
        admission_key = _admission_identity(
            board_id=board_id,
            root_task_id=root_task_id,
            required_task_ids=required_ids,
            checker_profile=canonical_profile,
            notification_route_identity_value=route_identity,
            notification_policy=notification_policy,
            retention_days=retention_days,
            repair_budget=repair_budget,
        )

        existing_rows = conn.execute(
            """
            SELECT * FROM project_finalizations
             WHERE board_id = ? AND root_task_id = ?
             ORDER BY generation
            """,
            (board_id, root_task_id),
        ).fetchall()
        if existing_rows:
            admitted = [row for row in existing_rows if row["admission_key"] is not None]
            if len(admitted) != 1 or admitted[0]["admission_key"] != admission_key:
                raise ValueError("project admission conflicts with existing durable project state")
            existing = admitted[0]
            persisted_required = {
                row["task_id"]
                for row in conn.execute(
                    """
                    SELECT task_id FROM project_finalization_members
                     WHERE board_id = ? AND root_task_id = ? AND generation = ?
                       AND membership_kind = 'required' AND required = 1
                    """,
                    (board_id, root_task_id, int(existing["generation"])),
                )
            }
            if persisted_required != set(required_ids):
                raise ValueError("project admission conflicts with required task membership")
            finalization = get_project_finalization(
                conn,
                board_id=board_id,
                root_task_id=root_task_id,
                generation=int(existing["generation"]),
            )
            assert finalization is not None
            return AtomicProjectAdmission(
                ADMISSION_ALREADY_ADMITTED, finalization, admission_key
            )

        pending_identity = _pending_checker_identity(
            admission_key, generation=1, repair_index=0
        )
        conn.execute(
            """
            INSERT INTO project_finalizations (
                board_id, root_task_id, generation, state, terminal_outcome,
                final_checker_task_id, checker_verdict,
                admission_key, checker_profile, notification_route_identity,
                checker_candidate_snapshot_version, checker_candidate_id,
                repair_generation, repair_budget, notification_policy,
                retention_days, created_at, updated_at, version
            ) VALUES (?, ?, 1, 'open', NULL, ?, NULL, ?, ?, ?, NULL, NULL,
                      0, ?, ?, ?, ?, ?, 1)
            """,
            (
                board_id,
                root_task_id,
                pending_identity,
                admission_key,
                canonical_profile,
                route_identity,
                repair_budget,
                notification_policy,
                retention_days,
                current_time,
                current_time,
            ),
        )
        _inject(inject_failure, "after_project")
        conn.executemany(
            """
            INSERT INTO project_finalization_members
                (board_id, root_task_id, generation, task_id,
                 membership_kind, required, created_at)
            VALUES (?, ?, 1, ?, 'required', 1, ?)
            """,
            (
                (board_id, root_task_id, task_id, current_time)
                for task_id in required_ids
            ),
        )
        kb._append_event(
            conn,
            root_task_id,
            "project_admitted",
            {
                "admission_key": admission_key,
                "board_id": board_id,
                "generation": 1,
                "required_task_ids": list(required_ids),
                "checker_profile": canonical_profile,
                "notification_route_identity": route_identity,
                "notification_policy": notification_policy,
                "retention_days": retention_days,
                "repair_budget": repair_budget,
            },
        )
        _inject(inject_failure, "after_membership")

    finalization = get_project_finalization(
        conn, board_id=board_id, root_task_id=root_task_id, generation=1
    )
    assert finalization is not None
    return AtomicProjectAdmission(ADMISSION_CREATED, finalization, admission_key)


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
        if project_row["admission_key"] is not None:
            try:
                repair_profile = normalize_profile_name(action.worker_profile)
            except ValueError:
                repair_profile = action.worker_profile.strip().lower()
            if repair_profile == project_row["checker_profile"]:
                raise ValueError(
                    "repair worker profile must remain independent from checker authority"
                )

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

        pending_checker: str | None = None
        if project_row["admission_key"] is not None:
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
            pending_checker = _pending_checker_identity(
                project_row["admission_key"],
                generation=action.project.generation,
                repair_index=action.repair_index,
            )

        updated = conn.execute(
            """
            UPDATE project_finalizations
               SET repair_generation = ?,
                   checker_verdict = NULL,
                   final_checker_task_id = COALESCE(?, final_checker_task_id),
                   checker_candidate_snapshot_version = CASE WHEN ? IS NULL
                       THEN checker_candidate_snapshot_version ELSE NULL END,
                   checker_candidate_id = CASE WHEN ? IS NULL
                       THEN checker_candidate_id ELSE NULL END,
                   state = 'repairing',
                   updated_at = ?,
                   version = version + 1
             WHERE board_id = ? AND root_task_id = ? AND generation = ?
               AND version = ? AND lock_owner = ?
               AND COALESCE(lock_expires_at, 0) >= ?
            """,
            (
                action.repair_index,
                pending_checker,
                pending_checker,
                pending_checker,
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

        if project_row["admission_key"] is not None:
            persisted_profile = project_row["checker_profile"]
            if action.worker_profile != persisted_profile:
                raise ValueError("checker worker profile does not match admitted checker authority")
            implementation_assignees = conn.execute(
                """
                SELECT t.id, t.assignee
                  FROM project_finalization_members AS m
                  JOIN tasks AS t ON t.id = m.task_id
                 WHERE m.board_id = ? AND m.root_task_id = ? AND m.generation = ?
                   AND m.membership_kind IN ('required', 'repair')
                   AND m.required = 1
                """,
                (
                    action.project.board_id,
                    action.project.root_task_id,
                    action.project.generation,
                ),
            ).fetchall()
            for assigned in implementation_assignees:
                if isinstance(assigned["assignee"], str) and assigned["assignee"].strip():
                    try:
                        assigned_profile = normalize_profile_name(assigned["assignee"])
                    except ValueError:
                        assigned_profile = assigned["assignee"].strip().lower()
                    if assigned_profile == persisted_profile:
                        raise ValueError(
                            "checker profile is no longer independent from required work"
                        )
            if tuple(action.notification_route_identities) != (
                project_row["notification_route_identity"],
            ):
                raise ValueError("checker notification route does not match admitted project route")
            if action.candidate_id != action.candidate_snapshot_version:
                raise ValueError("admitted checker candidate id must equal the stable candidate digest")

            # The evaluator is read-only and reuses our open write snapshot.
            # Its stable candidate digest excludes lock/evaluation/checker
            # state, so it is the final authority on the implementation set
            # this checker is about to inspect.
            from hermes_cli.project_finalizer import evaluate_project

            evaluation = evaluate_project(
                conn,
                board_id=action.project.board_id,
                root_task_id=action.project.root_task_id,
                generation=action.project.generation,
                evaluation_time=current_time,
            )
            if (
                evaluation.failure_reason != "checker_required"
                or evaluation.candidate_snapshot_version
                != action.candidate_snapshot_version
            ):
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

        if (
            project_row["terminal_candidate_snapshot_version"]
            == action.candidate_snapshot_version
        ):
            # A frozen candidate is already in publication/delivery. It must
            # finish through that terminal intent, not acquire new authority.
            return AtomicCheckerRegistration(REGISTRATION_STALE_SNAPSHOT)

        has_artifacts = any(
            project_row[field] is not None
            for field in (
                "final_report_path",
                "final_report_sha256",
                "manifest_path",
                "manifest_sha256",
                "usage_summary_json",
            )
        )
        reset_stale_artifacts = bool(
            (
                project_row["terminal_candidate_snapshot_version"] is not None
                and project_row["terminal_candidate_snapshot_version"]
                != action.candidate_snapshot_version
            )
            or (
                has_artifacts
                and project_row["artifact_candidate_snapshot_version"]
                != action.candidate_snapshot_version
            )
        )
        if reset_stale_artifacts:
            accepted = conn.execute(
                """
                SELECT 1 FROM project_delivery_attempts
                 WHERE board_id=? AND root_task_id=? AND generation=? AND accepted=1
                 LIMIT 1
                """,
                (
                    action.project.board_id,
                    action.project.root_task_id,
                    action.project.generation,
                ),
            ).fetchone()
            if accepted is not None:
                return AtomicCheckerRegistration(REGISTRATION_STALE_SNAPSHOT)
            # Disable a stale/pre-migration fence inside this atomic recovery
            # transaction so member triggers permit the replacement checker.
            # The final CAS below clears the obsolete artifact identity; all
            # previously published files remain immutable on disk.
            prepared = conn.execute(
                """
                UPDATE project_finalizations
                   SET terminal_intent=NULL,
                       terminal_candidate_snapshot_version=NULL
                 WHERE board_id=? AND root_task_id=? AND generation=?
                   AND version=? AND lock_owner=?
                   AND COALESCE(lock_expires_at,0)>=?
                """,
                (
                    action.project.board_id,
                    action.project.root_task_id,
                    action.project.generation,
                    expected_token.project_version,
                    expected_token.lock_token,
                    current_time,
                ),
            )
            if prepared.rowcount != 1:
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
                   checker_candidate_snapshot_version = ?,
                   checker_candidate_id = ?,
                   final_report_path=CASE WHEN ? THEN NULL ELSE final_report_path END,
                   final_report_sha256=CASE WHEN ? THEN NULL ELSE final_report_sha256 END,
                   manifest_path=CASE WHEN ? THEN NULL ELSE manifest_path END,
                   manifest_sha256=CASE WHEN ? THEN NULL ELSE manifest_sha256 END,
                   usage_summary_json=CASE WHEN ? THEN NULL ELSE usage_summary_json END,
                   artifact_candidate_snapshot_version=CASE WHEN ? THEN NULL ELSE artifact_candidate_snapshot_version END,
                   finalized_at=CASE WHEN ? THEN NULL ELSE finalized_at END,
                   state = 'evaluating',
                   updated_at = ?,
                   version = version + 1
             WHERE board_id = ? AND root_task_id = ? AND generation = ?
               AND version = ? AND lock_owner = ?
               AND COALESCE(lock_expires_at, 0) >= ?
            """,
            (
                task_id,
                action.candidate_snapshot_version,
                action.candidate_id,
                reset_stale_artifacts,
                reset_stale_artifacts,
                reset_stale_artifacts,
                reset_stale_artifacts,
                reset_stale_artifacts,
                reset_stale_artifacts,
                reset_stale_artifacts,
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


def _normalize_checker_evidence(evidence: Sequence[Mapping[str, object]]) -> tuple[dict[str, str], ...]:
    if isinstance(evidence, (str, bytes)) or not evidence:
        raise ValueError("checker evidence must be a non-empty array")
    normalized: list[dict[str, str]] = []
    for index, item in enumerate(evidence):
        if not isinstance(item, Mapping):
            raise ValueError(f"checker evidence item {index} must be an object")
        unknown = set(item) - {"kind", "reference", "summary"}
        if unknown:
            raise ValueError(
                f"checker evidence item {index} has unknown fields: {', '.join(sorted(unknown))}"
            )
        kind = item.get("kind")
        reference = item.get("reference")
        item_summary = item.get("summary")
        if kind not in CHECKER_EVIDENCE_KINDS:
            raise ValueError(f"checker evidence item {index} has invalid kind")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(f"checker evidence item {index} requires reference")
        if not isinstance(item_summary, str) or not item_summary.strip():
            raise ValueError(f"checker evidence item {index} requires summary")
        normalized.append(
            {
                "kind": str(kind),
                "reference": reference.strip(),
                "summary": item_summary.strip(),
            }
        )
    return tuple(normalized)


def submit_project_checker_verdict(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    task_id: str,
    run_id: int,
    worker_profile: str,
    verdict: str,
    reason: str,
    evidence: Sequence[Mapping[str, object]],
    summary: str | None = None,
    now: int | None = None,
    inject_failure: FailureInjector | None = None,
) -> CheckerVerdictSubmission:
    """Durably record an admitted checker's structured verdict, then complete it.

    The verdict event and project aggregate update commit before task completion.
    A crash at that boundary is therefore safely retried with the identical
    payload; a changed retry is rejected as an immutable-authority conflict.
    """

    if verdict not in CHECKER_VERDICTS:
        raise ValueError(f"invalid checker verdict: {verdict!r}")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("checker verdict reason is required")
    if summary is not None and not isinstance(summary, str):
        raise ValueError("checker verdict summary must be a string")
    if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id < 1:
        raise ValueError("checker run_id must be a positive integer")
    canonical_profile = normalize_profile_name(worker_profile)
    validate_profile_name(canonical_profile)
    normalized_evidence = _normalize_checker_evidence(evidence)
    current_time = _validated_now(now)
    clean_summary = summary.strip() if isinstance(summary, str) and summary.strip() else None

    ensure_project_finalization_schema(conn)
    disposition = "recorded"
    with write_txn(conn):
        authorities = conn.execute(
            """
            SELECT f.*
              FROM project_finalizations AS f
              JOIN project_finalization_members AS m
                ON m.board_id = f.board_id
               AND m.root_task_id = f.root_task_id
               AND m.generation = f.generation
             WHERE f.board_id = ?
               AND f.admission_key IS NOT NULL
               AND f.terminal_outcome IS NULL
               AND f.final_checker_task_id = ?
               AND m.task_id = ?
               AND m.membership_kind = 'checker'
               AND m.required = 1
            """,
            (board_id, task_id, task_id),
        ).fetchall()
        if len(authorities) != 1:
            raise ValueError("task is not the current admitted project checker")
        project_row = authorities[0]
        if project_row["checker_profile"] != canonical_profile:
            raise ValueError("worker profile does not own this checker authority")

        task_row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if task_row is None:
            raise ValueError("checker task does not exist")
        if task_row["assignee"] != canonical_profile:
            raise ValueError("checker task assignee does not match checker authority")

        registration_events = conn.execute(
            """
            SELECT payload FROM task_events
             WHERE task_id = ? AND kind = 'project_checker_registered'
             ORDER BY id
            """,
            (task_id,),
        ).fetchall()
        if len(registration_events) != 1:
            raise RuntimeError("checker registration authority is incomplete")
        registration = json.loads(registration_events[0]["payload"])
        if (
            registration.get("checker_profile") != canonical_profile
            or registration.get("candidate_snapshot_version")
            != project_row["checker_candidate_snapshot_version"]
            or registration.get("candidate_id") != project_row["checker_candidate_id"]
        ):
            raise RuntimeError("checker registration authority does not match project candidate")

        from hermes_cli.project_finalizer import evaluate_project

        evaluation = evaluate_project(
            conn,
            board_id=board_id,
            root_task_id=project_row["root_task_id"],
            generation=int(project_row["generation"]),
            evaluation_time=current_time,
        )
        if (
            evaluation.evaluation_state == "MALFORMED"
            or evaluation.checker_task_id != task_id
            or evaluation.candidate_snapshot_version
            != project_row["checker_candidate_snapshot_version"]
        ):
            raise ValueError("checker candidate is stale")

        payload: dict[str, object] = {
            "protocol": "project-checker-verdict-v1",
            "board_id": board_id,
            "root_task_id": project_row["root_task_id"],
            "generation": int(project_row["generation"]),
            "checker_task_id": task_id,
            "checker_profile": canonical_profile,
            "candidate_snapshot_version": project_row[
                "checker_candidate_snapshot_version"
            ],
            "candidate_id": project_row["checker_candidate_id"],
            "verdict": verdict,
            "reason": reason.strip(),
            "evidence": list(normalized_evidence),
            "summary": clean_summary,
        }
        prior_events = conn.execute(
            """
            SELECT run_id, payload FROM task_events
             WHERE task_id = ? AND kind = 'project_checker_verdict_recorded'
             ORDER BY id
            """,
            (task_id,),
        ).fetchall()
        if prior_events:
            if (
                len(prior_events) != 1
                or json.loads(prior_events[0]["payload"]) != payload
                or project_row["checker_verdict"] != verdict
            ):
                raise ValueError("checker verdict conflicts with durable submission")
            run_row = conn.execute(
                "SELECT task_id, profile, ended_at FROM task_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if (
                run_row is None
                or run_row["task_id"] != task_id
                or run_row["profile"] != canonical_profile
            ):
                raise ValueError("checker run does not belong to this authority")
            if task_row["status"] != "done" and (
                int(task_row["current_run_id"] or 0) != run_id
                or run_row["ended_at"] is not None
            ):
                raise ValueError("checker run is not current")
            disposition = "already_recorded"
        else:
            if project_row["checker_verdict"] is not None:
                raise RuntimeError("checker verdict aggregate is missing its durable event")
            if int(task_row["current_run_id"] or 0) != run_id:
                raise ValueError("checker run is not current")
            run_row = conn.execute(
                "SELECT task_id, profile, ended_at FROM task_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if (
                run_row is None
                or run_row["task_id"] != task_id
                or run_row["profile"] != canonical_profile
                or run_row["ended_at"] is not None
            ):
                raise ValueError("checker run does not own current authority")
            kb._append_event(
                conn,
                task_id,
                "project_checker_verdict_recorded",
                payload,
                run_id=run_id,
            )
            updated = conn.execute(
                """
                UPDATE project_finalizations
                   SET checker_verdict = ?, evaluated_at = ?, updated_at = ?,
                       version = version + 1
                 WHERE board_id = ? AND root_task_id = ? AND generation = ?
                   AND final_checker_task_id = ?
                   AND checker_verdict IS NULL
                   AND checker_candidate_snapshot_version = ?
                   AND checker_candidate_id = ?
                """,
                (
                    verdict,
                    current_time,
                    current_time,
                    board_id,
                    project_row["root_task_id"],
                    int(project_row["generation"]),
                    task_id,
                    project_row["checker_candidate_snapshot_version"],
                    project_row["checker_candidate_id"],
                ),
            )
            if updated.rowcount != 1:
                raise RuntimeError("checker verdict compare-and-set failed")

    _inject(inject_failure, "after_verdict_commit")
    task_after_verdict = kb.get_task(conn, task_id)
    if task_after_verdict is None:
        raise RuntimeError("checker task disappeared after verdict commit")
    completed = task_after_verdict.status == "done"
    if not completed:
        completed = kb.complete_task(
            conn,
            task_id,
            result=clean_summary or reason.strip(),
            summary=clean_summary or reason.strip(),
            metadata={
                "project_checker_verdict": verdict,
                "evidence": list(normalized_evidence),
            },
            expected_run_id=run_id,
        )
        if not completed:
            refreshed = kb.get_task(conn, task_id)
            if refreshed is None or refreshed.status != "done":
                raise RuntimeError("checker verdict persisted but task completion did not succeed")
            completed = True
    return CheckerVerdictSubmission(disposition, task_id, verdict, completed)


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
        and (
            project_row["admission_key"] is None
            or (
                project_row["checker_candidate_snapshot_version"]
                == action.candidate_snapshot_version
                and project_row["checker_candidate_id"] == action.candidate_id
                and project_row["checker_profile"] == action.worker_profile
            )
        )
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
    "ADMISSION_ALREADY_ADMITTED",
    "ADMISSION_CREATED",
    "CHECKER_EVIDENCE_KINDS",
    "DESTINATION_FOUND",
    "DESTINATION_MISSING",
    "AtomicCheckerRegistration",
    "AtomicProjectAdmission",
    "CheckerRegistrationAction",
    "CheckerVerdictSubmission",
    "ProjectTelegramDestination",
    "admit_existing_project",
    "checker_registration_identity",
    "notification_route_identity",
    "register_project_checker",
    "register_project_repair",
    "resolve_project_telegram_destination",
    "submit_project_checker_verdict",
]
