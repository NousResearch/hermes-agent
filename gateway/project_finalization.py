"""Gateway-owned, idempotent project-finalization lifecycle.

This module orchestrates accepted project subsystems only.  It neither creates a
provider nor executes cleanup; callers inject the delivery boundary and select a
narrow canary scope before any durable project is considered.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from hermes_cli import kanban_db as kb
from hermes_cli.project_delivery_ledger import (
    DELIVERY_ACCEPTED,
    DELIVERY_AMBIGUOUS,
    DELIVERY_ATTEMPTING,
    DELIVERY_PERMANENT_FAILURE,
    DELIVERY_REJECTED,
    DELIVERY_RETRY_SCHEDULED,
    MAX_DELIVERY_ATTEMPTS,
    create_delivery_attempt,
    create_next_delivery_attempt,
    get_latest_delivery_attempt,
    mark_delivery_attempt_accepted,
    mark_delivery_attempt_ambiguous,
    mark_delivery_attempt_attempting,
    mark_delivery_attempt_permanent_failure,
    mark_delivery_attempt_rejected,
    mark_delivery_attempt_retry_scheduled,
)
from hermes_cli.project_failure_envelope import record_failure_envelope
from hermes_cli.project_final_artifacts import (
    ProjectFinalArtifacts,
    ProjectFinalizationSnapshot,
    publish_project_final_artifacts,
)
from hermes_cli.project_finalization_contract import (
    acquire_finalization_lock,
    ensure_project_finalization_schema,
    get_project_finalization,
    list_project_finalizations,
    list_project_members,
    record_terminal_outcome,
    release_finalization_lock,
    schedule_project_cleanup,
)
from hermes_cli.project_finalizer import ProjectEvaluation, evaluate_project
from hermes_cli.project_repair_router import (
    ProjectIdentity,
    ProjectRepairRequest,
    ProjectVersionToken,
    RepairMembership,
    route_project_repair,
)
from hermes_cli.project_runtime_registration import (
    DESTINATION_FOUND,
    CheckerRegistrationAction,
    checker_registration_identity,
    notification_route_identity,
    register_project_checker,
    register_project_repair,
    resolve_project_telegram_destination,
)

DeliveryCallable = Callable[[str, str, str | None, str], Awaitable[Mapping[str, Any]]]
ConnectionFactory = Callable[[], sqlite3.Connection]
MAX_TERMINAL_MESSAGE_CHARS = 512


@dataclass(frozen=True)
class ProjectFinalizationTickResult:
    """Non-secret outcome summary for one bounded lifecycle tick."""

    processed: int = 0
    skipped: int = 0
    repaired: int = 0
    checkers_reconciled: int = 0
    delivered: int = 0
    terminalized: int = 0
    ambiguous: int = 0
    failures: tuple[str, ...] = field(default_factory=tuple)

    def plus(self, **changes: int | str) -> "ProjectFinalizationTickResult":
        values = self.__dict__.copy()
        failure = changes.pop("failure", None)
        for key, value in changes.items():
            values[key] = int(values[key]) + int(value)
        if failure:
            values["failures"] = (*self.failures, str(failure))
        return ProjectFinalizationTickResult(**values)


class ProjectFinalizationService:
    """One-tick project finalizer with explicit state and delivery boundaries."""

    def __init__(
        self,
        connection_factory: ConnectionFactory,
        *,
        owner: str,
        now: Callable[[], int],
        deliver: DeliveryCallable,
        enabled: bool = False,
        canary_scope: tuple[str, ...] = (),
        cleanup_enabled: bool = False,
        lease_seconds: int = 120,
    ) -> None:
        self._connection_factory = connection_factory
        self._owner = owner
        self._now = now
        self._deliver = deliver
        self._enabled = bool(enabled)
        self._canary_scope = frozenset(item for item in canary_scope if isinstance(item, str) and item)
        self._cleanup_enabled = bool(cleanup_enabled)
        self._lease_seconds = max(1, int(lease_seconds))

    async def tick(self, *, board_id: str | None = None) -> ProjectFinalizationTickResult:
        """Process one current nonterminal generation per eligible project."""
        if not self._enabled or not self._canary_scope:
            return ProjectFinalizationTickResult(skipped=1)
        conn = self._connection_factory()
        try:
            ensure_project_finalization_schema(conn)
            result = ProjectFinalizationTickResult()
            for finalization in list_project_finalizations(conn, board_id=board_id):
                if finalization.terminal_outcome is not None or not self._in_canary(finalization.board_id, finalization.root_task_id):
                    result = result.plus(skipped=1)
                    continue
                try:
                    result = await self._process(conn, finalization.board_id, finalization.root_task_id, finalization.generation, result)
                except Exception as exc:  # one bad project must not stop the watcher
                    result = result.plus(failure=f"{finalization.board_id}/{finalization.root_task_id}/{finalization.generation}: {type(exc).__name__}")
            return result
        finally:
            conn.close()

    def _in_canary(self, board_id: str, root_task_id: str) -> bool:
        return "*" in self._canary_scope or root_task_id in self._canary_scope or f"{board_id}/{root_task_id}" in self._canary_scope

    async def _process(self, conn: sqlite3.Connection, board_id: str, root_task_id: str, generation: int, result: ProjectFinalizationTickResult) -> ProjectFinalizationTickResult:
        now = self._now()
        if not acquire_finalization_lock(conn, board_id=board_id, root_task_id=root_task_id, generation=generation, owner=self._owner, lease_seconds=self._lease_seconds, now=str(now)):
            return result.plus(skipped=1)
        try:
            current = get_project_finalization(conn, board_id=board_id, root_task_id=root_task_id, generation=generation)
            if current is None or current.terminal_outcome is not None or not self._owns_lock(current, now):
                return result.plus(skipped=1)
            evaluation = evaluate_project(conn, board_id=board_id, root_task_id=root_task_id, generation=generation, evaluation_time=now)
            if evaluation.evaluation_state == "WAITING":
                if evaluation.failure_reason == "checker_required":
                    return self._reconcile_checker(conn, current, evaluation, result.plus(processed=1), now)
                return result.plus(processed=1)
            if evaluation.evaluation_state == "REPAIRABLE":
                return self._route_repair(conn, current, evaluation, result.plus(processed=1), now)
            if evaluation.evaluation_state == "COMPLETE_ELIGIBLE":
                return await self._finalize(conn, current, evaluation, "COMPLETE", result.plus(processed=1), now)
            if evaluation.evaluation_state in {"BLOCKED", "FAILED"}:
                outcome = "BLOCKED" if evaluation.evaluation_state == "BLOCKED" else "FAILED"
                return await self._finalize(conn, current, evaluation, outcome, result.plus(processed=1), now)
            # MALFORMED is deliberately non-repairing and never claims success.
            return result.plus(processed=1, skipped=1)
        finally:
            release_finalization_lock(conn, board_id=board_id, root_task_id=root_task_id, generation=generation, owner=self._owner)

    def _owns_lock(self, finalization: Any, now: int) -> bool:
        return finalization.lock_owner == self._owner and (finalization.lock_expires_at or 0) >= now

    def _identity(self, current: Any) -> ProjectIdentity:
        return ProjectIdentity(
            project_id=f"{current.board_id}:{current.root_task_id}:{current.generation}",
            board_id=current.board_id,
            root_task_id=current.root_task_id,
            generation=current.generation,
        )

    def _version_token(self, current: Any, evaluation: ProjectEvaluation) -> ProjectVersionToken:
        return ProjectVersionToken(snapshot_version=evaluation.snapshot_version, project_version=current.version, lock_token=self._owner)

    def _destination(self, conn: sqlite3.Connection, current: Any) -> Any:
        return resolve_project_telegram_destination(conn, board_id=current.board_id, root_task_id=current.root_task_id, generation=current.generation)

    def _task_contract(self, conn: sqlite3.Connection, root_task_id: str) -> Mapping[str, object]:
        contract = kb.get_task_contract(conn, root_task_id)
        if contract is not None:
            return contract
        # The registration boundary retains the admission decision; this fallback
        # is intentionally minimal and never elevates a task into a live runtime.
        return {"version": 1, "scope": "project finalization runtime", "allowed_files": [], "forbidden_files": [], "base_commit": "0" * 40, "required_evidence": [], "required_commands": [], "allow_child_creation": False, "forbidden_git_actions": ["push"], "notification_verified": True}

    def _worker_profile(self, conn: sqlite3.Connection, task_id: str, fallback: str) -> str:
        task = kb.get_task(conn, task_id)
        return (getattr(task, "assignee", None) or fallback).strip()

    def _route_identities(self, destination: Any) -> tuple[str, ...]:
        if destination.status != DESTINATION_FOUND or not destination.route_identity:
            return ()
        return (destination.route_identity,)

    def _reconcile_checker(self, conn: sqlite3.Connection, current: Any, evaluation: ProjectEvaluation, result: ProjectFinalizationTickResult, now: int) -> ProjectFinalizationTickResult:
        if not self._owns_lock(current, now):
            return result.plus(skipped=1)
        # A pending authoritative checker already owns this candidate. Its
        # completion/verdict is the only event that may advance finalization;
        # do not mint a new identity merely because evaluator time changed.
        # A *completed* checker with no current verdict is different: repair
        # registration clears the old verdict, so the completed authority must
        # be replaced by one checker bound to the repaired snapshot.
        members = list_project_members(conn, board_id=current.board_id, root_task_id=current.root_task_id, generation=current.generation)
        checker_is_authoritative = any(
            member.membership_kind == "checker"
            and member.task_id == current.final_checker_task_id
            and member.required
            for member in members
        )
        if checker_is_authoritative:
            checker_task = kb.get_task(conn, current.final_checker_task_id)
            if checker_task is not None and checker_task.status != "done":
                return result.plus(skipped=1)
        destination = self._destination(conn, current)
        project = self._identity(current)
        candidate_id = evaluation.candidate_snapshot_version
        identity = checker_registration_identity(project, candidate_snapshot_version=evaluation.candidate_snapshot_version, candidate_id=candidate_id)
        checker_profile = current.checker_profile or self._worker_profile(
            conn, current.root_task_id, "checker"
        )
        action = CheckerRegistrationAction(project=project, checker_identity=identity, idempotency_key=identity, candidate_snapshot_version=evaluation.candidate_snapshot_version, candidate_id=candidate_id, worker_profile=checker_profile, task_contract=self._task_contract(conn, current.root_task_id), notification_route_identities=self._route_identities(destination))
        registered = register_project_checker(conn, action, self._version_token(current, evaluation), now=now)
        return result.plus(checkers_reconciled=1 if registered.disposition in {"created", "already_exists"} else 0, skipped=1 if registered.disposition == "stale_snapshot" else 0)

    def _route_repair(self, conn: sqlite3.Connection, current: Any, evaluation: ProjectEvaluation, result: ProjectFinalizationTickResult, now: int) -> ProjectFinalizationTickResult:
        if not self._owns_lock(current, now) or not evaluation.checker_task_id:
            return result.plus(skipped=1)
        destination = self._destination(conn, current)
        # A checker FAIL_REPAIRABLE is a bounded protocol failure. Persist this
        # redacted envelope before routing so replay has a stable fingerprint.
        envelope = record_failure_envelope(conn, board_id=current.board_id, root_task_id=current.root_task_id, generation=current.generation, task_id=evaluation.checker_task_id, run_id=None, failure_class="protocol_violation", redacted_error="checker requested bounded repair")
        project = self._identity(current)
        # Once a repair registration succeeds the aggregate state advances to
        # ``repairing``; a subsequent evaluator pass will not re-enter this
        # branch.  The atomic registrar remains the replay fence, so no
        # synthetic repair-membership reconstruction (and no guessed identity)
        # is needed here.
        request = ProjectRepairRequest(project=project, evaluation=evaluation, failed_task_id=evaluation.checker_task_id, failed_run_id=None, failure_envelope=envelope, project_repair_budget=current.repair_budget, task_retry_limit=current.repair_budget, existing_repairs=(), worker_profile=self._worker_profile(conn, current.root_task_id, "builder"), allowed_worker_profiles=(self._worker_profile(conn, current.root_task_id, "builder"),), task_contract=self._task_contract(conn, current.root_task_id), notification_route_identities=self._route_identities(destination), version_token=self._version_token(current, evaluation))
        routed = route_project_repair(request, register_repair=lambda action, token: register_project_repair(conn, action, token, now=now))
        return result.plus(repaired=1 if routed.outcome in {"REPAIR_CREATED", "REPAIR_ALREADY_EXISTS"} else 0, skipped=1 if routed.outcome not in {"REPAIR_CREATED", "REPAIR_ALREADY_EXISTS"} else 0)

    async def _finalize(self, conn: sqlite3.Connection, current: Any, evaluation: ProjectEvaluation, outcome: str, result: ProjectFinalizationTickResult, now: int) -> ProjectFinalizationTickResult:
        if not self._owns_lock(current, now):
            return result.plus(skipped=1)
        snapshot = self._artifact_snapshot(conn, current, evaluation, outcome)
        published = publish_project_final_artifacts(conn, snapshot)
        durable = get_project_finalization(conn, board_id=current.board_id, root_task_id=current.root_task_id, generation=current.generation)
        if durable is None:
            raise ValueError("project finalization disappeared after artifact publication")
        destination = self._destination(conn, durable)
        if destination.status != DESTINATION_FOUND:
            # No route is safe to infer. Persist a terminal technical outcome only
            # for non-success states; success remains nonterminal until delivery.
            if outcome != "COMPLETE":
                record_terminal_outcome(conn, board_id=current.board_id, root_task_id=current.root_task_id, generation=current.generation, outcome=outcome, blocker_json=json.dumps({"reason": destination.reason}, sort_keys=True))
                return result.plus(terminalized=1)
            return result.plus(skipped=1)
        message_kind = f"project_{outcome.lower()}"
        terminal_message = self._terminal_message(durable, outcome, published)
        delivered = await self._deliver_once(conn, durable, destination, message_kind, terminal_message, now)
        if delivered == DELIVERY_AMBIGUOUS:
            return result.plus(ambiguous=1)
        if delivered == DELIVERY_RETRY_SCHEDULED:
            return result.plus(skipped=1)
        if delivered != DELIVERY_ACCEPTED:
            return result.plus(skipped=1)
        refreshed = get_project_finalization(conn, board_id=current.board_id, root_task_id=current.root_task_id, generation=current.generation)
        if refreshed is None or not self._owns_lock(refreshed, self._now()):
            return result.plus(skipped=1)
        record_terminal_outcome(conn, board_id=current.board_id, root_task_id=current.root_task_id, generation=current.generation, outcome=outcome, blocker_json=(json.dumps({"reason": evaluation.failure_reason}, sort_keys=True) if outcome != "COMPLETE" else None))
        if self._cleanup_enabled:
            cleanup_after = (datetime.fromtimestamp(now, UTC) + timedelta(days=current.retention_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
            schedule_project_cleanup(conn, board_id=current.board_id, root_task_id=current.root_task_id, generation=current.generation, cleanup_after=cleanup_after)
        return result.plus(delivered=1, terminalized=1)

    def _terminal_message(self, current: Any, outcome: str, published: ProjectFinalArtifacts) -> str:
        artifact_names = (
            Path(published.report_path).name,
            Path(published.manifest_path).name,
            Path(published.usage_summary_path).name,
        )
        if any(not name for name in artifact_names):
            raise ValueError("published artifact is missing a relative name")
        checker_verdict = current.checker_verdict or "NOT_RECORDED"
        message = "\n".join(
            (
                f"Result: {outcome}",
                f"Root: {current.root_task_id}",
                f"Checker: {checker_verdict}",
                f"Artifacts: {', '.join(artifact_names)}",
            )
        )
        if len(message) > MAX_TERMINAL_MESSAGE_CHARS:
            raise ValueError("terminal message exceeds the bounded delivery contract")
        return message

    async def _deliver_once(self, conn: sqlite3.Connection, current: Any, destination: Any, message_kind: str, terminal_message: str, now: int) -> str:
        kwargs = dict(board_id=current.board_id, root_task_id=current.root_task_id, generation=current.generation, platform=destination.platform, destination_reference=destination.chat_id, thread_reference=destination.thread_id, message_kind=message_kind)
        latest = get_latest_delivery_attempt(conn, **{key: value for key, value in kwargs.items() if key != "thread_reference"})
        if latest is not None:
            if latest.delivery_state == DELIVERY_ACCEPTED:
                return DELIVERY_ACCEPTED
            if latest.delivery_state in {DELIVERY_AMBIGUOUS, DELIVERY_ATTEMPTING}:
                return DELIVERY_AMBIGUOUS
            if latest.delivery_state == DELIVERY_PERMANENT_FAILURE:
                return DELIVERY_PERMANENT_FAILURE
            if latest.delivery_state == DELIVERY_RETRY_SCHEDULED and (latest.next_retry_at or 0) > now:
                return DELIVERY_RETRY_SCHEDULED
            if latest.delivery_state == DELIVERY_REJECTED:
                return DELIVERY_RETRY_SCHEDULED
        if latest is not None and latest.delivery_state == DELIVERY_RETRY_SCHEDULED:
            attempt = create_next_delivery_attempt(conn, delivery_state="pending", **kwargs)
        else:
            attempt = create_delivery_attempt(conn, attempt_number=1 if latest is None else latest.attempt_number, delivery_state="pending", **kwargs)
        mark_delivery_attempt_attempting(conn, attempt_number=attempt.attempt_number, now=now, **{key: value for key, value in kwargs.items() if key != "thread_reference"})
        try:
            receipt = await self._deliver(destination.platform, destination.chat_id, destination.thread_id, terminal_message)
        except Exception as exc:
            mark_delivery_attempt_ambiguous(conn, attempt_number=attempt.attempt_number, redacted_error=f"delivery exception: {type(exc).__name__}", now=now, **{key: value for key, value in kwargs.items() if key != "thread_reference"})
            return DELIVERY_AMBIGUOUS
        message_id = receipt.get("provider_message_id") if isinstance(receipt, Mapping) else None
        if isinstance(message_id, str) and message_id:
            mark_delivery_attempt_accepted(conn, attempt_number=attempt.attempt_number, provider_message_id=message_id, now=now, **{key: value for key, value in kwargs.items() if key != "thread_reference"})
            return DELIVERY_ACCEPTED
        if isinstance(receipt, Mapping) and receipt.get("rejected"):
            mark_delivery_attempt_rejected(conn, attempt_number=attempt.attempt_number, redacted_error=str(receipt.get("error") or "provider rejected delivery"), now=now, **{key: value for key, value in kwargs.items() if key != "thread_reference"})
            if attempt.attempt_number >= MAX_DELIVERY_ATTEMPTS:
                mark_delivery_attempt_permanent_failure(conn, attempt_number=attempt.attempt_number, redacted_error="delivery attempts exhausted", now=now, **{key: value for key, value in kwargs.items() if key != "thread_reference"})
                return DELIVERY_PERMANENT_FAILURE
            mark_delivery_attempt_retry_scheduled(conn, attempt_number=attempt.attempt_number, now=now, **{key: value for key, value in kwargs.items() if key != "thread_reference"})
            return DELIVERY_RETRY_SCHEDULED
        mark_delivery_attempt_ambiguous(conn, attempt_number=attempt.attempt_number, redacted_error="provider acceptance could not be proven", now=now, **{key: value for key, value in kwargs.items() if key != "thread_reference"})
        return DELIVERY_AMBIGUOUS

    def _artifact_snapshot(self, conn: sqlite3.Connection, current: Any, evaluation: ProjectEvaluation, outcome: str) -> ProjectFinalizationSnapshot:
        root = kb.get_task(conn, current.root_task_id)
        members = list_project_members(conn, board_id=current.board_id, root_task_id=current.root_task_id, generation=current.generation)
        required = [{"task_id": member.task_id, "membership_kind": member.membership_kind} for member in members if member.required]
        supports = [{"task_id": member.task_id, "membership_kind": member.membership_kind} for member in members if not member.required]
        evidence = [{"path": f"kanban:{current.board_id}:{current.root_task_id}:{current.generation}", "sha256": hashlib.sha256((current.board_id + current.root_task_id + str(current.generation)).encode()).hexdigest()}]
        return ProjectFinalizationSnapshot(board_id=current.board_id, root_task_id=current.root_task_id, generation=current.generation, goal=getattr(root, "title", "") or "project finalization", title=getattr(root, "title", "") or "project finalization", terminal_outcome=outcome, terminal_evaluation={"state": evaluation.evaluation_state, "reason": evaluation.failure_reason}, required_tasks=required, support_tasks=supports, repair_tasks=[], checker_task_id=evaluation.checker_task_id or current.final_checker_task_id, checker_verdict=evaluation.checker_verdict or "FAIL_TERMINAL", evidence=evidence, what_done=list(evaluation.successful_task_ids), what_verified=list(evaluation.successful_task_ids), blockers=[evaluation.blocker] if evaluation.blocker else [], next_step="review finalization artifacts", delivery={"status": "pending"}, cleanup={"status": "not_scheduled"}, limitations=["gateway lifecycle does not execute cleanup"], usage={"usage_status": "unknown"}, created_at=current.created_at)


async def reconcile_project_finalizations(service: ProjectFinalizationService, *, board_id: str | None = None) -> ProjectFinalizationTickResult:
    """Named startup/tick reconciliation entry point for watchers and tests."""
    return await service.tick(board_id=board_id)
