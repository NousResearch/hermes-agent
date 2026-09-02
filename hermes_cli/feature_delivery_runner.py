"""Durable, opt-in runner for the fixed Feature Delivery V1 workflow.

The runner owns root-state transitions. Stage executors only return typed
reports; they cannot select a next state or mark delivery complete.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from hermes_cli import kanban_db as kb
from hermes_cli.feature_delivery import (
    FEATURE_DELIVERY_WORKFLOW,
    MAX_FIX_LOOPS,
    AcceptanceReport,
    AcceptanceReportStatus,
    DeliveryCommitContext,
    DeveloperReport,
    DeveloperReportStatus,
    FeatureDeliveryState,
    StageReport,
    StageRole,
    TaskContract,
    TesterReport,
    TesterReportStatus,
    canonicalize_contract,
    compute_contract_hash,
    count_fix_loops,
    evaluate_delivery_gate,
    is_legal_transition,
    validate_stage_report,
)


BLOCKED_CODES = frozenset(
    {
        "contract_hash_mismatch",
        "profile_missing",
        "stage_executor_missing",
        "invalid_report",
        "commit_mismatch",
        "dirty_worktree",
        "tester_modified_source",
        "acceptance_gate_denied",
        "max_fix_loops_reached",
        "external_environment_missing",
        "stage_execution_failed",
    }
)
_REPORT_NAME_RE = re.compile(r"^reports/(\d{3})-(developer|tester|acceptance)\.json$")


class StageExecutor(Protocol):
    """The only seam a future profile adapter needs to implement."""

    def execute(
        self,
        *,
        role: StageRole,
        task_contract: TaskContract,
        workspace: Path,
        target_commit: str,
        feedback: tuple[str, ...],
        stage_task_id: str,
    ) -> object: ...


class DeliveryRunnerError(RuntimeError):
    pass


class ReportIntegrityError(DeliveryRunnerError):
    pass


class StageAlreadyRunning(DeliveryRunnerError):
    pass


@dataclass(frozen=True)
class StoredStageReport:
    role: StageRole
    stage_task_id: str
    run_id: int
    report_path: str
    report: StageReport


@dataclass(frozen=True)
class DeliverySnapshot:
    developer_commit: str | None = None
    tested_commit: str | None = None
    accepted_commit: str | None = None
    fix_loops: int = 0
    last_stage: str | None = None
    last_report_status: str | None = None
    blocked_code: str | None = None
    blocked_message: str | None = None
    reports: tuple[StoredStageReport, ...] = ()

    def feedback(self) -> tuple[str, ...]:
        for stored in reversed(self.reports):
            report = stored.report
            if isinstance(report, TesterReport) and report.status == TesterReportStatus.TEST_FAIL:
                return tuple(report.blocking_issues)
            if (
                isinstance(report, AcceptanceReport)
                and report.status == AcceptanceReportStatus.REJECT
            ):
                return tuple(report.blocking_issues)
        return ()


@dataclass(frozen=True)
class DeliveryStatus:
    task_id: str
    title: str
    current_state: str
    fix_loops: int
    branch: str
    base_commit: str
    developer_commit: str | None
    tested_commit: str | None
    accepted_commit: str | None
    contract_hash: str
    last_stage: str | None
    last_report_status: str | None
    blocked_reason: str | None

    @property
    def delivery_status(self) -> str:
        if self.current_state == FeatureDeliveryState.DELIVERED.value:
            return "Feature Delivery Gate Passed"
        if self.current_state == FeatureDeliveryState.BLOCKED.value:
            return "BLOCKED"
        return "IN PROGRESS"

    def render(self) -> str:
        values = (
            ("Task ID", self.task_id),
            ("Title", self.title),
            ("Current State", self.current_state),
            ("Fix Loops", str(self.fix_loops)),
            ("Branch", self.branch),
            ("Base Commit", self.base_commit),
            ("Developer Commit", self.developer_commit or "-"),
            ("Tested Commit", self.tested_commit or "-"),
            ("Accepted Commit", self.accepted_commit or "-"),
            ("Contract Hash", self.contract_hash),
            ("Last Stage", self.last_stage or "-"),
            ("Last Report Status", self.last_report_status or "-"),
            ("Blocked Reason", self.blocked_reason or "-"),
            ("Delivery Status", self.delivery_status),
        )
        return "\n".join(f"{label}: {value}" for label, value in values)


class FeatureDeliveryRunner:
    """Run only root tasks opted into ``feature_delivery_v1``."""

    def __init__(self, executor: StageExecutor | None = None, *, board: str | None = None):
        self.executor = executor
        self.board = board
        self._stage_busy = False

    def create(self, contract_path: str | Path) -> str:
        source = Path(contract_path).expanduser().resolve()
        try:
            contract = TaskContract.model_validate_json(source.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError) as exc:
            raise DeliveryRunnerError(f"invalid contract: {exc}") from exc

        repository = self._validate_repository(contract)
        canonical = canonicalize_contract(contract)
        contract_hash = compute_contract_hash(contract)

        with kb.connect(board=self.board) as conn:
            root_id = kb.create_task(
                conn,
                title=contract.title,
                body=contract.objective,
                created_by="feature-delivery-runner",
                workspace_kind="worktree",
                workspace_path=str(repository),
                branch_name=contract.branch,
                idempotency_key=f"feature-delivery:{contract.task_id}:{contract_hash}",
                board=self.board,
            )
            root = kb.get_task(conn, root_id)
            if root is None:
                raise DeliveryRunnerError("root task creation failed")
            if root.workflow_template_id is None:
                with kb.write_txn(conn):
                    conn.execute(
                        "UPDATE tasks SET workflow_template_id = ?, current_step_key = ? "
                        "WHERE id = ? AND workflow_template_id IS NULL",
                        (
                            FEATURE_DELIVERY_WORKFLOW,
                            FeatureDeliveryState.CONTRACT_READY.value,
                            root_id,
                        ),
                    )

            attachment_dir = kb.task_attachments_dir(root_id, board=self.board)
            destination = attachment_dir / "task-contract.json"
            attachment_dir.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.read_bytes() != canonical:
                raise DeliveryRunnerError("existing root contract attachment differs")
            destination.write_bytes(canonical)
            if not any(a.filename == "task-contract.json" for a in kb.list_attachments(conn, root_id)):
                kb.add_attachment(
                    conn,
                    root_id,
                    filename="task-contract.json",
                    stored_path=str(destination),
                    content_type="application/json",
                    size=len(canonical),
                    uploaded_by="feature-delivery-runner",
                )
            with kb.write_txn(conn):
                existing = conn.execute(
                    "SELECT 1 FROM task_events WHERE task_id = ? "
                    "AND kind = 'feature_delivery_created'",
                    (root_id,),
                ).fetchone()
                if not existing:
                    kb._append_event(
                        conn,
                        root_id,
                        "feature_delivery_created",
                        {
                            "contract_path": str(destination),
                            "contract_sha256": contract_hash,
                            "repository": str(repository),
                            "base_commit": contract.base_commit,
                            "branch": contract.branch,
                            "contract_task_id": contract.task_id,
                        },
                    )
        return root_id

    def run(self, task_id: str) -> DeliveryStatus:
        return self._drive(task_id)

    def resume(self, task_id: str) -> DeliveryStatus:
        return self._drive(task_id)

    def status(self, task_id: str) -> DeliveryStatus:
        with kb.connect(board=self.board) as conn:
            root, metadata = self._root_and_metadata(conn, task_id)
            try:
                snapshot = self._snapshot(conn, root, recover=False)
            except ReportIntegrityError as exc:
                if root.current_step_key != FeatureDeliveryState.BLOCKED.value:
                    raise
                snapshot = DeliverySnapshot(
                    last_report_status="INVALID_REPORT",
                    blocked_code="invalid_report",
                    blocked_message=str(exc),
                )
            return self._status(root, metadata, snapshot)

    def _drive(self, task_id: str) -> DeliveryStatus:
        self._stage_busy = False
        with kb.connect(board=self.board) as conn:
            for _ in range(64):
                root, metadata = self._root_and_metadata(conn, task_id)
                state = FeatureDeliveryState(root.current_step_key)
                try:
                    contract = self._load_contract(conn, root, metadata)
                except DeliveryRunnerError:
                    refreshed = kb.get_task(conn, task_id)
                    if refreshed and refreshed.current_step_key == FeatureDeliveryState.BLOCKED.value:
                        return self._status(
                            refreshed,
                            metadata,
                            DeliverySnapshot(
                                blocked_code="contract_hash_mismatch",
                                blocked_message="contract attachment hash changed",
                            ),
                        )
                    raise
                if state in {FeatureDeliveryState.BLOCKED, FeatureDeliveryState.DELIVERED}:
                    try:
                        snapshot = self._snapshot(conn, root, recover=True)
                    except ReportIntegrityError as exc:
                        if state != FeatureDeliveryState.BLOCKED:
                            raise
                        snapshot = DeliverySnapshot(
                            last_report_status="INVALID_REPORT",
                            blocked_code="invalid_report",
                            blocked_message=str(exc),
                        )
                    return self._status(root, metadata, snapshot)

                try:
                    snapshot = self._snapshot(conn, root, recover=True)
                except ReportIntegrityError as exc:
                    self._block(conn, root, "invalid_report", str(exc))
                    continue

                if state == FeatureDeliveryState.CONTRACT_READY:
                    self._transition(conn, root, FeatureDeliveryState.DEVELOPING)
                    continue
                if state == FeatureDeliveryState.DEVELOPING:
                    self._run_developer(conn, root, contract, metadata, snapshot)
                    if self._stage_busy:
                        return self._status(
                            root,
                            metadata,
                            self._snapshot(conn, root, recover=False),
                        )
                    continue
                if state == FeatureDeliveryState.READY_FOR_TEST:
                    if snapshot.developer_commit is None:
                        self._block(conn, root, "invalid_report", "developer commit is missing")
                    else:
                        self._transition(conn, root, FeatureDeliveryState.TESTING)
                    continue
                if state == FeatureDeliveryState.TESTING:
                    self._run_tester(conn, root, contract, metadata, snapshot)
                    if self._stage_busy:
                        return self._status(
                            root,
                            metadata,
                            self._snapshot(conn, root, recover=False),
                        )
                    continue
                if state == FeatureDeliveryState.TEST_FAILED:
                    self._return_to_development(conn, root, snapshot)
                    continue
                if state == FeatureDeliveryState.TEST_PASSED:
                    if snapshot.tested_commit != snapshot.developer_commit:
                        self._block(conn, root, "commit_mismatch", "test evidence is stale")
                    else:
                        self._transition(conn, root, FeatureDeliveryState.ACCEPTANCE)
                    continue
                if state == FeatureDeliveryState.ACCEPTANCE:
                    self._run_acceptance(conn, root, contract, metadata, snapshot)
                    if self._stage_busy:
                        return self._status(
                            root,
                            metadata,
                            self._snapshot(conn, root, recover=False),
                        )
                    continue
                if state == FeatureDeliveryState.REJECTED:
                    self._return_to_development(conn, root, snapshot)
                    continue
                raise DeliveryRunnerError(f"unsupported feature delivery state: {state.value}")
            root, metadata = self._root_and_metadata(conn, task_id)
            self._block(conn, root, "stage_execution_failed", "runner iteration limit reached")
            root, metadata = self._root_and_metadata(conn, task_id)
            return self._status(root, metadata, self._snapshot(conn, root, recover=True))

    def _run_developer(
        self,
        conn,
        root: kb.Task,
        contract: TaskContract,
        metadata: dict,
        snapshot: DeliverySnapshot,
    ) -> None:
        target = snapshot.developer_commit or contract.base_commit
        try:
            workspace = self._feature_workspace(conn, root, contract, metadata)
        except DeliveryRunnerError as exc:
            self._block(conn, root, "stage_execution_failed", str(exc))
            return
        stored = self._stage_report(
            conn,
            root,
            contract,
            "developer",
            target,
            snapshot.fix_loops + 1,
            workspace,
            snapshot.feedback(),
        )
        if stored is None:
            return
        report = stored.report
        if not isinstance(report, DeveloperReport):
            self._block(conn, root, "invalid_report", "developer returned the wrong report type")
            return
        if report.status == DeveloperReportStatus.BLOCKED:
            self._block(conn, root, "external_environment_missing", report.implementation_summary)
            return
        assert report.commit is not None
        error = self._validate_developer_commit(
            contract, Path(metadata["repository"]), workspace, report.commit
        )
        if error:
            self._block(conn, root, error[0], error[1])
            return
        self._transition(
            conn,
            root,
            FeatureDeliveryState.READY_FOR_TEST,
            {"developer_commit": report.commit},
        )

    def _run_tester(
        self,
        conn,
        root: kb.Task,
        contract: TaskContract,
        metadata: dict,
        snapshot: DeliverySnapshot,
    ) -> None:
        if snapshot.developer_commit is None:
            self._block(conn, root, "commit_mismatch", "developer commit is missing")
            return
        stage = self._ensure_stage(
            conn,
            root,
            "tester",
            snapshot.developer_commit,
            snapshot.fix_loops + 1,
        )
        existing = self._report_for_stage(conn, root, stage, recover=True)
        if existing is None:
            try:
                workspace = self._frozen_workspace(
                    conn, root, stage, metadata, snapshot.developer_commit
                )
            except DeliveryRunnerError as exc:
                self._block(conn, root, "stage_execution_failed", str(exc))
                return
            before = self._git(workspace, "status", "--porcelain", "--untracked-files=no")
            existing = self._execute_stage(
                conn,
                root,
                stage,
                contract,
                "tester",
                snapshot.developer_commit,
                workspace,
                (),
            )
            after = self._git(workspace, "status", "--porcelain", "--untracked-files=no")
            if before != after or after:
                self._block(
                    conn,
                    root,
                    "tester_modified_source",
                    "tester changed tracked files in the frozen worktree",
                )
                return
        if existing is None:
            return
        report = existing.report
        if not isinstance(report, TesterReport):
            self._block(conn, root, "invalid_report", "tester returned the wrong report type")
            return
        if report.status == TesterReportStatus.BLOCKED:
            self._block(conn, root, "external_environment_missing", "tester reported BLOCKED")
            return
        if report.tested_commit != snapshot.developer_commit:
            self._block(conn, root, "commit_mismatch", "tested commit does not match developer commit")
            return
        if report.status == TesterReportStatus.TEST_PASS:
            self._transition(
                conn,
                root,
                FeatureDeliveryState.TEST_PASSED,
                {"tested_commit": report.tested_commit},
            )
        else:
            self._transition(conn, root, FeatureDeliveryState.TEST_FAILED)

    def _run_acceptance(
        self,
        conn,
        root: kb.Task,
        contract: TaskContract,
        metadata: dict,
        snapshot: DeliverySnapshot,
    ) -> None:
        target = snapshot.tested_commit
        if target is None or target != snapshot.developer_commit:
            self._block(conn, root, "commit_mismatch", "tested commit is missing or stale")
            return
        stage = self._ensure_stage(
            conn,
            root,
            "acceptance",
            target,
            snapshot.fix_loops + 1,
        )
        stored = self._report_for_stage(conn, root, stage, recover=True)
        if stored is None:
            try:
                workspace = self._frozen_workspace(conn, root, stage, metadata, target)
            except DeliveryRunnerError as exc:
                self._block(conn, root, "stage_execution_failed", str(exc))
                return
            before = self._git(workspace, "status", "--porcelain", "--untracked-files=no")
            stored = self._execute_stage(
                conn, root, stage, contract, "acceptance", target, workspace, ()
            )
            after = self._git(workspace, "status", "--porcelain", "--untracked-files=no")
            if before != after or after:
                self._block(
                    conn,
                    root,
                    "dirty_worktree",
                    "acceptance changed tracked files in the frozen worktree",
                )
                return
        if stored is None:
            return
        report = stored.report
        if not isinstance(report, AcceptanceReport):
            self._block(conn, root, "invalid_report", "acceptance returned the wrong report type")
            return
        if report.status == AcceptanceReportStatus.BLOCKED:
            self._block(conn, root, "external_environment_missing", "acceptance reported BLOCKED")
            return
        if report.accepted_commit != target:
            self._block(conn, root, "commit_mismatch", "accepted commit does not match tested commit")
            return
        if report.status == AcceptanceReportStatus.REJECT:
            self._transition(conn, root, FeatureDeliveryState.REJECTED)
            return

        branch_head = self._branch_head(Path(metadata["repository"]), contract.branch)
        current = self._snapshot(conn, root, recover=True)
        tester_evidence: tuple[str, ...] = ()
        for prior in reversed(current.reports):
            if isinstance(prior.report, TesterReport) and prior.report.tested_commit == target:
                tester_evidence = tuple(prior.report.evidence)
                break
        result = evaluate_delivery_gate(
            contract,
            report,
            DeliveryCommitContext(
                developer_commit=target,
                tested_commit=current.tested_commit,
                accepted_commit=report.accepted_commit,
                branch_head=branch_head,
            ),
            workflow_template_id=root.workflow_template_id or "",
            current_state=FeatureDeliveryState.ACCEPTANCE,
            expected_contract_hash=metadata["contract_sha256"],
            stage_evidence=tester_evidence,
        )
        if not result.allowed:
            self._block(
                conn,
                root,
                "acceptance_gate_denied",
                "; ".join(result.reasons),
            )
            return
        self._transition(
            conn,
            root,
            FeatureDeliveryState.DELIVERED,
            {"accepted_commit": report.accepted_commit, "gate": "passed"},
        )

    def _return_to_development(
        self, conn, root: kb.Task, snapshot: DeliverySnapshot
    ) -> None:
        self._transition(conn, root, FeatureDeliveryState.DEVELOPING)
        refreshed = kb.get_task(conn, root.id)
        assert refreshed is not None
        loops = self._snapshot(conn, refreshed, recover=True).fix_loops
        if loops >= MAX_FIX_LOOPS:
            self._block(
                conn,
                refreshed,
                "max_fix_loops_reached",
                f"maximum fix loops reached ({MAX_FIX_LOOPS})",
            )

    def _stage_report(
        self,
        conn,
        root: kb.Task,
        contract: TaskContract,
        role: StageRole,
        target_commit: str,
        attempt: int,
        workspace: Path,
        feedback: tuple[str, ...],
    ) -> StoredStageReport | None:
        stage = self._ensure_stage(conn, root, role, target_commit, attempt)
        existing = self._report_for_stage(conn, root, stage, recover=True)
        if existing is not None:
            return existing
        return self._execute_stage(
            conn, root, stage, contract, role, target_commit, workspace, feedback
        )

    def _execute_stage(
        self,
        conn,
        root: kb.Task,
        stage: kb.Task,
        contract: TaskContract,
        role: StageRole,
        target_commit: str,
        workspace: Path,
        feedback: tuple[str, ...],
    ) -> StoredStageReport | None:
        if self.executor is None:
            self._block(
                conn,
                root,
                "stage_executor_missing",
                "No configured stage executor",
            )
            return None
        try:
            run_id = self._start_run(conn, root, stage, role, target_commit)
        except StageAlreadyRunning:
            self._stage_busy = True
            return None
        try:
            raw = self.executor.execute(
                role=role,
                task_contract=contract,
                workspace=workspace,
                target_commit=target_commit,
                feedback=feedback,
                stage_task_id=stage.id,
            )
            report = self._coerce_report(role, raw)
            if report.task_id != contract.task_id:
                raise ValueError("report task_id does not match contract")
            return self._record_report(conn, root, stage, run_id, role, target_commit, report)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            self._fail_run(conn, stage, run_id, str(exc))
            code = "invalid_report" if isinstance(exc, (ValidationError, ValueError)) else "stage_execution_failed"
            self._block(conn, root, code, str(exc))
            return None

    def _ensure_stage(
        self,
        conn,
        root: kb.Task,
        role: StageRole,
        target_commit: str,
        attempt: int,
    ) -> kb.Task:
        rows = conn.execute(
            "SELECT t.* FROM tasks t JOIN task_links l ON l.child_id = t.id "
            "WHERE l.parent_id = ? ORDER BY t.created_at, t.id",
            (root.id,),
        ).fetchall()
        for row in rows:
            task = kb.Task.from_row(row)
            info = self._stage_info(task)
            if info and info.get("role") == role and int(info.get("attempt", 0)) == attempt:
                return task

        key = f"feature-delivery-stage:{root.id}:{role}:{target_commit}:{attempt}"
        stage_id = kb.create_task(
            conn,
            title=f"{root.title} [{role} {attempt}]",
            body=json.dumps(
                {
                    "feature_delivery_stage": {
                        "root_task_id": root.id,
                        "role": role,
                        "input_commit": target_commit,
                        "attempt": attempt,
                    }
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            created_by="feature-delivery-runner",
            parents=[root.id],
            idempotency_key=key,
            board=self.board,
        )
        stage = kb.get_task(conn, stage_id)
        if stage is None:
            raise DeliveryRunnerError("stage task creation failed")
        return stage

    @staticmethod
    def _stage_info(stage: kb.Task) -> dict | None:
        try:
            body = json.loads(stage.body or "{}")
            info = body.get("feature_delivery_stage")
            return info if isinstance(info, dict) else None
        except (TypeError, json.JSONDecodeError):
            return None

    def _start_run(
        self,
        conn,
        root: kb.Task,
        stage: kb.Task,
        role: StageRole,
        target_commit: str,
    ) -> int:
        now = int(time.time())
        with kb.write_txn(conn):
            active = conn.execute(
                "SELECT * FROM task_runs WHERE task_id = ? AND ended_at IS NULL "
                "ORDER BY id DESC LIMIT 1",
                (stage.id,),
            ).fetchone()
            if active:
                pid = active["worker_pid"]
                if pid and self._pid_alive(int(pid)):
                    raise StageAlreadyRunning(f"stage {stage.id} is already running")
                conn.execute(
                    "UPDATE task_runs SET status = 'crashed', outcome = 'crashed', "
                    "ended_at = ? WHERE id = ?",
                    (now, active["id"]),
                )
            metadata = {
                "root_task_id": root.id,
                "stage_task_id": stage.id,
                "role": role,
                "input_commit": target_commit,
                "report_path": None,
                "report_status": None,
            }
            cursor = conn.execute(
                "INSERT INTO task_runs "
                "(task_id, profile, step_key, status, claim_lock, claim_expires, "
                "worker_pid, started_at, metadata) "
                "VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?)",
                (
                    stage.id,
                    role,
                    root.current_step_key,
                    uuid.uuid4().hex,
                    now + 3600,
                    os.getpid(),
                    now,
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                ),
            )
            run_id = int(cursor.lastrowid)
            conn.execute(
                "UPDATE tasks SET status = 'running', current_run_id = ?, "
                "started_at = COALESCE(started_at, ?) WHERE id = ?",
                (run_id, now, stage.id),
            )
            kb._append_event(
                conn,
                stage.id,
                "claimed",
                {"run_id": run_id, "role": role, "root_task_id": root.id},
                run_id=run_id,
            )
        return run_id

    def _record_report(
        self,
        conn,
        root: kb.Task,
        stage: kb.Task,
        run_id: int,
        role: StageRole,
        target_commit: str,
        report: StageReport,
    ) -> StoredStageReport:
        data = report.model_dump(mode="json")
        canonical = self._canonical_json(data)
        report_hash = hashlib.sha256(canonical).hexdigest()
        report_path = self._next_report_path(conn, root.id, role)
        metadata = {
            "root_task_id": root.id,
            "stage_task_id": stage.id,
            "role": role,
            "input_commit": target_commit,
            "report_path": report_path,
            "report_status": report.status.value,
            "report_sha256": report_hash,
            "report": data,
        }
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET status = 'report_persisting', metadata = ? "
                "WHERE id = ? AND task_id = ? AND ended_at IS NULL",
                (json.dumps(metadata, ensure_ascii=False, sort_keys=True), run_id, stage.id),
            )
        self._materialize_report(conn, root.id, report_path, canonical)
        self._finish_report_run(conn, root, stage, run_id, metadata)
        return StoredStageReport(role, stage.id, run_id, report_path, report)

    def _materialize_report(
        self, conn, root_id: str, report_path: str, canonical: bytes
    ) -> None:
        destination = kb.task_attachments_dir(root_id, board=self.board) / report_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            try:
                existing = self._canonical_json(json.loads(destination.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                raise ReportIntegrityError(f"malformed report attachment {report_path}") from exc
            if existing != canonical:
                raise ReportIntegrityError(f"report attachment mismatch: {report_path}")
        else:
            temporary = destination.with_suffix(destination.suffix + f".{uuid.uuid4().hex}.tmp")
            temporary.write_bytes(canonical)
            os.replace(temporary, destination)
        attachments = kb.list_attachments(conn, root_id)
        matches = [item for item in attachments if item.filename == report_path]
        if matches and any(Path(item.stored_path).resolve() != destination.resolve() for item in matches):
            raise ReportIntegrityError(f"report attachment path mismatch: {report_path}")
        if not matches:
            kb.add_attachment(
                conn,
                root_id,
                filename=report_path,
                stored_path=str(destination),
                content_type="application/json",
                size=len(canonical),
                uploaded_by="feature-delivery-runner",
            )

    def _finish_report_run(self, conn, root, stage, run_id: int, metadata: dict) -> None:
        now = int(time.time())
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET status = ?, outcome = 'completed', summary = ?, "
                "metadata = ?, ended_at = ?, claim_lock = NULL, claim_expires = NULL, "
                "worker_pid = NULL WHERE id = ? AND task_id = ?",
                (
                    metadata["report_status"],
                    f"{metadata['role']} report: {metadata['report_status']}",
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                    now,
                    run_id,
                    stage.id,
                ),
            )
            conn.execute(
                "UPDATE tasks SET status = 'done', current_run_id = NULL, completed_at = ? "
                "WHERE id = ?",
                (now, stage.id),
            )
            kb._append_event(
                conn,
                root.id,
                "feature_delivery_report_saved",
                {
                    "stage_task_id": stage.id,
                    "run_id": run_id,
                    "role": metadata["role"],
                    "report_path": metadata["report_path"],
                    "report_status": metadata["report_status"],
                },
                run_id=run_id,
            )

    def _report_for_stage(
        self, conn, root: kb.Task, stage: kb.Task, *, recover: bool
    ) -> StoredStageReport | None:
        row = conn.execute(
            "SELECT * FROM task_runs WHERE task_id = ? AND metadata IS NOT NULL "
            "ORDER BY id DESC LIMIT 1",
            (stage.id,),
        ).fetchone()
        if row is None:
            return None
        try:
            metadata = json.loads(row["metadata"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ReportIntegrityError(f"malformed run metadata for stage {stage.id}") from exc
        if not metadata.get("report"):
            return None
        if recover and row["ended_at"] is None:
            canonical = self._canonical_json(metadata["report"])
            self._materialize_report(conn, root.id, metadata["report_path"], canonical)
            self._finish_report_run(conn, root, stage, int(row["id"]), metadata)
        return self._validate_stored_report(
            conn, root.id, stage.id, int(row["id"]), metadata
        )

    def _validate_stored_report(
        self,
        conn,
        root_id: str,
        stage_id: str,
        run_id: int,
        metadata: dict,
    ) -> StoredStageReport:
        role = metadata.get("role")
        if role not in {"developer", "tester", "acceptance"}:
            raise ReportIntegrityError(f"invalid report role for stage {stage_id}")
        report_path = metadata.get("report_path")
        if not isinstance(report_path, str) or not _REPORT_NAME_RE.match(report_path):
            raise ReportIntegrityError(f"invalid report path for stage {stage_id}")
        canonical = self._canonical_json(metadata.get("report"))
        if hashlib.sha256(canonical).hexdigest() != metadata.get("report_sha256"):
            raise ReportIntegrityError(f"report metadata hash mismatch: {report_path}")
        destination = kb.task_attachments_dir(root_id, board=self.board) / report_path
        if not destination.is_file():
            raise ReportIntegrityError(f"report attachment missing: {report_path}")
        try:
            attachment_data = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReportIntegrityError(f"malformed report attachment: {report_path}") from exc
        if self._canonical_json(attachment_data) != canonical:
            raise ReportIntegrityError(f"report attachment mismatch: {report_path}")
        attachments = [a for a in kb.list_attachments(conn, root_id) if a.filename == report_path]
        if len(attachments) != 1 or Path(attachments[0].stored_path).resolve() != destination.resolve():
            raise ReportIntegrityError(f"report attachment metadata mismatch: {report_path}")
        try:
            report = self._coerce_report(role, metadata["report"])
        except (ValidationError, ValueError, TypeError) as exc:
            raise ReportIntegrityError(f"invalid stored report: {report_path}") from exc
        if report.status.value != metadata.get("report_status"):
            raise ReportIntegrityError(f"report status mismatch: {report_path}")
        return StoredStageReport(role, stage_id, run_id, report_path, report)

    def _snapshot(self, conn, root: kb.Task, *, recover: bool) -> DeliverySnapshot:
        transitions: list[tuple[FeatureDeliveryState, FeatureDeliveryState]] = []
        blocked_code = None
        blocked_message = None
        for event in kb.list_events(conn, root.id):
            payload = event.payload or {}
            if event.kind == "workflow_step_transitioned":
                try:
                    transitions.append(
                        (
                            FeatureDeliveryState(payload["from_step"]),
                            FeatureDeliveryState(payload["to_step"]),
                        )
                    )
                except (KeyError, ValueError):
                    continue
                if payload.get("to_step") == FeatureDeliveryState.BLOCKED.value:
                    blocked_code = payload.get("code")
                    blocked_message = payload.get("message")

        rows = conn.execute(
            "SELECT t.* FROM tasks t JOIN task_links l ON l.child_id = t.id "
            "WHERE l.parent_id = ? ORDER BY t.created_at, t.id",
            (root.id,),
        ).fetchall()
        reports: list[StoredStageReport] = []
        for row in rows:
            stage = kb.Task.from_row(row)
            if self._stage_info(stage) is None:
                continue
            stored = self._report_for_stage(conn, root, stage, recover=recover)
            if stored is not None:
                reports.append(stored)
        reports.sort(key=lambda item: item.run_id)

        developer_commit = tested_commit = accepted_commit = None
        for stored in reports:
            report = stored.report
            if isinstance(report, DeveloperReport) and report.status == DeveloperReportStatus.READY_FOR_TEST:
                developer_commit = report.commit
                tested_commit = None
                accepted_commit = None
            elif (
                isinstance(report, TesterReport)
                and report.status == TesterReportStatus.TEST_PASS
                and report.tested_commit == developer_commit
            ):
                tested_commit = report.tested_commit
                accepted_commit = None
            elif (
                isinstance(report, AcceptanceReport)
                and report.status == AcceptanceReportStatus.ACCEPT
                and report.accepted_commit == tested_commit
            ):
                accepted_commit = report.accepted_commit
        last = reports[-1] if reports else None
        return DeliverySnapshot(
            developer_commit=developer_commit,
            tested_commit=tested_commit,
            accepted_commit=accepted_commit,
            fix_loops=count_fix_loops(transitions),
            last_stage=last.role if last else None,
            last_report_status=last.report.status.value if last else None,
            blocked_code=blocked_code,
            blocked_message=blocked_message,
            reports=tuple(reports),
        )

    def _feature_workspace(
        self, conn, root: kb.Task, contract: TaskContract, metadata: dict
    ) -> Path:
        repository = Path(metadata["repository"])
        target = repository / ".worktrees" / root.id
        kb._ensure_git_worktree(
            repository,
            target,
            contract.branch,
            start_point=contract.base_commit,
        )
        kb.set_workspace_path(conn, root.id, target)
        return target

    def _frozen_workspace(
        self,
        conn,
        root: kb.Task,
        stage: kb.Task,
        metadata: dict,
        commit: str,
    ) -> Path:
        repository = Path(metadata["repository"])
        target = kb.task_attachments_dir(root.id, board=self.board) / "worktrees" / stage.id
        kb._ensure_git_worktree(
            repository,
            target,
            f"detached/{stage.id}",
            start_point=commit,
            detached=True,
        )
        if self._git(target, "rev-parse", "HEAD") != commit:
            raise DeliveryRunnerError("frozen worktree is not at the required commit")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET workspace_kind = 'dir', workspace_path = ? WHERE id = ?",
                (str(target), stage.id),
            )
        return target

    def _validate_developer_commit(
        self,
        contract: TaskContract,
        repository: Path,
        workspace: Path,
        commit: str,
    ) -> tuple[str, str] | None:
        if not self._commit_exists(repository, commit):
            return "commit_mismatch", "developer commit does not exist"
        if self._git(workspace, "status", "--porcelain"):
            return "dirty_worktree", "developer worktree is not clean"
        if self._git(workspace, "branch", "--show-current") != contract.branch:
            return "commit_mismatch", "developer worktree is on the wrong branch"
        if self._branch_head(repository, contract.branch) != commit:
            return "commit_mismatch", "feature branch HEAD does not match developer report"
        result = subprocess.run(
            ["git", "-C", str(repository), "merge-base", "--is-ancestor", contract.base_commit, commit],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if result.returncode != 0:
            return "commit_mismatch", "developer commit is not a descendant of base commit"
        return None

    def _validate_repository(self, contract: TaskContract) -> Path:
        requested = Path(contract.repository).expanduser().resolve()
        top_level = (
            self._git(requested, "rev-parse", "--show-toplevel", check=False)
            if requested.is_dir()
            else ""
        )
        if not top_level:
            raise DeliveryRunnerError(f"repository does not exist or is not Git: {requested}")
        repository = Path(top_level).resolve()
        if not self._commit_exists(repository, contract.base_commit):
            raise DeliveryRunnerError(f"base commit does not exist: {contract.base_commit}")
        if self._git(repository, "status", "--porcelain"):
            raise DeliveryRunnerError("repository is dirty")
        ref_check = subprocess.run(
            ["git", "check-ref-format", "--branch", contract.branch],
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if ref_check.returncode != 0:
            raise DeliveryRunnerError(f"invalid feature branch: {contract.branch}")
        if self._branch_exists(repository, contract.branch):
            raise DeliveryRunnerError(f"feature branch already exists: {contract.branch}")
        return repository

    @staticmethod
    def _commit_exists(repository: Path, commit: str) -> bool:
        result = subprocess.run(
            ["git", "-C", str(repository), "cat-file", "-e", f"{commit}^{{commit}}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        return result.returncode == 0

    @staticmethod
    def _branch_exists(repository: Path, branch: str) -> bool:
        result = subprocess.run(
            ["git", "-C", str(repository), "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        return result.returncode == 0

    def _branch_head(self, repository: Path, branch: str) -> str:
        value = self._git(repository, "rev-parse", "--verify", f"refs/heads/{branch}", check=False)
        return value if re.fullmatch(r"[0-9a-f]{40}", value) else "0" * 40

    @staticmethod
    def _git(path: Path, *args: str, check: bool = True) -> str:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if check and result.returncode != 0:
            message = (result.stderr or result.stdout or "git command failed").strip()
            raise DeliveryRunnerError(message)
        return result.stdout.strip() if result.returncode == 0 else ""

    def _root_and_metadata(self, conn, task_id: str) -> tuple[kb.Task, dict]:
        root = kb.get_task(conn, task_id)
        if root is None:
            raise DeliveryRunnerError(f"unknown task: {task_id}")
        if root.workflow_template_id != FEATURE_DELIVERY_WORKFLOW:
            raise DeliveryRunnerError(f"task {task_id} is not a feature delivery root")
        if root.current_step_key not in {state.value for state in FeatureDeliveryState}:
            raise DeliveryRunnerError(f"task {task_id} has an invalid feature delivery state")
        events = [
            event
            for event in kb.list_events(conn, task_id)
            if event.kind == "feature_delivery_created" and event.payload
        ]
        if not events:
            raise DeliveryRunnerError(f"task {task_id} is missing delivery metadata")
        return root, dict(events[0].payload or {})

    def _load_contract(self, conn, root: kb.Task, metadata: dict) -> TaskContract:
        try:
            data = Path(metadata["contract_path"]).read_bytes()
            contract = TaskContract.model_validate_json(data)
        except (KeyError, OSError, ValidationError, ValueError) as exc:
            self._block(conn, root, "contract_hash_mismatch", f"cannot read canonical contract: {exc}")
            raise DeliveryRunnerError("contract attachment is invalid") from exc
        actual_hash = compute_contract_hash(contract)
        if actual_hash != metadata.get("contract_sha256") or data != canonicalize_contract(contract):
            self._block(conn, root, "contract_hash_mismatch", "contract attachment hash changed")
            raise DeliveryRunnerError("contract hash mismatch")
        return contract

    def _transition(
        self,
        conn,
        root: kb.Task,
        target: FeatureDeliveryState,
        payload: dict | None = None,
    ) -> bool:
        current = FeatureDeliveryState(root.current_step_key)
        if not is_legal_transition(current, target):
            raise DeliveryRunnerError(f"illegal transition {current.value} -> {target.value}")
        changed = kb.transition_workflow_step_cas(
            conn,
            task_id=root.id,
            workflow_template_id=FEATURE_DELIVERY_WORKFLOW,
            expected_step=current.value,
            new_step=target.value,
            event_payload=payload,
        )
        if changed:
            root.current_step_key = target.value
            return True
        refreshed = kb.get_task(conn, root.id)
        return bool(refreshed and refreshed.current_step_key == target.value)

    def _block(self, conn, root: kb.Task, code: str, message: str) -> None:
        if code not in BLOCKED_CODES:
            raise ValueError(f"unknown blocked reason code: {code}")
        refreshed = kb.get_task(conn, root.id)
        if refreshed is None or refreshed.current_step_key in {
            FeatureDeliveryState.BLOCKED.value,
            FeatureDeliveryState.DELIVERED.value,
        }:
            return
        self._transition(
            conn,
            refreshed,
            FeatureDeliveryState.BLOCKED,
            {"code": code, "message": message},
        )

    def _status(self, root: kb.Task, metadata: dict, snapshot: DeliverySnapshot) -> DeliveryStatus:
        reason = None
        if snapshot.blocked_code:
            reason = snapshot.blocked_code
            if snapshot.blocked_message:
                reason += f": {snapshot.blocked_message}"
        return DeliveryStatus(
            task_id=root.id,
            title=root.title,
            current_state=root.current_step_key or "",
            fix_loops=snapshot.fix_loops,
            branch=str(metadata.get("branch", "")),
            base_commit=str(metadata.get("base_commit", "")),
            developer_commit=snapshot.developer_commit,
            tested_commit=snapshot.tested_commit,
            accepted_commit=snapshot.accepted_commit,
            contract_hash=str(metadata.get("contract_sha256", "")),
            last_stage=snapshot.last_stage,
            last_report_status=snapshot.last_report_status,
            blocked_reason=reason,
        )

    def _next_report_path(self, conn, root_id: str, role: StageRole) -> str:
        numbers = []
        for attachment in kb.list_attachments(conn, root_id):
            match = _REPORT_NAME_RE.match(attachment.filename)
            if match:
                numbers.append(int(match.group(1)))
        return f"reports/{max(numbers, default=0) + 1:03d}-{role}.json"

    @staticmethod
    def _coerce_report(role: StageRole, raw: object) -> StageReport:
        model = {
            "developer": DeveloperReport,
            "tester": TesterReport,
            "acceptance": AcceptanceReport,
        }[role]
        report = raw if isinstance(raw, model) else model.model_validate(raw)
        if not validate_stage_report(role, report):
            raise ValueError(f"report does not belong to {role}")
        return report

    @staticmethod
    def _canonical_json(value: object) -> bytes:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def _fail_run(self, conn, stage: kb.Task, run_id: int, error: str) -> None:
        now = int(time.time())
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET status = 'failed', outcome = 'failed', error = ?, "
                "ended_at = ?, claim_lock = NULL, claim_expires = NULL, worker_pid = NULL "
                "WHERE id = ?",
                (error[:4000], now, run_id),
            )
            conn.execute(
                "UPDATE tasks SET status = 'failed', current_run_id = NULL WHERE id = ?",
                (stage.id,),
            )

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid == os.getpid():
            return True
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
