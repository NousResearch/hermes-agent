from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from agent.task_contracts import TaskContract, validate_task_contract


TERMINAL_TASK_STATUSES = {"completed", "failed", "cancelled"}
RETRY_REQUESTED_CONTINUATION_STATUSES = {"pending", "retry_requested"}


logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    draft = "draft"
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


ALLOWED_STATUS_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.draft: {TaskStatus.queued, TaskStatus.cancelled},
    TaskStatus.queued: {TaskStatus.running, TaskStatus.failed, TaskStatus.cancelled},
    TaskStatus.running: {TaskStatus.completed, TaskStatus.failed, TaskStatus.cancelled},
    TaskStatus.completed: set(),
    TaskStatus.failed: set(),
    TaskStatus.cancelled: set(),
}


class TaskExecutionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: TaskStatus = TaskStatus.draft
    background: bool = False
    process_session_id: Optional[str] = None
    process_command: Optional[str] = None
    process_task_id: Optional[str] = None
    exit_code: Optional[int] = None
    queued_at: Optional[float] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    last_error: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    continuation: dict[str, Any] = Field(default_factory=dict)


class PersistentTaskRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    created_at: float
    updated_at: float
    owner_session_id: Optional[str] = None
    parent_session_id: Optional[str] = None
    session_delegation_id: Optional[str] = None
    goal: str
    context: Optional[str] = None
    activeForm: str = ""
    blocks: list[str] = Field(default_factory=list)
    blockedBy: list[str] = Field(default_factory=list)
    owner: str = ""
    threadID: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    archetype: Optional[str] = None
    specialist: Optional[str] = None
    route_category: Optional[str] = None
    delegation_profile: Optional[str] = None
    runtime_mode: Optional[str] = None
    skills: list[str] = Field(default_factory=list)
    task_contract: Optional[TaskContract] = None
    permissions: dict[str, Any] = Field(default_factory=dict)
    resolved_inputs: dict[str, Any] = Field(default_factory=dict)
    launch_spec: dict[str, Any] = Field(default_factory=dict)
    execution: TaskExecutionState = Field(default_factory=TaskExecutionState)
    summary: Optional[str] = None

    @field_validator("goal")
    @classmethod
    def _require_goal(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("goal is required")
        return value


class TaskStore:
    def __init__(self, root_dir: Optional[str | os.PathLike[str]] = None, hooks: Any = None):
        base = Path(root_dir) if root_dir is not None else self._default_root_dir()
        self.root_dir = base
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._hooks = hooks

    @staticmethod
    def _default_root_dir() -> Path:
        hermes_home = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
        return Path(os.environ.get("HERMES_TASK_STORE_DIR", str(hermes_home / "tasks")))

    def _task_path(self, task_id: str) -> Path:
        return self.root_dir / f"{task_id}.json"

    def _delegate_exit_path(self, task_id: str) -> Path:
        return self.root_dir / f"{task_id}.exit"

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def _normalize_task_refs(task_ids: Optional[list[str]], *, exclude: Optional[str] = None) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        excluded = str(exclude or "").strip()
        for task_id in task_ids or []:
            value = str(task_id or "").strip()
            if not value or value == excluded or value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        return normalized

    def _load_task_unlocked(self, task_id: str) -> Optional[PersistentTaskRecord]:
        path = self._task_path(task_id)
        if not path.exists():
            return None
        try:
            return PersistentTaskRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, ValidationError):
            return None

    def _write_task_unlocked(self, record: PersistentTaskRecord) -> PersistentTaskRecord:
        record.updated_at = time.time()
        self._write_json_atomic(self._task_path(record.id), record.model_dump(mode="json"))
        return record

    @staticmethod
    def _is_sensitive_key(key: Any) -> bool:
        normalized = str(key or "").strip().lower()
        if not normalized:
            return False
        return any(
            marker in normalized
            for marker in (
                "secret",
                "token",
                "password",
                "passwd",
                "api_key",
                "apikey",
                "authorization",
                "credential",
                "cookie",
                "session",
                "private_key",
                "bearer",
            )
        )

    @classmethod
    def _redact_sensitive_data(cls, value: Any, *, key: Any = None) -> Any:
        if cls._is_sensitive_key(key):
            return "[REDACTED]"
        if isinstance(value, dict):
            return {k: cls._redact_sensitive_data(v, key=k) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._redact_sensitive_data(item) for item in value]
        if isinstance(value, tuple):
            return [cls._redact_sensitive_data(item) for item in value]
        return value

    def _build_hook_payload(self, event_name: str, record: PersistentTaskRecord, **extra: Any) -> dict[str, Any]:
        payload = {
            "event": event_name,
            "task_id": record.id,
            "task": self._redact_sensitive_data(record.model_dump(mode="json")),
        }
        for key, value in extra.items():
            payload[key] = self._redact_sensitive_data(value, key=key)
        return payload

    def _emit_hook(self, event_name: str, record: PersistentTaskRecord, **extra: Any) -> None:
        if self._hooks is None:
            return
        payload = self._build_hook_payload(event_name, record, **extra)
        try:
            if callable(self._hooks):
                self._hooks(event_name, payload)
                return
            emitter = getattr(self._hooks, "emit", None) or getattr(self._hooks, "fire", None)
            if callable(emitter):
                emitter(event_name, payload)
        except Exception:
            logger.warning("task lifecycle hook failed for %s", event_name, exc_info=True)

    def _sync_block_relationships(self, record: PersistentTaskRecord) -> PersistentTaskRecord:
        previous = self._load_task_unlocked(record.id)
        previous_blocks = set(previous.blocks or []) if previous is not None else set()
        previous_blocked_by = set(previous.blockedBy or []) if previous is not None else set()

        record.blocks = self._normalize_task_refs(record.blocks, exclude=record.id)
        record.blockedBy = self._normalize_task_refs(record.blockedBy, exclude=record.id)

        other_records: dict[str, PersistentTaskRecord] = {}
        for path in sorted(self.root_dir.glob("*.json")):
            task_id = path.stem
            if task_id == record.id:
                continue
            other = self._load_task_unlocked(task_id)
            if other is None:
                continue
            other.blocks = self._normalize_task_refs(other.blocks, exclude=other.id)
            other.blockedBy = self._normalize_task_refs(other.blockedBy, exclude=other.id)
            other_records[task_id] = other

        desired_blocks = list(record.blocks)
        desired_blocked_by = list(record.blockedBy)
        desired_blocks_set = set(desired_blocks)
        desired_blocked_by_set = set(desired_blocked_by)

        for other in other_records.values():
            if (
                record.id in other.blockedBy
                and other.id not in desired_blocks_set
                and other.id not in previous_blocks
            ):
                desired_blocks.append(other.id)
                desired_blocks_set.add(other.id)
            if (
                record.id in other.blocks
                and other.id not in desired_blocked_by_set
                and other.id not in previous_blocked_by
            ):
                desired_blocked_by.append(other.id)
                desired_blocked_by_set.add(other.id)

        record.blocks = desired_blocks
        record.blockedBy = desired_blocked_by

        for other in other_records.values():
            other_changed = False

            if other.id in desired_blocks_set:
                if record.id not in other.blockedBy:
                    other.blockedBy.append(record.id)
                    other.blockedBy = self._normalize_task_refs(other.blockedBy, exclude=other.id)
                    other_changed = True
            elif record.id in other.blockedBy:
                other.blockedBy = [task_id for task_id in other.blockedBy if task_id != record.id]
                other_changed = True

            if other.id in desired_blocked_by_set:
                if record.id not in other.blocks:
                    other.blocks.append(record.id)
                    other.blocks = self._normalize_task_refs(other.blocks, exclude=other.id)
                    other_changed = True
            elif record.id in other.blocks:
                other.blocks = [task_id for task_id in other.blocks if task_id != record.id]
                other_changed = True

            if other_changed:
                self._write_task_unlocked(other)

        return record

    def _sync_session_delegation_visibility(self, record: PersistentTaskRecord) -> PersistentTaskRecord:
        session_id = str(record.owner_session_id or record.parent_session_id or "").strip()
        if not session_id:
            return record

        status = record.execution.status.value
        if status == TaskStatus.draft.value and not record.session_delegation_id:
            return record

        try:
            from agent.orchestration_state import record_delegation_end, record_delegation_start
        except Exception:
            return record

        toolsets = list(record.launch_spec.get("toolsets") or [])
        is_persistent = bool(record.launch_spec)
        result_payload = record.execution.result if isinstance(record.execution.result, dict) else {}
        summary = record.summary if record.summary is not None else record.execution.last_error
        model = result_payload.get("model") or record.launch_spec.get("model")
        api_calls = result_payload.get("api_calls")
        duration_seconds = result_payload.get("duration_seconds")

        if not record.session_delegation_id:
            record.session_delegation_id = record_delegation_start(
                session_id,
                goal=record.goal,
                status=status,
                task_index=record.launch_spec.get("task_index"),
                toolsets=toolsets,
                task_id=record.id,
                persistent=is_persistent,
                background=record.execution.background,
                process_session_id=record.execution.process_session_id,
                process_command=record.execution.process_command,
                queued_at=record.execution.queued_at,
                started_at=record.execution.started_at,
                finished_at=record.execution.finished_at,
                error=record.execution.last_error,
                exit_code=record.execution.exit_code,
            )

        record_delegation_end(
            session_id,
            record.session_delegation_id,
            status=status,
            summary=summary,
            api_calls=api_calls,
            duration_seconds=duration_seconds,
            model=model,
            task_id=record.id,
            persistent=is_persistent,
            background=record.execution.background,
            process_session_id=record.execution.process_session_id,
            process_command=record.execution.process_command,
            queued_at=record.execution.queued_at,
            started_at=record.execution.started_at,
            finished_at=record.execution.finished_at,
            error=record.execution.last_error,
            exit_code=record.execution.exit_code,
        )
        return record

    def save_task(self, record: PersistentTaskRecord) -> PersistentTaskRecord:
        with self._lock:
            record = self._sync_block_relationships(record)
            self._write_task_unlocked(record)
        return record

    def clear_delegate_exit_artifact(self, task_id: str) -> None:
        with self._lock:
            exit_path = self._delegate_exit_path(task_id)
            try:
                exit_path.unlink()
            except FileNotFoundError:
                return

    def create_task(
        self,
        *,
        goal: str,
        context: Optional[str] = None,
        activeForm: str = "",
        blocks: Optional[list[str]] = None,
        blockedBy: Optional[list[str]] = None,
        owner: str = "",
        threadID: str = "",
        metadata: Optional[dict[str, Any]] = None,
        owner_session_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        archetype: Optional[str] = None,
        specialist: Optional[str] = None,
        route_category: Optional[str] = None,
        delegation_profile: Optional[str] = None,
        runtime_mode: Optional[str] = None,
        skills: Optional[list[str]] = None,
        task_contract: Optional[dict[str, Any] | TaskContract] = None,
        permissions: Optional[dict[str, Any]] = None,
        resolved_inputs: Optional[dict[str, Any]] = None,
        launch_spec: Optional[dict[str, Any]] = None,
        task_id: Optional[str] = None,
    ) -> PersistentTaskRecord:
        now = time.time()
        contract_model = validate_task_contract(task_contract) if task_contract is not None else None
        record = PersistentTaskRecord(
            id=task_id or f"task_{uuid.uuid4().hex[:12]}",
            created_at=now,
            updated_at=now,
            owner_session_id=owner_session_id,
            parent_session_id=parent_session_id,
            goal=goal,
            context=context,
            activeForm=str(activeForm or ""),
            blocks=[str(item).strip() for item in (blocks or []) if str(item).strip()],
            blockedBy=[str(item).strip() for item in (blockedBy or []) if str(item).strip()],
            owner=str(owner or ""),
            threadID=str(threadID or ""),
            metadata=dict(metadata or {}),
            archetype=archetype,
            specialist=specialist,
            route_category=route_category,
            delegation_profile=delegation_profile,
            runtime_mode=runtime_mode,
            skills=list(skills or []),
            task_contract=contract_model,
            permissions=dict(permissions or {}),
            resolved_inputs=dict(resolved_inputs or {}),
            launch_spec=dict(launch_spec or {}),
        )
        saved = self.save_task(record)
        self._emit_hook("task.created", saved)
        return saved

    def get_task(self, task_id: str) -> Optional[PersistentTaskRecord]:
        return self._load_task_unlocked(task_id)

    def require_task(self, task_id: str) -> PersistentTaskRecord:
        record = self.get_task(task_id)
        if record is None:
            raise KeyError(f"task not found: {task_id}")
        return record

    def list_tasks(
        self,
        *,
        owner_session_id: Optional[str] = None,
        statuses: Optional[set[str] | list[str] | tuple[str, ...]] = None,
    ) -> list[PersistentTaskRecord]:
        normalized_statuses = {str(s) for s in statuses} if statuses else None
        records: list[PersistentTaskRecord] = []
        for path in sorted(self.root_dir.glob("*.json")):
            record = self.get_task(path.stem)
            if record is None:
                continue
            if owner_session_id and record.owner_session_id != owner_session_id:
                continue
            if normalized_statuses and record.execution.status.value not in normalized_statuses:
                continue
            records.append(record)
        return sorted(records, key=lambda r: r.created_at)

    def get_tasks_by_thread(self, thread_id: str) -> list[PersistentTaskRecord]:
        normalized_thread_id = str(thread_id or "").strip()
        if not normalized_thread_id:
            return []
        return [
            record
            for record in self.list_tasks()
            if str(record.threadID or "").strip() == normalized_thread_id
        ]

    def _dependencies_completed(self, record: PersistentTaskRecord) -> bool:
        dependency_ids = [str(task_id).strip() for task_id in (record.blockedBy or []) if str(task_id).strip()]
        if not dependency_ids:
            return True
        for dependency_id in dependency_ids:
            dependency = self.get_task(dependency_id)
            if dependency is None or dependency.execution.status != TaskStatus.completed:
                return False
        return True

    def _thread_is_available(self, record: PersistentTaskRecord) -> bool:
        thread_id = str(record.threadID or "").strip()
        if not thread_id:
            return True
        for sibling in self.get_tasks_by_thread(thread_id):
            if sibling.id == record.id:
                return True
            if sibling.created_at >= record.created_at:
                return True
            if sibling.execution.status.value not in TERMINAL_TASK_STATUSES:
                return False
        return True

    def get_runnable_tasks(
        self,
        *,
        owner_session_id: Optional[str] = None,
        statuses: Optional[set[str] | list[str] | tuple[str, ...]] = None,
    ) -> list[PersistentTaskRecord]:
        candidate_statuses = {str(s) for s in statuses} if statuses else {TaskStatus.draft.value, TaskStatus.queued.value}
        runnable: list[PersistentTaskRecord] = []
        for record in self.list_tasks(owner_session_id=owner_session_id, statuses=candidate_statuses):
            if record.execution.process_session_id:
                continue
            if not self._dependencies_completed(record):
                continue
            if not self._thread_is_available(record):
                continue
            runnable.append(record)
        return runnable

    def _update_blocked_dependents(self, completed_task_id: str) -> None:
        normalized_task_id = str(completed_task_id or "").strip()
        if not normalized_task_id:
            return
        for record in self.list_tasks():
            blocked_by = {str(task_id).strip() for task_id in (record.blockedBy or []) if str(task_id).strip()}
            if normalized_task_id not in blocked_by:
                continue
            if record.execution.status in {TaskStatus.running, TaskStatus.completed, TaskStatus.failed, TaskStatus.cancelled}:
                continue
            if not self._dependencies_completed(record):
                continue
            if record.execution.status == TaskStatus.draft:
                self.transition_task(record.id, TaskStatus.queued)

    def transition_task(
        self,
        task_id: str,
        new_status: TaskStatus | str,
        **execution_updates: Any,
    ) -> PersistentTaskRecord:
        record = self.require_task(task_id)
        current = record.execution.status
        target = TaskStatus(new_status)
        if target != current and target not in ALLOWED_STATUS_TRANSITIONS[current]:
            raise ValueError(f"invalid task status transition: {current.value} -> {target.value}")

        now = time.time()
        record.execution.status = target
        if target == TaskStatus.queued and record.execution.queued_at is None:
            record.execution.queued_at = now
        if target == TaskStatus.running and record.execution.started_at is None:
            record.execution.started_at = now
        if target.value in TERMINAL_TASK_STATUSES:
            record.execution.finished_at = execution_updates.pop("finished_at", now)
            record.execution.process_session_id = execution_updates.pop("process_session_id", None)
        for key, value in execution_updates.items():
            if hasattr(record.execution, key):
                setattr(record.execution, key, value)
        self._sync_session_delegation_visibility(record)
        saved = self.save_task(record)
        if target == TaskStatus.running and current != TaskStatus.running:
            self._emit_hook("task.started", saved)
        elif target == TaskStatus.cancelled and current != TaskStatus.cancelled:
            self._emit_hook("task.cancelled", saved)
        elif target == TaskStatus.completed and current != TaskStatus.completed:
            self._emit_hook("task.completed", saved)
        elif target == TaskStatus.failed and current != TaskStatus.failed:
            self._emit_hook("task.failed", saved)
        if target == TaskStatus.completed:
            self._update_blocked_dependents(task_id)
        return saved

    def attach_process(
        self,
        task_id: str,
        *,
        process_session_id: str,
        process_command: str,
        process_task_id: Optional[str] = None,
        background: bool = True,
    ) -> PersistentTaskRecord:
        record = self.require_task(task_id)
        if (
            record.execution.process_session_id
            and record.execution.status in {TaskStatus.queued, TaskStatus.running}
        ):
            raise ValueError(f"task already has an active process: {task_id}")
        return self.transition_task(
            task_id,
            TaskStatus.queued,
            background=background,
            process_session_id=process_session_id,
            process_command=process_command,
            process_task_id=process_task_id or task_id,
            exit_code=None,
            last_error=None,
        )

    def record_result(
        self,
        task_id: str,
        *,
        status: TaskStatus | str,
        result: Optional[dict[str, Any]] = None,
        summary: Optional[str] = None,
        error: Optional[str] = None,
        exit_code: Optional[int] = None,
    ) -> PersistentTaskRecord:
        record = self.require_task(task_id)
        current = record.execution.status
        target = TaskStatus(status)
        if target != current and target not in ALLOWED_STATUS_TRANSITIONS[current]:
            raise ValueError(f"invalid task status transition: {current.value} -> {target.value}")

        if result is not None:
            record.execution.result = result
        if summary is not None:
            record.summary = summary
        record.execution.last_error = error
        record.execution.exit_code = exit_code
        record.execution.status = target
        if target.value in TERMINAL_TASK_STATUSES:
            record.execution.finished_at = time.time()
            record.execution.process_session_id = None
            if target in {TaskStatus.completed, TaskStatus.cancelled}:
                continuation = dict(record.execution.continuation or {})
                if str(continuation.get("status") or "").strip().lower() in RETRY_REQUESTED_CONTINUATION_STATUSES:
                    continuation.pop("status", None)
                    continuation["retry_cleared_at"] = time.time()
                    record.execution.continuation = continuation
        self._sync_session_delegation_visibility(record)
        saved = self.save_task(record)
        if target == TaskStatus.completed and current != TaskStatus.completed:
            self._emit_hook("task.completed", saved)
        elif target == TaskStatus.failed and current != TaskStatus.failed:
            self._emit_hook("task.failed", saved)
        elif target == TaskStatus.cancelled and current != TaskStatus.cancelled:
            self._emit_hook("task.cancelled", saved)
        if target == TaskStatus.completed:
            self._update_blocked_dependents(task_id)
        return saved

    def update_continuation(
        self,
        task_id: str,
        *,
        mode: Optional[str] = None,
        status: Optional[str] = None,
        open_todos: Optional[list[dict[str, Any]]] = None,
        latest_response_preview: Optional[str] = None,
        last_outcome_status: Optional[str] = None,
        resume_count: Optional[int] = None,
        attempt_count: Optional[int] = None,
    ) -> PersistentTaskRecord:
        record = self.require_task(task_id)
        state = dict(record.execution.continuation or {})
        previous_status = str(state.get("status") or "").strip().lower()
        if mode is not None:
            state["mode"] = str(mode or "").strip() or None
        if status is not None:
            state["status"] = str(status or "").strip() or None
        if open_todos is not None:
            state["open_todos"] = [dict(item) for item in open_todos if isinstance(item, dict)]
        if latest_response_preview is not None:
            state["latest_response_preview"] = str(latest_response_preview or "").strip()
        if last_outcome_status is not None:
            state["last_outcome_status"] = str(last_outcome_status or "").strip() or None
        if resume_count is not None:
            state["resume_count"] = int(resume_count)
        if attempt_count is not None:
            state["attempt_count"] = int(attempt_count)
        state["updated_at"] = time.time()
        record.execution.continuation = state
        self._sync_session_delegation_visibility(record)
        saved = self.save_task(record)
        current_status = str(saved.execution.continuation.get("status") or "").strip().lower()
        if (
            current_status in RETRY_REQUESTED_CONTINUATION_STATUSES
            and previous_status not in RETRY_REQUESTED_CONTINUATION_STATUSES
        ):
            self._emit_hook("task.retry_requested", saved, continuation_status=current_status)
        return saved

    def clear_continuation(self, task_id: str) -> PersistentTaskRecord:
        record = self.require_task(task_id)
        record.execution.continuation = {}
        self._sync_session_delegation_visibility(record)
        return self.save_task(record)

    def prepare_for_retry(self, task_id: str) -> PersistentTaskRecord:
        record = self.require_task(task_id)
        continuation = dict(record.execution.continuation or {})
        if str(continuation.get("status") or "").strip().lower() not in {"pending", "retry_requested"}:
            raise ValueError(f"task {task_id} does not have a retry-requested continuation")
        if record.execution.status in {TaskStatus.queued, TaskStatus.running}:
            raise ValueError(f"task already active: {task_id}")
        record.execution.status = TaskStatus.draft
        record.execution.background = bool(record.launch_spec.get("background") or record.execution.background)
        record.execution.process_session_id = None
        record.execution.process_command = None
        record.execution.process_task_id = None
        record.execution.exit_code = None
        record.execution.queued_at = None
        record.execution.started_at = None
        record.execution.finished_at = None
        record.execution.last_error = None
        record.execution.result = None
        continuation.pop("status", None)
        continuation["retry_prepared_at"] = time.time()
        record.execution.continuation = continuation
        record.summary = None
        self.clear_delegate_exit_artifact(task_id)
        self._sync_session_delegation_visibility(record)
        return self.save_task(record)

    @staticmethod
    def _delegate_completion_snapshot(result: Any) -> Optional[dict[str, Any]]:
        if not isinstance(result, dict):
            return None
        if str(result.get("status") or "").strip():
            return result
        if any(key in result for key in ("continuation", "orchestration", "summary", "api_calls")):
            return result
        return None

    @classmethod
    def _derive_delegate_terminal_payload(
        cls,
        record: PersistentTaskRecord,
        *,
        exit_code: Optional[int],
        fallback_error: Optional[str],
    ) -> Optional[dict[str, Any]]:
        snapshot = cls._delegate_completion_snapshot(record.execution.result)
        if snapshot is None:
            return None

        continuation = dict(snapshot.get("continuation") or {})
        orchestration = dict(snapshot.get("orchestration") or {})
        open_todos = continuation.get("open_todos") or orchestration.get("activeTodos") or []
        child_status = str(snapshot.get("status") or "").strip().lower()
        is_complete = child_status == "completed" and not open_todos
        error = snapshot.get("error") or fallback_error
        if open_todos and not error:
            error = "unfinished work remains"
        if not is_complete and not error:
            error = "delegate completion snapshot did not record a completed outcome"

        return {
            "status": TaskStatus.completed if is_complete else TaskStatus.failed,
            "summary": snapshot.get("summary") or record.summary,
            "error": error,
            "result": snapshot,
            "exit_code": exit_code,
        }

    def _reconcile_delegate_exit(
        self,
        task_id: str,
        *,
        record: PersistentTaskRecord,
        exit_code: Optional[int],
        fallback_error: Optional[str],
    ) -> PersistentTaskRecord:
        derived = self._derive_delegate_terminal_payload(
            record,
            exit_code=exit_code,
            fallback_error=fallback_error,
        )
        if derived is None:
            result_payload = record.execution.result or {
                "output": "",
                "exit_code": exit_code,
                "command": record.execution.process_command,
            }
            summary = record.summary or (str(result_payload.get("output") or "").strip()[:400] or None)
            error = fallback_error
            if exit_code == 0 and not error:
                derived = {
                    "status": TaskStatus.completed,
                    "summary": summary,
                    "error": None,
                    "result": result_payload,
                    "exit_code": exit_code,
                }
            else:
                if not error:
                    error = "background task failed"
                derived = {
                    "status": TaskStatus.failed,
                    "summary": summary,
                    "error": error,
                    "result": result_payload,
                    "exit_code": exit_code,
                }
        return self.record_result(
            task_id,
            status=derived["status"],
            result=derived["result"],
            summary=derived["summary"],
            error=derived["error"],
            exit_code=derived["exit_code"],
        )

    def reconcile_task(self, task_id: str, process_registry=None) -> PersistentTaskRecord:
        from tools.process_registry import process_registry as default_process_registry

        def _load_delegate_exit_code() -> Optional[int]:
            exit_path = self._delegate_exit_path(task_id)
            if not exit_path.exists():
                return None
            try:
                exit_text = exit_path.read_text(encoding="utf-8").strip()
                if not exit_text:
                    return None
                return int(exit_text.splitlines()[-1].strip())
            except Exception:
                return None

        registry = process_registry or default_process_registry
        record = self.require_task(task_id)
        session_id = record.execution.process_session_id
        if not session_id:
            return record

        poll = registry.poll(session_id)
        status = poll.get("status")
        latest_record = self.require_task(task_id)
        if latest_record.execution.process_session_id != session_id:
            return latest_record
        if status == "running":
            if record.execution.status in {TaskStatus.draft, TaskStatus.queued}:
                record = self.transition_task(task_id, TaskStatus.running)
            self._emit_hook("task.reconciled", record, process_status=status)
            return record

        if status == "exited":
            exit_code = poll.get("exit_code")
            is_delegate_runner = record.launch_spec.get("runner") == "delegate"
            if exit_code is None and is_delegate_runner:
                exit_code = _load_delegate_exit_code()
            result_payload = record.execution.result or {
                "output": poll.get("output_preview", ""),
                "exit_code": exit_code,
                "command": record.execution.process_command,
            }
            if record.execution.status not in {TaskStatus.completed, TaskStatus.failed, TaskStatus.cancelled}:
                if is_delegate_runner:
                    reconciled = self._reconcile_delegate_exit(
                        task_id,
                        record=record,
                        exit_code=exit_code,
                        fallback_error=None if exit_code == 0 else (poll.get("output_preview") or "background task failed"),
                    )
                    self._emit_hook("task.reconciled", reconciled, process_status=status, exit_code=exit_code)
                    return reconciled
                if exit_code == 0 and record.execution.status in {TaskStatus.draft, TaskStatus.queued}:
                    record = self.transition_task(task_id, TaskStatus.running)
                summary = record.summary or ((poll.get("output_preview") or "").strip()[:400] or None)
                error = None if exit_code == 0 else (poll.get("output_preview") or "background task failed")
                reconciled = self.record_result(
                    task_id,
                    status=TaskStatus.completed if exit_code == 0 else TaskStatus.failed,
                    result=result_payload,
                    summary=summary,
                    error=error,
                    exit_code=exit_code,
                )
                self._emit_hook("task.reconciled", reconciled, process_status=status, exit_code=exit_code)
                return reconciled
            record.execution.process_session_id = None
            record.execution.exit_code = exit_code
            if record.execution.result is None:
                record.execution.result = result_payload
            self._sync_session_delegation_visibility(record)
            reconciled = self.save_task(record)
            self._emit_hook("task.reconciled", reconciled, process_status=status, exit_code=exit_code)
            return reconciled

        if status == "not_found":
            delegate_exit_code = _load_delegate_exit_code() if record.launch_spec.get("runner") == "delegate" else None
            if delegate_exit_code is not None and record.execution.status not in {TaskStatus.completed, TaskStatus.failed, TaskStatus.cancelled}:
                reconciled = self._reconcile_delegate_exit(
                    task_id,
                    record=record,
                    exit_code=delegate_exit_code,
                    fallback_error=None if delegate_exit_code == 0 else (record.execution.last_error or "background task failed"),
                )
                self._emit_hook("task.reconciled", reconciled, process_status=status, exit_code=delegate_exit_code)
                return reconciled
            if record.execution.status not in {TaskStatus.completed, TaskStatus.failed, TaskStatus.cancelled}:
                reconciled = self.record_result(
                    task_id,
                    status=TaskStatus.failed,
                    result=record.execution.result,
                    summary=record.summary,
                    error=record.execution.last_error or "background process record was lost",
                    exit_code=record.execution.exit_code,
                )
                self._emit_hook("task.reconciled", reconciled, process_status=status)
                return reconciled
            record.execution.process_session_id = None
            self._sync_session_delegation_visibility(record)
            reconciled = self.save_task(record)
            self._emit_hook("task.reconciled", reconciled, process_status=status)
            return reconciled

        return record

    def reconcile_tasks(
        self,
        *,
        owner_session_id: Optional[str] = None,
        task_ids: Optional[list[str]] = None,
        process_registry=None,
    ) -> list[PersistentTaskRecord]:
        if task_ids is not None:
            return [self.reconcile_task(task_id, process_registry=process_registry) for task_id in task_ids]
        return [
            self.reconcile_task(record.id, process_registry=process_registry)
            for record in self.list_tasks(owner_session_id=owner_session_id)
        ]


__all__ = [
    "ALLOWED_STATUS_TRANSITIONS",
    "PersistentTaskRecord",
    "TERMINAL_TASK_STATUSES",
    "TaskExecutionState",
    "TaskStatus",
    "TaskStore",
]
