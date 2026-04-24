from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Callable

from agent.task_store import PersistentTaskRecord, TaskStatus, TaskStore

LaunchPayload = dict[str, Any]
Launcher = Callable[[PersistentTaskRecord, dict[str, Any]], LaunchPayload]


class AtlasTaskScheduler:
    def __init__(
        self,
        *,
        task_store: TaskStore,
        launcher: Launcher,
        owner: str = "",
        agent_name: str = "",
        model: str = "",
        runtime_mode: str | None = None,
        default_max_launches: int = 1,
    ) -> None:
        self.task_store = task_store
        self.launcher = launcher
        self.owner = str(owner or "").strip()
        self.agent_name = str(agent_name or "").strip()
        self.model = str(model or "").strip()
        self.runtime_mode = str(runtime_mode or "").strip() or None
        self.default_max_launches = max(1, int(default_max_launches))

    def _continuation_status(self, record: PersistentTaskRecord) -> str:
        continuation = record.execution.continuation or {}
        return str(continuation.get("status") or "").strip().lower()

    def _retry_requested(self, record: PersistentTaskRecord) -> bool:
        return self._continuation_status(record) in {"pending", "retry_requested"}

    def _prepare_retry_tasks(self, *, owner_session_id: str | None = None) -> list[str]:
        prepared: list[str] = []
        for record in self.task_store.list_tasks(owner_session_id=owner_session_id):
            if not self._retry_requested(record):
                continue
            if record.execution.status in {TaskStatus.queued, TaskStatus.running}:
                continue
            self.task_store.prepare_for_retry(record.id)
            prepared.append(record.id)
        return prepared

    def _launch_runtime_mode(self, record: PersistentTaskRecord) -> str | None:
        return str(record.runtime_mode or self.runtime_mode or "").strip() or None

    def _record_scheduler_metadata(self, record: PersistentTaskRecord) -> PersistentTaskRecord:
        updated = False
        if self.owner and record.owner != self.owner:
            record.owner = self.owner
            updated = True

        metadata = dict(record.metadata or {})
        if self.agent_name and metadata.get("agent") != self.agent_name:
            metadata["agent"] = self.agent_name
            updated = True
        if self.model and metadata.get("model") != self.model:
            metadata["model"] = self.model
            updated = True

        runtime_mode = self._launch_runtime_mode(record)
        if runtime_mode and record.runtime_mode != runtime_mode:
            record.runtime_mode = runtime_mode
            updated = True

        if updated:
            record.metadata = metadata
            return self.task_store.save_task(record)
        return record

    def _build_launch_spec(self, record: PersistentTaskRecord) -> dict[str, Any]:
        launch_spec = deepcopy(record.launch_spec or {})
        launch_spec.setdefault("task_id", record.id)
        launch_spec.setdefault("goal", record.goal)
        launch_spec.setdefault("thread_id", record.threadID)
        if self.owner:
            launch_spec["owner"] = self.owner
        if self.agent_name:
            launch_spec["agent"] = self.agent_name
        if self.model:
            launch_spec["model"] = self.model
        runtime_mode = self._launch_runtime_mode(record)
        if runtime_mode:
            launch_spec["runtime_mode"] = runtime_mode
        return launch_spec

    @staticmethod
    def _normalize_process_payload(record: PersistentTaskRecord, launch_spec: dict[str, Any], payload: LaunchPayload) -> dict[str, Any]:
        process_session_id = str(payload.get("process_session_id") or payload.get("session_id") or "").strip()
        if not process_session_id:
            raise ValueError(f"launcher did not return a process/session id for task {record.id}")

        process_task_id = str(
            payload.get("process_task_id")
            or payload.get("task_id")
            or payload.get("process_id")
            or record.id
        ).strip()
        process_command = str(
            payload.get("process_command")
            or payload.get("command")
            or launch_spec.get("command")
            or launch_spec.get("task_id")
            or record.goal
        ).strip()
        background = bool(payload.get("background", launch_spec.get("background", True)))
        return {
            "process_session_id": process_session_id,
            "process_task_id": process_task_id,
            "process_command": process_command,
            "background": background,
        }

    def dry_run(self, *, max_launches: int | None = None, owner_session_id: str | None = None) -> dict[str, Any]:
        limit = max(0, self.default_max_launches if max_launches is None else int(max_launches))
        runnable = self.task_store.get_runnable_tasks(owner_session_id=owner_session_id)
        launchable = runnable[:limit] if limit else []
        return {
            "launchable_task_ids": [record.id for record in launchable],
            "prepared_retry_task_ids": [
                record.id
                for record in self.task_store.list_tasks(owner_session_id=owner_session_id)
                if self._retry_requested(record) and record.execution.status not in {TaskStatus.queued, TaskStatus.running}
            ],
        }

    def status(self, *, owner_session_id: str | None = None) -> dict[str, Any]:
        tasks = self.task_store.list_tasks(owner_session_id=owner_session_id)
        runnable_ids = {record.id for record in self.task_store.get_runnable_tasks(owner_session_id=owner_session_id)}
        counts = Counter(record.execution.status.value for record in tasks)
        blocked: dict[str, dict[str, Any]] = {}

        for record in tasks:
            if record.id in runnable_ids:
                continue
            if record.execution.status.value in {TaskStatus.queued.value, TaskStatus.running.value}:
                continue
            dependency_ids = [task_id for task_id in record.blockedBy if str(task_id or "").strip()]
            failed_dependencies = []
            incomplete_dependencies = []
            for dependency_id in dependency_ids:
                dependency = self.task_store.get_task(dependency_id)
                if dependency is None:
                    incomplete_dependencies.append(dependency_id)
                    continue
                if dependency.execution.status == TaskStatus.failed:
                    failed_dependencies.append(dependency_id)
                elif dependency.execution.status != TaskStatus.completed:
                    incomplete_dependencies.append(dependency_id)
            if failed_dependencies:
                blocked[record.id] = {"reason": "dependency_failed", "dependency_ids": failed_dependencies}
                continue
            if incomplete_dependencies:
                blocked[record.id] = {"reason": "waiting_on_dependencies", "dependency_ids": incomplete_dependencies}
                continue
            if record.threadID and record.execution.status in {TaskStatus.draft, TaskStatus.queued}:
                blocked[record.id] = {"reason": "thread_busy", "thread_id": record.threadID}

        return {
            "counts": dict(counts),
            "runnable_task_ids": sorted(runnable_ids),
            "blocked_tasks": blocked,
        }

    def run_once(self, *, max_launches: int | None = None, owner_session_id: str | None = None) -> dict[str, Any]:
        limit = max(0, self.default_max_launches if max_launches is None else int(max_launches))
        prepared_retry_task_ids = self._prepare_retry_tasks(owner_session_id=owner_session_id)
        launched_task_ids: list[str] = []

        while len(launched_task_ids) < limit:
            runnable = self.task_store.get_runnable_tasks(owner_session_id=owner_session_id)
            if not runnable:
                break
            record = self._record_scheduler_metadata(runnable[0])
            launch_spec = self._build_launch_spec(record)
            payload = self.launcher(record, launch_spec)
            reloaded = self.task_store.require_task(record.id)
            if reloaded.execution.process_session_id and reloaded.execution.status in {TaskStatus.queued, TaskStatus.running}:
                launched_task_ids.append(record.id)
                continue
            process_payload = self._normalize_process_payload(record, launch_spec, payload)
            self.task_store.attach_process(record.id, **process_payload)
            launched_task_ids.append(record.id)

        return {
            "prepared_retry_task_ids": prepared_retry_task_ids,
            "launched_task_ids": launched_task_ids,
            "status": self.status(owner_session_id=owner_session_id),
        }


__all__ = ["AtlasTaskScheduler", "LaunchPayload", "Launcher"]
