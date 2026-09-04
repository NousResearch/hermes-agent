"""Durable, event-driven coordinator for one autonomous engineering outcome."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import threading
import uuid
from collections.abc import Callable, Mapping
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from hermes_constants import get_hermes_home
from pydantic import BaseModel, ConfigDict, Field


class AutonomousTaskStatus(str, Enum):
    QUEUED = "QUEUED"
    DEVELOPING = "DEVELOPING"
    TESTING = "TESTING"
    REPAIRING = "REPAIRING"
    ACCEPTANCE_PRE_DEPLOY = "ACCEPTANCE_PRE_DEPLOY"
    DELIVERING = "DELIVERING"
    DEPLOYING = "DEPLOYING"
    ACCEPTANCE_RUNTIME = "ACCEPTANCE_RUNTIME"
    DONE = "DONE"
    BLOCKED = "BLOCKED"


OWNER_STOP_CONDITIONS = frozenset({
    "PRODUCT_DECISION_REQUIRED",
    "ENVIRONMENT_REQUIRED",
    "HIGH_RISK_PRODUCTION_APPROVAL_REQUIRED",
    "BLOCKED_BY_ARCHITECTURE",
    "BLOCKED_BY_REPOSITORY_LINEAGE",
})


class AutonomousTaskState(BaseModel):
    """The complete durable checkpoint needed to resume a coordinator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    project: str = Field(min_length=1)
    repository: str = Field(min_length=1)
    target_branch: str = Field(min_length=1)
    state: AutonomousTaskStatus = AutonomousTaskStatus.QUEUED
    current_agent: Literal["developer", "tester", "acceptance"] | None = None
    developer_commit: str | None = None
    tester_verdict: str | None = None
    acceptance_verdict: str | None = None
    pr_number: int | None = None
    ci_status: str | None = None
    deployment_status: str | None = None
    runtime_acceptance: str | None = None
    repair_loops: int = 0
    acceptance_loops: int = 0
    last_error: str | None = None
    next_action: str = "Dispatch Developer"
    operation_id: str | None = None
    repository_remote: str | None = None
    deployment_retries: int = 0
    revision: int = 0

    def render(self) -> str:
        owner_action = "YES" if self.state == AutonomousTaskStatus.BLOCKED else "NO"
        return "\n".join((
            f"TASK: {self.task_id}",
            f"STATE: {self.state.value}",
            f"REPAIR_LOOPS: {self.repair_loops}",
            f"NEXT_ACTION: {self.next_action}",
            f"OWNER_ACTION_REQUIRED: {owner_action}",
        ))


class CoordinatorEvent(BaseModel):
    """A completion or progress event from an agent, CI, or deployment watcher."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_id: str
    status: Literal["PASS", "FAIL", "RUNNING", "BLOCKED"]
    commit: str | None = None
    verdict: str | None = None
    pr_number: int | None = None
    ci_status: str | None = None
    deployment_status: str | None = None
    error: str | None = None
    blocker_code: str | None = None


class ConcurrentCoordinatorUpdate(RuntimeError):
    pass


class AutonomousTaskStateStore:
    """Small SQLite store with compare-and-swap updates for process-safe recovery."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or get_hermes_home() / "state" / "autonomous-tasks.db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5)
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _initialize(self) -> None:
        with self._init_lock, self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS autonomous_tasks ("
                "task_id TEXT PRIMARY KEY, revision INTEGER NOT NULL, body TEXT NOT NULL)"
            )

    def create(self, state: AutonomousTaskState) -> AutonomousTaskState:
        body = state.model_dump_json()
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO autonomous_tasks(task_id, revision, body) VALUES (?, ?, ?)",
                    (state.task_id, state.revision, body),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f"autonomous task already exists: {state.task_id}"
            ) from exc
        return state

    def load(self, task_id: str) -> AutonomousTaskState:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT body FROM autonomous_tasks WHERE task_id=?", (task_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown autonomous task: {task_id}")
        return AutonomousTaskState.model_validate_json(row[0])

    def save(self, state: AutonomousTaskState) -> AutonomousTaskState:
        updated = state.model_copy(update={"revision": state.revision + 1})
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE autonomous_tasks SET revision=?, body=? "
                "WHERE task_id=? AND revision=?",
                (
                    updated.revision,
                    updated.model_dump_json(),
                    updated.task_id,
                    state.revision,
                ),
            )
        if cursor.rowcount != 1:
            raise ConcurrentCoordinatorUpdate(state.task_id)
        return updated

    def active(self) -> tuple[AutonomousTaskState, ...]:
        with self._connect() as conn:
            rows = conn.execute("SELECT body FROM autonomous_tasks").fetchall()
        states = (AutonomousTaskState.model_validate_json(row[0]) for row in rows)
        return tuple(
            state
            for state in states
            if state.state
            not in {AutonomousTaskStatus.DONE, AutonomousTaskStatus.BLOCKED}
        )


CoordinatorAction = Callable[[AutonomousTaskState, str], CoordinatorEvent | None]


_AGENT_BY_STATE = {
    AutonomousTaskStatus.DEVELOPING: "developer",
    AutonomousTaskStatus.TESTING: "tester",
    AutonomousTaskStatus.REPAIRING: "developer",
    AutonomousTaskStatus.ACCEPTANCE_PRE_DEPLOY: "acceptance",
    AutonomousTaskStatus.ACCEPTANCE_RUNTIME: "acceptance",
}

_NEXT_ACTION = {
    AutonomousTaskStatus.QUEUED: "Dispatch Developer",
    AutonomousTaskStatus.DEVELOPING: "Wait for Developer completion",
    AutonomousTaskStatus.TESTING: "Dispatch Independent Tester",
    AutonomousTaskStatus.REPAIRING: "Dispatch Developer Repair",
    AutonomousTaskStatus.ACCEPTANCE_PRE_DEPLOY: "Dispatch Acceptance",
    AutonomousTaskStatus.DELIVERING: "Repository lock and Git/PR/CI delivery",
    AutonomousTaskStatus.DEPLOYING: "Monitor Dev deployment",
    AutonomousTaskStatus.ACCEPTANCE_RUNTIME: "Dispatch Runtime Acceptance",
    AutonomousTaskStatus.DONE: "DELIVERED_AND_VERIFIED",
    AutonomousTaskStatus.BLOCKED: "Owner resolution required",
}


class VesselMindRepositoryLock:
    """Resolve the approved VesselMind GitHub remote without rewriting user remotes."""

    expected = "github.com/jasoncheungcn/ceramic-ai-designer-h5"

    @staticmethod
    def _canonical(url: str) -> str:
        value = (
            url.strip().removesuffix(".git").replace("git@github.com:", "github.com/")
        )
        for prefix in ("https://", "http://", "ssh://git@"):
            if value.startswith(prefix):
                value = value[len(prefix) :]
        return value.rstrip("/").lower()

    def verify(self, repository: str | Path) -> str:
        repo = Path(repository).resolve()
        names = subprocess.run(
            ["git", "-C", str(repo), "remote"],
            capture_output=True,
            text=True,
            check=False,
        )
        if names.returncode:
            raise RuntimeError(
                "BLOCKED_BY_REPOSITORY_LINEAGE: repository is not readable"
            )
        for name in names.stdout.splitlines():
            urls = subprocess.run(
                ["git", "-C", str(repo), "remote", "get-url", "--all", name],
                capture_output=True,
                text=True,
                check=False,
            )
            if urls.returncode == 0 and any(
                self._canonical(url) == self.expected
                for url in urls.stdout.splitlines()
            ):
                return name
        raise RuntimeError(
            "BLOCKED_BY_REPOSITORY_LINEAGE: approved VesselMind remote is unavailable"
        )


class CoordinatorRunner:
    """Advance a persisted task until an external event is required or it is terminal."""

    def __init__(
        self,
        store: AutonomousTaskStateStore,
        actions: Mapping[AutonomousTaskStatus, CoordinatorAction],
        *,
        repository_lock: VesselMindRepositoryLock | None = None,
        max_repair_loops: int = 5,
        max_deployment_retries: int = 3,
    ) -> None:
        self.store = store
        self.actions = actions
        self.repository_lock = repository_lock or VesselMindRepositoryLock()
        self.max_repair_loops = max_repair_loops
        self.max_deployment_retries = max_deployment_retries

    def submit(
        self, *, task_id: str, project: str, repository: str, target_branch: str
    ) -> AutonomousTaskState:
        state = self.store.create(
            AutonomousTaskState(
                task_id=task_id,
                project=project,
                repository=repository,
                target_branch=target_branch,
            )
        )
        return self.run(state.task_id)

    def resume_all(self) -> tuple[AutonomousTaskState, ...]:
        return tuple(self.run(state.task_id) for state in self.store.active())

    def run(self, task_id: str) -> AutonomousTaskState:
        """Run immediate transitions; return silently when a watcher must wake us."""

        for _ in range(64):
            try:
                state = self.store.load(task_id)
                if state.state in {
                    AutonomousTaskStatus.DONE,
                    AutonomousTaskStatus.BLOCKED,
                }:
                    return state
                if state.state == AutonomousTaskStatus.QUEUED:
                    self._transition(state, AutonomousTaskStatus.DEVELOPING)
                    continue
                if state.repair_loops >= self.max_repair_loops:
                    return self._block(
                        state, "BLOCKED_BY_ARCHITECTURE", "repair loop limit reached"
                    )

                action = self.actions.get(state.state)
                if action is None:
                    return self._block(
                        state,
                        "BLOCKED_BY_ARCHITECTURE",
                        f"no coordinator action for {state.state.value}",
                    )
                if state.state in {
                    AutonomousTaskStatus.DELIVERING,
                    AutonomousTaskStatus.DEPLOYING,
                }:
                    locked = self._lock_repository(state)
                    if locked.state == AutonomousTaskStatus.BLOCKED:
                        return locked
                    state = locked
                if state.operation_id is None:
                    state = self.store.save(
                        state.model_copy(
                            update={
                                "operation_id": uuid.uuid4().hex,
                                "current_agent": _AGENT_BY_STATE.get(state.state),
                                "next_action": _NEXT_ACTION[state.state],
                                "last_error": None,
                            }
                        )
                    )
                try:
                    event = action(state, state.operation_id)
                # Action failures are ordinary repair/retry inputs, not owner stops.
                except Exception as exc:
                    event = CoordinatorEvent(
                        operation_id=state.operation_id,
                        status="FAIL",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                if event is None:
                    return state
                self._apply_event(state, event)
            except ConcurrentCoordinatorUpdate:
                continue
        state = self.store.load(task_id)
        return self._block(
            state, "BLOCKED_BY_ARCHITECTURE", "coordinator step limit reached"
        )

    def handle_event(
        self, task_id: str, event: CoordinatorEvent | Mapping[str, Any]
    ) -> AutonomousTaskState:
        """Persist one callback/watcher event and immediately re-enter the runner."""

        parsed = (
            event
            if isinstance(event, CoordinatorEvent)
            else CoordinatorEvent.model_validate(event)
        )
        for _ in range(2):
            state = self.store.load(task_id)
            if state.state in {AutonomousTaskStatus.DONE, AutonomousTaskStatus.BLOCKED}:
                return state
            if parsed.operation_id != state.operation_id:
                return state  # stale or duplicate completion from an earlier attempt
            try:
                updated = self._apply_event(state, parsed)
            except ConcurrentCoordinatorUpdate:
                continue
            if parsed.status == "RUNNING":
                return updated
            return self.run(task_id)
        return self.store.load(task_id)

    def completion_callback(self, task_id: str) -> Callable[[Mapping[str, Any]], None]:
        """Return the direct continuation used by async delegation and watchers."""

        def resume(result: Mapping[str, Any]) -> None:
            event = result.get("coordinator_event")
            if event is not None:
                self.handle_event(task_id, event)

        return resume

    def _apply_event(
        self, state: AutonomousTaskState, event: CoordinatorEvent
    ) -> AutonomousTaskState:
        updates: dict[str, Any] = {"last_error": event.error}
        if event.pr_number is not None:
            updates["pr_number"] = event.pr_number
        if event.ci_status is not None:
            updates["ci_status"] = event.ci_status
        if event.deployment_status is not None:
            updates["deployment_status"] = event.deployment_status
        if event.status == "RUNNING":
            return self.store.save(state.model_copy(update=updates))
        if event.status == "BLOCKED":
            code = event.blocker_code or "BLOCKED_BY_ARCHITECTURE"
            if code not in OWNER_STOP_CONDITIONS:
                code = "BLOCKED_BY_ARCHITECTURE"
            return self._block(
                state.model_copy(update=updates), code, event.error or code
            )

        current = state.state
        if current in {AutonomousTaskStatus.DEVELOPING, AutonomousTaskStatus.REPAIRING}:
            if event.status == "PASS":
                updates.update(developer_commit=event.commit or state.developer_commit)
                target = AutonomousTaskStatus.TESTING
            else:
                target = AutonomousTaskStatus.REPAIRING
                updates["repair_loops"] = state.repair_loops + 1
        elif current == AutonomousTaskStatus.TESTING:
            updates["tester_verdict"] = event.verdict or event.status
            target = (
                AutonomousTaskStatus.ACCEPTANCE_PRE_DEPLOY
                if event.status == "PASS"
                else AutonomousTaskStatus.REPAIRING
            )
            if event.status == "FAIL":
                updates["repair_loops"] = state.repair_loops + 1
        elif current == AutonomousTaskStatus.ACCEPTANCE_PRE_DEPLOY:
            updates["acceptance_verdict"] = event.verdict or event.status
            target = (
                AutonomousTaskStatus.DELIVERING
                if event.status == "PASS"
                else AutonomousTaskStatus.REPAIRING
            )
            if event.status == "FAIL":
                updates["acceptance_loops"] = state.acceptance_loops + 1
                updates["repair_loops"] = state.repair_loops + 1
        elif current == AutonomousTaskStatus.DELIVERING:
            target = (
                AutonomousTaskStatus.DEPLOYING
                if event.status == "PASS"
                else AutonomousTaskStatus.REPAIRING
            )
            if event.status == "FAIL":
                updates["repair_loops"] = state.repair_loops + 1
        elif current == AutonomousTaskStatus.DEPLOYING:
            if event.status == "PASS":
                target = AutonomousTaskStatus.ACCEPTANCE_RUNTIME
            else:
                retries = state.deployment_retries + 1
                if retries > self.max_deployment_retries:
                    return self._block(
                        state.model_copy(update=updates),
                        "ENVIRONMENT_REQUIRED",
                        event.error or "Dev deployment retry limit reached",
                    )
                target = AutonomousTaskStatus.DEPLOYING
                updates["deployment_retries"] = retries
        elif current == AutonomousTaskStatus.ACCEPTANCE_RUNTIME:
            updates["runtime_acceptance"] = event.verdict or event.status
            if event.status == "PASS":
                target = AutonomousTaskStatus.DONE
            else:
                target = AutonomousTaskStatus.REPAIRING
                updates["acceptance_loops"] = state.acceptance_loops + 1
                updates["repair_loops"] = state.repair_loops + 1
        else:  # QUEUED and terminal states never own an operation
            return state
        updates.update(
            state=target,
            operation_id=None,
            current_agent=None,
            next_action=_NEXT_ACTION[target],
        )
        return self.store.save(state.model_copy(update=updates))

    def _transition(
        self, state: AutonomousTaskState, target: AutonomousTaskStatus
    ) -> AutonomousTaskState:
        return self.store.save(
            state.model_copy(
                update={
                    "state": target,
                    "operation_id": None,
                    "current_agent": None,
                    "next_action": _NEXT_ACTION[target],
                }
            )
        )

    def _lock_repository(self, state: AutonomousTaskState) -> AutonomousTaskState:
        if (
            "vesselmind" not in state.project.lower()
            and "ceramic-ai-designer-h5" not in state.repository
        ):
            return state
        try:
            remote = self.repository_lock.verify(state.repository)
        except RuntimeError as exc:
            return self._block(state, "BLOCKED_BY_REPOSITORY_LINEAGE", str(exc))
        if remote == state.repository_remote:
            return state
        return self.store.save(state.model_copy(update={"repository_remote": remote}))

    def _block(
        self, state: AutonomousTaskState, code: str, error: str
    ) -> AutonomousTaskState:
        latest = self.store.load(state.task_id)
        if latest.revision != state.revision:
            return latest
        return self.store.save(
            state.model_copy(
                update={
                    "state": AutonomousTaskStatus.BLOCKED,
                    "current_agent": None,
                    "operation_id": None,
                    "last_error": f"{code}: {error}",
                    "next_action": _NEXT_ACTION[AutonomousTaskStatus.BLOCKED],
                }
            )
        )
