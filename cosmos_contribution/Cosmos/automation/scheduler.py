"""
cosmos Task Scheduler

"I've scheduled everything! Even my morning coffee is on autopilot!"

Advanced task scheduling with cron, interval, and event-based triggers.
"""

import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
from pathlib import Path
from loguru import logger

try:
    from croniter import croniter
    HAS_CRONITER = True
except ImportError:
    HAS_CRONITER = False


class ScheduleType(Enum):
    """Types of schedules."""
    CRON = "cron"
    INTERVAL = "interval"
    ONCE = "once"
    EVENT = "event"


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class ScheduledTask:
    """A scheduled task definition."""
    id: str
    name: str
    schedule_type: ScheduleType
    schedule_config: Dict[str, Any]
    callback: Optional[Callable] = None
    callback_name: str = ""  # For serialization
    callback_args: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    max_retries: int = 3
    retry_delay: int = 60  # seconds
    timeout: int = 300  # seconds
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "schedule_type": self.schedule_type.value,
            "schedule_config": self.schedule_config,
            "callback_name": self.callback_name,
            "callback_args": self.callback_args,
            "enabled": self.enabled,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "run_count": self.run_count,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "timeout": self.timeout,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScheduledTask":
        return cls(
            id=data["id"],
            name=data["name"],
            schedule_type=ScheduleType(data["schedule_type"]),
            schedule_config=data["schedule_config"],
            callback_name=data.get("callback_name", ""),
            callback_args=data.get("callback_args", {}),
            enabled=data.get("enabled", True),
            last_run=datetime.fromisoformat(data["last_run"]) if data.get("last_run") else None,
            next_run=datetime.fromisoformat(data["next_run"]) if data.get("next_run") else None,
            run_count=data.get("run_count", 0),
            error_count=data.get("error_count", 0),
            last_error=data.get("last_error"),
            max_retries=data.get("max_retries", 3),
            retry_delay=data.get("retry_delay", 60),
            timeout=data.get("timeout", 300),
            tags=data.get("tags", []),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
        )


@dataclass
class TaskExecution:
    """Record of a task execution."""
    id: str
    task_id: str
    status: TaskStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    attempt: int = 1


class TaskScheduler:
    """
    Advanced task scheduler for cosmos.

    Features:
    - Cron expressions
    - Interval scheduling
    - One-time execution
    - Event-driven triggers
    - Retry with backoff
    - Persistence
    """

    def __init__(self, storage_path: Path = None):
        self.storage_path = storage_path or Path("./data/scheduler")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.tasks: Dict[str, ScheduledTask] = {}
        self.executions: Dict[str, TaskExecution] = {}
        self.callbacks: Dict[str, Callable] = {}

        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        self._load_tasks()
        self._register_default_callbacks()

    def _load_tasks(self):
        """Load tasks from storage."""
        tasks_file = self.storage_path / "tasks.json"
        if tasks_file.exists():
            try:
                with open(tasks_file) as f:
                    data = json.load(f)
                    for task_data in data.get("tasks", []):
                        task = ScheduledTask.from_dict(task_data)
                        self.tasks[task.id] = task
                logger.info(f"Loaded {len(self.tasks)} scheduled tasks")
            except Exception as e:
                logger.error(f"Failed to load tasks: {e}")

    def _save_tasks(self):
        """Save tasks to storage."""
        tasks_file = self.storage_path / "tasks.json"
        try:
            with open(tasks_file, "w") as f:
                json.dump({
                    "tasks": [t.to_dict() for t in self.tasks.values()],
                    "updated_at": datetime.utcnow().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save tasks: {e}")

    def _register_default_callbacks(self):
        """Register default callback functions."""
        # These can be extended by the application

        async def log_callback(**kwargs):
            logger.info(f"Scheduled log: {kwargs}")
            return {"logged": True}

        async def http_callback(**kwargs):
            import aiohttp
            url = kwargs.get("url")
            method = kwargs.get("method", "GET")
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url) as response:
                    return {"status": response.status}

        async def shell_callback(**kwargs):
            import subprocess
            command = kwargs.get("command")
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            return {"stdout": result.stdout, "stderr": result.stderr, "code": result.returncode}

        self.register_callback("log", log_callback)
        self.register_callback("http", http_callback)
        self.register_callback("shell", shell_callback)

    # =========================================================================
    # CALLBACK MANAGEMENT
    # =========================================================================

    def register_callback(
        self,
        name: str,
        callback: Callable[..., Coroutine],
    ):
        """Register a callback function."""
        self.callbacks[name] = callback
        logger.debug(f"Registered scheduler callback: {name}")

    def unregister_callback(self, name: str):
        """Unregister a callback function."""
        if name in self.callbacks:
            del self.callbacks[name]

    # =========================================================================
    # TASK MANAGEMENT
    # =========================================================================

    def create_task(
        self,
        name: str,
        schedule_type: ScheduleType,
        schedule_config: Dict[str, Any],
        callback_name: str,
        callback_args: Dict[str, Any] = None,
        **kwargs,
    ) -> ScheduledTask:
        """Create a new scheduled task."""
        task = ScheduledTask(
            id=str(uuid.uuid4()),
            name=name,
            schedule_type=schedule_type,
            schedule_config=schedule_config,
            callback_name=callback_name,
            callback_args=callback_args or {},
            **kwargs,
        )

        # Calculate next run
        task.next_run = self._calculate_next_run(task)

        self.tasks[task.id] = task
        self._save_tasks()

        logger.info(f"Created scheduled task: {name} (next run: {task.next_run})")
        return task

    def create_cron_task(
        self,
        name: str,
        cron_expression: str,
        callback_name: str,
        callback_args: Dict[str, Any] = None,
        **kwargs,
    ) -> ScheduledTask:
        """Create a cron-based task."""
        return self.create_task(
            name=name,
            schedule_type=ScheduleType.CRON,
            schedule_config={"cron": cron_expression},
            callback_name=callback_name,
            callback_args=callback_args,
            **kwargs,
        )

    def create_interval_task(
        self,
        name: str,
        interval_seconds: int,
        callback_name: str,
        callback_args: Dict[str, Any] = None,
        **kwargs,
    ) -> ScheduledTask:
        """Create an interval-based task."""
        return self.create_task(
            name=name,
            schedule_type=ScheduleType.INTERVAL,
            schedule_config={"interval": interval_seconds},
            callback_name=callback_name,
            callback_args=callback_args,
            **kwargs,
        )

    def create_once_task(
        self,
        name: str,
        run_at: datetime,
        callback_name: str,
        callback_args: Dict[str, Any] = None,
        **kwargs,
    ) -> ScheduledTask:
        """Create a one-time task."""
        return self.create_task(
            name=name,
            schedule_type=ScheduleType.ONCE,
            schedule_config={"run_at": run_at.isoformat()},
            callback_name=callback_name,
            callback_args=callback_args,
            **kwargs,
        )

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get a task by ID."""
        return self.tasks.get(task_id)

    def list_tasks(
        self,
        enabled_only: bool = False,
        tags: List[str] = None,
    ) -> List[ScheduledTask]:
        """List all tasks."""
        tasks = list(self.tasks.values())

        if enabled_only:
            tasks = [t for t in tasks if t.enabled]

        if tags:
            tasks = [t for t in tasks if any(tag in t.tags for tag in tags)]

        return sorted(tasks, key=lambda t: t.next_run or datetime.max)

    def update_task(
        self,
        task_id: str,
        **updates,
    ) -> Optional[ScheduledTask]:
        """Update a task."""
        task = self.tasks.get(task_id)
        if not task:
            return None

        for key, value in updates.items():
            if hasattr(task, key):
                setattr(task, key, value)

        # Recalculate next run if schedule changed
        if "schedule_config" in updates or "schedule_type" in updates:
            task.next_run = self._calculate_next_run(task)

        self._save_tasks()
        return task

    def delete_task(self, task_id: str) -> bool:
        """Delete a task."""
        if task_id in self.tasks:
            del self.tasks[task_id]
            self._save_tasks()
            logger.info(f"Deleted scheduled task: {task_id}")
            return True
        return False

    def enable_task(self, task_id: str) -> bool:
        """Enable a task."""
        task = self.tasks.get(task_id)
        if task:
            task.enabled = True
            task.next_run = self._calculate_next_run(task)
            self._save_tasks()
            return True
        return False

    def disable_task(self, task_id: str) -> bool:
        """Disable a task."""
        task = self.tasks.get(task_id)
        if task:
            task.enabled = False
            self._save_tasks()
            return True
        return False

    # =========================================================================
    # SCHEDULING
    # =========================================================================

    def _calculate_next_run(self, task: ScheduledTask) -> Optional[datetime]:
        """Calculate the next run time for a task."""
        now = datetime.utcnow()

        if task.schedule_type == ScheduleType.CRON:
            if not HAS_CRONITER:
                logger.warning("croniter not installed, cron scheduling unavailable")
                return None
            cron = task.schedule_config.get("cron", "0 * * * *")
            iter = croniter(cron, now)
            return iter.get_next(datetime)

        elif task.schedule_type == ScheduleType.INTERVAL:
            interval = task.schedule_config.get("interval", 3600)
            base = task.last_run or now
            return base + timedelta(seconds=interval)

        elif task.schedule_type == ScheduleType.ONCE:
            run_at = task.schedule_config.get("run_at")
            if run_at:
                run_at = datetime.fromisoformat(run_at)
                return run_at if run_at > now else None
            return None

        return None

    # =========================================================================
    # EXECUTION
    # =========================================================================

    async def start(self):
        """Start the scheduler."""
        if self._running:
            return

        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Task scheduler started")

    async def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("Task scheduler stopped")

    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                now = datetime.utcnow()

                # Find tasks ready to run
                ready_tasks = [
                    task for task in self.tasks.values()
                    if task.enabled and task.next_run and task.next_run <= now
                ]

                # Execute ready tasks
                for task in ready_tasks:
                    asyncio.create_task(self._execute_task(task))

                # Sleep until next check
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(5)

    async def _execute_task(self, task: ScheduledTask):
        """Execute a scheduled task."""
        execution = TaskExecution(
            id=str(uuid.uuid4()),
            task_id=task.id,
            status=TaskStatus.RUNNING,
            started_at=datetime.utcnow(),
        )
        self.executions[execution.id] = execution

        logger.info(f"Executing scheduled task: {task.name}")

        try:
            # Get callback
            callback = self.callbacks.get(task.callback_name)
            if not callback:
                raise ValueError(f"Callback not found: {task.callback_name}")

            # Execute with timeout
            result = await asyncio.wait_for(
                callback(**task.callback_args),
                timeout=task.timeout
            )

            execution.status = TaskStatus.COMPLETED
            execution.result = result
            task.run_count += 1
            task.last_error = None

            logger.info(f"Task completed: {task.name}")

        except asyncio.TimeoutError:
            execution.status = TaskStatus.FAILED
            execution.error = "Task timed out"
            task.error_count += 1
            task.last_error = "Timeout"
            logger.error(f"Task timed out: {task.name}")

        except Exception as e:
            execution.status = TaskStatus.FAILED
            execution.error = str(e)
            task.error_count += 1
            task.last_error = str(e)
            logger.error(f"Task failed: {task.name} - {e}")

        finally:
            execution.completed_at = datetime.utcnow()
            task.last_run = datetime.utcnow()

            # Calculate next run (unless it was a one-time task)
            if task.schedule_type != ScheduleType.ONCE:
                task.next_run = self._calculate_next_run(task)
            else:
                task.enabled = False

            self._save_tasks()

    async def run_task_now(self, task_id: str) -> Optional[TaskExecution]:
        """Run a task immediately."""
        task = self.tasks.get(task_id)
        if not task:
            return None

        await self._execute_task(task)
        return list(self.executions.values())[-1]

    # =========================================================================
    # EXECUTION HISTORY
    # =========================================================================

    def get_executions(
        self,
        task_id: str = None,
        status: TaskStatus = None,
        limit: int = 100,
    ) -> List[TaskExecution]:
        """Get execution history."""
        executions = list(self.executions.values())

        if task_id:
            executions = [e for e in executions if e.task_id == task_id]

        if status:
            executions = [e for e in executions if e.status == status]

        return sorted(executions, key=lambda e: e.started_at, reverse=True)[:limit]

    def clear_execution_history(self, older_than_days: int = 30):
        """Clear old execution history."""
        cutoff = datetime.utcnow() - timedelta(days=older_than_days)
        self.executions = {
            eid: e for eid, e in self.executions.items()
            if e.started_at > cutoff
        }


# Singleton instance
task_scheduler = TaskScheduler()
