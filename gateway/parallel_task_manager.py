"""
Parallel Task Manager for Hermes Gateway

Manages concurrent task execution, allowing multiple independent tasks
to run simultaneously while properly handling dependencies and sequencing.
"""

import asyncio
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from gateway.task_classifier import TaskClassification, TaskType, classify_task

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a parallel task."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ParallelTask:
    """Represents a task managed by the ParallelTaskManager."""
    task_id: str
    session_key: str
    message: str
    classification: TaskClassification
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0
    progress_message: Optional[str] = None
    # Callback for status updates
    status_callback: Optional[Callable[["ParallelTask"], None]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "session_key": self.session_key,
            "message": self.message[:100] + "..." if len(self.message) > 100 else self.message,
            "classification": {
                "type": self.classification.task_type.value,
                "confidence": self.classification.confidence,
                "reasoning": self.classification.reasoning,
            },
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "progress_message": self.progress_message,
            "error": self.error,
        }


@dataclass
class TaskQueue:
    """Queue for tasks waiting to be executed."""
    pending: List[ParallelTask] = field(default_factory=list)
    queued: List[ParallelTask] = field(default_factory=list)


class ParallelTaskManager:
    """
    Manages parallel task execution for Hermes gateway.
    
    Allows multiple independent tasks to run concurrently while ensuring
    sequential tasks execute in order and dependent tasks wait for
    their dependencies.
    
    Usage:
        manager = ParallelTaskManager(max_concurrent=3)
        
        # Submit a task
        task = await manager.submit_task(
            session_key="user_123",
            message="Generate an image of a cat",
            task_runner=my_agent_runner
        )
        
        # Check task status
        status = manager.get_task_status(task.task_id)
        
        # Get all tasks for a session
        tasks = manager.get_session_tasks("user_123")
        
        # Cancel a task
        await manager.cancel_task(task.task_id)
    """
    
    def __init__(self, max_concurrent: int = 3):
        """
        Initialize the parallel task manager.
        
        Args:
            max_concurrent: Maximum number of tasks to run simultaneously
        """
        self.max_concurrent = max_concurrent
        
        # Task storage
        self._tasks: Dict[str, ParallelTask] = {}
        self._session_tasks: Dict[str, Set[str]] = defaultdict(set)
        self._running_tasks: Dict[str, asyncio.Task] = {}
        
        # Queue for pending tasks
        self._queue: asyncio.Queue = asyncio.Queue()
        
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
        
        # Background worker task
        self._worker_task: Optional[asyncio.Task] = None
        
        # Configuration
        self._enabled = True
        
        logger.info(f"ParallelTaskManager initialized with max_concurrent={max_concurrent}")
    
    async def start(self):
        """Start the background task processor."""
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._process_queue())
            logger.info("ParallelTaskManager worker started")
    
    async def stop(self):
        """Stop the task manager and cancel all running tasks."""
        # Cancel worker
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        
        # Cancel all running tasks
        async with self._lock:
            for task_id, asyncio_task in list(self._running_tasks.items()):
                asyncio_task.cancel()
            
            # Wait for tasks to complete cancellation
            if self._running_tasks:
                await asyncio.gather(
                    *self._running_tasks.values(),
                    return_exceptions=True
                )
            
            self._running_tasks.clear()
        
        logger.info("ParallelTaskManager stopped")
    
    async def submit_task(
        self,
        session_key: str,
        message: str,
        task_runner: Callable[[ParallelTask], asyncio.Future],
        conversation_history: Optional[List[Dict]] = None,
    ) -> ParallelTask:
        """
        Submit a new task for execution.
        
        Args:
            session_key: Unique identifier for the user session
            message: The user's message/request
            task_runner: Async callable that executes the task
            conversation_history: Previous conversation messages
            
        Returns:
            The submitted ParallelTask
        """
        # Classify the task
        running_task_messages = [
            self._tasks[tid].message 
            for tid in self._session_tasks[session_key]
            if self._tasks[tid].status == TaskStatus.RUNNING
        ]
        
        classification = classify_task(
            message=message,
            conversation_history=conversation_history,
            running_tasks=running_task_messages
        )
        
        # Create task object
        task = ParallelTask(
            task_id=str(uuid.uuid4()),
            session_key=session_key,
            message=message,
            classification=classification,
            status=TaskStatus.PENDING,
        )
        
        async with self._lock:
            self._tasks[task.task_id] = task
            self._session_tasks[session_key].add(task.task_id)
        
        # Determine if task can run in parallel
        can_parallel = self._can_run_in_parallel(task)
        
        if not self._enabled or not can_parallel:
            # Queue for sequential execution
            task.status = TaskStatus.QUEUED
            await self._queue.put((task, task_runner, False))  # False = wait for running tasks
        else:
            # Queue for parallel execution
            await self._queue.put((task, task_runner, True))  # True = can run now
        
        logger.info(
            f"Task {task.task_id[:8]} submitted: {classification.task_type.value} "
            f"(parallel={can_parallel}, conf={classification.confidence:.2f})"
        )
        
        # Ensure worker is running
        if self._worker_task is None:
            await self.start()
        
        return task
    
    async def _process_queue(self):
        """Background worker that processes the task queue."""
        while True:
            try:
                task, task_runner, can_run_now = await self._queue.get()
                
                # Check if we should wait for other tasks
                if not can_run_now:
                    await self._wait_for_session_tasks(task.session_key)
                
                # Check if we're at max concurrency
                while len(self._running_tasks) >= self.max_concurrent:
                    await asyncio.sleep(0.1)
                
                # Start the task
                asyncio_task = asyncio.create_task(
                    self._execute_task(task, task_runner)
                )
                
                async with self._lock:
                    self._running_tasks[task.task_id] = asyncio_task
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in queue processor: {e}")
    
    async def _execute_task(
        self,
        task: ParallelTask,
        task_runner: Callable[[ParallelTask], asyncio.Future]
    ):
        """Execute a single task and handle its lifecycle."""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow()
        
        logger.info(f"Task {task.task_id[:8]} started")
        
        try:
            # Run the task
            result = await task_runner(task)
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            task.progress = 1.0
            
            logger.info(f"Task {task.task_id[:8]} completed")
            
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            logger.info(f"Task {task.task_id[:8]} cancelled")
            raise
            
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.utcnow()
            
            logger.error(f"Task {task.task_id[:8]} failed: {e}")
            
        finally:
            # Clean up
            async with self._lock:
                if task.task_id in self._running_tasks:
                    del self._running_tasks[task.task_id]
            
            # Notify status change
            if task.status_callback:
                try:
                    task.status_callback(task)
                except Exception as e:
                    logger.error(f"Status callback error: {e}")
    
    async def _wait_for_session_tasks(self, session_key: str):
        """Wait for all running tasks in a session to complete."""
        while True:
            running = False
            async with self._lock:
                for task_id in self._session_tasks[session_key]:
                    if task_id in self._running_tasks:
                        running = True
                        break
            
            if not running:
                break
            
            await asyncio.sleep(0.1)
    
    def _can_run_in_parallel(self, new_task: ParallelTask) -> bool:
        """Check if a new task can run in parallel with existing tasks."""
        session_key = new_task.session_key
        
        # Get classifications of running tasks in the same session
        running_classifications = []
        for task_id in self._session_tasks[session_key]:
            task = self._tasks.get(task_id)
            if task and task.status == TaskStatus.RUNNING:
                running_classifications.append(task.classification)
        
        # Use the classifier to determine if parallel execution is safe
        from gateway.task_classifier import get_classifier
        return get_classifier().can_run_in_parallel(
            new_task.classification,
            running_classifications
        )
    
    def get_task(self, task_id: str) -> Optional[ParallelTask]:
        """Get a task by ID."""
        return self._tasks.get(task_id)
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the status of a task."""
        task = self._tasks.get(task_id)
        return task.status if task else None
    
    def get_session_tasks(
        self,
        session_key: str,
        status_filter: Optional[Set[TaskStatus]] = None
    ) -> List[ParallelTask]:
        """
        Get all tasks for a session.
        
        Args:
            session_key: The session identifier
            status_filter: Optional set of statuses to filter by
            
        Returns:
            List of matching tasks
        """
        tasks = []
        for task_id in self._session_tasks.get(session_key, set()):
            task = self._tasks.get(task_id)
            if task:
                if status_filter is None or task.status in status_filter:
                    tasks.append(task)
        
        # Sort by creation time
        tasks.sort(key=lambda t: t.created_at)
        return tasks
    
    def get_running_tasks(self, session_key: Optional[str] = None) -> List[ParallelTask]:
        """Get currently running tasks, optionally filtered by session."""
        tasks = []
        for task_id, asyncio_task in self._running_tasks.items():
            task = self._tasks.get(task_id)
            if task:
                if session_key is None or task.session_key == session_key:
                    tasks.append(task)
        return tasks
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running or queued task.
        
        Args:
            task_id: The task to cancel
            
        Returns:
            True if the task was cancelled
        """
        task = self._tasks.get(task_id)
        if not task:
            return False
        
        if task.status == TaskStatus.RUNNING:
            # Cancel the asyncio task
            asyncio_task = self._running_tasks.get(task_id)
            if asyncio_task:
                asyncio_task.cancel()
                return True
        
        elif task.status in (TaskStatus.PENDING, TaskStatus.QUEUED):
            # Mark as cancelled (will be handled by queue processor)
            task.status = TaskStatus.CANCELLED
            return True
        
        return False
    
    async def cancel_session_tasks(self, session_key: str) -> int:
        """
        Cancel all tasks for a session.
        
        Returns:
            Number of tasks cancelled
        """
        cancelled = 0
        for task_id in list(self._session_tasks.get(session_key, set())):
            if await self.cancel_task(task_id):
                cancelled += 1
        return cancelled
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the task manager."""
        stats = {
            "total_tasks": len(self._tasks),
            "running_tasks": len(self._running_tasks),
            "max_concurrent": self.max_concurrent,
            "enabled": self._enabled,
            "by_status": defaultdict(int),
            "by_session": {},
        }
        
        for task in self._tasks.values():
            stats["by_status"][task.status.value] += 1
        
        for session_key, task_ids in self._session_tasks.items():
            stats["by_session"][session_key] = len(task_ids)
        
        return dict(stats)
    
    def set_enabled(self, enabled: bool):
        """Enable or disable parallel task execution."""
        self._enabled = enabled
        logger.info(f"ParallelTaskManager enabled={enabled}")


# Global manager instance
_task_manager: Optional[ParallelTaskManager] = None


async def get_task_manager(max_concurrent: int = 3) -> ParallelTaskManager:
    """Get or create the global task manager instance."""
    global _task_manager
    if _task_manager is None:
        _task_manager = ParallelTaskManager(max_concurrent=max_concurrent)
        await _task_manager.start()
    return _task_manager


def get_task_manager_sync(max_concurrent: int = 3) -> ParallelTaskManager:
    """Synchronous version to get the task manager (requires prior initialization)."""
    global _task_manager
    if _task_manager is None:
        raise RuntimeError("Task manager not initialized. Call get_task_manager() first.")
    return _task_manager
