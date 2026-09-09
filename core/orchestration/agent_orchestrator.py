"""
Multi-Agent Orchestration System.

Coordinates specialized subagents (Research, Coding, Browser, Testing, Documentation, Reviewer)
over task dependency graphs with sequential and parallel execution capabilities.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import asyncio
import uuid
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class SubAgentType(str, Enum):
    MAIN = "main"
    RESEARCH = "research"
    CODING = "coding"
    BROWSER = "browser"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    REVIEWER = "reviewer"


class TaskStatus(str, Enum):
    QUEUED = "QUEUED"
    PLANNING = "PLANNING"
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    PAUSED = "PAUSED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"


@dataclass
class AgentTask:
    task_id: str
    session_id: str
    title: str
    description: str
    agent_type: SubAgentType = SubAgentType.MAIN
    priority: int = 1
    status: TaskStatus = TaskStatus.QUEUED
    dependencies: List[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    logs: List[str] = field(default_factory=list)


@dataclass
class SubAgent:
    agent_id: str
    name: str
    agent_type: SubAgentType
    description: str
    capabilities: List[str] = field(default_factory=list)


class AgentOrchestrator:
    """Orchestrates multi-agent tasks, dependency resolution, and parallel execution."""

    def __init__(self) -> None:
        self.subagents: Dict[SubAgentType, SubAgent] = {
            SubAgentType.MAIN: SubAgent(
                "main", "Main Agent", SubAgentType.MAIN, "Primary coordinator"
            ),
            SubAgentType.RESEARCH: SubAgent(
                "research",
                "Research Agent",
                SubAgentType.RESEARCH,
                "Information lookup and synthesis",
            ),
            SubAgentType.CODING: SubAgent(
                "coding",
                "Coding Agent",
                SubAgentType.CODING,
                "Code generation, refactoring, bug fixes",
            ),
            SubAgentType.BROWSER: SubAgent(
                "browser",
                "Browser Agent",
                SubAgentType.BROWSER,
                "Web browsing and page extraction",
            ),
            SubAgentType.TESTING: SubAgent(
                "testing",
                "Testing Agent",
                SubAgentType.TESTING,
                "Running and verifying test suites",
            ),
            SubAgentType.DOCUMENTATION: SubAgent(
                "documentation",
                "Documentation Agent",
                SubAgentType.DOCUMENTATION,
                "Writing docs and summaries",
            ),
            SubAgentType.REVIEWER: SubAgent(
                "reviewer",
                "Reviewer Agent",
                SubAgentType.REVIEWER,
                "Code review and safety validation",
            ),
        }
        self.tasks: Dict[str, AgentTask] = {}

    def create_task(
        self,
        session_id: str,
        title: str,
        description: str,
        agent_type: SubAgentType = SubAgentType.MAIN,
        dependencies: Optional[List[str]] = None,
    ) -> AgentTask:
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        task = AgentTask(
            task_id=task_id,
            session_id=session_id,
            title=title,
            description=description,
            agent_type=agent_type,
            dependencies=dependencies or [],
        )
        self.tasks[task_id] = task
        return task

    def get_task(self, task_id: str) -> Optional[AgentTask]:
        return self.tasks.get(task_id)

    def list_tasks(self, session_id: Optional[str] = None) -> List[AgentTask]:
        if session_id:
            return [t for t in self.tasks.values() if t.session_id == session_id]
        return list(self.tasks.values())

    async def execute_task_graph(
        self,
        task_ids: List[str],
        executor: Callable[[AgentTask], Any],
    ) -> Dict[str, AgentTask]:
        """Executes a dependency graph of tasks sequentially or in parallel."""
        completed: Dict[str, AgentTask] = {}

        while len(completed) < len(task_ids):
            runnable = [
                t_id
                for t_id in task_ids
                if t_id not in completed
                and all(dep in completed for dep in self.tasks[t_id].dependencies)
            ]

            if not runnable:
                # Unresolvable dependency loop or missing tasks
                break

            async def _run_one(t_id: str) -> None:
                task = self.tasks[t_id]
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now(timezone.utc).isoformat()
                try:
                    if asyncio.iscoroutinefunction(executor):
                        res = await executor(task)
                    else:
                        res = executor(task)
                    task.result = res
                    task.status = TaskStatus.COMPLETED
                except Exception as e:
                    task.error = str(e)
                    task.status = TaskStatus.FAILED
                finally:
                    task.completed_at = datetime.now(timezone.utc).isoformat()
                    completed[t_id] = task

            await asyncio.gather(*[_run_one(t_id) for t_id in runnable])

        return completed
