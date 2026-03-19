"""
cosmos Planner Agent - Task Decomposition & Orchestration

Novel Approaches:
1. Hierarchical Task Networks - Multi-level task breakdown
2. Dependency Graph - Automatic dependency detection
3. Progress Monitoring - Real-time status tracking
4. Dynamic Replanning - Adapt to failures and changes
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any, Callable
from collections import defaultdict
import json

from loguru import logger


# =============================================================================
# DYNAMIC REPLANNING (AGI Upgrade)
# =============================================================================

@dataclass
class ReplanTrigger:
    """
    Conditions that trigger automatic replanning.

    Dynamic replanning allows plans to adapt to changing circumstances,
    new information, or unexpected failures.
    """
    trigger_id: str
    name: str
    description: str

    # Detection parameters
    condition_type: str  # "failure_rate", "timeout", "drift", "external", "opportunity"
    threshold: float = 0.5
    check_interval_seconds: float = 30.0

    # Response
    replan_strategy: str = "partial"  # "partial", "full", "abort"
    max_replans: int = 3

    # State
    is_active: bool = True
    trigger_count: int = 0
    last_triggered: Optional[datetime] = None


@dataclass
class ReplanEvent:
    """Record of a replanning event."""
    event_id: str
    plan_id: str
    trigger_id: str
    timestamp: datetime = field(default_factory=datetime.now)

    # What changed
    reason: str = ""
    affected_tasks: list[str] = field(default_factory=list)
    original_task_count: int = 0
    new_task_count: int = 0

    # Outcome
    strategy_used: str = ""
    tasks_added: list[str] = field(default_factory=list)
    tasks_removed: list[str] = field(default_factory=list)
    tasks_modified: list[str] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class PlanHealth:
    """Health metrics for a plan."""
    plan_id: str
    measured_at: datetime = field(default_factory=datetime.now)

    # Progress metrics
    progress_ratio: float = 0.0
    velocity: float = 0.0  # Tasks/hour
    estimated_completion: Optional[datetime] = None

    # Health indicators
    failure_rate: float = 0.0  # Failed / Total attempted
    retry_rate: float = 0.0  # Retries / Attempts
    blocked_ratio: float = 0.0  # Blocked / Pending

    # Risk assessment
    critical_path_health: float = 1.0  # 1.0 = healthy, 0.0 = blocked
    bottleneck_tasks: list[str] = field(default_factory=list)
    at_risk_tasks: list[str] = field(default_factory=list)

    # Drift detection
    scope_drift: float = 0.0  # How much has scope changed
    time_drift: float = 0.0  # Behind/ahead of schedule

    def overall_health(self) -> float:
        """Calculate overall plan health score."""
        return (
            (1 - self.failure_rate) * 0.3 +
            (1 - self.blocked_ratio) * 0.2 +
            self.critical_path_health * 0.3 +
            (1 - min(1, abs(self.time_drift))) * 0.2
        )


class TaskStatus(Enum):
    PENDING = "pending"
    READY = "ready"  # All dependencies satisfied
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"  # Waiting on dependencies
    CANCELLED = "cancelled"
    REPLANNING = "replanning"  # AGI: Being reconsidered


class TaskPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class SubTask:
    """A single task in the plan."""
    id: str
    title: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM

    # Dependencies
    depends_on: list[str] = field(default_factory=list)  # Task IDs
    blocks: list[str] = field(default_factory=list)  # Tasks waiting on this

    # Assignment
    assigned_agent: Optional[str] = None
    agent_type: Optional[str] = None  # "code", "reasoning", "research", etc.

    # Execution
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None

    # Estimation
    estimated_complexity: float = 1.0  # 1.0 = simple, 5.0 = complex
    actual_duration_seconds: Optional[float] = None

    # Metadata
    metadata: dict = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "depends_on": self.depends_on,
            "blocks": self.blocks,
            "assigned_agent": self.assigned_agent,
            "agent_type": self.agent_type,
            "estimated_complexity": self.estimated_complexity,
            "metadata": self.metadata,
        }


@dataclass
class Plan:
    """A complete plan with tasks and structure."""
    id: str
    goal: str
    tasks: dict[str, SubTask] = field(default_factory=dict)

    # Status
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Hierarchy
    root_task_ids: list[str] = field(default_factory=list)

    # Progress
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0

    def get_progress(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.completed_tasks / self.total_tasks

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "goal": self.goal,
            "status": self.status.value,
            "progress": self.get_progress(),
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "tasks": {k: v.to_dict() for k, v in self.tasks.items()},
        }


class PlannerAgent:
    """
    Intelligent task planning and orchestration agent.

    Features:
    - Breaks complex goals into actionable sub-tasks
    - Builds dependency graphs automatically
    - Monitors progress and replans on failures
    - Coordinates with specialist agents
    """

    def __init__(
        self,
        llm_fn: Optional[Callable] = None,
        max_concurrent_tasks: int = 3,
    ):
        self.llm_fn = llm_fn
        self.max_concurrent_tasks = max_concurrent_tasks

        self.plans: dict[str, Plan] = {}
        self.active_plan_id: Optional[str] = None

        self._task_counter = 0
        self._plan_counter = 0
        self._lock = asyncio.Lock()

        # Callbacks
        self.on_task_complete: Optional[Callable] = None
        self.on_task_failed: Optional[Callable] = None
        self.on_plan_complete: Optional[Callable] = None

    async def create_plan(
        self,
        goal: str,
        context: Optional[str] = None,
        constraints: Optional[list[str]] = None,
    ) -> Plan:
        """
        Create a plan to achieve a goal.

        Uses LLM to break down the goal into sub-tasks.
        """
        async with self._lock:
            self._plan_counter += 1
            plan_id = f"plan_{self._plan_counter}"

            plan = Plan(id=plan_id, goal=goal)

            if self.llm_fn:
                # Use LLM to decompose the goal
                tasks = await self._decompose_goal(goal, context, constraints)
                for task in tasks:
                    plan.tasks[task.id] = task
                    if not task.depends_on:
                        plan.root_task_ids.append(task.id)

                # Build dependency graph
                self._build_dependency_graph(plan)
            else:
                # Create a single task for the whole goal
                task = await self._create_task(goal, goal)
                plan.tasks[task.id] = task
                plan.root_task_ids.append(task.id)

            plan.total_tasks = len(plan.tasks)
            self.plans[plan_id] = plan

            logger.info(f"Created plan {plan_id} with {plan.total_tasks} tasks")
            return plan

    async def _decompose_goal(
        self,
        goal: str,
        context: Optional[str],
        constraints: Optional[list[str]],
    ) -> list[SubTask]:
        """Use LLM to decompose goal into tasks."""
        prompt = f"""Break down this goal into specific, actionable sub-tasks.

Goal: {goal}

{f"Context: {context}" if context else ""}
{f"Constraints: {', '.join(constraints)}" if constraints else ""}

Return a JSON array of tasks, each with:
- title: Brief task title
- description: Detailed description of what to do
- agent_type: "code", "reasoning", "research", "creative", or "general"
- depends_on: Array of task indices (0-based) this task depends on
- complexity: 1-5 (1=simple, 5=very complex)

Example:
[
  {{"title": "Research API options", "description": "Find available APIs for...", "agent_type": "research", "depends_on": [], "complexity": 2}},
  {{"title": "Design schema", "description": "Create data model based on research", "agent_type": "reasoning", "depends_on": [0], "complexity": 3}}
]

Return ONLY the JSON array, no other text."""

        try:
            if asyncio.iscoroutinefunction(self.llm_fn):
                response = await self.llm_fn(prompt)
            else:
                response = self.llm_fn(prompt)

            # Parse JSON from response
            json_str = self._extract_json(response)
            task_data = json.loads(json_str)

            tasks = []
            task_ids = []

            for i, td in enumerate(task_data):
                task = await self._create_task(
                    title=td.get("title", f"Task {i+1}"),
                    description=td.get("description", ""),
                    agent_type=td.get("agent_type", "general"),
                    complexity=td.get("complexity", 1.0),
                )
                tasks.append(task)
                task_ids.append(task.id)

            # Resolve dependencies by index
            for i, td in enumerate(task_data):
                dep_indices = td.get("depends_on", [])
                for dep_idx in dep_indices:
                    if 0 <= dep_idx < len(task_ids):
                        tasks[i].depends_on.append(task_ids[dep_idx])

            return tasks

        except Exception as e:
            logger.error(f"Goal decomposition failed: {e}")
            # Fallback to single task
            task = await self._create_task(goal, goal)
            return [task]

    async def _create_task(
        self,
        title: str,
        description: str,
        agent_type: str = "general",
        complexity: float = 1.0,
    ) -> SubTask:
        """Create a new sub-task."""
        self._task_counter += 1
        return SubTask(
            id=f"task_{self._task_counter}",
            title=title,
            description=description,
            agent_type=agent_type,
            estimated_complexity=complexity,
        )

    def _build_dependency_graph(self, plan: Plan):
        """Build reverse dependency links (blocks)."""
        for task in plan.tasks.values():
            for dep_id in task.depends_on:
                if dep_id in plan.tasks:
                    plan.tasks[dep_id].blocks.append(task.id)

    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response."""
        # Find JSON array
        start = text.find('[')
        end = text.rfind(']') + 1
        if start >= 0 and end > start:
            return text[start:end]

        # Try object
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            return '[' + text[start:end] + ']'

        return '[]'

    async def execute_plan(
        self,
        plan_id: str,
        agent_executor: Optional[Callable] = None,
    ) -> Plan:
        """
        Execute a plan, coordinating task execution.

        Args:
            plan_id: ID of the plan to execute
            agent_executor: Async function(task, agent_type) -> result
        """
        if plan_id not in self.plans:
            raise ValueError(f"Unknown plan: {plan_id}")

        plan = self.plans[plan_id]
        plan.status = TaskStatus.IN_PROGRESS
        plan.started_at = datetime.now()
        self.active_plan_id = plan_id

        logger.info(f"Executing plan {plan_id}: {plan.goal}")

        try:
            while not self._is_plan_complete(plan):
                # Get ready tasks
                ready_tasks = self._get_ready_tasks(plan)

                if not ready_tasks:
                    # Check for deadlock
                    if self._has_unfinished_tasks(plan):
                        logger.warning("Plan appears deadlocked")
                        await self._handle_deadlock(plan)
                    break

                # Execute ready tasks (up to max concurrent)
                batch = ready_tasks[:self.max_concurrent_tasks]

                if agent_executor:
                    await asyncio.gather(*[
                        self._execute_task(plan, task, agent_executor)
                        for task in batch
                    ])
                else:
                    # Mark as completed (no executor)
                    for task in batch:
                        task.status = TaskStatus.COMPLETED
                        task.completed_at = datetime.now()
                        plan.completed_tasks += 1

            # Finalize plan
            if plan.failed_tasks == 0:
                plan.status = TaskStatus.COMPLETED
            else:
                plan.status = TaskStatus.FAILED

            plan.completed_at = datetime.now()

            if self.on_plan_complete:
                await self._call_async(self.on_plan_complete, plan)

            logger.info(f"Plan {plan_id} finished: {plan.status.value}")
            return plan

        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            plan.status = TaskStatus.FAILED
            raise
        finally:
            self.active_plan_id = None

    async def _execute_task(
        self,
        plan: Plan,
        task: SubTask,
        agent_executor: Callable,
    ):
        """Execute a single task."""
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()

        logger.info(f"Executing task {task.id}: {task.title}")

        try:
            if asyncio.iscoroutinefunction(agent_executor):
                result = await agent_executor(task, task.agent_type)
            else:
                result = agent_executor(task, task.agent_type)

            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.actual_duration_seconds = (
                task.completed_at - task.started_at
            ).total_seconds()

            plan.completed_tasks += 1

            if self.on_task_complete:
                await self._call_async(self.on_task_complete, task)

            logger.info(f"Task {task.id} completed")

        except Exception as e:
            task.error = str(e)
            task.retry_count += 1

            if task.retry_count >= task.max_retries:
                task.status = TaskStatus.FAILED
                plan.failed_tasks += 1

                if self.on_task_failed:
                    await self._call_async(self.on_task_failed, task)

                logger.error(f"Task {task.id} failed after {task.retry_count} retries")

                # Attempt replanning
                await self._handle_task_failure(plan, task)
            else:
                task.status = TaskStatus.PENDING
                logger.warning(f"Task {task.id} failed, will retry ({task.retry_count}/{task.max_retries})")

    async def _call_async(self, fn: Callable, *args):
        """Call function, handling both sync and async."""
        if asyncio.iscoroutinefunction(fn):
            await fn(*args)
        else:
            fn(*args)

    def _get_ready_tasks(self, plan: Plan) -> list[SubTask]:
        """Get tasks ready for execution (all dependencies satisfied)."""
        ready = []

        for task in plan.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue

            # Check all dependencies are completed
            deps_satisfied = all(
                plan.tasks.get(dep_id, SubTask(id="", title="", description="")).status == TaskStatus.COMPLETED
                for dep_id in task.depends_on
            )

            if deps_satisfied:
                task.status = TaskStatus.READY
                ready.append(task)

        # Sort by priority
        ready.sort(key=lambda t: t.priority.value)

        return ready

    def _is_plan_complete(self, plan: Plan) -> bool:
        """Check if plan is complete."""
        for task in plan.tasks.values():
            if task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                return False
        return True

    def _has_unfinished_tasks(self, plan: Plan) -> bool:
        """Check if there are unfinished tasks."""
        for task in plan.tasks.values():
            if task.status in (TaskStatus.PENDING, TaskStatus.READY, TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED):
                return True
        return False

    async def _handle_deadlock(self, plan: Plan):
        """Handle a deadlocked plan."""
        # Find blocked tasks
        blocked = [
            t for t in plan.tasks.values()
            if t.status in (TaskStatus.PENDING, TaskStatus.BLOCKED)
        ]

        for task in blocked:
            # Check if blocking dependencies failed
            failed_deps = [
                dep_id for dep_id in task.depends_on
                if plan.tasks.get(dep_id, SubTask(id="", title="", description="")).status == TaskStatus.FAILED
            ]

            if failed_deps:
                task.status = TaskStatus.BLOCKED
                task.error = f"Blocked by failed dependencies: {failed_deps}"
                plan.failed_tasks += 1

    async def _handle_task_failure(self, plan: Plan, failed_task: SubTask):
        """Handle a failed task with replanning."""
        logger.info(f"Attempting to replan after task {failed_task.id} failure")

        # Cancel dependent tasks
        for blocked_id in failed_task.blocks:
            if blocked_id in plan.tasks:
                task = plan.tasks[blocked_id]
                if task.status in (TaskStatus.PENDING, TaskStatus.READY):
                    task.status = TaskStatus.CANCELLED
                    task.error = f"Cancelled due to dependency failure: {failed_task.id}"

        # Optionally create alternative tasks via LLM
        if self.llm_fn:
            await self._create_recovery_plan(plan, failed_task)

    async def _create_recovery_plan(self, plan: Plan, failed_task: SubTask):
        """Create alternative tasks to recover from failure."""
        prompt = f"""A task in our plan has failed. Suggest an alternative approach.

Original task: {failed_task.title}
Description: {failed_task.description}
Error: {failed_task.error}

Plan goal: {plan.goal}

Suggest 1-2 alternative tasks that could achieve the same outcome.
Return JSON array with title, description, agent_type, complexity.
Return empty array [] if no alternatives are viable."""

        try:
            if asyncio.iscoroutinefunction(self.llm_fn):
                response = await self.llm_fn(prompt)
            else:
                response = self.llm_fn(prompt)

            json_str = self._extract_json(response)
            alternatives = json.loads(json_str)

            for alt in alternatives:
                new_task = await self._create_task(
                    title=alt.get("title", "Recovery task"),
                    description=alt.get("description", ""),
                    agent_type=alt.get("agent_type", "general"),
                    complexity=alt.get("complexity", 2.0),
                )

                # Inherit dependencies and blocked tasks
                new_task.depends_on = [
                    d for d in failed_task.depends_on
                    if plan.tasks.get(d, SubTask(id="", title="", description="")).status == TaskStatus.COMPLETED
                ]
                new_task.blocks = failed_task.blocks.copy()

                plan.tasks[new_task.id] = new_task
                plan.total_tasks += 1

                # Unblock dependent tasks
                for blocked_id in new_task.blocks:
                    if blocked_id in plan.tasks:
                        blocked_task = plan.tasks[blocked_id]
                        blocked_task.depends_on = [
                            d if d != failed_task.id else new_task.id
                            for d in blocked_task.depends_on
                        ]
                        if blocked_task.status == TaskStatus.CANCELLED:
                            blocked_task.status = TaskStatus.PENDING
                            blocked_task.error = None

                logger.info(f"Created recovery task {new_task.id}: {new_task.title}")

        except Exception as e:
            logger.error(f"Recovery planning failed: {e}")

    async def add_task(
        self,
        plan_id: str,
        title: str,
        description: str,
        depends_on: Optional[list[str]] = None,
        agent_type: str = "general",
    ) -> SubTask:
        """Add a new task to an existing plan."""
        if plan_id not in self.plans:
            raise ValueError(f"Unknown plan: {plan_id}")

        plan = self.plans[plan_id]

        async with self._lock:
            task = await self._create_task(title, description, agent_type)
            task.depends_on = depends_on or []

            plan.tasks[task.id] = task
            plan.total_tasks += 1

            if not task.depends_on:
                plan.root_task_ids.append(task.id)

            # Update dependency graph
            for dep_id in task.depends_on:
                if dep_id in plan.tasks:
                    plan.tasks[dep_id].blocks.append(task.id)

            return task

    async def cancel_task(self, plan_id: str, task_id: str) -> bool:
        """Cancel a task and its dependents."""
        if plan_id not in self.plans:
            return False

        plan = self.plans[plan_id]

        if task_id not in plan.tasks:
            return False

        task = plan.tasks[task_id]

        if task.status in (TaskStatus.COMPLETED, TaskStatus.IN_PROGRESS):
            return False

        async with self._lock:
            await self._cancel_task_recursive(plan, task_id)

        return True

    async def _cancel_task_recursive(self, plan: Plan, task_id: str):
        """Recursively cancel a task and dependents."""
        task = plan.tasks.get(task_id)
        if not task:
            return

        task.status = TaskStatus.CANCELLED

        for blocked_id in task.blocks:
            await self._cancel_task_recursive(plan, blocked_id)

    def get_plan_status(self, plan_id: str) -> Optional[dict]:
        """Get detailed status of a plan."""
        if plan_id not in self.plans:
            return None

        plan = self.plans[plan_id]

        status_counts = defaultdict(int)
        for task in plan.tasks.values():
            status_counts[task.status.value] += 1

        return {
            "id": plan_id,
            "goal": plan.goal,
            "status": plan.status.value,
            "progress": plan.get_progress(),
            "tasks": {
                "total": plan.total_tasks,
                "by_status": dict(status_counts),
            },
            "timing": {
                "created": plan.created_at.isoformat(),
                "started": plan.started_at.isoformat() if plan.started_at else None,
                "completed": plan.completed_at.isoformat() if plan.completed_at else None,
            },
        }

    def get_dependency_graph(self, plan_id: str) -> Optional[dict]:
        """Get dependency graph for visualization."""
        if plan_id not in self.plans:
            return None

        plan = self.plans[plan_id]

        nodes = []
        edges = []

        for task in plan.tasks.values():
            nodes.append({
                "id": task.id,
                "label": task.title,
                "status": task.status.value,
                "agent_type": task.agent_type,
            })

            for dep_id in task.depends_on:
                edges.append({
                    "from": dep_id,
                    "to": task.id,
                })

        return {"nodes": nodes, "edges": edges}

    def get_stats(self) -> dict:
        """Get planner statistics."""
        total_tasks = sum(len(p.tasks) for p in self.plans.values())
        completed_tasks = sum(p.completed_tasks for p in self.plans.values())

        return {
            "total_plans": len(self.plans),
            "active_plan": self.active_plan_id,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "success_rate": completed_tasks / total_tasks if total_tasks > 0 else 0,
        }

    # =========================================================================
    # DYNAMIC REPLANNING (AGI Upgrade)
    # =========================================================================

    def _init_replanning(self):
        """Initialize dynamic replanning state."""
        if not hasattr(self, '_replan_triggers'):
            self._replan_triggers: list[ReplanTrigger] = []
            self._replan_history: list[ReplanEvent] = []
            self._health_history: dict[str, list[PlanHealth]] = {}  # plan_id -> history
            self._replan_callbacks: list[Callable] = []

            # Register default triggers
            self._register_default_replan_triggers()

    def _register_default_replan_triggers(self):
        """Register default replanning triggers."""
        self._replan_triggers = [
            ReplanTrigger(
                trigger_id="high_failure_rate",
                name="High Failure Rate",
                description="Too many tasks failing consecutively",
                condition_type="failure_rate",
                threshold=0.3,  # 30% failure rate
                replan_strategy="partial",
            ),
            ReplanTrigger(
                trigger_id="timeout_risk",
                name="Timeout Risk",
                description="Plan at risk of exceeding time budget",
                condition_type="timeout",
                threshold=0.8,  # 80% of time used, <50% complete
                replan_strategy="partial",
            ),
            ReplanTrigger(
                trigger_id="scope_drift",
                name="Scope Drift",
                description="Plan scope has drifted significantly",
                condition_type="drift",
                threshold=0.5,  # 50% scope change
                replan_strategy="full",
            ),
            ReplanTrigger(
                trigger_id="blocked_cascade",
                name="Blocked Cascade",
                description="Many tasks blocked due to failures",
                condition_type="failure_rate",
                threshold=0.4,  # 40% blocked
                replan_strategy="partial",
            ),
            ReplanTrigger(
                trigger_id="opportunity_detected",
                name="Opportunity Detected",
                description="New information suggests better approach",
                condition_type="opportunity",
                threshold=0.7,  # High confidence opportunity
                replan_strategy="partial",
            ),
        ]

    def add_replan_trigger(self, trigger: ReplanTrigger):
        """Add a custom replanning trigger."""
        self._init_replanning()
        self._replan_triggers.append(trigger)
        logger.info(f"Added replan trigger: {trigger.name}")

    def on_replan(self, callback: Callable[[ReplanEvent], None]):
        """Register a callback for replan events."""
        self._init_replanning()
        self._replan_callbacks.append(callback)

    async def assess_plan_health(self, plan_id: str) -> PlanHealth:
        """
        Assess the current health of a plan.

        Args:
            plan_id: The plan to assess

        Returns:
            PlanHealth metrics
        """
        self._init_replanning()

        if plan_id not in self.plans:
            raise ValueError(f"Unknown plan: {plan_id}")

        plan = self.plans[plan_id]
        health = PlanHealth(plan_id=plan_id)

        # Progress metrics
        health.progress_ratio = plan.get_progress()

        # Calculate velocity (tasks per hour)
        if plan.started_at:
            elapsed_hours = (datetime.now() - plan.started_at).total_seconds() / 3600
            if elapsed_hours > 0:
                health.velocity = plan.completed_tasks / elapsed_hours

                # Estimate completion
                remaining = plan.total_tasks - plan.completed_tasks
                if health.velocity > 0:
                    remaining_hours = remaining / health.velocity
                    from datetime import timedelta
                    health.estimated_completion = datetime.now() + timedelta(hours=remaining_hours)

        # Failure rate
        attempted = plan.completed_tasks + plan.failed_tasks
        if attempted > 0:
            health.failure_rate = plan.failed_tasks / attempted

        # Retry rate
        total_retries = sum(t.retry_count for t in plan.tasks.values())
        if attempted > 0:
            health.retry_rate = total_retries / attempted

        # Blocked ratio
        pending_tasks = [t for t in plan.tasks.values() if t.status in (TaskStatus.PENDING, TaskStatus.BLOCKED)]
        blocked_tasks = [t for t in plan.tasks.values() if t.status == TaskStatus.BLOCKED]
        if pending_tasks:
            health.blocked_ratio = len(blocked_tasks) / len(pending_tasks)

        # Critical path analysis
        health.critical_path_health, health.bottleneck_tasks = self._analyze_critical_path(plan)

        # At-risk tasks (high retry count, complex, or blocked)
        health.at_risk_tasks = [
            t.id for t in plan.tasks.values()
            if (t.retry_count >= 2 or
                t.estimated_complexity >= 4 or
                t.status == TaskStatus.BLOCKED)
        ]

        # Scope drift (new tasks added vs original)
        if plan_id in self._health_history and self._health_history[plan_id]:
            initial_count = self._health_history[plan_id][0].progress_ratio * plan.total_tasks
            if initial_count > 0:
                health.scope_drift = abs(plan.total_tasks - initial_count) / initial_count

        # Time drift
        if plan.started_at and health.estimated_completion:
            # Assume original estimate was 2x the elapsed time per task
            original_estimate = plan.started_at + (datetime.now() - plan.started_at) * (plan.total_tasks / max(1, plan.completed_tasks))
            time_diff = (health.estimated_completion - original_estimate).total_seconds()
            original_duration = (original_estimate - plan.started_at).total_seconds()
            if original_duration > 0:
                health.time_drift = time_diff / original_duration

        # Store health history
        if plan_id not in self._health_history:
            self._health_history[plan_id] = []
        self._health_history[plan_id].append(health)

        # Keep only recent history (last 20)
        if len(self._health_history[plan_id]) > 20:
            self._health_history[plan_id] = self._health_history[plan_id][-20:]

        return health

    def _analyze_critical_path(self, plan: Plan) -> tuple[float, list[str]]:
        """
        Analyze the critical path of the plan.

        Returns:
            Tuple of (health_score, bottleneck_task_ids)
        """
        if not plan.tasks:
            return 1.0, []

        # Find longest path (critical path)
        def get_path_length(task_id: str, visited: set) -> int:
            if task_id in visited or task_id not in plan.tasks:
                return 0
            visited.add(task_id)
            task = plan.tasks[task_id]
            if not task.blocks:
                return 1
            return 1 + max(get_path_length(b, visited.copy()) for b in task.blocks)

        # Find critical path from root tasks
        max_length = 0
        critical_path_root = None
        for root_id in plan.root_task_ids:
            length = get_path_length(root_id, set())
            if length > max_length:
                max_length = length
                critical_path_root = root_id

        if not critical_path_root:
            return 1.0, []

        # Trace critical path and find bottlenecks
        bottlenecks = []
        current = critical_path_root
        health_score = 1.0

        while current and current in plan.tasks:
            task = plan.tasks[current]

            # Check if this task is a bottleneck
            if task.status == TaskStatus.FAILED:
                health_score *= 0.5
                bottlenecks.append(current)
            elif task.status == TaskStatus.BLOCKED:
                health_score *= 0.7
                bottlenecks.append(current)
            elif task.retry_count >= 2:
                health_score *= 0.8
                bottlenecks.append(current)

            # Move to next task in critical path
            if task.blocks:
                # Choose the blocking task with highest complexity
                next_task = max(
                    task.blocks,
                    key=lambda t_id: plan.tasks.get(t_id, SubTask(id="", title="", description="")).estimated_complexity
                )
                current = next_task
            else:
                break

        return max(0.1, health_score), bottlenecks

    async def check_replan_triggers(self, plan_id: str) -> list[ReplanEvent]:
        """
        Check all replan triggers against current plan health.

        Args:
            plan_id: The plan to check

        Returns:
            List of triggered replan events (may trigger replanning)
        """
        self._init_replanning()

        health = await self.assess_plan_health(plan_id)
        triggered_events = []

        for trigger in self._replan_triggers:
            if not trigger.is_active:
                continue
            if trigger.trigger_count >= trigger.max_replans:
                continue

            should_trigger = False

            if trigger.condition_type == "failure_rate":
                should_trigger = health.failure_rate >= trigger.threshold

            elif trigger.condition_type == "timeout":
                # Check if running out of time with low progress
                if health.progress_ratio < 0.5 and health.time_drift > trigger.threshold:
                    should_trigger = True

            elif trigger.condition_type == "drift":
                should_trigger = health.scope_drift >= trigger.threshold

            elif trigger.condition_type == "blocked_cascade":
                should_trigger = health.blocked_ratio >= trigger.threshold

            if should_trigger:
                event = await self._execute_replan(plan_id, trigger, health)
                triggered_events.append(event)

                trigger.trigger_count += 1
                trigger.last_triggered = datetime.now()

                logger.info(f"Replan triggered: {trigger.name} for plan {plan_id}")

        return triggered_events

    async def _execute_replan(
        self,
        plan_id: str,
        trigger: ReplanTrigger,
        health: PlanHealth,
    ) -> ReplanEvent:
        """
        Execute a replanning based on the trigger.

        Args:
            plan_id: The plan to replan
            trigger: The trigger that fired
            health: Current health metrics

        Returns:
            ReplanEvent record
        """
        plan = self.plans[plan_id]

        event = ReplanEvent(
            event_id=f"replan_{len(self._replan_history) + 1}_{int(datetime.now().timestamp())}",
            plan_id=plan_id,
            trigger_id=trigger.trigger_id,
            reason=trigger.description,
            original_task_count=plan.total_tasks,
            strategy_used=trigger.replan_strategy,
        )

        try:
            if trigger.replan_strategy == "partial":
                await self._partial_replan(plan, event, health)
            elif trigger.replan_strategy == "full":
                await self._full_replan(plan, event)
            elif trigger.replan_strategy == "abort":
                await self._abort_plan(plan, event)

            event.new_task_count = plan.total_tasks
            event.success = True

        except Exception as e:
            event.success = False
            event.error_message = str(e)
            logger.error(f"Replan failed: {e}")

        self._replan_history.append(event)

        # Notify callbacks
        for callback in self._replan_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Replan callback error: {e}")

        return event

    async def _partial_replan(self, plan: Plan, event: ReplanEvent, health: PlanHealth):
        """
        Partial replan: fix problem areas while preserving working parts.
        """
        # Identify tasks to reconsider
        problem_tasks = set(health.bottleneck_tasks + health.at_risk_tasks)

        # Mark them for replanning
        for task_id in problem_tasks:
            if task_id in plan.tasks:
                task = plan.tasks[task_id]
                if task.status in (TaskStatus.PENDING, TaskStatus.BLOCKED, TaskStatus.FAILED):
                    task.status = TaskStatus.REPLANNING
                    event.affected_tasks.append(task_id)

        # Use LLM to suggest alternatives for problem tasks
        if self.llm_fn and event.affected_tasks:
            await self._generate_alternative_tasks(plan, event)

    async def _generate_alternative_tasks(self, plan: Plan, event: ReplanEvent):
        """Generate alternative tasks for problematic ones."""
        problem_descriptions = []
        for task_id in event.affected_tasks:
            task = plan.tasks.get(task_id)
            if task:
                problem_descriptions.append(
                    f"- {task.title}: {task.description} (Error: {task.error or 'blocked/at-risk'})"
                )

        if not problem_descriptions:
            return

        prompt = f"""Several tasks in a plan are having problems. Suggest alternatives or simplifications.

PLAN GOAL: {plan.goal}

PROBLEMATIC TASKS:
{chr(10).join(problem_descriptions)}

REQUIREMENTS:
1. Suggest simpler alternatives that achieve the same outcomes
2. Consider breaking complex tasks into smaller steps
3. Identify if any tasks can be skipped or merged
4. Provide dependencies based on existing successful tasks

Return JSON array:
[
  {{"original_task_id": "...", "action": "replace|simplify|skip|split", "new_tasks": [{{"title": "...", "description": "...", "agent_type": "...", "complexity": 1-5}}], "reasoning": "..."}}
]

Return only necessary changes. Empty array if no good alternatives exist."""

        try:
            if asyncio.iscoroutinefunction(self.llm_fn):
                response = await self.llm_fn(prompt)
            else:
                response = self.llm_fn(prompt)

            json_str = self._extract_json(response)
            alternatives = json.loads(json_str)

            for alt in alternatives:
                original_id = alt.get("original_task_id")
                action = alt.get("action", "skip")
                new_task_defs = alt.get("new_tasks", [])

                if original_id not in plan.tasks:
                    continue

                original_task = plan.tasks[original_id]

                if action == "skip":
                    # Remove task and its dependents
                    original_task.status = TaskStatus.CANCELLED
                    event.tasks_removed.append(original_id)

                elif action == "replace" or action == "simplify":
                    # Create new tasks to replace
                    for new_def in new_task_defs:
                        new_task = await self._create_task(
                            title=new_def.get("title", "Recovery task"),
                            description=new_def.get("description", ""),
                            agent_type=new_def.get("agent_type", "general"),
                            complexity=new_def.get("complexity", 2.0),
                        )

                        # Inherit dependencies
                        new_task.depends_on = [
                            d for d in original_task.depends_on
                            if plan.tasks.get(d, SubTask(id="", title="", description="")).status == TaskStatus.COMPLETED
                        ]
                        new_task.blocks = original_task.blocks.copy()

                        plan.tasks[new_task.id] = new_task
                        plan.total_tasks += 1
                        event.tasks_added.append(new_task.id)

                        # Update dependencies of blocked tasks
                        for blocked_id in new_task.blocks:
                            if blocked_id in plan.tasks:
                                blocked_task = plan.tasks[blocked_id]
                                blocked_task.depends_on = [
                                    d if d != original_id else new_task.id
                                    for d in blocked_task.depends_on
                                ]

                    # Mark original as cancelled
                    original_task.status = TaskStatus.CANCELLED
                    event.tasks_removed.append(original_id)

                elif action == "split":
                    # Split into multiple smaller tasks
                    prev_task_id = None
                    for i, new_def in enumerate(new_task_defs):
                        new_task = await self._create_task(
                            title=new_def.get("title", f"Split task {i+1}"),
                            description=new_def.get("description", ""),
                            agent_type=new_def.get("agent_type", "general"),
                            complexity=new_def.get("complexity", 1.0),
                        )

                        # Chain dependencies
                        if i == 0:
                            new_task.depends_on = original_task.depends_on.copy()
                        else:
                            new_task.depends_on = [prev_task_id] if prev_task_id else []

                        if i == len(new_task_defs) - 1:
                            new_task.blocks = original_task.blocks.copy()

                        plan.tasks[new_task.id] = new_task
                        plan.total_tasks += 1
                        event.tasks_added.append(new_task.id)
                        prev_task_id = new_task.id

                    original_task.status = TaskStatus.CANCELLED
                    event.tasks_removed.append(original_id)

                event.tasks_modified.append(original_id)

        except Exception as e:
            logger.error(f"Alternative task generation failed: {e}")

    async def _full_replan(self, plan: Plan, event: ReplanEvent):
        """
        Full replan: regenerate the entire plan from scratch.
        """
        # Preserve completed task results
        completed_results = {
            task.id: task.result
            for task in plan.tasks.values()
            if task.status == TaskStatus.COMPLETED
        }

        # Cancel all non-completed tasks
        for task in plan.tasks.values():
            if task.status not in (TaskStatus.COMPLETED,):
                task.status = TaskStatus.CANCELLED
                event.tasks_removed.append(task.id)

        # Create new plan with same goal
        if self.llm_fn:
            # Include completed work in context
            completed_summary = "\n".join([
                f"- {plan.tasks[tid].title}: COMPLETED"
                for tid in completed_results.keys()
                if tid in plan.tasks
            ])

            context = f"Previously completed work:\n{completed_summary}\n\nContinue from here."

            new_tasks = await self._decompose_goal(plan.goal, context, None)

            for task in new_tasks:
                # Don't duplicate completed work
                if not any(
                    t.title.lower() == task.title.lower()
                    for t in plan.tasks.values()
                    if t.status == TaskStatus.COMPLETED
                ):
                    plan.tasks[task.id] = task
                    plan.total_tasks += 1
                    event.tasks_added.append(task.id)

                    if not task.depends_on:
                        plan.root_task_ids.append(task.id)

            self._build_dependency_graph(plan)

    async def _abort_plan(self, plan: Plan, event: ReplanEvent):
        """
        Abort plan: cancel all remaining tasks.
        """
        for task in plan.tasks.values():
            if task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                task.status = TaskStatus.CANCELLED
                event.tasks_removed.append(task.id)

        plan.status = TaskStatus.CANCELLED

    async def trigger_opportunity_replan(
        self,
        plan_id: str,
        opportunity_description: str,
        confidence: float = 0.8,
    ) -> Optional[ReplanEvent]:
        """
        Manually trigger a replan based on new information/opportunity.

        Args:
            plan_id: The plan to potentially replan
            opportunity_description: What new information or opportunity was found
            confidence: How confident we are this is worth replanning for

        Returns:
            ReplanEvent if replanning was triggered, None otherwise
        """
        self._init_replanning()

        if plan_id not in self.plans:
            return None

        # Find opportunity trigger
        opportunity_trigger = next(
            (t for t in self._replan_triggers if t.condition_type == "opportunity"),
            None
        )

        if not opportunity_trigger or confidence < opportunity_trigger.threshold:
            return None

        health = await self.assess_plan_health(plan_id)

        # Create a modified event for opportunity-based replan
        event = ReplanEvent(
            event_id=f"replan_opp_{len(self._replan_history) + 1}_{int(datetime.now().timestamp())}",
            plan_id=plan_id,
            trigger_id="opportunity_manual",
            reason=opportunity_description,
            original_task_count=self.plans[plan_id].total_tasks,
            strategy_used="partial",
        )

        # Use LLM to determine how to incorporate the opportunity
        if self.llm_fn:
            plan = self.plans[plan_id]

            prompt = f"""A new opportunity or information has been discovered that may improve this plan.

PLAN GOAL: {plan.goal}

CURRENT TASKS:
{json.dumps([t.to_dict() for t in plan.tasks.values() if t.status in (TaskStatus.PENDING, TaskStatus.READY)], indent=2)}

NEW OPPORTUNITY:
{opportunity_description}

INSTRUCTIONS:
1. Determine if this opportunity should change the plan
2. Suggest specific task modifications or additions
3. Consider if any pending tasks should be reprioritized or skipped

Return JSON:
{{
    "should_replan": true/false,
    "reasoning": "why or why not",
    "task_changes": [
        {{"action": "add|modify|remove|reprioritize", "task_id": "..." (for modify/remove), "new_task": {{...}} (for add/modify), "new_priority": 1-5 (for reprioritize)}}
    ]
}}"""

            try:
                if asyncio.iscoroutinefunction(self.llm_fn):
                    response = await self.llm_fn(prompt)
                else:
                    response = self.llm_fn(prompt)

                # Extract JSON
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    data = json.loads(response[start:end])

                    if not data.get("should_replan", False):
                        return None

                    # Apply changes
                    for change in data.get("task_changes", []):
                        action = change.get("action")

                        if action == "add":
                            new_def = change.get("new_task", {})
                            new_task = await self._create_task(
                                title=new_def.get("title", "New task"),
                                description=new_def.get("description", ""),
                                agent_type=new_def.get("agent_type", "general"),
                                complexity=new_def.get("complexity", 2.0),
                            )
                            plan.tasks[new_task.id] = new_task
                            plan.total_tasks += 1
                            event.tasks_added.append(new_task.id)

                        elif action == "remove":
                            task_id = change.get("task_id")
                            if task_id in plan.tasks:
                                plan.tasks[task_id].status = TaskStatus.CANCELLED
                                event.tasks_removed.append(task_id)

                        elif action == "reprioritize":
                            task_id = change.get("task_id")
                            new_priority = change.get("new_priority", 3)
                            if task_id in plan.tasks:
                                plan.tasks[task_id].priority = TaskPriority(new_priority)
                                event.tasks_modified.append(task_id)

                    event.success = True
                    event.new_task_count = plan.total_tasks

            except Exception as e:
                logger.error(f"Opportunity replan failed: {e}")
                event.success = False
                event.error_message = str(e)

        self._replan_history.append(event)

        # Notify callbacks
        for callback in self._replan_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Replan callback error: {e}")

        return event

    def get_replan_history(self, plan_id: Optional[str] = None, limit: int = 20) -> list[ReplanEvent]:
        """Get replanning history, optionally filtered by plan."""
        self._init_replanning()

        events = self._replan_history
        if plan_id:
            events = [e for e in events if e.plan_id == plan_id]

        return events[-limit:]

    def get_replan_stats(self) -> dict:
        """Get replanning statistics."""
        self._init_replanning()

        if not self._replan_history:
            return {"total_replans": 0}

        successful = sum(1 for e in self._replan_history if e.success)
        by_strategy = {}
        for event in self._replan_history:
            by_strategy[event.strategy_used] = by_strategy.get(event.strategy_used, 0) + 1

        return {
            "total_replans": len(self._replan_history),
            "successful_replans": successful,
            "success_rate": successful / len(self._replan_history),
            "by_strategy": by_strategy,
            "total_tasks_added": sum(len(e.tasks_added) for e in self._replan_history),
            "total_tasks_removed": sum(len(e.tasks_removed) for e in self._replan_history),
            "active_triggers": len([t for t in self._replan_triggers if t.is_active]),
        }
