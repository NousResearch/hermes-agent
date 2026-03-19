"""
cosmos Hierarchical Agent Teams - Multi-Level Agent Organization

Novel Approaches:
1. Manager Agents - Coordinate specialist teams
2. Dynamic Team Formation - Create teams for specific tasks
3. Load Balancing - Distribute work across agents
4. Escalation Protocols - Handle failures gracefully
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any, Callable
from collections import defaultdict
import json

from loguru import logger


class AgentRole(Enum):
    """Roles in the hierarchy."""
    EXECUTIVE = "executive"     # Top-level coordinator
    MANAGER = "manager"         # Team coordinator
    SPECIALIST = "specialist"   # Task executor
    SUPPORT = "support"         # Auxiliary functions


class TeamStatus(Enum):
    """Team operational status."""
    IDLE = "idle"
    FORMING = "forming"
    ACTIVE = "active"
    WORKING = "working"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    DISBANDED = "disbanded"


@dataclass
class AgentNode:
    """An agent in the hierarchy."""
    id: str
    name: str
    role: AgentRole
    specializations: list[str] = field(default_factory=list)

    # Hierarchy
    manager_id: Optional[str] = None
    subordinates: list[str] = field(default_factory=list)

    # Capacity
    max_concurrent_tasks: int = 3
    current_tasks: int = 0
    workload: float = 0.0  # 0.0 to 1.0

    # Performance
    success_rate: float = 0.5
    avg_completion_time: float = 0.0

    # Status
    is_available: bool = True
    status_message: str = ""

    def is_overloaded(self) -> bool:
        return self.current_tasks >= self.max_concurrent_tasks

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role.value,
            "specializations": self.specializations,
            "manager": self.manager_id,
            "subordinates": self.subordinates,
            "workload": round(self.workload, 2),
            "available": self.is_available,
        }


@dataclass
class Team:
    """A team of agents working together."""
    id: str
    name: str
    purpose: str
    status: TeamStatus = TeamStatus.IDLE

    # Composition
    manager_id: Optional[str] = None
    member_ids: list[str] = field(default_factory=list)

    # Task
    current_task_id: Optional[str] = None
    completed_tasks: list[str] = field(default_factory=list)

    # Performance
    success_count: int = 0
    failure_count: int = 0

    # Lifecycle
    created_at: datetime = field(default_factory=datetime.now)
    disbanded_at: Optional[datetime] = None

    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "purpose": self.purpose,
            "status": self.status.value,
            "manager": self.manager_id,
            "members": self.member_ids,
            "success_rate": self.success_rate(),
        }


@dataclass
class TaskAssignment:
    """Assignment of a task to an agent/team."""
    task_id: str
    task_type: str
    description: str

    # Assignment
    assigned_to: str  # Agent or team ID
    assigned_by: Optional[str] = None  # Manager ID
    assigned_at: datetime = field(default_factory=datetime.now)

    # Status
    status: str = "pending"  # pending, in_progress, completed, failed, escalated
    result: Optional[Any] = None
    error: Optional[str] = None

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    deadline: Optional[datetime] = None

    # Escalation
    escalation_count: int = 0
    escalated_from: Optional[str] = None


class HierarchicalTeams:
    """
    Hierarchical agent team management system.

    Features:
    - Multi-level agent hierarchy
    - Dynamic team formation
    - Intelligent load balancing
    - Escalation handling
    """

    def __init__(
        self,
        llm_fn: Optional[Callable] = None,
        max_escalations: int = 3,
    ):
        self.llm_fn = llm_fn
        self.max_escalations = max_escalations

        self.agents: dict[str, AgentNode] = {}
        self.teams: dict[str, Team] = {}
        self.assignments: dict[str, TaskAssignment] = {}

        # Hierarchy root
        self.executive_id: Optional[str] = None

        # Counters
        self._team_counter = 0
        self._assignment_counter = 0

        self._lock = asyncio.Lock()

    async def create_agent(
        self,
        agent_id: str,
        name: str,
        role: AgentRole,
        specializations: Optional[list[str]] = None,
        manager_id: Optional[str] = None,
    ) -> AgentNode:
        """Create and register an agent in the hierarchy."""
        agent = AgentNode(
            id=agent_id,
            name=name,
            role=role,
            specializations=specializations or [],
            manager_id=manager_id,
        )

        async with self._lock:
            self.agents[agent_id] = agent

            # Set as executive if first executive
            if role == AgentRole.EXECUTIVE and self.executive_id is None:
                self.executive_id = agent_id

            # Add to manager's subordinates
            if manager_id and manager_id in self.agents:
                self.agents[manager_id].subordinates.append(agent_id)

        logger.info(f"Created agent {agent_id} ({role.value})")
        return agent

    async def form_team(
        self,
        purpose: str,
        required_specializations: list[str],
        manager_preference: Optional[str] = None,
        team_size: int = 3,
    ) -> Team:
        """
        Dynamically form a team for a specific purpose.

        Args:
            purpose: What the team will work on
            required_specializations: Skills needed
            manager_preference: Preferred manager agent
            team_size: Target team size

        Returns:
            Formed Team
        """
        async with self._lock:
            self._team_counter += 1
            team_id = f"team_{self._team_counter}"

        # Select manager
        manager_id = manager_preference
        if not manager_id:
            manager_id = await self._select_manager(required_specializations)

        # Select members
        members = await self._select_members(
            required_specializations,
            manager_id,
            team_size,
        )

        team = Team(
            id=team_id,
            name=f"Team for: {purpose[:30]}",
            purpose=purpose,
            status=TeamStatus.FORMING,
            manager_id=manager_id,
            member_ids=members,
        )

        # Assign members to team
        async with self._lock:
            self.teams[team_id] = team

            for member_id in members:
                if member_id in self.agents:
                    agent = self.agents[member_id]
                    # Temporarily assign to this manager
                    agent.status_message = f"Assigned to {team_id}"

        team.status = TeamStatus.ACTIVE
        logger.info(f"Formed team {team_id} with {len(members)} members")

        return team

    async def _select_manager(
        self,
        required_specializations: list[str],
    ) -> Optional[str]:
        """Select the best manager for a team."""
        best_manager = None
        best_score = -1

        for agent_id, agent in self.agents.items():
            if agent.role not in (AgentRole.MANAGER, AgentRole.EXECUTIVE):
                continue

            if not agent.is_available or agent.is_overloaded():
                continue

            # Score based on specialization match and availability
            spec_match = len(set(agent.specializations) & set(required_specializations))
            availability = 1 - agent.workload

            score = spec_match * 0.5 + availability * 0.3 + agent.success_rate * 0.2

            if score > best_score:
                best_score = score
                best_manager = agent_id

        return best_manager

    async def _select_members(
        self,
        required_specializations: list[str],
        manager_id: Optional[str],
        team_size: int,
    ) -> list[str]:
        """Select team members based on requirements."""
        candidates = []

        for agent_id, agent in self.agents.items():
            if agent_id == manager_id:
                continue

            if agent.role != AgentRole.SPECIALIST:
                continue

            if not agent.is_available or agent.is_overloaded():
                continue

            # Score candidate
            spec_match = len(set(agent.specializations) & set(required_specializations))
            availability = 1 - agent.workload

            score = spec_match * 0.6 + availability * 0.2 + agent.success_rate * 0.2

            candidates.append((agent_id, score))

        # Sort by score and take top N
        candidates.sort(key=lambda x: x[1], reverse=True)

        return [c[0] for c in candidates[:team_size]]

    async def assign_task(
        self,
        task_type: str,
        description: str,
        preferred_agent: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> TaskAssignment:
        """
        Assign a task to an agent or team.

        Uses intelligent routing if no preference given.
        """
        async with self._lock:
            self._assignment_counter += 1
            task_id = f"task_{self._assignment_counter}"

        # Determine assignee
        if team_id and team_id in self.teams:
            assignee = team_id
            assigner = self.teams[team_id].manager_id
        elif preferred_agent and preferred_agent in self.agents:
            assignee = preferred_agent
            assigner = self.agents[preferred_agent].manager_id
        else:
            # Auto-route
            assignee = await self._route_task(task_type)
            assigner = self.executive_id

        assignment = TaskAssignment(
            task_id=task_id,
            task_type=task_type,
            description=description,
            assigned_to=assignee,
            assigned_by=assigner,
        )

        async with self._lock:
            self.assignments[task_id] = assignment

            # Update agent workload
            if assignee in self.agents:
                agent = self.agents[assignee]
                agent.current_tasks += 1
                agent.workload = agent.current_tasks / agent.max_concurrent_tasks

        logger.info(f"Assigned task {task_id} to {assignee}")
        return assignment

    async def _route_task(self, task_type: str) -> str:
        """Route a task to the best available agent."""
        best_agent = None
        best_score = -1

        for agent_id, agent in self.agents.items():
            if agent.role != AgentRole.SPECIALIST:
                continue

            if not agent.is_available or agent.is_overloaded():
                continue

            # Check specialization match
            if task_type in agent.specializations:
                spec_score = 1.0
            elif any(task_type in s for s in agent.specializations):
                spec_score = 0.5
            else:
                spec_score = 0.1

            # Combine with availability
            availability = 1 - agent.workload
            score = spec_score * 0.6 + availability * 0.3 + agent.success_rate * 0.1

            if score > best_score:
                best_score = score
                best_agent = agent_id

        return best_agent or self.executive_id or ""

    async def complete_task(
        self,
        task_id: str,
        success: bool,
        result: Optional[Any] = None,
        error: Optional[str] = None,
    ) -> TaskAssignment:
        """Mark a task as completed."""
        if task_id not in self.assignments:
            raise ValueError(f"Unknown task: {task_id}")

        assignment = self.assignments[task_id]

        async with self._lock:
            assignment.status = "completed" if success else "failed"
            assignment.result = result
            assignment.error = error
            assignment.completed_at = datetime.now()

            # Update agent
            if assignment.assigned_to in self.agents:
                agent = self.agents[assignment.assigned_to]
                agent.current_tasks = max(0, agent.current_tasks - 1)
                agent.workload = agent.current_tasks / agent.max_concurrent_tasks

                # Update success rate
                if success:
                    agent.success_rate = agent.success_rate * 0.9 + 0.1
                else:
                    agent.success_rate = agent.success_rate * 0.9

            # Update team if applicable
            for team in self.teams.values():
                if assignment.assigned_to in team.member_ids or assignment.assigned_to == team.id:
                    if success:
                        team.success_count += 1
                    else:
                        team.failure_count += 1
                    team.completed_tasks.append(task_id)

        return assignment

    async def escalate_task(
        self,
        task_id: str,
        reason: str,
    ) -> TaskAssignment:
        """
        Escalate a task to a higher-level agent.

        Args:
            task_id: Task to escalate
            reason: Why it's being escalated

        Returns:
            Updated assignment
        """
        if task_id not in self.assignments:
            raise ValueError(f"Unknown task: {task_id}")

        assignment = self.assignments[task_id]

        if assignment.escalation_count >= self.max_escalations:
            assignment.status = "failed"
            assignment.error = f"Max escalations reached. Reason: {reason}"
            return assignment

        async with self._lock:
            current_agent = assignment.assigned_to

            # Find manager to escalate to
            new_assignee = None

            if current_agent in self.agents:
                agent = self.agents[current_agent]
                new_assignee = agent.manager_id

                # Release current agent
                agent.current_tasks = max(0, agent.current_tasks - 1)
                agent.workload = agent.current_tasks / agent.max_concurrent_tasks

            if not new_assignee:
                new_assignee = self.executive_id

            if new_assignee:
                assignment.escalated_from = current_agent
                assignment.assigned_to = new_assignee
                assignment.escalation_count += 1
                assignment.status = "escalated"

                # Assign to new agent
                if new_assignee in self.agents:
                    new_agent = self.agents[new_assignee]
                    new_agent.current_tasks += 1
                    new_agent.workload = new_agent.current_tasks / new_agent.max_concurrent_tasks

        logger.info(f"Escalated task {task_id} from {current_agent} to {new_assignee}")
        return assignment

    async def rebalance_workload(self):
        """
        Rebalance workload across agents.

        Moves tasks from overloaded to underutilized agents.
        """
        # Find overloaded and underutilized agents
        overloaded = []
        underutilized = []

        for agent_id, agent in self.agents.items():
            if agent.role != AgentRole.SPECIALIST:
                continue

            if agent.workload > 0.8:
                overloaded.append(agent_id)
            elif agent.workload < 0.3 and agent.is_available:
                underutilized.append(agent_id)

        if not overloaded or not underutilized:
            return

        # Find pending tasks from overloaded agents
        reassignments = []

        for task_id, assignment in self.assignments.items():
            if assignment.status not in ("pending", "in_progress"):
                continue

            if assignment.assigned_to not in overloaded:
                continue

            # Find compatible underutilized agent
            overloaded_agent = self.agents[assignment.assigned_to]

            for under_id in underutilized:
                under_agent = self.agents[under_id]

                # Check specialization compatibility
                if set(overloaded_agent.specializations) & set(under_agent.specializations):
                    reassignments.append((task_id, assignment.assigned_to, under_id))
                    underutilized.remove(under_id)
                    break

        # Execute reassignments
        async with self._lock:
            for task_id, from_id, to_id in reassignments:
                assignment = self.assignments[task_id]
                assignment.assigned_to = to_id

                # Update agent workloads
                self.agents[from_id].current_tasks -= 1
                self.agents[from_id].workload = self.agents[from_id].current_tasks / self.agents[from_id].max_concurrent_tasks

                self.agents[to_id].current_tasks += 1
                self.agents[to_id].workload = self.agents[to_id].current_tasks / self.agents[to_id].max_concurrent_tasks

                logger.info(f"Rebalanced task {task_id}: {from_id} -> {to_id}")

    async def delegate_to_team(
        self,
        team_id: str,
        task_type: str,
        description: str,
    ) -> dict:
        """
        Delegate a task to a team for collaborative execution.

        Manager coordinates, specialists execute subtasks.
        """
        if team_id not in self.teams:
            raise ValueError(f"Unknown team: {team_id}")

        team = self.teams[team_id]

        if team.status != TeamStatus.ACTIVE:
            raise ValueError(f"Team {team_id} is not active")

        team.status = TeamStatus.WORKING

        # Create main task
        main_task = await self.assign_task(
            task_type=task_type,
            description=description,
            team_id=team_id,
        )

        team.current_task_id = main_task.task_id

        # Manager decomposes task (if LLM available)
        subtasks = []
        if self.llm_fn and team.manager_id:
            subtasks = await self._decompose_for_team(description, team)

        # Assign subtasks to members
        member_assignments = {}

        for i, (subtask_desc, member_id) in enumerate(
            zip(subtasks, team.member_ids)
        ):
            sub_assignment = await self.assign_task(
                task_type=task_type,
                description=subtask_desc,
                preferred_agent=member_id,
            )
            member_assignments[member_id] = sub_assignment.task_id

        return {
            "main_task_id": main_task.task_id,
            "subtasks": member_assignments,
            "team_id": team_id,
        }

    async def _decompose_for_team(
        self,
        description: str,
        team: Team,
    ) -> list[str]:
        """Use LLM to decompose task for team members."""
        member_specs = []
        for member_id in team.member_ids:
            if member_id in self.agents:
                agent = self.agents[member_id]
                member_specs.append(f"- {agent.name}: {', '.join(agent.specializations)}")

        prompt = f"""Decompose this task into subtasks for team members.

Task: {description}

Team members:
{chr(10).join(member_specs)}

Return JSON array with one subtask per member:
["Subtask for member 1", "Subtask for member 2", ...]"""

        try:
            if asyncio.iscoroutinefunction(self.llm_fn):
                response = await self.llm_fn(prompt)
            else:
                response = self.llm_fn(prompt)

            # Extract JSON
            start = response.find('[')
            end = response.rfind(']') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])

        except Exception as e:
            logger.error(f"Task decomposition failed: {e}")

        # Fallback: equal distribution
        return [f"Part of: {description}" for _ in team.member_ids]

    async def disband_team(self, team_id: str) -> bool:
        """Disband a team and release members."""
        if team_id not in self.teams:
            return False

        team = self.teams[team_id]

        async with self._lock:
            team.status = TeamStatus.DISBANDED
            team.disbanded_at = datetime.now()

            for member_id in team.member_ids:
                if member_id in self.agents:
                    self.agents[member_id].status_message = ""

        logger.info(f"Disbanded team {team_id}")
        return True

    def get_hierarchy_tree(self) -> dict:
        """Get the full hierarchy as a tree structure."""
        def build_subtree(agent_id: str) -> dict:
            agent = self.agents.get(agent_id)
            if not agent:
                return {}

            return {
                "id": agent.id,
                "name": agent.name,
                "role": agent.role.value,
                "workload": agent.workload,
                "subordinates": [
                    build_subtree(sub_id)
                    for sub_id in agent.subordinates
                ],
            }

        if self.executive_id:
            return build_subtree(self.executive_id)

        return {}

    def get_stats(self) -> dict:
        """Get team statistics."""
        by_role = defaultdict(int)
        for agent in self.agents.values():
            by_role[agent.role.value] += 1

        active_teams = sum(1 for t in self.teams.values() if t.status == TeamStatus.ACTIVE)
        pending_tasks = sum(1 for a in self.assignments.values() if a.status == "pending")

        return {
            "total_agents": len(self.agents),
            "agents_by_role": dict(by_role),
            "total_teams": len(self.teams),
            "active_teams": active_teams,
            "total_tasks": len(self.assignments),
            "pending_tasks": pending_tasks,
            "avg_workload": sum(a.workload for a in self.agents.values()) / len(self.agents)
                if self.agents else 0,
        }
