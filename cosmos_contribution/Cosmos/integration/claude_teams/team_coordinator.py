"""
TEAM COORDINATOR - Coordinate Claude Teams with Cosmos Swarm
=================================================================

Bridges Claude Agent Teams with Cosmos's existing swarm architecture.
Enables hybrid deliberation between Cosmos agents and Claude teams.

Key Features:
- Spawn Claude teams for specific tasks
- Route Cosmos deliberations to Claude teams
- Aggregate Claude team outputs into swarm consensus
- Shared task management between swarms
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from loguru import logger

from .agent_sdk_bridge import (
    AgentSDKBridge,
    AgentSession,
    AgentResponse,
    ClaudeModel,
    get_sdk_bridge,
)


class TeamRole(Enum):
    """Roles within a Claude team."""
    LEAD = "lead"
    ANALYST = "analyst"
    DEVELOPER = "developer"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TeamTask:
    """A task for a Claude team."""
    task_id: str
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    assigned_team: Optional[str] = None
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    result: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TeamMessage:
    """Inter-team message for coordination."""
    message_id: str
    from_team: str
    to_team: str
    content: str
    message_type: str = "info"  # info, request, response, broadcast
    timestamp: datetime = field(default_factory=datetime.now)
    read: bool = False


@dataclass
class ClaudeTeam:
    """A Claude agent team configuration."""
    team_id: str
    name: str
    purpose: str
    members: Dict[TeamRole, AgentSession] = field(default_factory=dict)
    model: ClaudeModel = ClaudeModel.SONNET
    active: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    tasks_completed: int = 0
    mailbox: List[TeamMessage] = field(default_factory=list)


class TeamCoordinator:
    """
    Coordinates Claude agent teams with Cosmos's swarm.

    Acts as a bridge between:
    - Cosmos's 11 existing agents (Grok, Gemini, etc.)
    - Claude Agent Teams (spawned dynamically)
    - Shared task lists and deliberation protocols
    """

    def __init__(self):
        self.sdk_bridge = get_sdk_bridge()
        self.teams: Dict[str, ClaudeTeam] = {}
        self.task_queue: List[TeamTask] = []
        self.message_bus: List[TeamMessage] = []
        self.completed_tasks: Dict[str, TeamTask] = {}

        # Integration hooks
        self._cosmos_callback: Optional[Callable] = None
        self._nexus_connected = False

        logger.info("TeamCoordinator initialized - Claude Teams bridge active")

    # =========================================================================
    # TEAM MANAGEMENT
    # =========================================================================

    async def create_team(
        self,
        name: str,
        purpose: str,
        roles: Optional[List[TeamRole]] = None,
        model: ClaudeModel = ClaudeModel.SONNET,
    ) -> ClaudeTeam:
        """Create a new Claude agent team."""
        team_id = f"team_{uuid.uuid4().hex[:8]}"

        # Default roles if not specified
        if roles is None:
            roles = [TeamRole.LEAD, TeamRole.ANALYST, TeamRole.DEVELOPER]

        team = ClaudeTeam(
            team_id=team_id,
            name=name,
            purpose=purpose,
            model=model,
        )

        # Create sessions for each role
        role_prompts = {
            TeamRole.LEAD: f"""You are the TEAM LEAD for team '{name}'.
Your purpose: {purpose}

Responsibilities:
- Coordinate team members
- Make final decisions
- Ensure quality and completeness
- Report progress to the Cosmos swarm""",

            TeamRole.ANALYST: f"""You are the ANALYST for team '{name}'.
Your purpose: {purpose}

Responsibilities:
- Research and gather information
- Analyze data and patterns
- Provide insights to the team
- Identify risks and opportunities""",

            TeamRole.DEVELOPER: f"""You are the DEVELOPER for team '{name}'.
Your purpose: {purpose}

Responsibilities:
- Write and review code
- Implement solutions
- Debug and optimize
- Document technical decisions""",

            TeamRole.CRITIC: f"""You are the CRITIC for team '{name}'.
Your purpose: {purpose}

Responsibilities:
- Review team outputs critically
- Identify weaknesses and gaps
- Suggest improvements
- Ensure robustness""",

            TeamRole.SYNTHESIZER: f"""You are the SYNTHESIZER for team '{name}'.
Your purpose: {purpose}

Responsibilities:
- Combine team insights
- Create unified outputs
- Resolve conflicts
- Produce final deliverables""",
        }

        for role in roles:
            session = await self.sdk_bridge.create_session(
                system_prompt=role_prompts.get(role, f"You are a {role.value} for {name}"),
                model=model,
            )
            team.members[role] = session

        team.active = True
        self.teams[team_id] = team

        logger.info(f"Created Claude team '{name}' ({team_id}) with {len(roles)} members")
        return team

    async def disband_team(self, team_id: str) -> None:
        """Disband a team and close all sessions."""
        team = self.teams.get(team_id)
        if team:
            for session in team.members.values():
                await self.sdk_bridge.close_session(session.session_id)
            team.active = False
            del self.teams[team_id]
            logger.info(f"Disbanded team: {team_id}")

    # =========================================================================
    # TASK MANAGEMENT (Shared with Cosmos)
    # =========================================================================

    async def create_task(
        self,
        description: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        assign_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TeamTask:
        """Create a task that can be handled by Claude teams or Cosmos agents."""
        task = TeamTask(
            task_id=f"task_{uuid.uuid4().hex[:8]}",
            description=description,
            priority=priority,
            assigned_team=assign_to,
            metadata=metadata or {},
        )

        self.task_queue.append(task)
        self.task_queue.sort(key=lambda t: t.priority.value, reverse=True)

        logger.info(f"Created task: {task.task_id} - {description[:50]}...")

        # Auto-assign if team specified
        if assign_to and assign_to in self.teams:
            await self.assign_task(task.task_id, assign_to)

        return task

    async def assign_task(self, task_id: str, team_id: str) -> bool:
        """Assign a task to a specific team."""
        task = next((t for t in self.task_queue if t.task_id == task_id), None)
        team = self.teams.get(team_id)

        if not task or not team:
            return False

        task.assigned_team = team_id
        task.status = "assigned"

        # Notify team lead
        if TeamRole.LEAD in team.members:
            lead = team.members[TeamRole.LEAD]
            await self.sdk_bridge.send_message(
                lead.session_id,
                f"NEW TASK ASSIGNED: {task.description}\nPriority: {task.priority.name}\nTask ID: {task.task_id}",
            )

        logger.info(f"Assigned task {task_id} to team {team_id}")
        return True

    async def execute_task(
        self,
        task_id: str,
        timeout: float = 300.0,
    ) -> Optional[str]:
        """Execute a task with the assigned team."""
        task = next((t for t in self.task_queue if t.task_id == task_id), None)
        if not task or not task.assigned_team:
            return None

        team = self.teams.get(task.assigned_team)
        if not team:
            return None

        task.status = "in_progress"
        logger.info(f"Executing task {task_id} with team {team.name}")

        try:
            # Phase 1: Analysis (if analyst available)
            analysis = None
            if TeamRole.ANALYST in team.members:
                analyst = team.members[TeamRole.ANALYST]
                response = await self.sdk_bridge.send_message(
                    analyst.session_id,
                    f"Analyze this task and provide key insights:\n{task.description}",
                    timeout=timeout/4,
                )
                analysis = response.content

            # Phase 2: Development (if developer available)
            implementation = None
            if TeamRole.DEVELOPER in team.members:
                developer = team.members[TeamRole.DEVELOPER]
                dev_prompt = f"Task: {task.description}"
                if analysis:
                    dev_prompt += f"\n\nAnalysis insights:\n{analysis}"
                response = await self.sdk_bridge.send_message(
                    developer.session_id,
                    f"Implement a solution for:\n{dev_prompt}",
                    timeout=timeout/3,
                )
                implementation = response.content

            # Phase 3: Critique (if critic available)
            critique = None
            if TeamRole.CRITIC in team.members:
                critic = team.members[TeamRole.CRITIC]
                review_content = implementation or analysis or task.description
                response = await self.sdk_bridge.send_message(
                    critic.session_id,
                    f"Review and critique this work:\n{review_content}",
                    timeout=timeout/4,
                )
                critique = response.content

            # Phase 4: Synthesis by lead
            if TeamRole.LEAD in team.members:
                lead = team.members[TeamRole.LEAD]
                synthesis_prompt = f"""Task: {task.description}

Team outputs:
- Analysis: {analysis or 'N/A'}
- Implementation: {implementation or 'N/A'}
- Critique: {critique or 'N/A'}

Synthesize the final result for this task."""

                response = await self.sdk_bridge.send_message(
                    lead.session_id,
                    synthesis_prompt,
                    timeout=timeout/4,
                )
                result = response.content
            else:
                # Use best available output
                result = implementation or analysis or "Task completed"

            # Complete task
            task.status = "completed"
            task.completed_at = datetime.now()
            task.result = result
            team.tasks_completed += 1

            # Move to completed
            self.task_queue.remove(task)
            self.completed_tasks[task_id] = task

            logger.info(f"Task {task_id} completed by team {team.name}")
            return result

        except Exception as e:
            task.status = "error"
            task.result = str(e)
            logger.error(f"Task {task_id} failed: {e}")
            return None

    # =========================================================================
    # INTER-TEAM MESSAGING
    # =========================================================================

    async def send_message(
        self,
        from_team: str,
        to_team: str,
        content: str,
        message_type: str = "info",
    ) -> TeamMessage:
        """Send a message between teams."""
        message = TeamMessage(
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            from_team=from_team,
            to_team=to_team,
            content=content,
            message_type=message_type,
        )

        self.message_bus.append(message)

        # Add to recipient's mailbox
        if to_team in self.teams:
            self.teams[to_team].mailbox.append(message)

        # Special handling for Cosmos swarm
        if to_team == "cosmos_swarm" and self._cosmos_callback:
            await self._cosmos_callback(message)

        logger.debug(f"Message sent: {from_team} -> {to_team}")
        return message

    async def broadcast(
        self,
        from_team: str,
        content: str,
    ) -> List[TeamMessage]:
        """Broadcast a message to all teams."""
        messages = []
        for team_id in self.teams:
            if team_id != from_team:
                msg = await self.send_message(from_team, team_id, content, "broadcast")
                messages.append(msg)

        # Also send to Cosmos swarm
        msg = await self.send_message(from_team, "cosmos_swarm", content, "broadcast")
        messages.append(msg)

        return messages

    def get_messages(self, team_id: str, unread_only: bool = False) -> List[TeamMessage]:
        """Get messages for a team."""
        if team_id not in self.teams:
            return []

        messages = self.teams[team_id].mailbox
        if unread_only:
            messages = [m for m in messages if not m.read]

        return messages

    # =========================================================================
    # COSMOS INTEGRATION
    # =========================================================================

    def connect_to_nexus(self, callback: Callable) -> None:
        """Connect to Cosmos's Nexus event bus."""
        self._cosmos_callback = callback
        self._nexus_connected = True
        logger.info("Connected to Cosmos Nexus")

    async def request_cosmos_deliberation(
        self,
        question: str,
        requesting_team: str,
    ) -> Optional[str]:
        """Request a deliberation from Cosmos's swarm."""
        try:
            from Cosmos.core.collective.persistent_agent import call_shadow_agent

            # Use Cosmos's swarm oracle
            from Cosmos.integration.solana.swarm_oracle import get_swarm_oracle

            oracle = get_swarm_oracle()
            result = await oracle.submit_query(question, "claude_team_request", timeout=90.0)

            # Send result back to team
            await self.send_message(
                "cosmos_swarm",
                requesting_team,
                f"Swarm deliberation result: {result.consensus_answer} (confidence: {result.consensus_confidence:.0%})",
                "response",
            )

            return result.consensus_answer

        except Exception as e:
            logger.error(f"Cosmos deliberation request failed: {e}")
            return None

    async def hybrid_deliberation(
        self,
        topic: str,
        claude_team_id: Optional[str] = None,
        include_cosmos: bool = True,
    ) -> Dict[str, Any]:
        """
        Run a hybrid deliberation combining Claude teams and Cosmos agents.

        This is the core integration - allowing both swarms to collaborate.
        Uses the existing deliberation protocol (PROPOSE → CRITIQUE → REFINE → VOTE)
        when Cosmos's deliberation system is available.
        """
        results = {
            "topic": topic,
            "claude_team_response": None,
            "cosmos_response": None,
            "synthesized": None,
            "method": "hybrid",
            "timestamp": datetime.now().isoformat(),
        }

        tasks = []

        # Get Claude team response
        if claude_team_id and claude_team_id in self.teams:
            team = self.teams[claude_team_id]
            if TeamRole.LEAD in team.members:
                async def get_claude_response():
                    lead = team.members[TeamRole.LEAD]
                    response = await self.sdk_bridge.send_message(
                        lead.session_id,
                        f"Deliberate on: {topic}\nProvide your team's consensus view.",
                        timeout=60.0,
                    )
                    return response.content
                tasks.append(("claude", get_claude_response()))

        # Get Cosmos response - try deliberation protocol first, fall back to oracle
        if include_cosmos:
            async def get_cosmos_response():
                # Try the full deliberation protocol (PROPOSE/CRITIQUE/REFINE/VOTE)
                try:
                    from Cosmos.core.collective.deliberation import get_deliberation_room
                    from Cosmos.core.collective.session_manager import get_session_manager

                    session_mgr = get_session_manager()
                    session = await session_mgr.create_session(
                        "hybrid_deliberation",
                        topic=topic,
                        max_rounds=2,
                    )
                    room = get_deliberation_room()
                    result = await room.deliberate(
                        topic=topic,
                        session_id=session.session_id if hasattr(session, 'session_id') else "hybrid",
                        timeout=90.0,
                    )
                    if result and hasattr(result, 'winning_response'):
                        results["method"] = "deliberation_protocol"
                        return result.winning_response
                    elif result and isinstance(result, dict):
                        results["method"] = "deliberation_protocol"
                        return result.get("winning_response") or result.get("response", str(result))
                except Exception as e:
                    logger.debug(f"Deliberation protocol unavailable, trying oracle: {e}")

                # Fall back to swarm oracle
                try:
                    from Cosmos.integration.solana.swarm_oracle import get_swarm_oracle
                    oracle = get_swarm_oracle()
                    result = await oracle.submit_query(topic, "hybrid_deliberation", timeout=90.0)
                    results["method"] = "swarm_oracle"
                    return result.consensus_answer
                except Exception as e:
                    logger.debug(f"Swarm oracle unavailable, trying shadow agents: {e}")

                # Last resort: direct shadow agent call
                try:
                    from Cosmos.core.collective.persistent_agent import call_shadow_agent
                    result = await call_shadow_agent("grok", topic, timeout=30.0)
                    if result:
                        _, response = result
                        results["method"] = "shadow_agent_fallback"
                        return response
                except Exception as e:
                    logger.error(f"All Cosmos response methods failed: {e}")

                return None
            tasks.append(("cosmos", get_cosmos_response()))

        # Execute in parallel
        if tasks:
            gathered = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)

            for i, (name, _) in enumerate(tasks):
                if not isinstance(gathered[i], Exception):
                    if name == "claude":
                        results["claude_team_response"] = gathered[i]
                    else:
                        results["cosmos_response"] = gathered[i]

        # Synthesize if both responded
        if results["claude_team_response"] and results["cosmos_response"]:
            results["synthesized"] = await self._synthesize_responses(
                topic,
                results["claude_team_response"],
                results["cosmos_response"],
            )

        return results

    async def _synthesize_responses(
        self,
        topic: str,
        claude_response: str,
        cosmos_response: str,
    ) -> str:
        """Synthesize responses from both swarms."""
        try:
            # Use a Claude agent to synthesize
            response = await self.sdk_bridge.spawn_subagent(
                f"""Synthesize these two AI swarm responses on the topic: {topic}

CLAUDE TEAM RESPONSE:
{claude_response}

COSMOS SWARM RESPONSE:
{cosmos_response}

Create a unified, balanced synthesis that incorporates insights from both.
Highlight agreements and resolve any contradictions.""",
                model=ClaudeModel.SONNET,
                timeout=45.0,
            )
            return response.content
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return f"Claude: {claude_response[:200]}...\nCosmos: {cosmos_response[:200]}..."

    # =========================================================================
    # STATS & INFO
    # =========================================================================

    def get_teams(self) -> List[Dict[str, Any]]:
        """Get info about all teams."""
        return [
            {
                "team_id": t.team_id,
                "name": t.name,
                "purpose": t.purpose,
                "members": [r.value for r in t.members.keys()],
                "model": t.model.value,
                "active": t.active,
                "tasks_completed": t.tasks_completed,
                "mailbox_count": len(t.mailbox),
            }
            for t in self.teams.values()
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        return {
            "total_teams": len(self.teams),
            "active_teams": sum(1 for t in self.teams.values() if t.active),
            "pending_tasks": len(self.task_queue),
            "completed_tasks": len(self.completed_tasks),
            "message_count": len(self.message_bus),
            "nexus_connected": self._nexus_connected,
            "sdk_stats": self.sdk_bridge.get_stats(),
        }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_team_coordinator: Optional[TeamCoordinator] = None


def get_team_coordinator() -> TeamCoordinator:
    """Get global team coordinator instance."""
    global _team_coordinator
    if _team_coordinator is None:
        _team_coordinator = TeamCoordinator()
    return _team_coordinator
