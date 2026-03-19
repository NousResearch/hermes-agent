"""
cosmos Agent Specialization Learning - Adaptive Skill Development

Novel Approaches:
1. Performance Tracking - Monitor success across task types
2. Skill Discovery - Identify emergent capabilities
3. Adaptive Routing - Route tasks to best-suited agents
4. Transfer Learning - Share skills between agents
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any, Callable
import json
from collections import defaultdict
import math

from loguru import logger


class SkillLevel(Enum):
    """Proficiency levels."""
    NOVICE = 1
    BEGINNER = 2
    INTERMEDIATE = 3
    ADVANCED = 4
    EXPERT = 5


@dataclass
class TaskOutcome:
    """Outcome of a task attempt."""
    task_id: str
    task_type: str
    agent_id: str
    success: bool
    score: float  # 0.0 to 1.0
    duration_seconds: float
    timestamp: datetime = field(default_factory=datetime.now)

    # Feedback
    user_rating: Optional[float] = None
    error_type: Optional[str] = None

    # Context
    complexity: float = 1.0
    required_skills: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "agent_id": self.agent_id,
            "success": self.success,
            "score": self.score,
            "complexity": self.complexity,
        }


@dataclass
class Skill:
    """A skill that an agent can have."""
    id: str
    name: str
    category: str  # "code", "reasoning", "creativity", etc.
    description: str = ""

    # Related skills
    prerequisites: list[str] = field(default_factory=list)
    related_skills: list[str] = field(default_factory=list)

    # Learning
    practice_count: int = 0
    success_count: int = 0

    def success_rate(self) -> float:
        if self.practice_count == 0:
            return 0.0
        return self.success_count / self.practice_count


@dataclass
class AgentProfile:
    """Profile of an agent's capabilities."""
    agent_id: str
    agent_type: str

    # Skills and proficiency
    skills: dict[str, float] = field(default_factory=dict)  # skill_id -> proficiency
    skill_levels: dict[str, SkillLevel] = field(default_factory=dict)

    # Performance history
    task_history: list[str] = field(default_factory=list)  # Task IDs
    success_rate: float = 0.0
    avg_score: float = 0.0

    # Specializations
    specializations: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)

    # Learning rate
    learning_rate: float = 0.1
    adaptation_speed: float = 1.0

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    def get_level(self, skill_id: str) -> SkillLevel:
        proficiency = self.skills.get(skill_id, 0.0)
        if proficiency >= 0.9:
            return SkillLevel.EXPERT
        elif proficiency >= 0.7:
            return SkillLevel.ADVANCED
        elif proficiency >= 0.5:
            return SkillLevel.INTERMEDIATE
        elif proficiency >= 0.3:
            return SkillLevel.BEGINNER
        return SkillLevel.NOVICE

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "skills": {k: round(v, 2) for k, v in self.skills.items()},
            "specializations": self.specializations,
            "success_rate": round(self.success_rate, 2),
            "avg_score": round(self.avg_score, 2),
        }


@dataclass
class SkillTransfer:
    """Record of skill transfer between agents."""
    source_agent: str
    target_agent: str
    skill_id: str
    transfer_amount: float
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = False


class SpecializationLearning:
    """
    Agent skill development and specialization system.

    Features:
    - Track agent performance across task types
    - Automatically discover specializations
    - Route tasks to optimal agents
    - Enable skill transfer between agents
    """

    def __init__(
        self,
        skill_decay_rate: float = 0.01,
        specialization_threshold: float = 0.8,
    ):
        self.skill_decay_rate = skill_decay_rate
        self.specialization_threshold = specialization_threshold

        self.agents: dict[str, AgentProfile] = {}
        self.skills: dict[str, Skill] = {}
        self.outcomes: list[TaskOutcome] = []

        # Task type to skills mapping
        self.task_skill_map: dict[str, list[str]] = {}

        # Performance matrices
        self._agent_task_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self):
        """Initialize with default skills."""
        if self._initialized:
            return

        # Define core skills
        default_skills = [
            Skill(id="python", name="Python Programming", category="code"),
            Skill(id="javascript", name="JavaScript Programming", category="code"),
            Skill(id="debugging", name="Code Debugging", category="code", prerequisites=["python"]),
            Skill(id="logic", name="Logical Reasoning", category="reasoning"),
            Skill(id="math", name="Mathematical Reasoning", category="reasoning"),
            Skill(id="research", name="Information Research", category="research"),
            Skill(id="synthesis", name="Information Synthesis", category="research"),
            Skill(id="writing", name="Technical Writing", category="creative"),
            Skill(id="planning", name="Task Planning", category="organization"),
            Skill(id="analysis", name="Data Analysis", category="analysis"),
        ]

        for skill in default_skills:
            self.skills[skill.id] = skill

        # Map task types to skills
        self.task_skill_map = {
            "code": ["python", "javascript", "debugging"],
            "reasoning": ["logic", "math"],
            "research": ["research", "synthesis"],
            "creative": ["writing"],
            "planning": ["planning", "analysis"],
        }

        self._initialized = True

    async def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        initial_skills: Optional[dict[str, float]] = None,
    ) -> AgentProfile:
        """Register a new agent with initial capabilities."""
        await self.initialize()

        if agent_id in self.agents:
            return self.agents[agent_id]

        profile = AgentProfile(
            agent_id=agent_id,
            agent_type=agent_type,
            skills=initial_skills or {},
        )

        # Set type-based initial skills
        type_skills = {
            "code": {"python": 0.5, "debugging": 0.3},
            "reasoning": {"logic": 0.5, "math": 0.4},
            "research": {"research": 0.5, "synthesis": 0.4},
            "creative": {"writing": 0.5},
            "general": {"logic": 0.3, "research": 0.3, "writing": 0.3},
        }

        base_skills = type_skills.get(agent_type, type_skills["general"])
        for skill_id, proficiency in base_skills.items():
            if skill_id not in profile.skills:
                profile.skills[skill_id] = proficiency

        self.agents[agent_id] = profile
        logger.info(f"Registered agent {agent_id} ({agent_type})")

        return profile

    async def record_outcome(
        self,
        task_id: str,
        task_type: str,
        agent_id: str,
        success: bool,
        score: float,
        duration_seconds: float = 0.0,
        complexity: float = 1.0,
        user_rating: Optional[float] = None,
    ) -> TaskOutcome:
        """Record the outcome of a task attempt."""
        await self.initialize()

        # Ensure agent exists
        if agent_id not in self.agents:
            await self.register_agent(agent_id, "unknown")

        # Get required skills for task type
        required_skills = self.task_skill_map.get(task_type, [])

        outcome = TaskOutcome(
            task_id=task_id,
            task_type=task_type,
            agent_id=agent_id,
            success=success,
            score=score,
            duration_seconds=duration_seconds,
            complexity=complexity,
            user_rating=user_rating,
            required_skills=required_skills,
        )

        async with self._lock:
            self.outcomes.append(outcome)
            self._agent_task_scores[agent_id][task_type].append(score)

            # Update agent profile
            await self._update_agent_skills(outcome)

        return outcome

    async def _update_agent_skills(self, outcome: TaskOutcome):
        """Update agent skills based on outcome."""
        profile = self.agents.get(outcome.agent_id)
        if not profile:
            return

        # Update skills used in task
        for skill_id in outcome.required_skills:
            if skill_id not in profile.skills:
                profile.skills[skill_id] = 0.1

            current = profile.skills[skill_id]

            # Learning formula: weighted update based on outcome
            if outcome.success:
                # Skill increases on success
                delta = profile.learning_rate * (1 - current) * outcome.score
                profile.skills[skill_id] = min(1.0, current + delta)
            else:
                # Small decrease on failure
                delta = profile.learning_rate * current * 0.1
                profile.skills[skill_id] = max(0.0, current - delta)

            # Update skill practice count
            if skill_id in self.skills:
                self.skills[skill_id].practice_count += 1
                if outcome.success:
                    self.skills[skill_id].success_count += 1

        # Update overall stats
        profile.task_history.append(outcome.task_id)

        scores = self._agent_task_scores[outcome.agent_id]
        all_scores = [s for task_scores in scores.values() for s in task_scores]

        if all_scores:
            profile.avg_score = sum(all_scores) / len(all_scores)
            profile.success_rate = sum(1 for o in self.outcomes if o.agent_id == outcome.agent_id and o.success) / len(all_scores)

        # Detect specializations
        await self._update_specializations(profile)

        profile.last_updated = datetime.now()

    async def _update_specializations(self, profile: AgentProfile):
        """Detect and update agent specializations."""
        profile.specializations = []
        profile.weaknesses = []

        for skill_id, proficiency in profile.skills.items():
            if proficiency >= self.specialization_threshold:
                profile.specializations.append(skill_id)
            elif proficiency < 0.3 and skill_id in self.task_skill_map.get(profile.agent_type, []):
                profile.weaknesses.append(skill_id)

    async def get_best_agent(
        self,
        task_type: str,
        available_agents: Optional[list[str]] = None,
        complexity: float = 1.0,
    ) -> Optional[str]:
        """
        Get the best agent for a task type.

        Returns agent_id of best-suited agent.
        """
        await self.initialize()

        candidates = available_agents or list(self.agents.keys())

        if not candidates:
            return None

        required_skills = self.task_skill_map.get(task_type, [])

        best_agent = None
        best_score = -1

        for agent_id in candidates:
            if agent_id not in self.agents:
                continue

            profile = self.agents[agent_id]

            # Calculate suitability score
            skill_score = sum(
                profile.skills.get(s, 0) for s in required_skills
            ) / len(required_skills) if required_skills else 0.5

            # Factor in past performance on similar tasks
            task_scores = self._agent_task_scores[agent_id].get(task_type, [])
            historical_score = sum(task_scores[-10:]) / len(task_scores[-10:]) if task_scores else 0.5

            # Combine scores
            total_score = 0.6 * skill_score + 0.4 * historical_score

            # Adjust for complexity
            if complexity > 2 and profile.avg_score < 0.6:
                total_score *= 0.8  # Penalize for complex tasks

            if total_score > best_score:
                best_score = total_score
                best_agent = agent_id

        return best_agent

    async def get_agent_ranking(
        self,
        task_type: str,
        available_agents: Optional[list[str]] = None,
    ) -> list[tuple[str, float]]:
        """Get ranked list of agents for a task type."""
        await self.initialize()

        candidates = available_agents or list(self.agents.keys())
        required_skills = self.task_skill_map.get(task_type, [])

        rankings = []

        for agent_id in candidates:
            if agent_id not in self.agents:
                continue

            profile = self.agents[agent_id]

            skill_score = sum(
                profile.skills.get(s, 0) for s in required_skills
            ) / len(required_skills) if required_skills else 0.5

            rankings.append((agent_id, skill_score))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    async def transfer_skill(
        self,
        source_agent: str,
        target_agent: str,
        skill_id: str,
        transfer_rate: float = 0.3,
    ) -> SkillTransfer:
        """
        Transfer a skill from one agent to another.

        Simulates teaching/learning between agents.
        """
        await self.initialize()

        transfer = SkillTransfer(
            source_agent=source_agent,
            target_agent=target_agent,
            skill_id=skill_id,
            transfer_amount=0.0,
        )

        if source_agent not in self.agents or target_agent not in self.agents:
            return transfer

        source = self.agents[source_agent]
        target = self.agents[target_agent]

        source_skill = source.skills.get(skill_id, 0)
        target_skill = target.skills.get(skill_id, 0)

        if source_skill <= target_skill:
            # Nothing to transfer
            return transfer

        # Transfer amount based on gap and rate
        gap = source_skill - target_skill
        transfer_amount = gap * transfer_rate * source_skill

        # Apply transfer
        async with self._lock:
            target.skills[skill_id] = min(1.0, target_skill + transfer_amount)

            # Slight decay for source (teaching takes effort)
            source.skills[skill_id] = max(0, source_skill - transfer_amount * 0.1)

        transfer.transfer_amount = transfer_amount
        transfer.success = True

        logger.info(f"Transferred {skill_id} skill ({transfer_amount:.2f}) from {source_agent} to {target_agent}")

        return transfer

    async def apply_skill_decay(self):
        """Apply skill decay to all agents (call periodically)."""
        async with self._lock:
            for profile in self.agents.values():
                for skill_id in profile.skills:
                    current = profile.skills[skill_id]
                    # Exponential decay
                    profile.skills[skill_id] = current * (1 - self.skill_decay_rate)

    async def discover_emergent_skills(
        self,
        agent_id: str,
        min_success_streak: int = 3,
    ) -> list[str]:
        """
        Discover skills an agent has developed through practice.

        Returns newly discovered skill IDs.
        """
        if agent_id not in self.agents:
            return []

        # Analyze recent task patterns
        agent_outcomes = [
            o for o in self.outcomes
            if o.agent_id == agent_id
        ][-50:]

        if len(agent_outcomes) < min_success_streak:
            return []

        # Look for success patterns in new task types
        discovered = []
        task_types_seen = defaultdict(list)

        for outcome in agent_outcomes:
            task_types_seen[outcome.task_type].append(outcome.success)

        for task_type, successes in task_types_seen.items():
            # Check for consistent success in unexpected areas
            required_skills = self.task_skill_map.get(task_type, [])

            for skill_id in required_skills:
                profile = self.agents[agent_id]
                current = profile.skills.get(skill_id, 0)

                # Count recent successes
                recent_success_rate = sum(successes[-min_success_streak:]) / min_success_streak

                # If performing well but skill is low, it's emergent
                if recent_success_rate > 0.8 and current < 0.5:
                    # Boost skill
                    profile.skills[skill_id] = max(current, recent_success_rate * 0.8)
                    discovered.append(skill_id)
                    logger.info(f"Agent {agent_id} discovered emergent skill: {skill_id}")

        return discovered

    async def suggest_learning_path(
        self,
        agent_id: str,
        target_skills: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Suggest a learning path for an agent.

        Returns ordered list of skills to develop.
        """
        if agent_id not in self.agents:
            return []

        profile = self.agents[agent_id]

        # Determine target skills
        if not target_skills:
            # Target weaknesses and prerequisites
            target_skills = profile.weaknesses.copy()

            # Add prerequisites of current specializations
            for spec in profile.specializations:
                if spec in self.skills:
                    for prereq in self.skills[spec].prerequisites:
                        if prereq not in target_skills and profile.skills.get(prereq, 0) < 0.7:
                            target_skills.append(prereq)

        if not target_skills:
            return []

        # Build learning path
        path = []

        for skill_id in target_skills:
            if skill_id not in self.skills:
                continue

            skill = self.skills[skill_id]
            current = profile.skills.get(skill_id, 0)

            # Check prerequisites
            prereqs_met = all(
                profile.skills.get(p, 0) >= 0.5
                for p in skill.prerequisites
            )

            path.append({
                "skill_id": skill_id,
                "skill_name": skill.name,
                "current_level": current,
                "target_level": 0.8,
                "prerequisites_met": prereqs_met,
                "missing_prerequisites": [
                    p for p in skill.prerequisites
                    if profile.skills.get(p, 0) < 0.5
                ],
                "estimated_practice": max(1, int((0.8 - current) * 20)),
            })

        # Sort by prerequisites and current level
        path.sort(key=lambda x: (not x["prerequisites_met"], x["current_level"]))

        return path

    def get_skill_distribution(self) -> dict:
        """Get distribution of skills across all agents."""
        distribution = defaultdict(list)

        for profile in self.agents.values():
            for skill_id, proficiency in profile.skills.items():
                distribution[skill_id].append(proficiency)

        return {
            skill_id: {
                "mean": sum(scores) / len(scores),
                "max": max(scores),
                "min": min(scores),
                "count": len(scores),
            }
            for skill_id, scores in distribution.items()
        }

    def get_stats(self) -> dict:
        """Get learning system statistics."""
        return {
            "total_agents": len(self.agents),
            "total_skills": len(self.skills),
            "total_outcomes": len(self.outcomes),
            "avg_success_rate": sum(a.success_rate for a in self.agents.values()) / len(self.agents)
                if self.agents else 0,
            "skill_distribution": self.get_skill_distribution(),
        }
