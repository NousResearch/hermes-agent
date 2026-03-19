"""
cosmos Agents Module

LangGraph-based agent swarm with:
- Dynamic specialist spawning and handoffs
- Emergent behavior through genetic optimization
- User avatar sub-agent for preference modeling
- Meta-cognitive reflection loops

Q2 2025 Additions:
- Planner Agent for task decomposition
- Critic Agent for quality assurance
- Web Agent for intelligent browsing
- File System Agent for codebase navigation
- Agent Debates for multi-perspective synthesis
- Specialization Learning for skill development
- Hierarchical Teams for coordinated execution
"""

from Cosmos.agents.base_agent import BaseAgent, AgentState, AgentCapability
from Cosmos.agents.swarm_orchestrator import SwarmOrchestrator
from Cosmos.agents.specialist_agents import (
    CodeAgent,
    ReasoningAgent,
    ResearchAgent,
    CreativeAgent,
)
from Cosmos.agents.user_avatar import UserAvatar
from Cosmos.agents.meta_cognition import MetaCognitionAgent

# Q2 2025 - Advanced Agents
from Cosmos.agents.planner_agent import (
    PlannerAgent,
    Plan,
    SubTask,
    TaskStatus,
    TaskPriority,
)
from Cosmos.agents.critic_agent import (
    CriticAgent,
    Review,
    QualityScore,
    QualityDimension,
    ReviewType,
    RefinementResult,
)
from Cosmos.agents.web_agent import (
    WebAgent,
    BrowsingSession,
    PageState,
    WebAction,
    ActionType,
)
from Cosmos.agents.filesystem_agent import (
    FileSystemAgent,
    FileInfo,
    ProjectStructure,
    FileChange,
    SearchResult as FileSearchResult,
)
from Cosmos.agents.proactive_agent import ProactiveAgent, Suggestion

# Q2 2025 - Agent Collaboration
from Cosmos.agents.agent_debates import (
    AgentDebates,
    Debate,
    Argument,
    DebateRole,
    ArgumentType,
)
from Cosmos.agents.specialization_learning import (
    SpecializationLearning,
    AgentProfile,
    Skill,
    SkillLevel,
    TaskOutcome,
)
from Cosmos.agents.hierarchical_teams import (
    HierarchicalTeams,
    Team,
    AgentNode,
    AgentRole,
    TeamStatus,
    TaskAssignment,
)

__all__ = [
    # Core agents
    "BaseAgent",
    "AgentState",
    "AgentCapability",
    "SwarmOrchestrator",
    "CodeAgent",
    "ReasoningAgent",
    "ResearchAgent",
    "CreativeAgent",
    "UserAvatar",
    "MetaCognitionAgent",
    # Q2 2025 - New Agent Types
    "PlannerAgent",
    "Plan",
    "SubTask",
    "TaskStatus",
    "TaskPriority",
    "CriticAgent",
    "Review",
    "QualityScore",
    "QualityDimension",
    "ReviewType",
    "RefinementResult",
    "WebAgent",
    "BrowsingSession",
    "PageState",
    "WebAction",
    "ActionType",
    "FileSystemAgent",
    "FileInfo",
    "ProjectStructure",
    "FileChange",
    "FileSearchResult",
    # Q2 2025 - Agent Collaboration
    "AgentDebates",
    "Debate",
    "Argument",
    "DebateRole",
    "ArgumentType",
    "SpecializationLearning",
    "AgentProfile",
    "Skill",
    "SkillLevel",
    "TaskOutcome",
    "HierarchicalTeams",
    "Team",
    "AgentNode",
    "AgentRole",
    "TeamStatus",
    "TaskAssignment",
    # Q2 2025 - Proactive Intelligence
    "ProactiveAgent",
    "Suggestion",
]
