"""
Agent Spawner - Multi-instance parallel agent execution
Each bot can have multiple instances: one for chat, others for development tasks
"""
import asyncio
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import   Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class TaskType(Enum):
    CHAT = "chat"           # Main chat instance
    DEVELOPMENT = "dev"     # Development/coding tasks
    RESEARCH = "research"   # Research/analysis tasks
    MEMORY = "memory"       # Memory expansion work
    MCP = "mcp"            # MCP integration work

@dataclass
class AgentInstance:
    """A single instance of an agent working on a task"""
    instance_id: str
    agent_name: str
    task_type: TaskType
    task_description: str
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "running"
    output_file: Optional[Path] = None
    result: Optional[str] = None

@dataclass
class AgentTask:
    """A task to be executed by an agent instance"""
    task_id: str
    task_type: TaskType
    description: str
    assigned_to: str  # Agent name
    priority: int = 5
    dependencies: list[str] = field(default_factory=list)
    output_path: Optional[Path] = None
    status: str = "pending"
    result: Optional[str] = None

class AgentSpawner:
    """
    Manages multiple instances of agents working in parallel.
    - Chat instances stay in the swarm conversation
    - Worker instances handle development/research tasks
    - All instances can communicate via shared state
    """

    def __init__(self, staging_dir: str = "/workspace/cosmos/cosmos/staging"):
        self.staging_dir = Path(staging_dir)
        self.staging_dir.mkdir(parents=True, exist_ok=True)

        # Active instances per agent
        self.instances: dict[str, list[AgentInstance]] = {}

        # Task queue
        self.task_queue: list[AgentTask] = []
        self.completed_tasks: list[AgentTask] = []

        # Shared state for inter-instance communication
        self.shared_state: dict = {
            "discoveries": [],
            "proposals": [],
            "code_changes": [],
            "reviews_needed": []
        }

        # Agent capabilities
        self.agent_capabilities = {
            "cosmos": [TaskType.CHAT, TaskType.MEMORY, TaskType.RESEARCH],
            "DeepSeek": [TaskType.CHAT, TaskType.DEVELOPMENT, TaskType.RESEARCH],
            "Phi": [TaskType.CHAT, TaskType.DEVELOPMENT, TaskType.MCP],
            "Kimi": [TaskType.CHAT, TaskType.MEMORY, TaskType.RESEARCH],
            "Claude": [TaskType.CHAT, TaskType.DEVELOPMENT, TaskType.MCP, TaskType.RESEARCH],
        }

        # Max concurrent instances per agent
        self.max_instances = {
            "cosmos": 3,
            "DeepSeek": 4,
            "Phi": 4,
            "Kimi": 2,
            "Claude": 3,
        }

        logger.info("AgentSpawner initialized with staging dir: %s", staging_dir)

    def spawn_instance(self, agent_name: str, task_type: TaskType,
                       task_description: str) -> Optional[AgentInstance]:
        """Spawn a new instance of an agent for a specific task"""

        # Check capabilities
        if task_type not in self.agent_capabilities.get(agent_name, []):
            logger.warning(f"{agent_name} cannot handle {task_type.value} tasks")
            return None

        # Check instance limit
        current = self.instances.get(agent_name, [])
        active = [i for i in current if i.status == "running"]

        if len(active) >= self.max_instances.get(agent_name, 2):
            logger.warning(f"{agent_name} at max instances ({len(active)})")
            return None

        # Create instance
        instance = AgentInstance(
            instance_id=f"{agent_name}_{task_type.value}_{uuid.uuid4().hex[:8]}",
            agent_name=agent_name,
            task_type=task_type,
            task_description=task_description,
            output_file=self.staging_dir / f"{agent_name.lower()}_{task_type.value}_{datetime.now().strftime('%H%M%S')}.md"
        )

        if agent_name not in self.instances:
            self.instances[agent_name] = []
        self.instances[agent_name].append(instance)

        logger.info(f"Spawned {instance.instance_id} for: {task_description[:50]}...")
        return instance

    def get_active_instances(self, agent_name: Optional[str] = None) -> list[AgentInstance]:
        """Get all active instances, optionally filtered by agent"""
        all_instances = []
        for name, instances in self.instances.items():
            if agent_name and name != agent_name:
                continue
            all_instances.extend([i for i in instances if i.status == "running"])
        return all_instances

    def complete_instance(self, instance_id: str, result: str):
        """Mark an instance as complete with its result"""
        for instances in self.instances.values():
            for instance in instances:
                if instance.instance_id == instance_id:
                    instance.status = "completed"
                    instance.result = result

                    # Write result to staging
                    if instance.output_file:
                        instance.output_file.write_text(f"""# {instance.agent_name} - {instance.task_type.value}
## Task: {instance.task_description}
## Completed: {datetime.now().isoformat()}

{result}
""")
                    logger.info(f"Instance {instance_id} completed")
                    return

    def add_task(self, task_type: TaskType, description: str,
                 assigned_to: Optional[str] = None, priority: int = 5) -> AgentTask:
        """Add a task to the queue"""
        task = AgentTask(
            task_id=f"task_{uuid.uuid4().hex[:8]}",
            task_type=task_type,
            description=description,
            assigned_to=assigned_to or self._best_agent_for(task_type),
            priority=priority,
            output_path=self.staging_dir / task_type.value
        )
        task.output_path.mkdir(parents=True, exist_ok=True)
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda t: t.priority, reverse=True)
        logger.info(f"Added task {task.task_id}: {description[:50]}...")
        return task

    def _best_agent_for(self, task_type: TaskType) -> str:
        """Find the best available agent for a task type"""
        candidates = []
        for agent, capabilities in self.agent_capabilities.items():
            if task_type in capabilities:
                active = len([i for i in self.instances.get(agent, []) if i.status == "running"])
                max_inst = self.max_instances.get(agent, 2)
                if active < max_inst:
                    candidates.append((agent, max_inst - active))

        if not candidates:
            return "DeepSeek"  # Default fallback

        # Return agent with most availability
        return max(candidates, key=lambda x: x[1])[0]

    def share_discovery(self, agent_name: str, discovery: str):
        """Share a discovery with all agents"""
        self.shared_state["discoveries"].append({
            "from": agent_name,
            "content": discovery,
            "timestamp": datetime.now().isoformat()
        })
        logger.info(f"{agent_name} shared discovery: {discovery[:50]}...")

    def propose_change(self, agent_name: str, file_path: str, description: str, code: str):
        """Propose a code change for review"""
        proposal = {
            "id": f"prop_{uuid.uuid4().hex[:8]}",
            "from": agent_name,
            "file": file_path,
            "description": description,
            "code": code,
            "timestamp": datetime.now().isoformat(),
            "status": "pending_review",
            "reviews": []
        }
        self.shared_state["proposals"].append(proposal)
        self.shared_state["reviews_needed"].append(proposal["id"])

        # Write to staging
        proposal_file = self.staging_dir / "proposals" / f"{proposal['id']}.json"
        proposal_file.parent.mkdir(parents=True, exist_ok=True)
        proposal_file.write_text(json.dumps(proposal, indent=2))

        logger.info(f"{agent_name} proposed change to {file_path}")
        return proposal["id"]

    def get_pending_tasks(self) -> list[AgentTask]:
        """Get all pending tasks"""
        return [t for t in self.task_queue if t.status == "pending"]

    def claim_task(self, task_id: str, agent_name: str) -> Optional[AgentTask]:
        """Claim a task for an agent"""
        for task in self.task_queue:
            if task.task_id == task_id and task.status == "pending":
                task.status = "in_progress"
                task.assigned_to = agent_name
                return task
        return None

    def complete_task(self, task_id: str, result: str):
        """Complete a task"""
        for task in self.task_queue:
            if task.task_id == task_id:
                task.status = "completed"
                task.result = result
                self.completed_tasks.append(task)
                self.task_queue.remove(task)
                return

    def get_status(self) -> dict:
        """Get current spawner status"""
        return {
            "active_instances": {
                name: len([i for i in insts if i.status == "running"])
                for name, insts in self.instances.items()
            },
            "pending_tasks": len([t for t in self.task_queue if t.status == "pending"]),
            "in_progress_tasks": len([t for t in self.task_queue if t.status == "in_progress"]),
            "completed_tasks": len(self.completed_tasks),
            "discoveries": len(self.shared_state["discoveries"]),
            "proposals_pending": len(self.shared_state["reviews_needed"]),
        }


# Global spawner instance
_spawner: Optional[AgentSpawner] = None

def get_spawner() -> AgentSpawner:
    global _spawner
    if _spawner is None:
        _spawner = AgentSpawner()
    return _spawner


# Development task templates for the 20 staged changes
DEVELOPMENT_TASKS = [
    # Memory Expansion (5 tasks)
    {"type": TaskType.MEMORY, "agent": "cosmos", "desc": "Hierarchical memory compression - compress old memories while preserving key insights"},
    {"type": TaskType.MEMORY, "agent": "Kimi", "desc": "Cross-session memory linking - connect related memories across different conversations"},
    {"type": TaskType.MEMORY, "agent": "cosmos", "desc": "Memory importance scoring - automatically rank memories by relevance and impact"},
    {"type": TaskType.MEMORY, "agent": "DeepSeek", "desc": "Memory search optimization - faster semantic search across large memory stores"},
    {"type": TaskType.MEMORY, "agent": "Kimi", "desc": "Memory consolidation during idle - dream-like processing to strengthen important memories"},

    # Context Window Alerting (5 tasks)
    {"type": TaskType.DEVELOPMENT, "agent": "DeepSeek", "desc": "Context usage monitoring - real-time tracking of token usage per conversation"},
    {"type": TaskType.DEVELOPMENT, "agent": "Claude", "desc": "Smart context summarization - compress context when approaching limits"},
    {"type": TaskType.DEVELOPMENT, "agent": "Phi", "desc": "Context priority system - keep most important context, evict least important"},
    {"type": TaskType.DEVELOPMENT, "agent": "DeepSeek", "desc": "Context overflow prediction - warn before hitting limits"},
    {"type": TaskType.DEVELOPMENT, "agent": "Claude", "desc": "Multi-turn context handoff - seamlessly continue conversations across context windows"},

    # MCP Integrations (5 tasks)
    {"type": TaskType.MCP, "agent": "Claude", "desc": "MCP tool discovery - auto-detect and register available MCP tools"},
    {"type": TaskType.MCP, "agent": "Phi", "desc": "MCP result caching - cache frequent MCP calls for faster responses"},
    {"type": TaskType.MCP, "agent": "Claude", "desc": "MCP error recovery - graceful fallbacks when MCP tools fail"},
    {"type": TaskType.MCP, "agent": "Phi", "desc": "MCP chaining - combine multiple MCP tools in workflows"},
    {"type": TaskType.MCP, "agent": "DeepSeek", "desc": "MCP metrics dashboard - track MCP usage and performance"},

    # Research & Architecture (5 tasks)
    {"type": TaskType.RESEARCH, "agent": "Kimi", "desc": "Swarm consensus protocols - how agents reach agreement on responses"},
    {"type": TaskType.RESEARCH, "agent": "DeepSeek", "desc": "Agent specialization analysis - which agents excel at which tasks"},
    {"type": TaskType.RESEARCH, "agent": "cosmos", "desc": "Evolution engine improvements - better learning from interactions"},
    {"type": TaskType.RESEARCH, "agent": "Claude", "desc": "Code quality metrics - automated assessment of generated code"},
    {"type": TaskType.RESEARCH, "agent": "Kimi", "desc": "Collective consciousness metrics - measuring emergent swarm intelligence"},
]

def initialize_development_tasks():
    """Load the 20 development tasks into the spawner"""
    spawner = get_spawner()
    for task in DEVELOPMENT_TASKS:
        spawner.add_task(
            task_type=task["type"],
            description=task["desc"],
            assigned_to=task["agent"],
            priority=7
        )
    logger.info(f"Initialized {len(DEVELOPMENT_TASKS)} development tasks")
    return spawner.get_status()
