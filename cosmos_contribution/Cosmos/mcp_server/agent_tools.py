"""
cosmos Agent Tools - MCP Tool Implementations for Agent Operations

Provides agent swarm capabilities:
- Task delegation to specialists
- Agent status monitoring
- Multi-agent task coordination
"""

from dataclasses import dataclass
from typing import Optional, Any
from datetime import datetime

from loguru import logger


@dataclass
class AgentToolResult:
    """Result from an agent tool operation."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0

    def to_dict(self) -> dict:
        result = {"success": self.success}
        if self.data is not None:
            result["data"] = self.data
        if self.error:
            result["error"] = self.error
        result["execution_time"] = self.execution_time
        return result


class AgentTools:
    """
    Agent tool implementations for the MCP server.

    Provides access to the agent swarm for task delegation.
    """

    def __init__(self, swarm_orchestrator, model_manager=None):
        self.swarm = swarm_orchestrator
        self.model_manager = model_manager
        self._setup_agents()

    def _setup_agents(self):
        """Set up agent factories."""
        from Cosmos.agents.specialist_agents import (
            create_code_agent,
            create_reasoning_agent,
            create_research_agent,
            create_creative_agent,
        )

        self.swarm.register_agent_factory("code", create_code_agent)
        self.swarm.register_agent_factory("reasoning", create_reasoning_agent)
        self.swarm.register_agent_factory("research", create_research_agent)
        self.swarm.register_agent_factory("creative", create_creative_agent)

        # Set model manager
        if self.model_manager:
            self.swarm.llm_backend = self.model_manager

    async def delegate_task(
        self,
        task: str,
        agent_type: str = "auto",
        context: Optional[dict] = None,
        priority: int = 5,
        timeout: float = 120.0,
    ) -> AgentToolResult:
        """
        Delegate a task to a specialist agent.

        Args:
            task: Task description
            agent_type: Type of agent ("code", "reasoning", "research", "creative", "auto")
            context: Optional context dictionary
            priority: Task priority (1-10)
            timeout: Maximum wait time in seconds

        Returns:
            AgentToolResult with output and metadata
        """
        import time
        start_time = time.time()

        try:
            # Infer agent type if auto
            if agent_type == "auto":
                agent_type = self._infer_agent_type(task)

            # Spawn agent if needed
            agent = await self.swarm.spawn_agent(agent_type)
            if agent is None:
                # Try to use existing agent
                for existing in self.swarm.state.active_agents.values():
                    if existing.name.lower().startswith(agent_type):
                        agent = existing
                        break

            if agent is None:
                return AgentToolResult(
                    success=False,
                    error=f"Could not spawn agent of type: {agent_type}",
                    execution_time=time.time() - start_time,
                )

            # Execute task
            result = await agent.execute(task, context)

            return AgentToolResult(
                success=result.success,
                data={
                    "output": result.output,
                    "confidence": result.confidence,
                    "agent_type": agent_type,
                    "agent_id": agent.agent_id,
                    "metadata": result.metadata,
                },
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Delegate task failed: {e}")
            return AgentToolResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    def _infer_agent_type(self, task: str) -> str:
        """Infer the best agent type for a task."""
        task_lower = task.lower()

        if any(kw in task_lower for kw in ["code", "function", "implement", "debug", "fix bug"]):
            return "code"
        elif any(kw in task_lower for kw in ["reason", "think", "analyze", "math", "logic"]):
            return "reasoning"
        elif any(kw in task_lower for kw in ["research", "find", "search", "look up"]):
            return "research"
        elif any(kw in task_lower for kw in ["write", "creative", "story", "compose"]):
            return "creative"

        return "reasoning"  # Default

    async def execute_multi_task(
        self,
        main_task: str,
        subtasks: list[str],
        context: Optional[dict] = None,
    ) -> AgentToolResult:
        """
        Execute a main task with parallel subtasks.

        Args:
            main_task: Main task description
            subtasks: List of subtask descriptions
            context: Optional shared context

        Returns:
            AgentToolResult with combined results
        """
        import time
        start_time = time.time()

        try:
            results = await self.swarm.execute_with_subtasks(
                main_task=main_task,
                subtasks=subtasks,
                context=context,
            )

            return AgentToolResult(
                success=all(r.success for r in results),
                data={
                    "main_task": main_task,
                    "subtask_count": len(subtasks),
                    "results": [
                        {
                            "success": r.success,
                            "output": r.output[:500] if r.output else None,
                            "confidence": r.confidence,
                        }
                        for r in results
                    ],
                },
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Multi-task execution failed: {e}")
            return AgentToolResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    async def get_agent_status(self, agent_id: Optional[str] = None) -> AgentToolResult:
        """
        Get status of agents.

        Args:
            agent_id: Specific agent ID, or None for all agents

        Returns:
            AgentToolResult with agent status information
        """
        try:
            if agent_id:
                agent = self.swarm.state.active_agents.get(agent_id)
                if agent:
                    return AgentToolResult(
                        success=True,
                        data=agent.get_status(),
                    )
                else:
                    return AgentToolResult(
                        success=False,
                        error=f"Agent not found: {agent_id}",
                    )
            else:
                status = self.swarm.get_swarm_status()
                return AgentToolResult(
                    success=True,
                    data=status,
                )

        except Exception as e:
            logger.error(f"Get agent status failed: {e}")
            return AgentToolResult(success=False, error=str(e))

    async def get_task_status(self, task_id: str) -> AgentToolResult:
        """
        Get status of a specific task.
        """
        try:
            status = self.swarm.get_task_status(task_id)
            if status:
                return AgentToolResult(success=True, data=status)
            else:
                return AgentToolResult(
                    success=False,
                    error=f"Task not found: {task_id}",
                )

        except Exception as e:
            logger.error(f"Get task status failed: {e}")
            return AgentToolResult(success=False, error=str(e))

    async def list_available_agents(self) -> AgentToolResult:
        """
        List available agent types and their capabilities.
        """
        try:
            from Cosmos.agents.base_agent import AgentCapability

            agent_info = {
                "code": {
                    "description": "Programming specialist for code generation, analysis, and debugging",
                    "capabilities": [
                        AgentCapability.CODE_GENERATION.value,
                        AgentCapability.CODE_ANALYSIS.value,
                        AgentCapability.CODE_DEBUGGING.value,
                    ],
                },
                "reasoning": {
                    "description": "Logic specialist for step-by-step reasoning and analysis",
                    "capabilities": [
                        AgentCapability.REASONING.value,
                        AgentCapability.MATH.value,
                        AgentCapability.PLANNING.value,
                    ],
                },
                "research": {
                    "description": "Research specialist for information gathering and synthesis",
                    "capabilities": [
                        AgentCapability.RESEARCH.value,
                        AgentCapability.SUMMARIZATION.value,
                    ],
                },
                "creative": {
                    "description": "Creative specialist for writing and ideation",
                    "capabilities": [
                        AgentCapability.CREATIVE_WRITING.value,
                    ],
                },
            }

            return AgentToolResult(
                success=True,
                data={
                    "agent_types": agent_info,
                    "active_count": len(self.swarm.state.active_agents),
                    "max_concurrent": self.swarm.max_concurrent,
                },
            )

        except Exception as e:
            logger.error(f"List agents failed: {e}")
            return AgentToolResult(success=False, error=str(e))
