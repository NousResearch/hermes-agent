"""
cosmos Resources - MCP Resource Implementations

Provides streaming resources:
- Memory context
- Knowledge graph
- Agent status
- Evolution metrics
"""

import json
from typing import Optional, Any
from datetime import datetime

from loguru import logger


class cosmosResources:
    """
    Resource implementations for the MCP server.

    Provides read-only access to cosmos's internal state.
    """

    def __init__(
        self,
        memory_system=None,
        swarm_orchestrator=None,
        fitness_tracker=None,
    ):
        self.memory = memory_system
        self.swarm = swarm_orchestrator
        self.fitness = fitness_tracker

    async def get_recent_context(self, max_length: int = 4000) -> str:
        """
        Get recent context from all memory systems.

        Returns formatted context suitable for LLM consumption.
        """
        try:
            if self.memory is None:
                return "Memory system not initialized"

            context = self.memory.get_context()
            if len(context) > max_length:
                context = context[:max_length] + "\n...[truncated]"

            return context

        except Exception as e:
            logger.error(f"Get recent context failed: {e}")
            return f"Error: {e}"

    async def get_memory_summary(self) -> str:
        """
        Get a summary of memory system state.
        """
        try:
            if self.memory is None:
                return "Memory system not initialized"

            return await self.memory.get_memory_summary()

        except Exception as e:
            logger.error(f"Get memory summary failed: {e}")
            return f"Error: {e}"

    async def get_knowledge_graph_json(self) -> str:
        """
        Get knowledge graph as JSON.
        """
        try:
            if self.memory is None:
                return "{}"

            stats = self.memory.knowledge_graph.get_stats()

            # Get recent entities
            recent_entities = []
            for eid, entity in list(self.memory.knowledge_graph.entities.items())[:20]:
                recent_entities.append({
                    "id": eid,
                    "name": entity.name,
                    "type": entity.entity_type,
                    "mentions": entity.mention_count,
                })

            data = {
                "stats": stats,
                "recent_entities": recent_entities,
            }

            return json.dumps(data, indent=2)

        except Exception as e:
            logger.error(f"Get knowledge graph failed: {e}")
            return json.dumps({"error": str(e)})

    async def get_active_agents_json(self) -> str:
        """
        Get active agents as JSON.
        """
        try:
            if self.swarm is None:
                return "{}"

            status = self.swarm.get_swarm_status()
            return json.dumps(status, indent=2)

        except Exception as e:
            logger.error(f"Get active agents failed: {e}")
            return json.dumps({"error": str(e)})

    async def get_fitness_metrics_json(self) -> str:
        """
        Get fitness metrics as JSON.
        """
        try:
            if self.fitness is None:
                return "{}"

            stats = self.fitness.get_stats()
            return json.dumps(stats, indent=2)

        except Exception as e:
            logger.error(f"Get fitness metrics failed: {e}")
            return json.dumps({"error": str(e)})

    async def get_system_status_json(self) -> str:
        """
        Get comprehensive system status as JSON.
        """
        try:
            status = {
                "timestamp": datetime.now().isoformat(),
                "memory": {},
                "agents": {},
                "evolution": {},
            }

            if self.memory:
                status["memory"] = self.memory.get_stats()

            if self.swarm:
                status["agents"] = self.swarm.get_swarm_status()

            if self.fitness:
                status["evolution"] = self.fitness.get_stats()

            return json.dumps(status, indent=2)

        except Exception as e:
            logger.error(f"Get system status failed: {e}")
            return json.dumps({"error": str(e)})

    async def get_conversation_history(self, max_turns: int = 20) -> str:
        """
        Get recent conversation history.
        """
        try:
            if self.memory is None:
                return "Memory system not initialized"

            turns = self.memory.recall_memory.get_recent(max_turns)

            lines = ["# Recent Conversation\n"]
            for turn in turns:
                lines.append(f"**{turn.role}**: {turn.content}\n")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Get conversation history failed: {e}")
            return f"Error: {e}"

    async def get_working_memory_summary(self) -> str:
        """
        Get working memory contents.
        """
        try:
            if self.memory is None:
                return "Memory system not initialized"

            return self.memory.working_memory.to_context_string()

        except Exception as e:
            logger.error(f"Get working memory failed: {e}")
            return f"Error: {e}"


# Resource URI handlers
RESOURCE_HANDLERS = {
    "cosmos://memory/recent": "get_recent_context",
    "cosmos://memory/summary": "get_memory_summary",
    "cosmos://memory/graph": "get_knowledge_graph_json",
    "cosmos://memory/conversation": "get_conversation_history",
    "cosmos://memory/working": "get_working_memory_summary",
    "cosmos://agents/active": "get_active_agents_json",
    "cosmos://evolution/fitness": "get_fitness_metrics_json",
    "cosmos://status": "get_system_status_json",
}


async def handle_resource_read(resources: cosmosResources, uri: str) -> str:
    """
    Handle a resource read request.

    Args:
        resources: cosmosResources instance
        uri: Resource URI

    Returns:
        Resource content as string
    """
    handler_name = RESOURCE_HANDLERS.get(uri)

    if handler_name is None:
        return f"Unknown resource: {uri}"

    handler = getattr(resources, handler_name, None)

    if handler is None:
        return f"Handler not found for: {uri}"

    return await handler()
