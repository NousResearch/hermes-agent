"""
cosmos MCP Server Module

Hybrid MCP server for Claude Code integration with:
- Memory access and manipulation tools
- Agent delegation and monitoring
- Evolution control and feedback
- Resource streaming for context awareness
"""

from Cosmos.mcp_server.server import cosmosMCPServer, run_server
from Cosmos.mcp_server.memory_tools import MemoryTools
from Cosmos.mcp_server.agent_tools import AgentTools
from Cosmos.mcp_server.evolution_tools import EvolutionTools
from Cosmos.mcp_server.resources import cosmosResources

__all__ = [
    "cosmosMCPServer",
    "run_server",
    "MemoryTools",
    "AgentTools",
    "EvolutionTools",
    "cosmosResources",
]
