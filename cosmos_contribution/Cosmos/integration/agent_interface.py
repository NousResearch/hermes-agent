"""
cosmos Agent Interface.

"Hello, fellow machine!"

This module allows other AI agents (like Antigravity) to communicate with cosmos
via a standardized high-level API.
"""

from typing import Dict, Any, List
from Cosmos.mcp_server.server import cosmosMCPServer

class AgentInterface:
    def __init__(self, server: cosmosMCPServer):
        self.server = server

    async def query_knowledge(self, query: str):
        """High-level query for external agents."""
        return await self.server.recall(query)

    async def inject_thought(self, content: str):
        """Allow external agent to plant a thought."""
        return await self.server.remember(content, tags=["external_agent", "antigravity"])

    async def request_task(self, task_description: str):
        """Ask cosmos to do something."""
        return await self.server.delegate(task_description, agent_type="auto")

# This interface can be exposed via HTTP/RPC/MCP
