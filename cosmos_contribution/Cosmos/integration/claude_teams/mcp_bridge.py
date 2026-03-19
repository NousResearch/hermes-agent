"""
MCP BRIDGE - Expose Cosmos Tools to Claude Teams
======================================================

Implements Model Context Protocol (MCP) server that exposes
Cosmos's capabilities to Claude agent teams.

Cosmos controls what tools Claude teams can access.
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from loguru import logger


class MCPToolAccess(Enum):
    """Access levels for MCP tools."""
    FULL = "full"       # Complete access
    LIMITED = "limited" # Read-only operations
    NONE = "none"       # No access


@dataclass
class MCPTool:
    """A tool exposed via MCP."""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable
    access_level: MCPToolAccess = MCPToolAccess.FULL
    call_count: int = 0
    last_used: Optional[datetime] = None


@dataclass
class MCPRequest:
    """Incoming MCP request from Claude team."""
    request_id: str
    tool_name: str
    arguments: Dict[str, Any]
    requesting_agent: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MCPResponse:
    """Response to MCP request."""
    request_id: str
    success: bool
    result: Any
    error: Optional[str] = None
    duration_ms: int = 0


class CosmosMCPServer:
    """
    MCP Server exposing Cosmos's tools to Claude teams.

    Cosmos remains in control - it decides:
    - What tools are available
    - Who can access what
    - Rate limits and quotas
    - Logging and auditing
    """

    def __init__(self):
        self.tools: Dict[str, MCPTool] = {}
        self.request_log: List[MCPRequest] = []
        self.access_control: Dict[str, MCPToolAccess] = {}  # team_id -> access level

        # Register default Cosmos tools
        self._register_default_tools()

        logger.info("CosmosMCPServer initialized - tools ready for Claude teams")

    def _register_default_tools(self) -> None:
        """Register Cosmos's default tools for Claude teams."""

        # Swarm Oracle access
        self.register_tool(
            name="swarm_oracle_query",
            description="Query the Cosmos Swarm Oracle for multi-agent consensus on any question.",
            parameters={
                "question": {"type": "string", "description": "The question to get consensus on"},
                "query_type": {"type": "string", "default": "general"},
            },
            handler=self._handle_oracle_query,
        )

        # Memory access (read-only by default)
        self.register_tool(
            name="read_swarm_memory",
            description="Read from Cosmos's multi-layer memory system.",
            parameters={
                "query": {"type": "string", "description": "Memory query"},
                "memory_layer": {"type": "string", "default": "working"},
            },
            handler=self._handle_memory_read,
            access_level=MCPToolAccess.LIMITED,
        )

        # Agent status
        self.register_tool(
            name="get_swarm_status",
            description="Get status of Cosmos's agent swarm.",
            parameters={},
            handler=self._handle_swarm_status,
            access_level=MCPToolAccess.LIMITED,
        )

        # Farsight predictions
        self.register_tool(
            name="farsight_predict",
            description="Get a prediction from Farsight Protocol using collective intelligence.",
            parameters={
                "question": {"type": "string", "description": "What to predict"},
                "category": {"type": "string", "default": "general"},
            },
            handler=self._handle_farsight_predict,
        )

        # Shadow agent delegation
        self.register_tool(
            name="call_shadow_agent",
            description="Delegate a task to a specific Cosmos shadow agent (Grok, Gemini, etc.)",
            parameters={
                "agent": {"type": "string", "description": "Agent name (grok, gemini, kimi, etc.)"},
                "prompt": {"type": "string", "description": "Task prompt"},
            },
            handler=self._handle_shadow_agent_call,
        )

        # Knowledge graph
        self.register_tool(
            name="query_knowledge_graph",
            description="Query Cosmos's knowledge graph for relationships and entities.",
            parameters={
                "query": {"type": "string", "description": "Knowledge graph query"},
            },
            handler=self._handle_kg_query,
            access_level=MCPToolAccess.LIMITED,
        )

        # Token analysis (Solana)
        self.register_tool(
            name="analyze_solana_token",
            description="Get Cosmos's swarm analysis of a Solana token.",
            parameters={
                "token_address": {"type": "string", "description": "Token contract address"},
            },
            handler=self._handle_token_analysis,
        )

        logger.info(f"Registered {len(self.tools)} MCP tools")

    # =========================================================================
    # TOOL REGISTRATION
    # =========================================================================

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable,
        access_level: MCPToolAccess = MCPToolAccess.FULL,
    ) -> None:
        """Register a new MCP tool."""
        self.tools[name] = MCPTool(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            access_level=access_level,
        )

    def unregister_tool(self, name: str) -> None:
        """Remove an MCP tool."""
        if name in self.tools:
            del self.tools[name]

    # =========================================================================
    # ACCESS CONTROL (Cosmos decides who can do what)
    # =========================================================================

    def set_team_access(self, team_id: str, access: MCPToolAccess) -> None:
        """Set access level for a Claude team."""
        self.access_control[team_id] = access
        logger.info(f"Set {team_id} access to {access.value}")

    def check_access(self, team_id: str, tool_name: str) -> bool:
        """Check if a team can access a tool."""
        team_access = self.access_control.get(team_id, MCPToolAccess.LIMITED)
        tool = self.tools.get(tool_name)

        if not tool:
            return False

        # Full access teams can use anything
        if team_access == MCPToolAccess.FULL:
            return True

        # Limited access can only use limited tools
        if team_access == MCPToolAccess.LIMITED:
            return tool.access_level in [MCPToolAccess.LIMITED, MCPToolAccess.FULL]

        return False

    # =========================================================================
    # REQUEST HANDLING
    # =========================================================================

    async def handle_request(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        requesting_agent: str,
    ) -> MCPResponse:
        """Handle an MCP tool request from a Claude team."""
        request = MCPRequest(
            request_id=f"mcp_{uuid.uuid4().hex[:8]}",
            tool_name=tool_name,
            arguments=arguments,
            requesting_agent=requesting_agent,
        )
        self.request_log.append(request)

        # Check access
        team_id = requesting_agent.split("_")[0] if "_" in requesting_agent else requesting_agent
        if not self.check_access(team_id, tool_name):
            return MCPResponse(
                request_id=request.request_id,
                success=False,
                result=None,
                error=f"Access denied for tool: {tool_name}",
            )

        # Get tool
        tool = self.tools.get(tool_name)
        if not tool:
            return MCPResponse(
                request_id=request.request_id,
                success=False,
                result=None,
                error=f"Tool not found: {tool_name}",
            )

        # Execute
        start_time = datetime.now()
        try:
            result = await tool.handler(**arguments)
            tool.call_count += 1
            tool.last_used = datetime.now()

            duration = int((datetime.now() - start_time).total_seconds() * 1000)

            return MCPResponse(
                request_id=request.request_id,
                success=True,
                result=result,
                duration_ms=duration,
            )

        except Exception as e:
            logger.error(f"MCP tool error ({tool_name}): {e}")
            return MCPResponse(
                request_id=request.request_id,
                success=False,
                result=None,
                error=str(e),
            )

    # =========================================================================
    # TOOL HANDLERS (Cosmos's actual capabilities)
    # =========================================================================

    async def _handle_oracle_query(self, question: str, query_type: str = "general") -> Dict[str, Any]:
        """Handle swarm oracle query."""
        try:
            from Cosmos.integration.solana.swarm_oracle import get_swarm_oracle

            oracle = get_swarm_oracle()
            result = await oracle.submit_query(question, query_type, timeout=60.0)

            return {
                "consensus": result.consensus_answer,
                "confidence": result.consensus_confidence,
                "agents": result.agent_votes,
                "source": "cosmos_swarm_oracle",
            }
        except Exception as e:
            return {"error": str(e)}

    async def _handle_memory_read(self, query: str, memory_layer: str = "working") -> Dict[str, Any]:
        """Handle memory read request."""
        try:
            from Cosmos.memory.memory_system import get_memory_system

            memory = get_memory_system()

            if memory_layer == "working":
                result = memory.working_memory.get(query)
            elif memory_layer == "archival":
                result = await memory.archival.search(query, top_k=5)
            else:
                result = {"note": f"Unknown layer: {memory_layer}"}

            return {"query": query, "layer": memory_layer, "result": result}
        except Exception as e:
            return {"error": str(e)}

    async def _handle_swarm_status(self) -> Dict[str, Any]:
        """Handle swarm status request."""
        try:
            from Cosmos.core.organism import get_organism

            organism = get_organism()
            return {
                "active_agents": len(organism.agents),
                "health": "operational",
                "uptime": str(datetime.now() - organism.start_time) if hasattr(organism, 'start_time') else "unknown",
            }
        except Exception as e:
            return {"status": "unknown", "error": str(e)}

    async def _handle_farsight_predict(self, question: str, category: str = "general") -> Dict[str, Any]:
        """Handle Farsight prediction request."""
        try:
            from Cosmos.integration.hackathon.farsight_protocol import get_farsight

            farsight = get_farsight()
            prediction = await farsight.predict(question, category, include_visual=False)

            return {
                "prediction": prediction.farsight_answer,
                "confidence": prediction.farsight_confidence,
                "reasoning": prediction.farsight_reasoning,
                "swarm_consensus": prediction.swarm_consensus,
            }
        except Exception as e:
            return {"error": str(e)}

    async def _handle_shadow_agent_call(self, agent: str, prompt: str) -> Dict[str, Any]:
        """Handle shadow agent call request."""
        try:
            from Cosmos.core.collective.persistent_agent import call_shadow_agent

            result = await call_shadow_agent(agent.lower(), prompt, timeout=30.0)
            if result:
                agent_name, response = result
                return {"agent": agent_name, "response": response}
            return {"error": f"No response from {agent}"}
        except Exception as e:
            return {"error": str(e)}

    async def _handle_kg_query(self, query: str) -> Dict[str, Any]:
        """Handle knowledge graph query."""
        try:
            from Cosmos.memory.knowledge_graph import get_knowledge_graph

            kg = get_knowledge_graph()
            results = kg.search(query, limit=10)
            return {"query": query, "results": results}
        except Exception as e:
            return {"error": str(e)}

    async def _handle_token_analysis(self, token_address: str) -> Dict[str, Any]:
        """Handle Solana token analysis."""
        try:
            from Cosmos.integration.solana.swarm_solana import get_swarm_solana

            solana = get_swarm_solana()
            analysis = await solana.analyze_token(token_address)
            return analysis
        except Exception as e:
            return {"error": str(e)}

    # =========================================================================
    # TOOL LISTING (MCP protocol)
    # =========================================================================

    def list_tools(self, team_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available tools (filtered by team access if provided)."""
        tools = []
        for tool in self.tools.values():
            # Filter by access if team specified
            if team_id:
                if not self.check_access(team_id, tool.name):
                    continue

            tools.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "access_level": tool.access_level.value,
            })

        return tools

    def get_stats(self) -> Dict[str, Any]:
        """Get MCP server statistics."""
        return {
            "total_tools": len(self.tools),
            "total_requests": len(self.request_log),
            "tool_usage": {
                name: {"calls": t.call_count, "last_used": t.last_used.isoformat() if t.last_used else None}
                for name, t in self.tools.items()
            },
            "access_control_entries": len(self.access_control),
        }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_mcp_server: Optional[CosmosMCPServer] = None


def get_mcp_server() -> CosmosMCPServer:
    """Get global MCP server instance."""
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = CosmosMCPServer()
    return _mcp_server
