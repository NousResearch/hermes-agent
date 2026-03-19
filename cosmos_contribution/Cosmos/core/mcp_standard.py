"""
Cosmos MCP Standardization - Standard Protocol for Agent-Tool Communication.

AGI v1.8 Feature: Implements the Model Context Protocol standard for
unified tool discovery, invocation, and agent-to-agent communication.

Features:
- MCPCapability: Standard capability enumeration
- MCPToolSchema: Standardized tool schema with version and performance metadata
- MCPStandardProtocol: Convert existing tools to MCP standard
- AgentMCPClient: Bidirectional agent-to-agent MCP communication
- MCPToolRegistry: Central registry with discovery
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import  Callable, Optional, Set, Awaitable, Union

from loguru import logger


# =============================================================================
# MCP CAPABILITY DEFINITIONS
# =============================================================================

class MCPCapability(Enum):
    """Standard MCP capabilities."""
    TOOLS = "tools"             # Expose callable tools
    RESOURCES = "resources"     # Expose readable resources
    PROMPTS = "prompts"         # Expose prompt templates
    SAMPLING = "sampling"       # Support LLM sampling


class MCPToolCategory(Enum):
    """Categories of MCP tools."""
    MEMORY = "memory"           # Memory operations
    REASONING = "reasoning"     # Reasoning/inference
    EXTERNAL = "external"       # External API calls
    CODE = "code"               # Code execution/analysis
    DATA = "data"               # Data processing
    COMMUNICATION = "communication"  # Agent communication
    SYSTEM = "system"           # System operations


class MCPResourceType(Enum):
    """Types of MCP resources."""
    TEXT = "text/plain"
    JSON = "application/json"
    MARKDOWN = "text/markdown"
    BINARY = "application/octet-stream"


# =============================================================================
# MCP DATA STRUCTURES
# =============================================================================

@dataclass
class MCPToolSchema:
    """
    Standardized tool schema with version and performance metadata.

    Compatible with MCP specification for tool definitions.
    """
    name: str
    description: str
    input_schema: dict
    version: str = "1.0.0"
    category: MCPToolCategory = MCPToolCategory.SYSTEM
    capabilities_required: list[MCPCapability] = field(default_factory=list)

    # Performance metadata
    avg_latency_ms: float = 0.0
    success_rate: float = 1.0
    invocation_count: int = 0

    # Handler reference
    handler: Optional[Callable[..., Awaitable[dict]]] = None

    # Additional metadata
    tags: list[str] = field(default_factory=list)
    deprecated: bool = False
    replacement: Optional[str] = None

    def to_mcp_format(self) -> dict:
        """Convert to MCP protocol format."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }

    def to_dict(self) -> dict:
        """Convert to full dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "version": self.version,
            "category": self.category.value,
            "capabilities_required": [c.value for c in self.capabilities_required],
            "avg_latency_ms": self.avg_latency_ms,
            "success_rate": self.success_rate,
            "invocation_count": self.invocation_count,
            "tags": self.tags,
            "deprecated": self.deprecated,
            "replacement": self.replacement,
        }


@dataclass
class MCPResource:
    """Standardized MCP resource definition."""
    uri: str
    name: str
    description: str
    mime_type: MCPResourceType = MCPResourceType.TEXT
    handler: Optional[Callable[[], Awaitable[str]]] = None
    metadata: dict = field(default_factory=dict)

    def to_mcp_format(self) -> dict:
        """Convert to MCP protocol format."""
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type.value,
        }


@dataclass
class MCPPrompt:
    """Standardized MCP prompt template."""
    name: str
    description: str
    arguments: list[dict] = field(default_factory=list)
    template: str = ""
    metadata: dict = field(default_factory=dict)

    def to_mcp_format(self) -> dict:
        """Convert to MCP protocol format."""
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments,
        }


@dataclass
class MCPInvocationResult:
    """Result of an MCP tool invocation."""
    success: bool
    result: dict
    error: Optional[str] = None
    latency_ms: float = 0.0
    tool_name: str = ""
    metadata: dict = field(default_factory=dict)


# =============================================================================
# MCP TOOL REGISTRY
# =============================================================================

class MCPToolRegistry:
    """
    Central registry for MCP tools with discovery and invocation.

    Maintains a catalog of all available tools, their schemas,
    performance metrics, and provides discovery mechanisms.
    """

    def __init__(self):
        self._tools: dict[str, MCPToolSchema] = {}
        self._resources: dict[str, MCPResource] = {}
        self._prompts: dict[str, MCPPrompt] = {}

        # Capability tracking
        self._capabilities: Set[MCPCapability] = set()

        # Discovery index
        self._category_index: dict[MCPToolCategory[str]] = {}
        self._tag_index: dict[str[str]] = {}

        logger.info("MCPToolRegistry initialized")

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: dict,
        handler: Callable[..., Awaitable[dict]],
        version: str = "1.0.0",
        category: MCPToolCategory = MCPToolCategory.SYSTEM,
        tags: Optional[list[str]] = None,
    ) -> MCPToolSchema:
        """Register a new tool in the registry."""
        schema = MCPToolSchema(
            name=name,
            description=description,
            input_schema=input_schema,
            version=version,
            category=category,
            handler=handler,
            tags=tags or [],
        )

        self._tools[name] = schema
        self._capabilities.add(MCPCapability.TOOLS)

        # Update indices
        if category not in self._category_index:
            self._category_index[category] = []
        self._category_index[category].append(name)

        for tag in schema.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = []
            self._tag_index[tag].append(name)

        logger.debug(f"Registered MCP tool: {name}")
        return schema

    def register_resource(
        self,
        uri: str,
        name: str,
        description: str,
        handler: Callable[[], Awaitable[str]],
        mime_type: MCPResourceType = MCPResourceType.TEXT,
    ) -> MCPResource:
        """Register a new resource in the registry."""
        resource = MCPResource(
            uri=uri,
            name=name,
            description=description,
            mime_type=mime_type,
            handler=handler,
        )

        self._resources[uri] = resource
        self._capabilities.add(MCPCapability.RESOURCES)

        logger.debug(f"Registered MCP resource: {uri}")
        return resource

    def register_prompt(
        self,
        name: str,
        description: str,
        arguments: list[dict],
        template: str,
    ) -> MCPPrompt:
        """Register a new prompt template in the registry."""
        prompt = MCPPrompt(
            name=name,
            description=description,
            arguments=arguments,
            template=template,
        )

        self._prompts[name] = prompt
        self._capabilities.add(MCPCapability.PROMPTS)

        logger.debug(f"Registered MCP prompt: {name}")
        return prompt

    # =========================================================================
    # TOOL INVOCATION
    # =========================================================================

    async def invoke_tool(
        self,
        name: str,
        arguments: dict,
    ) -> MCPInvocationResult:
        """Invoke a tool by name with arguments."""
        import time

        if name not in self._tools:
            return MCPInvocationResult(
                success=False,
                result=None,
                error=f"Unknown tool: {name}",
                tool_name=name,
            )

        schema = self._tools[name]
        if not schema.handler:
            return MCPInvocationResult(
                success=False,
                result=None,
                error=f"Tool {name} has no handler",
                tool_name=name,
            )

        start_time = time.time()

        try:
            result = await schema.handler(**arguments)
            latency = (time.time() - start_time) * 1000

            # Update metrics
            schema.invocation_count += 1
            n = schema.invocation_count
            schema.avg_latency_ms = (schema.avg_latency_ms * (n - 1) + latency) / n
            schema.success_rate = (schema.success_rate * (n - 1) + 1) / n

            return MCPInvocationResult(
                success=True,
                result=result,
                latency_ms=latency,
                tool_name=name,
            )

        except Exception as e:
            latency = (time.time() - start_time) * 1000

            # Update metrics
            schema.invocation_count += 1
            n = schema.invocation_count
            schema.avg_latency_ms = (schema.avg_latency_ms * (n - 1) + latency) / n
            schema.success_rate = (schema.success_rate * (n - 1)) / n

            return MCPInvocationResult(
                success=False,
                result=None,
                error=str(e),
                latency_ms=latency,
                tool_name=name,
            )

    async def read_resource(self, uri: str) -> Optional[str]:
        """Read a resource by URI."""
        if uri not in self._resources:
            return None

        resource = self._resources[uri]
        if not resource.handler:
            return None

        try:
            return await resource.handler()
        except Exception as e:
            logger.error(f"Failed to read resource {uri}: {e}")
            return None

    def get_prompt(self, name: str, arguments: dict) -> Optional[str]:
        """Get a rendered prompt template."""
        if name not in self._prompts:
            return None

        prompt = self._prompts[name]

        try:
            # Simple template substitution
            result = prompt.template
            for key, value in arguments.items():
                result = result.replace(f"{{{key}}}", str(value))
            return result
        except Exception as e:
            logger.error(f"Failed to render prompt {name}: {e}")
            return None

    # =========================================================================
    # DISCOVERY
    # =========================================================================

    def list_tools(self) -> list[dict]:
        """list all tools in MCP format."""
        return [schema.to_mcp_format() for schema in self._tools.values()]

    def list_resources(self) -> list[dict]:
        """list all resources in MCP format."""
        return [res.to_mcp_format() for res in self._resources.values()]

    def list_prompts(self) -> list[dict]:
        """list all prompts in MCP format."""
        return [prompt.to_mcp_format() for prompt in self._prompts.values()]

    def discover_by_category(self, category: MCPToolCategory) -> list[MCPToolSchema]:
        """Discover tools by category."""
        tool_names = self._category_index.get(category, [])
        return [self._tools[name] for name in tool_names if name in self._tools]

    def discover_by_tag(self, tag: str) -> list[MCPToolSchema]:
        """Discover tools by tag."""
        tool_names = self._tag_index.get(tag, [])
        return [self._tools[name] for name in tool_names if name in self._tools]

    def search_tools(self, query: str) -> list[MCPToolSchema]:
        """Search tools by name or description."""
        query_lower = query.lower()
        results = []

        for schema in self._tools.values():
            if (
                query_lower in schema.name.lower() or
                query_lower in schema.description.lower()
            ):
                results.append(schema)

        return results

    def get_capabilities(self) -> list[str]:
        """Get list of supported capabilities."""
        return [cap.value for cap in self._capabilities]

    # =========================================================================
    # METRICS
    # =========================================================================

    def get_tool_metrics(self, name: str) -> Optional[dict]:
        """Get performance metrics for a tool."""
        if name not in self._tools:
            return None

        schema = self._tools[name]
        return {
            "name": name,
            "invocation_count": schema.invocation_count,
            "avg_latency_ms": schema.avg_latency_ms,
            "success_rate": schema.success_rate,
        }

    def get_stats(self) -> dict:
        """Get registry statistics."""
        return {
            "total_tools": len(self._tools),
            "total_resources": len(self._resources),
            "total_prompts": len(self._prompts),
            "capabilities": self.get_capabilities(),
            "categories": list(self._category_index.keys()),
            "tags": list(self._tag_index.keys()),
        }


# =============================================================================
# AGENT MCP CLIENT
# =============================================================================

class AgentMCPClient:
    """
    Bidirectional MCP client for agent-to-agent communication.

    Allows agents to expose and consume tools from other agents
    using the MCP protocol.
    """

    def __init__(
        self,
        agent_id: str,
        registry: MCPToolRegistry,
    ):
        self.agent_id = agent_id
        self.registry = registry

        # Connected agents
        self._connected_agents: dict[str, "AgentMCPClient"] = {}

        # Remote tool cache
        self._remote_tools: dict[str[str, MCPToolSchema]] = {}

        # Message queue for async communication
        self._message_queue: asyncio.Queue = asyncio.Queue()

        # Nexus integration
        self._nexus = None

        logger.info(f"AgentMCPClient initialized for {agent_id}")

    def connect_nexus(self, nexus) -> None:
        """Connect to the Nexus event bus."""
        self._nexus = nexus

    # =========================================================================
    # AGENT CONNECTIONS
    # =========================================================================

    def connect_agent(self, other_agent: "AgentMCPClient") -> None:
        """Establish connection with another agent."""
        self._connected_agents[other_agent.agent_id] = other_agent
        other_agent._connected_agents[self.agent_id] = self

        # Exchange tool catalogs
        self._remote_tools[other_agent.agent_id] = {
            name: schema for name, schema in other_agent.registry._tools.items()
        }
        other_agent._remote_tools[self.agent_id] = {
            name: schema for name, schema in self.registry._tools.items()
        }

        logger.info(f"Agent {self.agent_id} connected to {other_agent.agent_id}")

    def disconnect_agent(self, agent_id: str) -> None:
        """Disconnect from an agent."""
        if agent_id in self._connected_agents:
            other = self._connected_agents.pop(agent_id)
            other._connected_agents.pop(self.agent_id, None)

            self._remote_tools.pop(agent_id, None)
            other._remote_tools.pop(self.agent_id, None)

            logger.info(f"Agent {self.agent_id} disconnected from {agent_id}")

    # =========================================================================
    # REMOTE TOOL INVOCATION
    # =========================================================================

    async def invoke_remote_tool(
        self,
        target_agent_id: str,
        tool_name: str,
        arguments: dict,
    ) -> MCPInvocationResult:
        """Invoke a tool on a remote agent."""
        if target_agent_id not in self._connected_agents:
            return MCPInvocationResult(
                success=False,
                result=None,
                error=f"Not connected to agent: {target_agent_id}",
                tool_name=tool_name,
            )

        target = self._connected_agents[target_agent_id]

        # Emit signal for tracking
        await self._emit_signal("MCP_TOOL_CALLED", {
            "source_agent": self.agent_id,
            "target_agent": target_agent_id,
            "tool_name": tool_name,
        })

        # Invoke on target's registry
        return await target.registry.invoke_tool(tool_name, arguments)

    def get_remote_tools(self, agent_id: str) -> list[MCPToolSchema]:
        """Get list of tools available from a remote agent."""
        return list(self._remote_tools.get(agent_id, {}).values())

    def discover_remote_tool(
        self,
        tool_name: str,
    ) -> list[tuple[str, MCPToolSchema]]:
        """Discover which agents have a specific tool."""
        results = []

        for agent_id, tools in self._remote_tools.items():
            if tool_name in tools:
                results.append((agent_id, tools[tool_name]))

        return results

    # =========================================================================
    # CAPABILITY DISCOVERY
    # =========================================================================

    async def discover_capabilities(self, agent_id: str) -> dict:
        """Discover capabilities of a connected agent."""
        if agent_id not in self._connected_agents:
            return {"error": f"Not connected to agent: {agent_id}"}

        target = self._connected_agents[agent_id]

        await self._emit_signal("MCP_CAPABILITY_DISCOVERED", {
            "source_agent": self.agent_id,
            "target_agent": agent_id,
            "capabilities": target.registry.get_capabilities(),
        })

        return {
            "agent_id": agent_id,
            "capabilities": target.registry.get_capabilities(),
            "tool_count": len(target.registry._tools),
            "resource_count": len(target.registry._resources),
        }

    # =========================================================================
    # NEXUS INTEGRATION
    # =========================================================================

    async def _emit_signal(self, signal_type: str, payload: dict) -> None:
        """Emit a signal to Nexus."""
        if not self._nexus:
            return

        try:
            from Cosmos.core.nexus import SignalType

            signal_enum = getattr(SignalType, signal_type, None)
            if signal_enum:
                await self._nexus.emit(
                    type=signal_enum,
                    payload=payload,
                    source=f"mcp_client_{self.agent_id}",
                    urgency=0.5,
                )
        except Exception as e:
            logger.debug(f"Failed to emit signal {signal_type}: {e}")

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_status(self) -> dict:
        """Get client status."""
        return {
            "agent_id": self.agent_id,
            "connected_agents": list(self._connected_agents.keys()),
            "local_tools": len(self.registry._tools),
            "remote_tools_cached": sum(
                len(tools) for tools in self._remote_tools.values()
            ),
        }


# =============================================================================
# MCP STANDARD PROTOCOL
# =============================================================================

class MCPStandardProtocol:
    """
    Main protocol coordinator for MCP standardization.

    Converts existing tools to MCP standard and provides
    unified access to the tool ecosystem.
    """

    def __init__(self, data_dir: str = "./data/mcp"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.registry = MCPToolRegistry()
        self._agent_clients: dict[str, AgentMCPClient] = {}

        # Nexus integration
        self._nexus = None

        logger.info("MCPStandardProtocol initialized")

    def connect_nexus(self, nexus) -> None:
        """Connect to the Nexus event bus."""
        self._nexus = nexus
        self.registry._nexus = nexus

        for client in self._agent_clients.values():
            client.connect_nexus(nexus)

    # =========================================================================
    # TOOL CONVERSION
    # =========================================================================

    def convert_to_mcp(
        self,
        name: str,
        handler: Callable[..., Awaitable[dict]],
        description: str,
        parameters: dict,
        category: MCPToolCategory = MCPToolCategory.SYSTEM,
        tags: Optional[list[str]] = None,
    ) -> MCPToolSchema:
        """
        Convert an existing tool to MCP standard format.

        Args:
            name: Tool name
            handler: Async handler function
            description: Tool description
            parameters: Parameter definitions (JSON Schema format)
            category: Tool category
            tags: Optional tags for discovery
        """
        # Build input schema in JSON Schema format
        input_schema = {
            "type": "object",
            "properties": parameters,
            "required": [
                k for k, v in parameters.items()
                if v.get("required", False)
            ],
        }

        schema = self.registry.register_tool(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
            category=category,
            tags=tags,
        )

        # Emit registration signal
        asyncio.create_task(self._emit_signal("MCP_TOOL_REGISTERED", {
            "tool_name": name,
            "category": category.value,
        }))

        return schema

    # =========================================================================
    # AGENT CLIENT MANAGEMENT
    # =========================================================================

    def create_agent_client(self, agent_id: str) -> AgentMCPClient:
        """Create an MCP client for an agent."""
        if agent_id in self._agent_clients:
            return self._agent_clients[agent_id]

        client = AgentMCPClient(agent_id, self.registry)

        if self._nexus:
            client.connect_nexus(self._nexus)

        self._agent_clients[agent_id] = client

        # Emit signal
        asyncio.create_task(self._emit_signal("MCP_AGENT_CONNECTED", {
            "agent_id": agent_id,
        }))

        return client

    def get_agent_client(self, agent_id: str) -> Optional[AgentMCPClient]:
        """Get the MCP client for an agent."""
        return self._agent_clients.get(agent_id)

    # =========================================================================
    # NEXUS INTEGRATION
    # =========================================================================

    async def _emit_signal(self, signal_type: str, payload: dict) -> None:
        """Emit a signal to Nexus."""
        if not self._nexus:
            return

        try:
            from Cosmos.core.nexus import SignalType

            signal_enum = getattr(SignalType, signal_type, None)
            if signal_enum:
                await self._nexus.emit(
                    type=signal_enum,
                    payload=payload,
                    source="mcp_standard",
                    urgency=0.5,
                )
        except Exception as e:
            logger.debug(f"Failed to emit signal {signal_type}: {e}")

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_stats(self) -> dict:
        """Get protocol statistics."""
        return {
            "registry": self.registry.get_stats(),
            "agent_clients": len(self._agent_clients),
            "agent_ids": list(self._agent_clients.keys()),
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_mcp_protocol(data_dir: str = "./data/mcp") -> MCPStandardProtocol:
    """Factory function to create an MCPStandardProtocol instance."""
    return MCPStandardProtocol(data_dir=data_dir)


def create_tool_registry() -> MCPToolRegistry:
    """Factory function to create an MCPToolRegistry instance."""
    return MCPToolRegistry()
