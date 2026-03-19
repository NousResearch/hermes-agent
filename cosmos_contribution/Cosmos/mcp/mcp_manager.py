"""
Cosmos MCP Manager
======================

Manages MCP server connections and tool discovery.
"""

import os
import json
import asyncio
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from loguru import logger


@dataclass
class MCPTool:
    """An MCP tool definition."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str


@dataclass
class MCPServer:
    """An MCP server configuration."""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    process: Optional[subprocess.Popen] = None
    tools: List[MCPTool] = field(default_factory=list)
    is_connected: bool = False


class MCPManager:
    """
    Manages MCP server lifecycle and tool discovery.

    Features:
    - Start/stop MCP servers
    - Discover available tools
    - Route tool calls to servers
    - Handle stdio communication
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(__file__).parent.parent.parent / "mcp_config.json"
        self.servers: Dict[str, MCPServer] = {}
        self.tools: Dict[str, MCPTool] = {}
        self._load_config()

    def _load_config(self):
        """Load MCP server configurations."""
        if self.config_path.exists():
            try:
                config = json.loads(self.config_path.read_text())
                for name, server_config in config.get("mcpServers", {}).items():
                    self.servers[name] = MCPServer(
                        name=name,
                        command=server_config.get("command", "npx"),
                        args=server_config.get("args", []),
                        env=self._resolve_env(server_config.get("env", {})),
                    )
                logger.info(f"Loaded {len(self.servers)} MCP server configs")
            except Exception as e:
                logger.error(f"Failed to load MCP config: {e}")

    def _resolve_env(self, env: Dict[str, str]) -> Dict[str, str]:
        """Resolve environment variable references."""
        resolved = {}
        for key, value in env.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # Extract var name and default
                var_spec = value[2:-1]
                if ":-" in var_spec:
                    var_name, default = var_spec.split(":-", 1)
                else:
                    var_name, default = var_spec, ""
                resolved[key] = os.environ.get(var_name, default)
            else:
                resolved[key] = value
        return resolved

    async def start_server(self, name: str) -> bool:
        """Start an MCP server."""
        if name not in self.servers:
            logger.error(f"Unknown MCP server: {name}")
            return False

        server = self.servers[name]
        if server.is_connected:
            logger.info(f"MCP server {name} already running")
            return True

        try:
            # Build command
            cmd = [server.command] + server.args

            # Merge environment
            env = os.environ.copy()
            env.update(server.env)

            # Start process
            server.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )

            server.is_connected = True
            logger.info(f"Started MCP server: {name}")

            # Discover tools
            await self._discover_tools(server)

            return True

        except Exception as e:
            logger.error(f"Failed to start MCP server {name}: {e}")
            return False

    async def stop_server(self, name: str):
        """Stop an MCP server."""
        if name in self.servers:
            server = self.servers[name]
            if server.process:
                server.process.terminate()
                server.process = None
            server.is_connected = False
            logger.info(f"Stopped MCP server: {name}")

    async def _discover_tools(self, server: MCPServer):
        """Discover tools from an MCP server."""
        # Send tools/list request via JSON-RPC
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        }

        try:
            if server.process and server.process.stdin and server.process.stdout:
                # Write request
                request_str = json.dumps(request) + "\n"
                server.process.stdin.write(request_str.encode())
                server.process.stdin.flush()

                # Read response (with timeout)
                response_line = server.process.stdout.readline()
                if response_line:
                    response = json.loads(response_line.decode())
                    tools = response.get("result", {}).get("tools", [])

                    for tool_data in tools:
                        tool = MCPTool(
                            name=tool_data["name"],
                            description=tool_data.get("description", ""),
                            input_schema=tool_data.get("inputSchema", {}),
                            server_name=server.name,
                        )
                        server.tools.append(tool)
                        self.tools[tool.name] = tool

                    logger.info(f"Discovered {len(tools)} tools from {server.name}")

        except Exception as e:
            logger.error(f"Failed to discover tools from {server.name}: {e}")

    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool."""
        if tool_name not in self.tools:
            return {"error": f"Unknown tool: {tool_name}"}

        tool = self.tools[tool_name]
        server = self.servers.get(tool.server_name)

        if not server or not server.is_connected:
            # Try to start the server
            if server:
                await self.start_server(server.name)
            else:
                return {"error": f"Server for tool {tool_name} not found"}

        # Send tool call via JSON-RPC
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": params
            }
        }

        try:
            if server.process and server.process.stdin and server.process.stdout:
                request_str = json.dumps(request) + "\n"
                server.process.stdin.write(request_str.encode())
                server.process.stdin.flush()

                response_line = server.process.stdout.readline()
                if response_line:
                    response = json.loads(response_line.decode())
                    return response.get("result", {})

        except Exception as e:
            logger.error(f"Failed to call tool {tool_name}: {e}")
            return {"error": str(e)}

        return {"error": "No response from MCP server"}

    def list_tools(self) -> List[str]:
        """List all available tools."""
        return list(self.tools.keys())

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def add_server(self, name: str, command: str, args: List[str] = None, env: Dict[str, str] = None):
        """Add a new MCP server configuration."""
        self.servers[name] = MCPServer(
            name=name,
            command=command,
            args=args or [],
            env=env or {},
        )
        self._save_config()

    def _save_config(self):
        """Save current configuration to file."""
        config = {
            "mcpServers": {
                name: {
                    "command": server.command,
                    "args": server.args,
                    "env": server.env,
                }
                for name, server in self.servers.items()
            }
        }
        self.config_path.write_text(json.dumps(config, indent=2))


# Global MCP manager instance
_mcp_manager: Optional[MCPManager] = None


def get_mcp_manager() -> MCPManager:
    """Get or create the global MCP manager."""
    global _mcp_manager
    if _mcp_manager is None:
        _mcp_manager = MCPManager()
    return _mcp_manager
