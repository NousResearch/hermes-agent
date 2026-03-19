"""
Cosmos MCP Integration
==========================

Model Context Protocol (MCP) server management and tool routing.
Enables the collective to dynamically load and use external tools.
"""

from .mcp_manager import MCPManager, MCPServer, get_mcp_manager
from .tool_router import ToolRouter, route_tool_call

__all__ = ['MCPManager', 'MCPServer', 'get_mcp_manager', 'ToolRouter', 'route_tool_call']
