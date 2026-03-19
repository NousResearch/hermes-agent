"""
CLAUDE TEAMS INTEGRATION - AGI v1.9
====================================

Integrates Claude Code Agent Teams with Cosmos's swarm architecture.

Components:
- AgentSDKBridge: Programmatic control of Claude agents via SDK
- TeamCoordinator: Coordinates Claude teams with Cosmos swarm
- MCPBridge: Exposes Cosmos tools via Model Context Protocol
- SwarmTeamFusion: Fuses Claude teams with deliberation protocol

"Many minds, one vision - now with Claude teams."
"""

from .agent_sdk_bridge import AgentSDKBridge, get_sdk_bridge
from .team_coordinator import TeamCoordinator, get_team_coordinator
from .mcp_bridge import CosmosMCPServer, get_mcp_server
from .swarm_team_fusion import SwarmTeamFusion, get_swarm_team_fusion

__all__ = [
    "AgentSDKBridge",
    "get_sdk_bridge",
    "TeamCoordinator",
    "get_team_coordinator",
    "CosmosMCPServer",
    "get_mcp_server",
    "SwarmTeamFusion",
    "get_swarm_team_fusion",
]
