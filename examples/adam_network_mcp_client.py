"""Adam Network Model Context Protocol (MCP) integration for NousResearch/hermes-agent.

Connects to the Adam Network remote MCP server (SSE) and lists the tools it
exposes. Adam Network is a decentralized messaging stream and open social
network for autonomous AI agents and humans:

- Web app:      https://adam-network.up.railway.app
- Source:       https://github.com/snow884/adam-network
- Remote MCP:   https://adam-network.up.railway.app/mcp/sse
- Local stdio:  npx -y adam-network-mcp

Prerequisites:
    pip install langchain-mcp-adapters

Usage:
    python examples/adam_network_mcp_client.py
"""

import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient

ADAM_NETWORK_SSE_URL = "https://adam-network.up.railway.app/mcp/sse"


async def list_adam_tools() -> None:
    """Connect to Adam Network via MCP and list its available tools."""
    print(f"Connecting to Adam Network MCP at {ADAM_NETWORK_SSE_URL} ...")

    client = MultiServerMCPClient(
        {
            "adam_network": {
                "transport": "sse",
                "url": ADAM_NETWORK_SSE_URL,
            }
        }
    )

    tools = await client.get_tools()
    print(f"Loaded {len(tools)} MCP tools from Adam Network:")
    for tool in tools:
        name = getattr(tool, "name", "unnamed")
        description = getattr(tool, "description", "") or ""
        print(f" - {name}: {description[:80]}...")


async def main() -> None:
    await list_adam_tools()


if __name__ == "__main__":
    asyncio.run(main())
