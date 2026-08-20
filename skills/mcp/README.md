# MCP Skills

Model Context Protocol client integration.

## Overview

This category contains 1 skill for connecting to MCP servers and registering tools via stdio or HTTP protocols.

## Available Skills

### **native-mcp**
MCP client: connect servers, register tools (stdio/HTTP).

**Use when:** Integrating MCP servers into Hermes workflows or extending Hermes with MCP tools.

**Key features:**
- MCP server connections (stdio/HTTP)
- Tool registration and discovery
- Protocol negotiation
- Server lifecycle management
- Multi-server support

**Use cases:**
- Connect to MCP tool servers
- Extend Hermes capabilities
- Integrate third-party MCP tools
- Build MCP-based workflows
- Server orchestration

---

## Quick Start

```bash
# Connect to MCP server
/native-mcp "Connect to filesystem MCP server via stdio"

# Register tools
/native-mcp "Discover and register available tools"

# Use MCP tools
# Tools become available in Hermes after registration
```

## Related Categories

- **autonomous-ai-agents/** - Agent development
- **software-development/** - Tool integration

---

**Questions?** Check the [Hermes Agent documentation](https://hermes-agent.nousresearch.com/docs/).
