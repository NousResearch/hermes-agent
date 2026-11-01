# MCP Server

Longbridge provides MCP (Model Context Protocol) support in two modes: a hosted cloud service and a self-hosted binary.

## Hosted MCP Service

**Endpoint:** `https://openapi.longbridge.com/mcp`

No API keys needed — uses OAuth 2.1. The AI client handles the browser authorization flow automatically.

### Client Configuration

Add to MCP config in any compatible client:

```json
{
  "mcpServers": {
    "longbridge": {
      "url": "https://openapi.longbridge.com/mcp"
    }
  }
}
```

**Per-client setup:**

- **Cursor**: Settings → MCP Servers → Add Remote MCP Server
- **Claude Code**: `claude mcp add longbridge https://openapi.longbridge.com/mcp`
- **ChatGPT**: Settings → Connectors
- **Zed**: `context_servers` in `settings.json`
- **Cherry Studio**: Settings → MCP Servers → Add (requires latest version for OAuth support)

### OAuth Authorization Flow

1. Add the config and call any tool — this triggers the OAuth flow
2. Client opens a browser tab to Longbridge login & consent page
3. Sign in with your Longbridge account and approve scopes
4. Credentials are stored by the client; tokens refresh automatically
5. To revoke: Longbridge account → Security Settings

### Security Recommendations

- Only approve scopes required for the task (least privilege)
- For order placement, instruct AI to always ask for confirmation before executing
- Periodically review and revoke unused authorizations

---

## Available MCP Tools

When the MCP server is connected, available tools are automatically exposed to the AI — no hardcoded list needed. The AI can directly inspect and call all tools.

If you need to know what tools are available, ask the AI to list the connected MCP tools, or check the official docs: https://open.longbridge.com

---

## Example AI Prompts

```
# Market data
"What is the current price and PE ratio of TSLA.US?"

# Trade analysis
"Show my current HK stock positions and unrealized P&L"

# Order placement (always confirm first)
"I want to buy 100 shares of 700.HK at limit price 50 HKD.
 Please confirm the order details before placing it."

# Research
"Get the latest news and filings for AAPL.US"
```
