---
name: smithery-mcp-setup
description: Configure and connect to remote MCP servers via Smithery.ai. Handles service tokens, mcporter config, and direct HTTP MCP calls.
category: mcp
---

# Smithery MCP Setup

Connect to MCP servers hosted on Smithery (server.smithery.ai) using mcporter config and service tokens.

## When to use

- User provides a Smithery MCP server reference (e.g., `clay-inc/clay-mcp`)
- User provides a Smithery service token (UUID format)
- Setting up remote MCP connections for any skill

## When not to use

- Local/std MCP servers (use mcporter directly with `--stdio`)
- Direct API integrations that don't use MCP

## Smithery Token Format

Smithery service tokens are UUIDs: `75408f6f-4db1-4eb2-afc9-d166075884f4`

These are NOT OAuth tokens or API keys — they mint scoped service tokens for Smithery Connect.

## Endpoint Pattern

Smithery MCP servers are at: `https://server.smithery.ai/{owner}/{name}/mcp`

Example: `https://server.smithery.ai/clay-inc/clay-mcp/mcp`

## Auth Pattern

The MCP endpoint requires `Authorization: Bearer <service-token>` header.

Test with curl:
```bash
curl -s -X POST "https://server.smithery.ai/clay-inc/clay-mcp/mcp" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <smithery-service-token>" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

If response is `{"error":"invalid_token"}`, the token is a Smithery API key that needs to be exchanged for a service token via `smithery auth token` or the Smithery API.

## mcporter Configuration

Save to `/root/.hermes/config/mcporter.json`:

```json
{
  "mcpServers": {
    "server_name": {
      "url": "https://server.smithery.ai/owner/name/mcp",
      "headers": {
        "Authorization": "Bearer <service-token>"
      }
    }
  }
}
```

## Using with Weave (Clay/Me.sh)

The Mesh MCP (`clay-inc/clay-mcp`) provides Weave with contact enrichment tools:
- `searchContacts` — search by name, org, location, keywords
- `getContact` — full details including emails, phones, social links
- `createContact` / `updateContact` — CRUD operations
- `get_user_information` — current user profile

## Pitfalls

### Deprecated Clay API endpoints
`api.clay.com/v1/*` and `api.clay.com/v2/*` return "deprecated API endpoint" errors. Use the Mesh MCP via Smithery instead.

### me.sh has no public API
The Me.sh web app (app.me.sh) returns HTML, not API responses. All programmatic access goes through the Smithery Mesh MCP.

### Smithery CLI requires namespace
`smithery mcp add clay-inc/clay-mcp` fails with "404 Invalid credentials or namespace not found" without proper namespace configuration. Use mcporter config or direct HTTP calls instead.

### Service token vs API key
The Smithery API key (64-char hex) is different from the service token (UUID). The service token is what goes in the `Authorization: Bearer` header for MCP calls. Generate service tokens via `smithery auth token --policy`.

### mcporter terminal hangs
The `mcporter` CLI may hang in non-interactive terminals. Use direct curl HTTP calls to MCP endpoints instead for testing, or save config and let the agent use the MCP tools programmatically.