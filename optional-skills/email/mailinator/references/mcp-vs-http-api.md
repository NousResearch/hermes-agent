# Mailinator MCP vs HTTP API

## MCP Endpoint Limitations

The Mailinator MCP endpoint at `https://www.mailinator.com/mcp` requires a persistent WebSocket/SSE connection and cannot be used with stateless HTTP requests.

### MCP Endpoint Details

- **WebSocket/SSE-based**: Requires `initialize` call first, then `tools/call` with a persistent session
- **Session not persistent**: Each HTTP request gets a new session that expires immediately
- **Streaming support**: MCP supports Server-Sent Events (SSE) for real-time message delivery

### Test Results

```bash
# MCP initialize works
curl -X POST https://www.mailinator.com/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"initialize","params":{},"id":1}'

# Response includes protocolVersion and instructions
# But tools/call fails with "Session not initialized" or "Session not found"
```

### SSE Streaming (requires persistent connection)

```bash
# This only works with a WebSocket or SSE client
curl --no-buffer "https://www.mailinator.com/mcp?subscribe=mailinator://joe/public"
# Returns "Session not found or expired" on one-off HTTP requests
```

## HTTP API (Recommended for Stateless Access)

The HTTP API v2 is more reliable for polling use cases:

### Inbox Listing

```bash
GET https://www.mailinator.com/api/v2/domains/public/inboxes/{inbox}
```

Example:
```bash
curl -s "https://www.mailinator.com/api/v2/domains/public/inboxes/joe" | python3 -m json.tool
```

### Email Content

```bash
GET https://www.mailinator.com/api/v2/domains/public/inboxes/{inbox}/messages/{message_id}?format=raw
```

Example:
```bash
curl -s "https://www.mailinator.com/api/v2/domains/public/inboxes/joe/messages/{id}?format=raw" | python3 -m json.tool
```

## When to Use Each

| Use Case | Recommended Method |
|----------|-------------------|
| One-off polling (cron jobs, scripts) | HTTP API v2 |
| Real-time streaming | MCP WebSocket/SSE |
| Public domain access | HTTP API (no auth) |
| Private domain access | HTTP API with API token |
| Interactive CLI tools | HTTP API |

## Key Takeaways

1. **Use HTTP API for polling** - It's stateless, reliable, and doesn't require session management
2. **MCP requires persistent connection** - Only use MCP if you need real-time streaming
3. **Public domains don't need auth** - Private domains require an API token
4. **Format parameter** - Use `?format=raw` to get the full email body

## References

- Mailinator HTTP API v2: https://www.mailinator.com/mailinator-api/
- MCP Endpoint: https://www.mailinator.com/mcp