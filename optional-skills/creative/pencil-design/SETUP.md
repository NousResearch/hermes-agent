# Pencil MCP Configuration Guide

This guide helps you set up Pencil MCP for use with Hermes Agent.

## Quick Start (3 steps)

### 1. Identify Your Pencil Server Transport

- **Command-line (stdio)**: Pencil MCP runs as a CLI command
- **HTTP/REST**: Pencil MCP runs as a web service
- **SSH**: Pencil MCP runs on a remote machine

### 2. Add to `~/.hermes/config.yaml`

**Stdio example:**
```yaml
mcp_servers:
  pencil:
    command: "npx"
    args: ["@pencil-mcp/server"]
    timeout: 120
    connect_timeout: 60
    env: {}  # Add env vars here if needed
```

**HTTP example:**
```yaml
mcp_servers:
  pencil:
    url: "http://localhost:5000/mcp"
    timeout: 120
    connect_timeout: 60
    headers:
      Authorization: "Bearer your-token-here"  # if needed
```

**SSH example:**
```yaml
mcp_servers:
  pencil:
    command: "ssh"
    args: ["user@example.com", "pencil-mcp-server"]
    timeout: 120
    connect_timeout: 60
    env:
      SSH_KEY: "/path/to/key"  # optional
```

### 3. Verify It Works

```bash
hermes tools
# Filter for "pencil" or "design"
# Should list Pencil operations like:
#   - open_document
#   - batch_design
#   - get_screenshot
#   - export_nodes
#   etc.
```

## Environment Variable Substitution

Config values support `${VAR}` placeholder substitution:

```yaml
mcp_servers:
  pencil:
    url: "${PENCIL_SERVER_URL}"  # reads from env or ~/.hermes/.env
    headers:
      Authorization: "Bearer ${PENCIL_API_KEY}"
```

Environment variables are loaded from:
1. Process environment (`export VAR=value`)
2. `~/.hermes/.env` (one `KEY=value` per line)
3. System environment

## Troubleshooting

### Issue: "mcp_servers not found" or config not loading

**Solution**: Ensure `mcp_servers:` is at the **top level** of config.yaml, not nested under `auxiliary:` or `model:`.

```yaml
# ✅ CORRECT
mcp_servers:
  pencil:
    url: "..."

model: "gpt-4"
agent: {}

# ❌ WRONG (mcp_servers is nested)
agent:
  mcp_servers:
    pencil:
      url: "..."
```

### Issue: Pencil tools don't appear after adding config

1. **Check logs**:
   ```bash
   hermes logs --level DEBUG | grep -i pencil
   ```

2. **Reload MCP manually**:
   ```bash
   hermes /reload-mcp
   ```

3. **Test server connectivity**:
   - **HTTP**: `curl -I http://localhost:5000/mcp`
   - **Stdio**: Run the command directly: `npx @pencil-mcp/server`

### Issue: "Connection timeout" errors

- Increase `timeout` value: `timeout: 180` (default 120)
- Increase `connect_timeout`: `connect_timeout: 90` (default 60)
- Check server is actually running and accessible
- For HTTP, verify firewall/network rules allow connections

### Issue: Authentication/Headers not working

Ensure headers are under the `headers:` key:

```yaml
mcp_servers:
  pencil:
    url: "http://..."
    headers:
      Authorization: "Bearer token-here"
      X-Custom-Header: "value"
```

## Advanced Configuration

### Per-Server Timeouts

Each MCP server can have its own timeout:

```yaml
mcp_servers:
  pencil:
    url: "http://localhost:5000/mcp"
    timeout: 120        # per-tool-call timeout
    connect_timeout: 60  # initial connection timeout
```

### Multiple Pencil Servers

You can connect to multiple Pencil instances with different names:

```yaml
mcp_servers:
  pencil-primary:
    url: "http://server1.example.com/mcp"
  pencil-staging:
    url: "http://server2.example.com/mcp"
```

All operations become available, prefixed with their server name if disambiguation is needed.

### Environment Variables in Config

Pencil server URLs, ports, and credentials can reference env vars:

```yaml
mcp_servers:
  pencil:
    url: "${PENCIL_BASE_URL}/mcp"  # reads PENCIL_BASE_URL from env
    headers:
      X-API-Key: "${PENCIL_API_KEY}"  # reads PENCIL_API_KEY
```

Set these in `~/.hermes/.env`:

```
PENCIL_BASE_URL=http://localhost:5000
PENCIL_API_KEY=sk-pencil-xxxx
```

## Next Steps

Once configured:

1. **Load the skill**: `hermes skills install official/creative/pencil-design`
2. **Test it**: `hermes` → ask to create a design
3. **Check logs**: `hermes logs --follow` while running a design task

## Related Documentation

- **Hermes MCP Integration**: `tools/mcp_tool.py` (source code)
- **MCP Specification**: https://modelcontextprotocol.io/
- **Pencil MCP Docs**: https://github.com/pencil-mcp/docs (adjust URL as needed)
- **Pencil Skills**: See `pencil-design` SKILL.md in this directory
