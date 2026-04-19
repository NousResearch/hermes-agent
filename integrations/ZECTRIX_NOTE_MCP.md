# Zectrix Note 4 MCP Server

This package exposes Zectrix Note 4 cloud functions as MCP tools for Hermes Agent or any MCP client.

## Prerequisites

Set these environment variables:

```bash
export ZECTRIX_API_KEY="zt_xxx"
export ZECTRIX_API_BASE_URL="https://cloud.zectrix.com"   # optional
export ZECTRIX_DEFAULT_DEVICE_ID="AA:BB:CC:DD:EE:FF"      # optional
```

## Available tools

- `health_check`
- `list_devices`
- `list_todos`
- `create_todo`
- `update_todo`
- `toggle_todo_complete`
- `delete_todo`
- `push_text`
- `push_structured_text`
- `push_image`
- `clear_pages`

## Hermes config

```yaml
mcp_servers:
  zectrix_note:
    command: "zectrix-note-mcp"
    env:
      ZECTRIX_API_KEY: "zt_xxx"
      ZECTRIX_DEFAULT_DEVICE_ID: "AA:BB:CC:DD:EE:FF"
```

If you prefer, you can also run it directly:

```bash
python -m integrations.zectrix_note_mcp
```
