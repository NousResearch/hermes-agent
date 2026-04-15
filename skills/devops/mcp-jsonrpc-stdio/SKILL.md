---
name: mcp-jsonrpc-stdio
description: >
  Call MCP servers via JSON-RPC over stdin/stdout from execute_code or terminal.
  Use when you need to invoke MCP tools programmatically but the native MCP client
  is unavailable or you're in an execute_code sandbox that can't import the MCP
  library directly. Works with any MCP server that supports stdio transport.
metadata:
  author: Indigo Karasu
  version: "1.0.0"
  hermes:
    tags: [mcp, integration, jsonrpc]
    category: devops
---

# MCP JSON-RPC Stdio Client

Call any MCP server via the JSON-RPC stdin/stdout protocol from `execute_code` or `terminal`. This is needed when:
- You're in an `execute_code` sandbox and can't use the native MCP client
- The MCP server runs on a different Python version than the sandbox
- You need to script multiple MCP calls in sequence

## Pattern

### 1. Write requests to a temp file (avoids shell escaping hell)

```python
import json

def mcp_call(server_command, tool_name, arguments=None):
    """Call an MCP tool via JSON-RPC stdin/stdout protocol.
    
    Args:
        server_command: Shell command to start the MCP server (e.g., 'python3.13 -m mempalace.mcp_server')
        tool_name: Name of the MCP tool to call
        arguments: Dict of arguments for the tool
    
    Returns:
        Parsed JSON result from the MCP server
    """
    if arguments is None:
        arguments = {}
    
    init_msg = {
        "jsonrpc": "2.0", "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "hermes-execute", "version": "1.0.0"}
        }
    }
    tool_msg = {
        "jsonrpc": "2.0", "id": 2,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments}
    }
    
    # Write to temp file to avoid shell escaping issues
    with open('/tmp/mcp_request.jsonl', 'w') as f:
        f.write(json.dumps(init_msg) + '\n')
        f.write(json.dumps(tool_msg) + '\n')
    
    from hermes_tools import terminal
    r = terminal(
        command=f"cat /tmp/mcp_request.jsonl | timeout 15 {server_command} 2>/dev/null | tail -1"
    )
    
    output = r.get('output', '').strip()
    if output:
        try:
            result = json.loads(output)
            return result.get('result', result)
        except json.JSONDecodeError:
            return {"error": "parse_error", "raw": output[:500]}
    return {"error": "no_output"}
```

### 2. Call multiple tools in sequence

For multiple calls, chain them in the JSONL file:

```python
def mcp_multi_call(server_command, calls):
    """Call multiple MCP tools in a single server session.
    
    Args:
        calls: list of (tool_name, arguments) tuples
    
    Returns:
        List of results in the same order
    """
    init_msg = {
        "jsonrpc": "2.0", "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "hermes-execute", "version": "1.0.0"}
        }
    }
    
    with open('/tmp/mcp_request.jsonl', 'w') as f:
        f.write(json.dumps(init_msg) + '\n')
        for i, (tool_name, arguments) in enumerate(calls, start=1):
            tool_msg = {
                "jsonrpc": "2.0", "id": i,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments or {}}
            }
            f.write(json.dumps(tool_msg) + '\n')
    
    from hermes_tools import terminal
    r = terminal(
        command=f"cat /tmp/mcp_request.jsonl | timeout 30 {server_command} 2>/dev/null"
    )
    
    results = []
    for line in r.get('output', '').strip().split('\n'):
        if line.strip():
            try:
                parsed = json.loads(line)
                results.append(parsed.get('result', parsed))
            except json.JSONDecodeError:
                results.append({"error": "parse_error"})
    # Skip the initialize response (index 0)
    return results[1:]
```

### 3. List available tools

```python
def mcp_list_tools(server_command):
    """List all tools available on an MCP server."""
    init_msg = {
        "jsonrpc": "2.0", "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "hermes-execute", "version": "1.0.0"}
        }
    }
    list_msg = {
        "jsonrpc": "2.0", "id": 2,
        "method": "tools/list",
        "params": {}
    }
    
    with open('/tmp/mcp_request.jsonl', 'w') as f:
        f.write(json.dumps(init_msg) + '\n')
        f.write(json.dumps(list_msg) + '\n')
    
    from hermes_tools import terminal
    r = terminal(
        command=f"cat /tmp/mcp_request.jsonl | timeout 10 {server_command} 2>/dev/null | tail -1"
    )
    
    output = r.get('output', '').strip()
    if output:
        try:
            result = json.loads(output)
            return result.get('result', {}).get('tools', [])
        except json.JSONDecodeError:
            return []
    return []
```

## Pitfalls

- **Always use a temp file for the request payload.** Piping inline JSON through `echo` causes shell escaping issues, especially with nested quotes and special characters.
- **Use `tail -1`** for single-call responses. The MCP server outputs one JSON-RPC response per request; `tail -1` grabs the tool call result (id:2), skipping the initialize acknowledgment (id:1).
- **Set a timeout.** MCP servers are long-running processes; without `timeout`, the command hangs forever waiting for more stdin. 15 seconds is a reasonable default for most tool calls.
- **Python version mismatch.** If the MCP server is installed under Python 3.13 but the sandbox runs 3.11, start the server with `python3.13 -m module.mcp_server`, not `python3 -m module.mcp_server`.
- **Stderr suppression.** Use `2>/dev/null` because MCP servers may log startup messages to stderr that interfere with parsing stdout.
- **Empty responses.** Some MCP tools return empty content arrays (`{"content": []}`). Check for this before processing results.
- **The `MemPalace` class is NOT the public API.** For MemPalace specifically, don't try `from mempalace import MemPalace` — the public interface is the MCP server, not the Python class.

## Known working MCP servers

| Server | Command | Notes |
|--------|---------|-------|
| MemPalace | `python3.13 -m mempalace.mcp_server` | Installed under py3.13, not py3.11 |

## When not to use

- When the native MCP client is available (use it instead — it handles connection pooling, timeouts, and error recovery)
- For one-off interactive calls (use `mcporter` CLI if installed)
- When the MCP server requires environment variables not set in the terminal session (set them in the command: `VAR=val python3.13 -m ...`)