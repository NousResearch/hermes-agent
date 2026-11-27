# Hermes MCP server

Expose Hermes core tools to any MCP client — ChatGPT Desktop, Claude Desktop,
Codex, Cursor, or a custom MCP host — so ChatGPT can call Hermes functions
(read_file, write_file, patch, search_files, terminal, web_search, web_extract,
delegate_task, memory, todo_list, vision_analyze, image_generate, execute_code,
…).

## What this is

`tools/mcp_server.py` is a self-contained MCP server (stdio transport) that reads
the Hermes tool registry (`tools.registry`) and exposes every registered tool via
MCP `tools/list` and `tools/call`. No core Hermes files are edited.

## Run it

Stdio (default, works everywhere):

```bash
cd <HERMES_AGENT_DIR>
python -m tools.mcp_server
```

Log level via `HERMES_MCP_LOG` (default `WARNING`).

## Connect an MCP client

Add a server entry to your MCP client's config (example in
`openapi/mcp_client_config.json`), replacing `<HERMES_AGENT_DIR>` and the secrets:

```json
{
  "mcpServers": {
    "hermes": {
      "command": "python",
      "args": ["-m", "tools.mcp_server"],
      "cwd": "<HERMES_AGENT_DIR>",
      "env": {
        "HERMES_HOME": "~/.hermes",
        "OPENAI_API_KEY": "<your-openai-key>",
        "API_SERVER_KEY": "<your-hermes-api-server-key>"
      }
    }
  }
}
```

## Auth model

MCP stdio carries no auth of its own; trust is the client↔server process boundary.
Individual Hermes tools gate themselves via their `check_fn` / `requires_env` (reachability
and opt-in), so a tool whose secret isn't configured simply won't be exposable and
returns a clean error when called.

## Optional Streamable HTTP

Pass `--http` to request HTTP transport, which requires the `mcp` SDK
(`pip install mcp`). The stdlib build defaults to stdio; HTTP is a documented
extension point rather than a required path.

## Verify

```bash
# From another terminal, send an initialize then tools/list:
printf 'Content-Length: 62\r\n\r\n{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  | python -m tools.mcp_server
```
