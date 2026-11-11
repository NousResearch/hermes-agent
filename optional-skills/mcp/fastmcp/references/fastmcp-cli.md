# FastMCP CLI Reference

Use this file when the task needs exact FastMCP CLI workflows rather than the higher-level guidance in `SKILL.md`.

## Install and Verify

```bash
pip install fastmcp
fastmcp version
```

FastMCP documents `pip install fastmcp` and `fastmcp version` as the baseline installation and verification path.

## Run a Server

Run a server object from a Python file:

```bash
fastmcp run server.py:mcp
```

Run the same server over HTTP:

```bash
fastmcp run server.py:mcp --transport http --host 127.0.0.1 --port 8000
```

## Inspect a Server

Inspect what FastMCP will expose:

```bash
fastmcp inspect server.py:mcp
```

This is also the check FastMCP recommends before deploying to Prefect Horizon.

## List and Call Tools

List tools from a Python file:

```bash
fastmcp list server.py --json
```

List tools from an HTTP endpoint:

```bash
fastmcp list http://127.0.0.1:8000/mcp --json
```

Call a tool with key-value arguments:

```bash
fastmcp call server.py search_resources query=router limit=5 --json
```

Call a tool with a full JSON input payload:

```bash
fastmcp call server.py create_item '{"name": "Widget", "tags": ["sale"]}' --json
```

## Discover Named MCP Servers

Find named servers already configured in local MCP-aware tools:

```bash
fastmcp discover
```

FastMCP documents name-based resolution for Claude Desktop, Claude Code, Cursor, Gemini, Goose, and `./mcp.json`.

## Install into MCP Clients

Register a server with common clients:

```bash
fastmcp install claude-code server.py
fastmcp install claude-desktop server.py
fastmcp install cursor server.py -e .
```

FastMCP notes that client installs run in isolated environments, so declare dependencies explicitly when needed with flags such as `--with`, `--env-file`, or editable installs.

## Deployment Checks

### Prefect Horizon

Before pushing to Horizon:

```bash
fastmcp inspect server.py:mcp
```

FastMCP’s Horizon docs expect:

- a GitHub repo
- a Python file containing the FastMCP server object
- dependencies declared in `requirements.txt` or `pyproject.toml`
- an entrypoint like `main.py:mcp`

### Generic HTTP Hosting

Before shipping to any other host:

1. Start the server locally with HTTP transport.
2. Verify `fastmcp list` against the local `/mcp` URL.
3. Verify at least one `fastmcp call`.
4. Document required environment variables.
