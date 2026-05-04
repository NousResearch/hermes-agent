---
name: mcporter
description: Use the mcporter CLI to list, configure, auth, and call MCP servers/tools directly (HTTP or stdio), including ad-hoc servers, config edits, and CLI/type generation.
version: 1.0.0
author: community
license: MIT
metadata:
  hermes:
    tags: [MCP, Tools, API, Integrations, Interop]
    homepage: https://mcporter.dev
prerequisites:
  commands: [npx]
---

# mcporter

Use `mcporter` to discover, call, and manage [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) servers and tools directly from the terminal.

## Prerequisites

Requires Node.js:
```bash
# No install needed (runs via npx)
npx mcporter list

# Or install globally
npm install -g mcporter
```

## Quick Start

```bash
# List MCP servers already configured on this machine
mcporter list

# List tools for a specific server with schema details
mcporter list <server> --schema

# Call a tool
mcporter call <server.tool> key=value
```

## Discovering MCP Servers

mcporter auto-discovers servers configured by other MCP clients (Claude Desktop, Cursor, etc.) on the machine. To find new servers to use, browse registries like [mcpfinder.dev](https://mcpfinder.dev) or [mcp.so](https://mcp.so), then connect ad-hoc:

```bash
# Connect to any MCP server by URL (no config needed)
mcporter list --http-url https://some-mcp-server.com --name my_server

# Or run a stdio server on the fly
mcporter list --stdio "npx -y @modelcontextprotocol/server-filesystem" --name fs
```

## Calling Tools

```bash
# Key=value syntax
mcporter call linear.list_issues team=ENG limit:5

# Function syntax
mcporter call "linear.create_issue(title: \"Bug fix needed\")"

# Ad-hoc HTTP server (no config needed)
mcporter call https://api.example.com/mcp.fetch url=https://example.com

# Ad-hoc stdio server
mcporter call --stdio "bun run ./server.ts" scrape url=https://example.com

# JSON payload
mcporter call <server.tool> --args '{"limit": 5}'

# Machine-readable output (recommended for Hermes)
mcporter call <server.tool> key=value --output json
```

## Auth and Config

```bash
# OAuth login for a server
mcporter auth <server | url> [--reset]

# Manage config
mcporter config list
mcporter config get <key>
mcporter config add <server>
mcporter config remove <server>
mcporter config import <path>
```

Config file location: `./config/mcporter.json` (override with `--config`).

## Daemon

For persistent server connections:
```bash
mcporter daemon start
mcporter daemon status
mcporter daemon stop
mcporter daemon restart
```

## Code Generation

```bash
# Generate a CLI wrapper for an MCP server
mcporter generate-cli --server <name>
mcporter generate-cli --command <url>

# Inspect a generated CLI
mcporter inspect-cli <path> [--json]

# Generate TypeScript types/client
mcporter emit-ts <server> --mode client
mcporter emit-ts <server> --mode types
```

## Notes

- Use `--output json` for structured output that's easier to parse
- Ad-hoc servers (HTTP URL or `--stdio` command) work without any config — useful for one-off calls
- OAuth auth may require interactive browser flow — use `terminal(command="mcporter auth <server>", pty=true)` if needed

## Typical Steps

1. Confirm Node.js/npx is available: `node --version && npx --version`.
2. Discover configured MCP servers: `npx mcporter list --output json`.
3. Inspect the target server's tools and schemas: `npx mcporter list <server> --schema --output json`.
4. Call the smallest safe read-only tool first with `--output json` to validate auth and argument shape.
5. For state-changing tools, pass a complete JSON payload with `--args '{...}'` and record the returned ID/status for verification.

## Pitfalls

1. `mcporter` vs `npx mcporter`: many machines do not have a global `mcporter` binary. Prefer `npx mcporter ...` unless you have already verified the global install.
2. Shell quoting breaks JSON easily. For complex arguments, write the payload in single quotes around valid JSON and escape only inner single quotes.
3. OAuth flows may need a pseudo-terminal. Run auth with `pty=true` when using the terminal tool.
4. Do not assume a configured server name. Always list servers first; MCP client configs differ across Claude Desktop, Cursor, Hermes, and project-local configs.
5. Use `--output json` for automation. Human-formatted tables are harder to parse and can hide tool errors.

## Verification Checklist

- [ ] `npx mcporter list --output json` succeeds or the missing Node/npm prerequisite is explicit.
- [ ] The target server appears in the list, or the ad-hoc `--http-url` / `--stdio` command was used deliberately.
- [ ] Tool schema was inspected before calling a non-trivial tool.
- [ ] Tool call output was captured in JSON and includes a success indicator, result object, or actionable error.
- [ ] Any state-changing call was verified by reading back the created/updated resource.
