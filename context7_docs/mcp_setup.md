### List MCP servers and tools with mcporter

Source: https://github.com/nousresearch/hermes-agent/blob/main/optional-skills/mcp/mcporter/SKILL.md

Display configured servers, optionally with schema details for tools.

```bash
# List MCP servers already configured on this machine
mcporter list

# List tools for a specific server with schema details
mcporter list <server> --schema
```

--------------------------------

### Install and list MCP servers with mcporter

Source: https://github.com/nousresearch/hermes-agent/blob/main/optional-skills/mcp/mcporter/SKILL.md

Run mcporter via npx without installation, or install globally. Use 'list' to discover configured servers and their tools.

```bash
# No install needed (runs via npx)
npx mcporter list

# Or install globally
npm install -g mcporter
```

--------------------------------

### MCP Servers Configuration

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/getting-started/nix-setup.md

Define and configure Model Context Protocol (MCP) servers for the Hermes Agent. Supports multiple transport types (stdio, HTTP, StreamableHTTP), authentication methods, tool filtering, and sampling configuration.

```APIDOC
## Configuration Options: MCP Servers

### Description
Define and configure Model Context Protocol (MCP) servers for the Hermes Agent.

### Options

#### mcpServers
- **Type**: `attrsOf submodule`
- **Default**: `{}`
- **Description**: MCP server definitions, merged into `settings.mcp_servers`

#### mcpServers.<name>.command
- **Type**: `null or str`
- **Default**: `null`
- **Description**: Server command (stdio transport)

#### mcpServers.<name>.args
- **Type**: `listOf str`
- **Default**: `[]`
- **Description**: Command arguments

#### mcpServers.<name>.env
- **Type**: `attrsOf str`
- **Default**: `{}`
- **Description**: Environment variables for the server process

#### mcpServers.<name>.url
- **Type**: `null or str`
- **Default**: `null`
- **Description**: Server endpoint URL (HTTP/StreamableHTTP transport)

#### mcpServers.<name>.headers
- **Type**: `attrsOf str`
- **Default**: `{}`
- **Description**: HTTP headers, e.g. `Authorization`

#### mcpServers.<name>.auth
- **Type**: `null or "oauth"`
- **Default**: `null`
- **Description**: Authentication method. `"oauth"` enables OAuth 2.1 PKCE

#### mcpServers.<name>.enabled
- **Type**: `bool`
- **Default**: `true`
- **Description**: Enable or disable this server

#### mcpServers.<name>.timeout
- **Type**: `null or int`
- **Default**: `null`
- **Description**: Tool call timeout in seconds (default: 120)

#### mcpServers.<name>.connect_timeout
- **Type**: `null or int`
- **Default**: `null`
- **Description**: Connection timeout in seconds (default: 60)

#### mcpServers.<name>.tools
- **Type**: `null or submodule`
- **Default**: `null`
- **Description**: Tool filtering (`include`/`exclude` lists)

#### mcpServers.<name>.sampling
- **Type**: `null or submodule`
- **Default**: `null`
- **Description**: Sampling config for server-initiated LLM requests
```

### Troubleshooting > Hermes cannot see the deployed server

Source: https://github.com/nousresearch/hermes-agent/blob/main/optional-skills/mcp/fastmcp/SKILL.md

If Hermes cannot see the deployed server, the issue might be with the Hermes configuration rather than the server build. Load the `native-mcp` skill, configure the server in `~/.hermes/config.yaml`, and then restart Hermes.

--------------------------------

### mcp > native-mcp

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/reference/skills-catalog.md

Built-in MCP (Model Context Protocol) client that connects to external MCP servers, discovers their tools, and registers them as native Hermes Agent tools. Supports stdio and HTTP transports with automatic reconnection, security filtering, and zero-config tool injection.