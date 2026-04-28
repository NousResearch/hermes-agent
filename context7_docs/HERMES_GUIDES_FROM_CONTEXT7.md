# Hermes Agent installation and setup guide

### Install and Quick Start Hermes Agent

Source: https://github.com/nousresearch/hermes-agent/blob/main/skills/autonomous-ai-agents/hermes-agent/SKILL.md

Commands for installing the agent via shell script and performing common initial actions like starting a chat or running the setup wizard.

```bash
# Install
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash

# Interactive chat (default)
hermes

# Single query
hermes chat -q "What is the capital of France?"

# Setup wizard
hermes setup

# Change model/provider
hermes model

# Check health
hermes doctor
```

--------------------------------

### Installation and Quick Start

Source: https://github.com/nousresearch/hermes-agent/blob/main/skills/autonomous-ai-agents/hermes-agent/SKILL.md

Installation script and basic commands to get Hermes Agent running. Includes interactive chat, single query execution, setup wizard, model configuration, and health checks.

```APIDOC
## Installation

### Description
Install Hermes Agent using the official installation script.

### Command
```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

---

## Interactive Chat

### Description
Start Hermes Agent in interactive chat mode (default behavior).

### Command
```bash
hermes
```

---

## Single Query

### Description
Execute a single query without entering interactive mode.

### Command
```bash
hermes chat -q "What is the capital of France?"
```

### Parameters
- **-q, --query** (string) - Required - The query text to execute

---

## Setup Wizard

### Description
Run the interactive setup wizard to configure Hermes Agent.

### Command
```bash
hermes setup
```

---

## Change Model/Provider

### Description
Interactively select and configure the LLM model and provider.

### Command
```bash
hermes model
```

---

## Health Check

### Description
Verify Hermes Agent installation and configuration status.

### Command
```bash
hermes doctor
```
```

--------------------------------

### Install Hermes Agent

Source: https://github.com/nousresearch/hermes-agent/blob/main/README.md

This command downloads and executes the Hermes Agent installation script. It works on Linux, macOS, WSL2, and Android via Termux.

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

--------------------------------

### Install Hermes Agent via one-line script

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/getting-started/installation.md

The installer automatically handles dependencies like Python, Node.js, and ripgrep for Linux, macOS, WSL2, and Android.

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

--------------------------------

### Quick Start Development Setup for Hermes Agent (Bash)

Source: https://github.com/nousresearch/hermes-agent/blob/main/README.md

Clone the Hermes Agent repository and use the `setup-hermes.sh` script for a quick development environment setup, including `uv` installation, virtual environment creation, and package installation.

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
./setup-hermes.sh     # installs uv, creates venv, installs .[all], symlinks ~/.local/bin/hermes
./hermes              # auto-detects the venv, no need to `source` first
```

---

# Hermes gateway configuration for Matrix, Telegram, Discord

### Start Gateway and Configure Voice Mode

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/user-guide/features/voice-mode.md

Start the Hermes gateway to connect to messaging platforms and enable voice mode. The gateway command connects to configured Telegram and Discord bots.

```bash
hermes gateway        # Start the gateway (connects to configured platforms)
hermes gateway setup  # Interactive setup wizard for first-time configuration
```

--------------------------------

### Set Up Hermes Gateway for Messaging Platforms

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/getting-started/quickstart.md

Configure Hermes to connect with messaging platforms like Telegram, Discord, Slack, WhatsApp, Signal, Email, or Home Assistant.

```bash
hermes gateway setup    # Interactive platform configuration
```

--------------------------------

### Docker Compose Configuration for Hermes Matrix Proxy

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/user-guide/messaging/matrix.md

This `docker-compose.yml` defines the Hermes Matrix proxy service, configuring Matrix credentials and the URL for forwarding requests to the host agent. It uses environment variables for sensitive data.

```yaml
services:
  hermes-matrix:
    build: .
    environment:
      # Matrix credentials
      MATRIX_HOMESERVER: "https://matrix.example.org"
      MATRIX_ACCESS_TOKEN: "syt_..."
      MATRIX_ALLOWED_USERS: "@you:matrix.example.org"
      MATRIX_ENCRYPTION: "true"
      MATRIX_DEVICE_ID: "HERMES_BOT"

      # Proxy mode — forward to host agent
      GATEWAY_PROXY_URL: "http://192.168.1.100:8642"
      GATEWAY_PROXY_KEY: "your-secret-key-here"
    volumes:
      - ./matrix-store:/root/.hermes/platforms/matrix/store
```

--------------------------------

### Manual Configuration in .env File

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/user-guide/messaging/discord.md

Add Discord credentials directly to ~/.hermes/.env. DISCORD_BOT_TOKEN is required; DISCORD_ALLOWED_USERS accepts a single user ID or comma-separated list of user IDs.

```bash
# Required
DISCORD_BOT_TOKEN=your-bot-token
DISCORD_ALLOWED_USERS=284102345871466496

# Multiple allowed users (comma-separated)
# DISCORD_ALLOWED_USERS=284102345871466496,198765432109876543
```

--------------------------------

### Configure Hermes with Access Token

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/user-guide/messaging/matrix.md

Environment variables for ~/.hermes/.env using token-based authentication. MATRIX_USER_ID is optional and auto-detected from the token if omitted. MATRIX_ALLOWED_USERS restricts bot interaction to specified users.

```bash
# Required
MATRIX_HOMESERVER=https://matrix.example.org
MATRIX_ACCESS_TOKEN=***

# Optional: user ID (auto-detected from token if omitted)
# MATRIX_USER_ID=@hermes:matrix.example.org

# Security: restrict who can interact with the bot
MATRIX_ALLOWED_USERS=@alice:matrix.example.org

# Multiple allowed users (comma-separated)
# MATRIX_ALLOWED_USERS=@alice:matrix.example.org,@bob:matrix.example.org
```

---

# Hermes Matrix bot setup and authentication

### Configure Hermes with Access Token

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/user-guide/messaging/matrix.md

Environment variables for ~/.hermes/.env using token-based authentication. MATRIX_USER_ID is optional and auto-detected from the token if omitted. MATRIX_ALLOWED_USERS restricts bot interaction to specified users.

```bash
# Required
MATRIX_HOMESERVER=https://matrix.example.org
MATRIX_ACCESS_TOKEN=***

# Optional: user ID (auto-detected from token if omitted)
# MATRIX_USER_ID=@hermes:matrix.example.org

# Security: restrict who can interact with the bot
MATRIX_ALLOWED_USERS=@alice:matrix.example.org

# Multiple allowed users (comma-separated)
# MATRIX_ALLOWED_USERS=@alice:matrix.example.org,@bob:matrix.example.org
```

--------------------------------

### Docker Compose Configuration for Hermes Matrix Proxy

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/user-guide/messaging/matrix.md

This `docker-compose.yml` defines the Hermes Matrix proxy service, configuring Matrix credentials and the URL for forwarding requests to the host agent. It uses environment variables for sensitive data.

```yaml
services:
  hermes-matrix:
    build: .
    environment:
      # Matrix credentials
      MATRIX_HOMESERVER: "https://matrix.example.org"
      MATRIX_ACCESS_TOKEN: "syt_..."
      MATRIX_ALLOWED_USERS: "@you:matrix.example.org"
      MATRIX_ENCRYPTION: "true"
      MATRIX_DEVICE_ID: "HERMES_BOT"

      # Proxy mode — forward to host agent
      GATEWAY_PROXY_URL: "http://192.168.1.100:8642"
      GATEWAY_PROXY_KEY: "your-secret-key-here"
    volumes:
      - ./matrix-store:/root/.hermes/platforms/matrix/store
```

--------------------------------

### Generate new Matrix access token via login API

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/user-guide/messaging/matrix.md

Obtain a fresh access token and device ID by authenticating with the Matrix homeserver. Required as the first step of E2EE migration when upgrading Hermes to the new SQLite crypto store.

```bash
curl -X POST https://your-server/_matrix/client/v3/login \
  -H "Content-Type: application/json" \
  -d '{
    "type": "m.login.password",
    "identifier": {"type": "m.id.user", "user": "@hermes:your-server.org"},
    "password": "***",
    "initial_device_display_name": "Hermes Agent"
  }'
```

--------------------------------

### Install `hermes-agent` with Matrix Extras

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/user-guide/messaging/matrix.md

Install the `hermes-agent` package with the `matrix` extra, which includes `mautrix[encryption]` and other Matrix-specific dependencies.

```bash
pip install 'hermes-agent[matrix]'
```

### Step 4: Configure Hermes Agent > Option B: Manual Configuration

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/user-guide/messaging/matrix.md

Manual configuration involves adding Matrix settings to the ~/.hermes/.env file, including the homeserver URL, authentication credentials (either access token or user ID and password), and a list of allowed user IDs to restrict bot interactions. The MATRIX_ALLOWED_USERS setting supports multiple users as a comma-separated list.

---

# Hermes Honcho memory integration and configuration

### Configure Memory with hermes honcho

Source: https://context7.com/nousresearch/hermes-agent/llms.txt

Set up Honcho AI integration for dialectic user modeling and cross-session memory. Supports hybrid, honcho, and local memory modes.

```bash
# Setup Honcho integration
hermes honcho setup

# Check connection status
hermes honcho status

# Map current directory to session name
hermes honcho map my-project

# Configure memory mode
hermes honcho mode hybrid  # hybrid|honcho|local
```

--------------------------------

### Manage Honcho memory integration with hermes honcho

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/reference/cli-commands.md

Use `hermes honcho` to manage cross-session memory when `memory.provider` is set to `honcho`. The `--target-profile` flag allows managing another profile's config without switching.

```bash
hermes honcho [--target-profile NAME] <subcommand>
```

--------------------------------

### CLI Command: hermes honcho

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/reference/cli-commands.md

Manage Honcho cross-session memory integration.

```APIDOC
## CLI Command: hermes honcho

### Description
Manage Honcho cross-session memory integration.

### Method
CLI

### Command Syntax
`hermes honcho`

### Example Usage
```bash
hermes honcho status
```
```

--------------------------------

### Setup Honcho Memory Provider

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/user-guide/features/honcho.md

Initialize Honcho as the memory provider using the interactive setup command or manual YAML configuration.

```bash
hermes memory setup    # select "honcho" from the provider list
```

```yaml
# ~/.hermes/config.yaml
memory:
  provider: honcho
```

```bash
echo "HONCHO_API_KEY=*** >> ~/.hermes/.env
```

### CLI Commands Reference > Top-level commands > Memory Management

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/reference/cli-commands.md

The `hermes honcho` command manages Honcho cross-session memory integration, and `hermes memory` configures external memory providers for persistent context across sessions.

---

# Hermes MCP server configuration native-mcp and mcporter

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

---

# Hermes troubleshooting common errors and logs

### hermes logs

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/reference/cli-commands.md

Allows viewing, tailing, and filtering Hermes log files. Users can specify which log to view, the number of lines, follow in real-time, filter by log level, session ID, time, or component.

```APIDOC
## hermes logs

### Description
View, tail, and filter Hermes log files. All logs are stored in `~/.hermes/logs/` (or `<profile>/logs/` for non-default profiles).

### Command
`hermes logs [log_name] [options]`

### Arguments
- **log_name** (string) - Optional - Which log to view: `agent` (default), `errors`, `gateway`, or `list` to show available files with sizes.

### Options
- **-n, --lines <N>** (integer) - Optional - Number of lines to show (default: 50).
- **-f, --follow** (boolean) - Optional - Follow the log in real time, like `tail -f`. Press Ctrl+C to stop.
- **--level <LEVEL>** (string) - Optional - Minimum log level to show: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
- **--session <ID>** (string) - Optional - Filter lines containing a session ID substring.
- **--since <TIME>** (string) - Optional - Show lines from a relative time ago: `30m`, `1h`, `2d`, etc. Supports `s` (seconds), `m` (minutes), `h` (hours), `d` (days).
- **--component <NAME>** (string) - Optional - Filter by component: `gateway`, `agent`, `tools`, `cli`, `cron`.

### CLI Examples
```bash
# View the last 50 lines of agent.log (default)
hermes logs

# Follow agent.log in real time
hermes logs -f

# View the last 100 lines of gateway.log
hermes logs gateway -n 100

# Show only warnings and errors from the last hour
hermes logs --level WARNING --since 1h

# Filter by a specific session
hermes logs --session abc123

# Follow errors.log, starting from 30 minutes ago
hermes logs errors --since 30m -f

# List all log files with their sizes
hermes logs list
```
```

--------------------------------

### Hermes Logs Command Examples

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/reference/cli-commands.md

Demonstrates various uses of the `hermes logs` command, including viewing default logs, following logs in real-time, filtering by log name, level, session, and time, and listing all log files.

```bash
hermes logs
```

```bash
hermes logs -f
```

```bash
hermes logs gateway -n 100
```

```bash
hermes logs --level WARNING --since 1h
```

```bash
hermes logs --session abc123
```

```bash
hermes logs errors --since 30m -f
```

```bash
hermes logs list
```

--------------------------------

### Common Hermes Cron Diagnostic Commands (Bash)

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/guides/cron-troubleshooting.md

Provides a set of commands for listing cron jobs, testing job execution, editing job configurations, viewing logs, and verifying installed skills. Use these to troubleshoot cron job issues.

```bash
hermes cron list                    # Show all jobs, states, next_run times
```

```bash
hermes cron run <job_id>            # Schedule for next tick (for testing)
```

```bash
hermes cron edit <job_id>           # Fix configuration issues
```

```bash
hermes logs                         # View recent Hermes logs
```

```bash
hermes skills list                  # Verify installed skills
```

--------------------------------

### Troubleshooting Hermes Agent Docker Commands

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/user-guide/docker.md

These commands provide various ways to check the health, logs, version, and resource usage of your Hermes Agent Docker container.

```sh
docker logs --tail 50 hermes
```

```sh
docker run -it --rm nousresearch/hermes-agent:latest version
```

```sh
docker stats hermes
```

--------------------------------

### CLI Command: hermes logs

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/reference/cli-commands.md

View, tail, and filter agent/gateway/error log files.

```APIDOC
## CLI Command: hermes logs

### Description
View, tail, and filter agent/gateway/error log files.

### Method
CLI

### Command Syntax
`hermes logs`

### Example Usage
```bash
hermes logs tail
```
```

---
