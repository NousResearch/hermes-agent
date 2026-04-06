---
name: acp-agents
description: Use external coding agents (Claude Code, Codex CLI, Gemini CLI, Cursor, Copilot, and more) as Hermes backend providers via the Agent Client Protocol (ACP). Switch between 14 supported agents with /model --provider.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [ACP, Agents, Providers, CLI, Claude, Codex, Gemini]
    related_skills: [native-mcp]
---

# ACP Agent Providers

Hermes can use external coding agents as backend providers via the **Agent Client Protocol (ACP)**. Instead of calling an LLM API directly, Hermes spawns a coding agent (Claude Code, Codex CLI, Gemini CLI, etc.) and communicates with it over the standardized ACP protocol.

This means you get the full capabilities of these agents — their tool use, file editing, terminal access, and reasoning — orchestrated through Hermes.

## Supported Agents

| Provider ID | Agent | ACP Adapter |
|-------------|-------|-------------|
| `claude-acp` | Claude Code | `acpx claude` |
| `codex-acp` | Codex CLI | `acpx codex` |
| `gemini-acp` | Gemini CLI | `acpx gemini` |
| `copilot-acp` | GitHub Copilot | `acpx copilot` |
| `cursor-acp` | Cursor | `acpx cursor` |
| `kiro-acp` | Kiro | `acpx kiro` |
| `kilocode-acp` | KiloCode | `acpx kilocode` |
| `opencode-acp` | OpenCode | `acpx opencode` |
| `kimi-acp` | Kimi | `acpx kimi` |
| `qwen-acp` | Qwen | `acpx qwen` |
| `cline-acp` | Cline | `acpx cline` |
| `amp-acp` | Amp | `acpx amp` |
| `droid-acp` | Droid | `acpx droid` |
| `iflow-acp` | iFlow | `acpx iflow` |

All agents are spawned via `acpx`, the standalone ACP headless client. It handles protocol negotiation, authentication (including OAuth), and session management.

## How to Switch

Use the `/model` command with the `--provider` flag:

```
/model --provider claude-acp
/model --provider codex-acp
/model --provider gemini-acp
```

To persist the switch across sessions:

```
/model --provider claude-acp --global
```

To switch back to a regular API provider:

```
/model sonnet --provider openrouter
/model opus --provider anthropic
```

## Prerequisites and Authentication

Each ACP agent has its own auth requirements. **ACP adapters use API keys, not CLI subscription/OAuth tokens.**

| Agent | Auth Required | How to Set |
|-------|--------------|------------|
| **Claude** | `ANTHROPIC_API_KEY` (paid API key) | [console.anthropic.com](https://console.anthropic.com) |
| **Codex** | OpenAI API key or ChatGPT subscription | `codex auth` |
| **Gemini** | Google Cloud API key or `gcloud auth` | `gemini auth login` |
| **Copilot** | GitHub Copilot subscription | `copilot auth` |

Authentication is handled by `acpx`, which supports both API keys and OAuth tokens. If you have a Claude Pro/Max subscription with an OAuth token, `acpx` will handle the gateway authentication automatically. A separate API key from the Anthropic console also works.

Agents distributed via npx (Claude, Codex, KiloCode, OpenCode, Cline) are auto-installed on first use. Node.js is required for npx-based agents. Agents like Gemini, Cursor, Copilot, Kiro, Kimi, and Qwen must be installed globally.

## Custom Agent Commands

Override the launch command for any agent via environment variables:

```bash
# Use a specific Claude Code binary
export HERMES_ACP_CLAUDE_COMMAND="/usr/local/bin/claude-agent-acp"

# Use a custom Codex path
export HERMES_ACP_CODEX_COMMAND="/opt/codex/bin/codex-acp"

# Point to a local dev build
export HERMES_ACP_GEMINI_COMMAND="/home/me/gemini-dev/gemini --acp --debug"
```

The env var pattern is `HERMES_ACP_{AGENT_NAME}_COMMAND` where the agent name is uppercased with hyphens replaced by underscores.

## How It Works

When you select an ACP provider, Hermes:

1. **Spawns** the agent's ACP adapter as a subprocess with stdin/stdout pipes
2. **Initializes** an ACP session (JSON-RPC handshake over NDJSON)
3. **Sends** your conversation as a prompt through the ACP protocol
4. **Streams** the agent's response back (text chunks and reasoning)
5. **Handles callbacks** — file reads, file writes, and permission requests from the agent are handled by Hermes within the project working directory

The agent runs with full access to its native capabilities (tool calling, code editing, terminal commands) scoped to the current working directory. File operations outside the working directory are blocked for security.

## Differences from API Providers

| | API Providers | ACP Agent Providers |
|---|---|---|
| **Communication** | HTTPS to cloud API | Stdio to local process |
| **Authentication** | API key / OAuth | Agent's own auth (local) |
| **Capabilities** | Chat completions | Full agent (tools, files, terminal) |
| **Billing** | Per-token API usage | Agent's subscription model |
| **Latency** | Network round-trip | Local process, no network |
| **Model selection** | You choose the model | Agent decides (its default model) |

## Troubleshooting

**"Could not start ACP agent"** — The agent binary is not installed or not on PATH. Install it or set the `HERMES_ACP_{NAME}_COMMAND` env var.

**"Unknown ACP agent"** — The agent name is not in the built-in registry. Use `HERMES_ACP_{NAME}_COMMAND` to define a custom agent.

**"did not return a sessionId"** — The ACP adapter started but failed during initialization. Check the agent's own auth status (e.g., `claude auth status`, `codex auth`).

**Timeout** — The agent took too long to respond. Default timeout is 15 minutes. Some agents need time for first-run setup (npx download, auth prompts).

**Permission errors on file operations** — File reads/writes are restricted to the session working directory. The agent cannot access files outside the project root.
