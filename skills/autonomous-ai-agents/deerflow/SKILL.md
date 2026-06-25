---
name: deerflow
description: "Use when bootstrapping, running, or delegating work to ByteDance DeerFlow: an open-source super-agent harness for deep research, sub-agents, memory, sandboxed execution, skills, and messaging channels."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [DeerFlow, Agent-Harness, Research-Agent, LangGraph, Skills, Sandboxes]
    related_skills: [hermes-agent, codex, claude-code, native-mcp]
---

# DeerFlow

## Overview

[DeerFlow](https://github.com/bytedance/deer-flow) is ByteDance's open-source super-agent harness. DeerFlow 2.0 orchestrates a lead agent, sub-agents, persistent memory, sandboxed filesystem/command execution, MCP servers, skills, and optional IM channels. Use it when the user wants a DeerFlow instance set up, evaluated, compared to Hermes, or used as a long-running research/build harness.

Canonical upstream:

- Repo: `https://github.com/bytedance/deer-flow`
- Website: `https://deerflow.tech`
- Install guide: `https://raw.githubusercontent.com/bytedance/deer-flow/main/Install.md`
- Default UI endpoint after startup: `http://localhost:2026`

## When to Use

- The user explicitly mentions DeerFlow, `deer-flow`, or ByteDance's super-agent harness.
- The task is a long-running deep research or multi-agent exploration where DeerFlow is the requested runtime.
- The user asks to compare Hermes, OpenClaw, DeerFlow, LangGraph-based agent harnesses, or sandboxed agent execution systems.
- The user wants to wire DeerFlow to Claude Code, Codex, Cursor, Windsurf, Copilot, MCP, or messaging channels.

Do not use this skill just because a task involves research. Use Hermes' native web/search/delegation tools unless DeerFlow itself is part of the requested environment.

## Prerequisites

DeerFlow 2.0 expects current language runtimes:

| Requirement | Baseline |
|---|---|
| Python | 3.12+ |
| Node.js | 22+ |
| Package tooling | `uv`, `pnpm` / repo-managed scripts |
| Persistent server | Docker-capable Linux preferred |
| Local evaluation | 4 vCPU / 8 GB RAM minimum; 8 vCPU / 16 GB recommended |

Run prerequisite checks from the repo root, not from a parent directory.

```bash
git clone https://github.com/bytedance/deer-flow.git
cd deer-flow
make check
```

If Docker commands fail with `permission denied while trying to connect to the Docker daemon socket`, add the user to the `docker` group and re-login before retrying.

## Fast Setup Path

For an agentic setup handoff, use the upstream one-line instruction verbatim:

```text
Help me clone DeerFlow if needed, then bootstrap it for local development by following https://raw.githubusercontent.com/bytedance/deer-flow/main/Install.md
```

For direct setup:

```bash
git clone https://github.com/bytedance/deer-flow.git
cd deer-flow
make setup
make doctor
```

`make setup` opens an interactive wizard for LLM provider, optional search provider, sandbox mode, bash access, and write-tool preferences. It creates `config.yaml` and writes credentials to `.env`.

Manual config path:

```bash
make config          # copies the full config template
$EDITOR config.yaml
make doctor
```

## Running DeerFlow

### Docker Development

Use this for most local evaluation. It hot-reloads source and follows `config.yaml` sandbox settings.

```bash
make docker-init     # pull sandbox image, first run only or after image updates
make docker-start    # start services
```

Stop/restart:

```bash
make docker-stop
make docker-start
```

### Docker Production

Use this for a persistent shared instance.

```bash
make up              # build images and start production services
make down            # stop and remove containers
```

### Local Development

Use this when Docker is unavailable or when debugging the Python/Node services directly.

```bash
make install
make dev
```

Startup variants exposed by the repo:

```bash
./scripts/serve.sh --dev
./scripts/serve.sh --dev --daemon
./scripts/serve.sh --prod
./scripts/serve.sh --stop
./scripts/serve.sh --restart
```

Access the UI at `http://localhost:2026` unless the user configured a different endpoint.

## Model Configuration Notes

DeerFlow is model-agnostic and uses OpenAI-compatible / LangChain-compatible model entries in `config.yaml`.

Common hosted model shape:

```yaml
models:
  - name: gpt-4o
    display_name: GPT-4o
    use: langchain_openai:ChatOpenAI
    model: gpt-4o
    api_key: $OPENAI_API_KEY
```

OpenRouter / compatible gateways:

```yaml
models:
  - name: openrouter-gemini-2.5-flash
    display_name: Gemini 2.5 Flash (OpenRouter)
    use: langchain_openai:ChatOpenAI
    model: google/gemini-2.5-flash-preview
    api_key: $OPENROUTER_API_KEY
    base_url: https://openrouter.ai/api/v1
```

OpenAI Responses API:

```yaml
models:
  - name: gpt-5-responses
    display_name: GPT-5 (Responses API)
    use: langchain_openai:ChatOpenAI
    model: gpt-5
    api_key: $OPENAI_API_KEY
    use_responses_api: true
    output_version: responses/v1
```

CLI-backed providers:

```yaml
models:
  - name: gpt-5.4
    display_name: GPT-5.4 (Codex CLI)
    use: deerflow.models.openai_codex_provider:CodexChatModel
    model: gpt-5.4
    supports_thinking: true
    supports_reasoning_effort: true

  - name: claude-sonnet-4.6
    display_name: Claude Sonnet 4.6 (Claude Code OAuth)
    use: deerflow.models.claude_provider:ClaudeChatModel
    model: claude-sonnet-4-6
    max_tokens: 4096
    supports_thinking: true
```

Credential locations from upstream docs:

- Codex CLI reads `~/.codex/auth.json`.
- Claude Code accepts `CLAUDE_CODE_OAUTH_TOKEN`, `ANTHROPIC_AUTH_TOKEN`, `CLAUDE_CODE_CREDENTIALS_PATH`, or `~/.claude/.credentials.json`.
- ACP agent entries are separate from model provider entries.

## Sandboxes and Security

DeerFlow can run with local execution, Docker execution, or Docker/Kubernetes provisioner execution. Treat these as material security choices.

- `AioSandboxProvider` runs shell execution inside isolated containers.
- `LocalSandboxProvider` maps file tools to per-thread host directories, but host bash is disabled by default because it is not a secure isolation boundary.
- Re-enable host bash only for trusted local workflows.
- Do not expose DeerFlow publicly without authentication, IP allowlists, or a proper access gateway.

The upstream security notice is explicit: DeerFlow has high-privilege capabilities including command execution, resource operations, and business logic invocation, and is designed by default for local trusted access.

## MCP and Messaging Channels

DeerFlow supports configurable MCP servers and optional IM channels.

Relevant docs in the upstream repo:

- `backend/docs/MCP_SERVER.md`
- `backend/docs/IM_CHANNEL_CONNECTIONS.md`
- `backend/docs/CONFIGURATION.md`

Supported IM channels include Telegram, Slack, Feishu/Lark, DingTalk, WeChat, and WeCom. Channels call the Gateway API internally. In Docker Compose, do not point channel URLs at `localhost`; use service names such as:

```yaml
channels:
  langgraph_url: http://gateway:8001/api
  gateway_url: http://gateway:8001
```

## Claude Code Integration

DeerFlow ships a `claude-to-deerflow` skill for controlling a running DeerFlow instance from Claude Code.

```bash
npx skills add https://github.com/bytedance/deer-flow --skill claude-to-deerflow
```

Then ensure DeerFlow is running and use the `/claude-to-deerflow` command from Claude Code. Optional endpoint variables:

```bash
DEERFLOW_URL=http://localhost:2026
DEERFLOW_GATEWAY_URL=http://localhost:2026
DEERFLOW_LANGGRAPH_URL=http://localhost:2026/api/langgraph
```

## Evaluation Workflow

When asked to evaluate DeerFlow rather than merely install it:

1. Clone a fresh copy or inspect the existing checkout.
2. Read `README.md`, `Install.md`, `backend/docs/CONFIGURATION.md`, and any requested integration docs.
3. Run `make check` and `make doctor` before attempting a full run.
4. Start with Docker dev (`make docker-init && make docker-start`) unless the environment lacks Docker.
5. Verify the UI/API responds at `http://localhost:2026`.
6. Run a small research prompt and inspect generated workspace/output artifacts.
7. Report exact commands, paths, endpoint, and blockers.

## Common Pitfalls

1. **Using DeerFlow v1 docs for DeerFlow 2.0.** DeerFlow 2.0 is a ground-up rewrite. The older deep-research framework lives on the `main-1.x` branch.
2. **Under-sizing the host.** Two CPU cores and 4 GB RAM is usually not enough. Start at 4 vCPU / 8 GB for evaluation and 8 vCPU / 16 GB for real use.
3. **Exposing the UI/API on a public interface without access control.** DeerFlow can execute commands and manipulate files.
4. **Pointing Docker channels at `localhost`.** Inside Compose, `localhost` is the container itself; use `gateway` service URLs.
5. **Treating CLI auth as API-key auth.** Codex/Claude Code provider entries can rely on their own CLI OAuth files instead of `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`.
6. **Skipping `make doctor`.** It is the fastest way to get actionable setup fixes.

## Verification Checklist

- [ ] Repo is cloned from `https://github.com/bytedance/deer-flow`.
- [ ] Runtime versions satisfy Python 3.12+ and Node 22+.
- [ ] `make check` and `make doctor` have been run and quoted in the report.
- [ ] `config.yaml` and `.env` exist without secrets being printed.
- [ ] Selected startup mode is explicit: Docker dev, Docker prod, or local dev.
- [ ] UI/API endpoint responds, usually `http://localhost:2026`.
- [ ] If exposed beyond localhost, access control is documented.
