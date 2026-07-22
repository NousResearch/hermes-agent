---
name: hermes-agent
description: "Configure, extend, or contribute to Hermes Agent."
version: 2.4.0
author: Hermes Agent + Teknium
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [hermes, setup, configuration, multi-agent, spawning, cli, gateway, development]
    homepage: https://github.com/NousResearch/hermes-agent
    related_skills: [claude-code, codex, opencode]
---

# Hermes Agent

Hermes Agent is an open-source AI agent framework by Nous Research that runs in your terminal, a native desktop app, messaging platforms, and IDEs. It's in the same category as Claude Code (Anthropic), Codex (OpenAI), and OpenClaw — autonomous coding and task-execution agents that use tool calling to interact with your system. Hermes works with any LLM provider (OpenRouter, Anthropic, OpenAI, Google, DeepSeek, xAI, local models, and 20+ others) and runs on Linux, macOS, Windows, and WSL.

What makes Hermes different:

- **Self-improving through skills** — Hermes learns from experience by saving reusable procedures as skills. When it solves a complex problem, discovers a workflow, or gets corrected, it can persist that knowledge as a skill document that loads into future sessions. Skills accumulate over time, making the agent better at your specific tasks and environment.
- **Persistent memory across sessions** — remembers who you are, your preferences, environment details, and lessons learned. Pluggable memory backends (built-in, Honcho, Mem0, and more) let you choose how memory works.
- **Multi-platform gateway** — the same agent runs on Telegram, Discord, Slack, WhatsApp, iMessage, Signal, Matrix, Teams, Email, and a dozen more platforms with full tool access, not just chat.
- **Many surfaces** — the same agent core drives the CLI, the Ink TUI, a native Electron desktop app, a web dashboard, and an ACP server for IDEs (VS Code / Zed / JetBrains).
- **Provider-agnostic** — swap models and providers mid-workflow without changing anything else. Credential pools rotate across multiple API keys automatically.
- **Profiles** — run multiple independent Hermes instances with isolated configs, sessions, skills, and memory.
- **Extensible** — plugins, MCP servers, custom tools, webhook triggers, cron scheduling, and the full Python ecosystem.

People use Hermes for software development, research, system administration, data analysis, content creation, home automation, and anything else that benefits from an AI agent with persistent context and full system access.

**This skill helps you work with Hermes Agent effectively** — setting it up, configuring features, spawning additional agent instances, troubleshooting issues, finding the right commands and settings, and understanding how the system works when you need to extend or contribute to it.

**Docs:** https://hermes-agent.nousresearch.com/docs/
## Scope & Verification

This skill is a concise operating guide, not the complete source of truth for every Hermes feature. If a Hermes feature, command, or setting is not mentioned here, do not treat that absence as evidence that it does not exist. Check the live repository and official docs before giving a negative answer.

Good verification targets:

- CLI commands: `hermes --help`, `hermes <command> --help`, and `hermes_cli/main.py`
- User documentation: https://hermes-agent.nousresearch.com/docs/
- Source tree: https://github.com/NousResearch/hermes-agent

## Progressive loading contract

This top-level file is intentionally a compact router. Do not load every reference by default.

1. Use `skills_list(query="<task>")` when the right workflow is unclear.
2. Load only the reference or exact Markdown section that covers the task.
3. Treat a successful `skill_view` result as active for the rest of the conversation. Do not load the same unchanged skill/file/section again.
4. Use `force_reload: true` only for a deliberate reread. Context compression starts a child session and may load instructions again.
5. For current Hermes facts, the documentation at <https://hermes-agent.nousresearch.com/docs> is authoritative. Confirm CLI surfaces with `hermes --help` or `hermes <command> --help` before changing configuration.

## Task router

| Task | Load |
|---|---|
| Install, CLI commands, slash commands, profiles, auth | `references/cli-reference.md` |
| Config keys, providers, toolsets, project context files | `references/configuration-and-context.md` |
| Redaction, approval prompts, allowlists, browser/tool safety | `references/security-and-privacy.md` |
| Voice, subagents, cron, curator, Kanban, desktop/dashboard | `references/automation-and-surfaces.md` |
| Windows terminals, paths, sandboxing, tests | `references/windows.md` |
| Models, OAuth, gateway, tool and skill failures | `references/troubleshooting.md` |
| Repository layout, adding tools/commands, tests, commits | `references/contributing.md` |
| Native MCP implementation and debugging | `references/native-mcp.md` |
| Webhook setup and operation | `references/webhooks.md` |

Use focused retrieval for large references, for example:

```text
skill_view(name="hermes-agent", file_path="references/troubleshooting.md", section="Model/provider issues")
skill_view(name="hermes-agent", file_path="references/contributing.md", section="Testing")
```

## Safe operating workflow

1. Inspect the live state before asserting it: `hermes status`, `hermes doctor`, `hermes config`, logs, or the relevant source file.
2. Prefer supported CLI/config paths over editing internal files directly.
3. Never request or place passwords, OAuth tokens, API keys, recovery codes, or payment details in chat. Use Hermes credential flows or the configured secret store.
4. Keep profile scope explicit. Do not edit another profile's skills, plugins, cron jobs, or memories without direct user instruction.
5. After a change, exercise the real path and report the command/output that verified it.

## Essential commands

```bash
hermes setup
hermes model
hermes tools
hermes config
hermes status --all
hermes doctor
hermes gateway status
hermes sessions stats
```

Use the routed references for exact flags and platform-specific behavior.
