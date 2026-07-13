---
name: hermes-agent
description: "Configure, extend, or contribute to Hermes Agent."
version: 2.3.0
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

Hermes Agent is an open-source AI agent framework by Nous Research for terminal sessions, messaging platforms, desktop/web surfaces, and IDE integrations. It supports many model providers, persistent memory, profiles, skills, plugins, MCP servers, gateways, cron jobs, and delegated agents.

This skill is a navigation index. Load only the documentation relevant to the question. Do not treat a missing entry here as evidence that a feature does not exist: check the repository, the official docs, or the authoritative CLI help.

## Documentation lookup

The canonical documentation is under `website/docs/` in a Hermes checkout.

1. If the current workspace contains `website/docs/`, use that checkout and read paths below relative to the repository root.
2. If local docs are unavailable, use the live CLI surface first: `hermes --help` and the relevant `hermes <command> [<subcommand> ...] --help`. Preserve the target profile when it matters: plain `hermes` follows `hermes profile use <name>`, while `hermes -p <name> <command> [<subcommand> ...] --help` and the generated `<name> <command> --help` alias target a named profile explicitly.
3. If the CLI help does not answer the question, fetch the mapped file by appending its path to <https://raw.githubusercontent.com/NousResearch/hermes-agent/main/>. If raw docs are unavailable, use the official site: <https://hermes-agent.nousresearch.com/docs/>.

For source-level questions, inspect the repository at <https://github.com/NousResearch/hermes-agent>.

Do not assume a fixed checkout such as `~/.hermes/hermes-agent`; users may be working from another clone, a worktree, or an installed package.

## Quick documentation map

| User asks about | Read first |
|---|---|
| Installation, setup, updating | `website/docs/getting-started/quickstart.md`, `website/docs/getting-started/installation.md`, `website/docs/getting-started/updating.md` |
| CLI commands and flags | `website/docs/reference/cli-commands.md`, `website/docs/user-guide/cli.md` |
| Configuration, YAML, environment variables | `website/docs/user-guide/configuration.md`, `website/docs/reference/environment-variables.md` |
| Models and providers | `website/docs/user-guide/configuring-models.md`, `website/docs/integrations/providers.md`, `website/docs/user-guide/features/provider-routing.md`, `website/docs/user-guide/features/fallback-providers.md` |
| Authentication and credential pools | `website/docs/user-guide/features/credential-pools.md`, `website/docs/reference/cli-commands.md` |
| Sessions and profiles | `website/docs/user-guide/sessions.md`, `website/docs/user-guide/profiles.md`, `website/docs/reference/profile-commands.md` |
| Gateway and messaging | `website/docs/user-guide/messaging/index.md`, `website/docs/user-guide/messaging/<platform>.md` |
| Telegram topics, DMs, and setup | `website/docs/user-guide/messaging/telegram.md` |
| Cron jobs and failures | `website/docs/user-guide/features/cron.md`, `website/docs/guides/cron-troubleshooting.md`, `website/docs/guides/cron-script-only.md` |
| Tools and toolsets | `website/docs/user-guide/features/tools.md`, `website/docs/reference/tools-reference.md`, `website/docs/reference/toolsets-reference.md` |
| Skills and progressive disclosure | `website/docs/user-guide/features/skills.md`, `website/docs/developer-guide/creating-skills.md` |
| Adding a built-in tool | `website/docs/developer-guide/adding-tools.md` |
| MCP servers | `website/docs/user-guide/features/mcp.md`, `website/docs/reference/mcp-config-reference.md` |
| Memory and memory providers | `website/docs/user-guide/features/memory.md`, `website/docs/user-guide/features/memory-providers.md` |
| Voice, TTS, STT, or vision | `website/docs/user-guide/features/voice-mode.md`, `website/docs/user-guide/features/tts.md`, `website/docs/user-guide/features/vision.md` |
| Browser, web search, image, or video tools | `website/docs/user-guide/features/browser.md`, `website/docs/user-guide/features/web-search.md`, `website/docs/developer-guide/web-search-provider-plugin.md`, `website/docs/developer-guide/image-gen-provider-plugin.md`, `website/docs/developer-guide/video-gen-provider-plugin.md` |
| Delegation and subagents | `website/docs/user-guide/features/delegation.md`, `website/docs/guides/delegation-patterns.md` |
| Kanban and multi-agent work queues | `website/docs/user-guide/features/kanban.md`, `website/docs/user-guide/features/kanban-worker-lanes.md` |
| Dashboard, desktop, TUI, or proxy | `website/docs/user-guide/features/web-dashboard.md`, `website/docs/user-guide/tui.md`, `website/docs/user-guide/features/subscription-proxy.md` |
| Security, secrets, or redaction | `website/docs/user-guide/security.md`, `website/docs/user-guide/secrets/index.md` |
| Context files and prompt behavior | `website/docs/user-guide/features/context-files.md`, `website/docs/developer-guide/prompt-assembly.md` |
| Windows, WSL, or cross-platform behavior | `website/docs/user-guide/windows-wsl-quickstart.md`, `website/docs/user-guide/windows-native.md` |
| Troubleshooting | `website/docs/reference/faq.md`, then the feature-specific guide |
| Developing or contributing | `website/docs/developer-guide/contributing.md`, `website/docs/developer-guide/architecture.md` |
| Agent loop internals | `website/docs/developer-guide/agent-loop.md`, `website/docs/developer-guide/context-compression-and-caching.md` |
| Plugins and provider extensions | `website/docs/developer-guide/model-provider-plugin.md`, `website/docs/developer-guide/provider-runtime.md`, `website/docs/user-guide/features/plugins.md` |

For platform-specific messaging questions, replace `<platform>` with the matching file in the messaging docs directory. If a mapped path is absent in the checkout, use the official docs site or inspect the docs tree instead of guessing.

## Fast verification

- Run `hermes --help` and the relevant subcommand help before asserting the current CLI surface.
- Read the relevant documentation section before answering configuration or troubleshooting questions.
- Inspect source when the docs and runtime disagree.
- Treat user configuration as runtime truth: read the active config rather than relying on stale examples.

## Safety and operating rules

- Never restart a gateway, change user configuration, install dependencies, or perform an external write without the required user approval.
- Do not fabricate commands, providers, paths, or feature support.
- Use `get_hermes_home()` for Hermes state paths in code; never hardcode `~/.hermes`.
- Keep secrets in `.env` and configuration values in `config.yaml`.
- Do not change tools, skills, or system-prompt inputs mid-session when prompt-cache stability matters; start a fresh session when the docs require it.
- Preserve message role alternation when working on conversation or gateway code.
- New tools need a requirements check so unavailable dependencies do not expose broken tools.

## Useful live commands

```bash
hermes --help
hermes <command> --help
hermes doctor
hermes config check
hermes config path
hermes config env-path
hermes skills list
hermes tools list
```

For contribution work, keep the PR focused, run the relevant tests, test the changed path manually when possible, and check cross-platform impact. The canonical workflow is in `website/docs/developer-guide/contributing.md`.
