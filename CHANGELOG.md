# Changelog

All notable changes to Hermes Agent are documented in per-release notes linked below.
Each release note includes commit counts, merged PRs, contributor credits, and detailed feature descriptions.

## Releases

| Version | Date | Highlights |
|---------|------|------------|
| [v0.14.0](RELEASE_v0.14.0.md) | 2026-05-16 | PyPI install, xAI Grok OAuth, `x_search` tool, Microsoft Teams, 180x faster browser CDP, cold-start perf wave, LINE + SimpleX Chat, debloating wave |
| [v0.13.0](RELEASE_v0.13.0.md) | 2026-04-08 | Kanban multi-agent orchestration, Gemini 2.5 Pro, MCP serve, `delegate_task` orchestrator mode, Unicode-safe Windows, `hermes update --check` |
| [v0.12.0](RELEASE_v0.12.0.md) | 2026-03-18 | ACP adapter (VS Code/Zed/JetBrains), Ink TUI beta, `hermes dashboard`, video generation tool, `hermes proxy`, LSP diagnostics, `hermes claw migrate` |
| [v0.11.0](RELEASE_v0.11.0.md) | 2026-02-24 | Bedrock provider, MCP stdio servers, `hermes mcp serve`, cron delivery to Signal/Matrix, context engine plugin, `hermes profile export/import` |
| [v0.10.0](RELEASE_v0.10.0.md) | 2026-02-03 | Credential pools, `hermes auth` CLI, provider rotation, Anthropic prompt caching, `hermes sessions prune`, `hermes insights` |
| [v0.9.0](RELEASE_v0.9.0.md) | 2026-01-13 | Honcho dialectic memory, Skills Hub v2, `hermes skills publish`, conditional skill activation, skin engine, `hermes doctor --fix` |
| [v0.8.0](RELEASE_v0.8.0.md) | 2025-12-16 | `delegate_task` subagents, `execute_code` sandboxed Python, `hermes webhook`, `hermes cron`, checkpoint/rollback, `hermes dashboard` |
| [v0.7.0](RELEASE_v0.7.0.md) | 2025-11-25 | Signal CLI adapter, Matrix adapter, Mattermost adapter, DingTalk adapter, Feishu adapter, BlueBubbles (iMessage) adapter |
| [v0.6.0](RELEASE_v0.6.0.md) | 2025-11-04 | OpenRouter provider routing, `hermes tools` interactive UI, toolset presets per platform, `hermes config migrate`, `hermes sessions browse` |
| [v0.5.0](RELEASE_v0.5.0.md) | 2025-10-14 | WhatsApp (Baileys bridge), Email (IMAP/SMTP), SMS, `hermes gateway install`, `hermes gateway setup`, session compression |
| [v0.4.0](RELEASE_v0.4.0.md) | 2025-09-22 | Discord adapter, Slack adapter, `hermes gateway run`, `hermes sessions list`, `hermes sessions export`, multi-platform gateway |
| [v0.3.0](RELEASE_v0.3.0.md) | 2025-09-02 | Telegram adapter (first messaging platform), `hermes setup` wizard, `hermes model` picker, `hermes config edit`, memory system |
| [v0.2.0](RELEASE_v0.2.0.md) | 2025-08-12 | Initial public release — CLI chat, OpenRouter/Anthropic/OpenAI providers, tool registry, terminal/file/web tools, skills system |

## Format

Each release note follows this structure:
- **Highlights** — top features with PR links
- **Breaking changes** — migration steps when needed
- **New tools & skills** — what shipped
- **Bug fixes** — notable fixes with issue links
- **Contributors** — everyone who contributed (including co-authors)

## See Also

- [GitHub Releases](https://github.com/NousResearch/hermes-agent/releases) — tags and assets
- [Contributing Guide](CONTRIBUTING.md) — how to contribute
- [Security Policy](SECURITY.md) — vulnerability reporting
