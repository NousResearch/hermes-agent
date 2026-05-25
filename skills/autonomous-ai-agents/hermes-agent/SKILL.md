---
name: hermes-agent
description: "Configure, extend, or contribute to Hermes Agent."
version: 2.1.0
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

Hermes Agent is an open-source, provider-agnostic AI agent framework by Nous Research. It runs in the terminal, gateways such as Telegram/Discord/Slack, IDE integrations, cron jobs, webhooks, MCP, and profile-isolated multi-agent setups.

Use this skill when Alexander asks to configure, troubleshoot, extend, operate, or reason about Hermes Agent itself: CLI, setup, config, models/providers, gateway platforms, tools, skills, voice, profiles, plugins, MCP, cron, webhooks, or contributor workflows.

Docs: https://hermes-agent.nousresearch.com/docs/

## Load Policy

Keep the main skill compact. Load detailed references only for the task at hand:

- `references/cli-and-configuration.md` — CLI commands, slash commands, config paths, providers, toolsets, security/privacy, voice.
- `references/agent-operations.md` — spawning Hermes instances, tmux/PTY patterns, cron jobs, webhooks, background systems, Windows quirks.
- `references/troubleshooting.md` — common failures: tools, gateways, x_search, providers, skills, voice, config changes.
- `references/contributor-reference.md` — repo layout, adding tools/commands, tests, commits, contributor rules.

Do not load all references by default; that defeats the token-hygiene split.

## Quick Start

```bash
# Install
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash

# Interactive chat
hermes

# Single query
hermes chat -q "What can you do?"

# Setup / health
hermes setup
hermes doctor
hermes status --all
```

Core commands:

```bash
hermes model                  # choose/change provider and model
hermes config                 # view config
hermes config edit            # edit config.yaml
hermes tools list             # show toolsets
hermes skills list            # list skills
hermes gateway status         # messaging gateway status
hermes logs --level warning   # inspect logs
```

## High-Value Workflows

### Profiles

Use profiles for isolated agents with separate config, skills, memory, sessions, and gateways:

```bash
hermes -p research chat -q "Use X search for ..."
hermes -p assistant gateway status
hermes -p research gateway restart
```

For Alexander's setup, profile isolation matters. Do not assume `assistant`, `research`, `hans-catering`, `zeiterfassung`, or other profiles share auth, toolsets, memory, or Telegram state.

### Gateway Operations

```bash
hermes gateway status
hermes gateway restart
hermes -p <profile> gateway status
hermes -p <profile> gateway restart
```

When Telegram receives messages but no answer arrives, inspect the relevant profile gateway status and logs before changing code or tokens.

### Tools and Skills

```bash
hermes tools list
hermes tools enable <toolset>
hermes tools disable <toolset>
hermes skills list
hermes skills search <query>
```

When a tool appears unavailable, verify the active profile and platform toolset. Gateway sessions may need a restart or active-session reset after config/SOUL/toolset changes.

### X Search vs xurl

- `x_search`: Grok/X Search style research through xAI credentials.
- `xurl`: official X API action/search path and may fail due to X Developer credits even when xAI/X Search works.

Do not conflate `xurl CreditsDepleted` with Grok/xAI auth failure.

## Troubleshooting Triage

1. Identify the exact profile: `hermes -p <profile> status`.
2. Check config and env paths: `hermes -p <profile> config path`, `hermes -p <profile> config env-path`.
3. Check gateway/service if platform messages are involved: `hermes -p <profile> gateway status`.
4. Check logs: `hermes -p <profile> logs --level warning` or profile log files.
5. Reproduce with a narrow CLI smoke test before changing persistent config.
6. Only then repair auth, toolsets, gateway, skills, or code.

Load `references/troubleshooting.md` for detailed cases.

## Development / Contribution Rules

When editing Hermes itself:

- Follow repo `AGENTS.md`.
- Prefer focused changes; avoid unrelated refactors.
- Use explicit-path staging; never `git add .`.
- Run the narrowest relevant validation, then broader tests when code changes require it.
- For skill changes, validate frontmatter and ensure references are linked.

Load `references/contributor-reference.md` for implementation details.

## Common Pitfalls

1. **Wrong profile.** Many issues are per-profile config/auth/toolset/session problems.
2. **Gateway state not refreshed.** Tool/SOUL/config changes may not affect existing gateway sessions until restart/reset.
3. **Assistant vs research auth drift.** One profile can be logged in while another is broken.
4. **x_search vs xurl confusion.** They use different paths and failure modes.
5. **Overloading this skill.** If a task needs full CLI or contributor details, load the targeted reference instead of expanding the main skill again.

## Verification Checklist

- [ ] Active profile identified.
- [ ] Relevant config/env/log path checked when troubleshooting.
- [ ] Gateway status checked for platform issues.
- [ ] Toolset/skill availability verified in the same profile/platform that will run the task.
- [ ] For repo edits: affected files and validation/test commands reported.
