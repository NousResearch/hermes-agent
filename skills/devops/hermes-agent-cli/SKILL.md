---
name: hermes-agent-cli
description: Use when working with Hermes Agent CLI commands, configuration, profiles, cron jobs, gateway control, or interactive chat sessions.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [cli, commands, configuration, gateway, profiles, cron, setup]
    related_skills: [hermes-agent]
---

# Hermes Agent CLI

## Overview

Practical reference for the `hermes` command-line interface. Covers the most-used commands, flag conventions, config keys, and Windows-specific gotchas for day-to-day operation of Hermes Agent.

## When to Use

- Running `hermes ...` in a terminal.
- Checking health, logs, or component status.
- Editing config, credentials, or environment variables.
- Managing cron jobs, skills, MCP servers, or profiles.
- Starting/stopping the messaging gateway.
- Spawning additional Hermes instances or delegating subtasks.
- Diagnosing model/provider/tool issues.

**Don't use for:** high-level architecture decisions, in-repo contribution workflows, or skill authoring. Those belong in `hermes-agent` skill docs.

## Quick Reference

```bash
hermes                          # interactive chat
hermes chat -q "..."            # single question
hermes setup                    # guided setup
hermes doctor                   # health check
hermes cron list                # scheduled jobs
hermes gateway status           # messaging gateway health
hermes skills list              # installed skills
hermes profile list             # named profiles
```

## Chat & Query

```bash
hermes chat -q "..."                          # non-interactive query
hermes chat -q "..." -m MODEL --provider P    # override model/provider
hermes chat -q "..." -t toolsets              # restrict toolsets
hermes --resume SESSION_ID                   # continue a session
hermes --continue NAME                       # resume most recent named session
hermes -w                                    # worktree mode (isolated git)
hermes -s skill-a,skill-b                    # preload skills
hermes --yolo                                 # skip dangerous command approval
```

## Configuration

```bash
hermes config                    # show current config
hermes config edit               # open config.yaml in $EDITOR
hermes config set KEY VAL        # set a config value
hermes config path               # print config.yaml path
hermes config env-path           # print .env path
hermes config check              # missing/outdated config
hermes config migrate            # update config with new options
hermes model                     # interactive model/provider picker
hermes auth                      # interactive credential manager
hermes auth add PROVIDER         # add OAuth or API-key credential
hermes auth list                 # list stored credentials
hermes auth remove PROVIDER IDX  # remove by provider + index
```

### Key Config Sections

| Section | Common keys |
|---------|-------------|
| `model` | `default`, `provider`, `base_url`, `api_key`, `context_length` |
| `agent` | `max_turns` |
| `terminal` | `backend`, `cwd`, `timeout` |
| `security` | `tirith_enabled`, `website_blocklist` |
| `privacy` | `redact_pii` |
| `approvals` | `mode` (`manual` / `smart` / `off`) |
| `memory` | `memory_enabled`, `user_profile_enabled`, `provider` |
| `gateway` | platform routing and delivery settings |
| `delegation` | `model`, `provider`, `max_iterations`, `reasoning_effort` |

## Status & Diagnostics

```bash
hermes status            # current component state
hermes status --all      # extended diagnostics
hermes doctor [--fix]    # dependencies and config check
hermes insights [--days N]   # usage analytics
hermes sessions list         # recent sessions
hermes sessions stats        # session store stats
hermes sessions prune --older-than 30d   # cleanup old sessions
```

Log files live under `~/.hermes/logs/`. Read exact log paths before broad discovery to avoid rotated/fragment files.

## Cron Jobs

```bash
hermes cron list            # list jobs
hermes cron list --all      # include disabled jobs
hermes cron status          # scheduler substrate health
hermes cron create SCHED    # create job ('30m', '0 9 * * *', 'every monday 9am')
hermes cron edit JOB_ID     # edit schedule/prompt/delivery
hermes cron pause JOB_ID    # pause
hermes cron resume JOB_ID   # resume
hermes cron run JOB_ID      # trigger on next scheduler tick
hermes cron remove JOB_ID   # delete
```

Cron output files are written under `~/AppData/Local/hermes/cron/output/<job_id>/`. To inspect the latest run file for a given job, prefer `ls -t ~/AppData/Local/hermes/cron/output/<job_id>/*.md | head -1` rather than filename inference.

## Gateway (Messaging Platforms)

```bash
hermes gateway run          # foreground (for logs)
hermes gateway install      # install as background service
hermes gateway start        # start service
hermes gateway stop         # stop service
hermes gateway restart      # restart service
hermes gateway status       # check state
hermes gateway setup        # configure platforms
```

Supported: Telegram, Discord, Slack, WhatsApp, Signal, Email, SMS, Matrix, Mattermost, Home Assistant, DingTalk, Feishu, WeCom, BlueBubbles, Weixin, API Server, Webhooks.

## Tools & Skills

```bash
hermes tools                # interactive enable/disable (curses UI)
hermes tools list           # show all tools and status
hermes tools enable NAME    # enable a toolset
hermes tools disable NAME   # disable a toolset
hermes skills list          # installed skills
hermes skills search QUERY  # search the skills hub
hermes skills install ID    # install from hub or direct SKILL.md URL
```

Changes to tools/skills require a new session (`/reset` in chat, or start a new `hermes` invocation).

## Profiles

```bash
hermes profile list         # list all profiles
hermes profile create NAME  # create new profile
hermes profile use NAME     # set sticky default
hermes profile show NAME    # show profile details
hermes profile delete NAME  # delete a profile
hermes profile export NAME  # export to tar.gz
hermes profile import FILE  # import archive
```

Profiles keep isolated configs, sessions, skills, and memory. Use `-p NAME`/`--profile NAME` to select one per invocation.

## Credential Pools

```bash
hermes auth add             # interactive credential wizard
hermes auth list [PROVIDER] # list pooled credentials
hermes auth remove P INDEX  # remove by provider + index
hermes auth reset PROVIDER  # clear exhaustion status
```

Logged-in auth only proves the credential is valid; whether calls are free depends on the selected model and provider pricing.

## Spawning Subprocesses

```bash
hermes chat -q "..."                        # one-shot/fire-and-forget
hermes chat -q "..." background=true        # long task in background
hermes -w                                   # worktree isolation for code edits
```

For interactive long-lived agents, prefer tmux/nohup over raw PTY; `terminal(pty=true)` is available when spawning from Hermes itself.

## MCP Servers

```bash
hermes mcp list              # configured servers
hermes mcp add NAME          # add server (--url or --command)
hermes mcp remove NAME       # remove
hermes mcp test NAME         # test connection
hermes mcp configure NAME    # toggle tool selection
```

The built-in MCP client auto-discovers tools from configured servers. Catalog install is supported via `hermes mcp install <name>`.

## Windows-Specific Notes

- **Alt+Enter:** toggles fullscreen in Windows Terminal; use **Ctrl+Enter** for newline insertion.
- **BOM config:** first-run `HTTP 400 No models provided` often means `config.yaml` was saved with UTF-8 BOM. Re-save without BOM.
- **Sandbox env:** `WinError 10106` is usually caused by Hermes scrubbing `SYSTEMROOT` from the sandbox child env. The built-in fix is in `_WINDOWS_ESSENTIAL_ENV_VARS`.
- **Python/pytest:** the Hermes venv can be stripped of pip/pytest. Use the system interpreter or `uv run --with pytest ...` for test runs, and prefer `-n 0` on Windows.
- **Rich CLI tables:** avoid piping Hermes CLI output into `head` in cron/background contexts; use file/SQLite/safe enumerators instead.

## Common Pitfalls

1. **BOM in `config.yaml`.** Notepad and some Windows editors write UTF-8 with BOM, which can break first-run config parsing on Windows.
2. **Tool/skill changes taking effect.** Changes require a fresh session (`/reset` in chat) or new process; they do not apply mid-conversation.
3. **Model/provider hot-reload.** `hermes config set model.default ...` does not change the active session model until you start a fresh `hermes` process.
4. **Windows `python` alias.** `python - <<'PY' ...` can resolve to an app launcher alias in Git Bash. Use the explicit Hermes venv interpreter: `C:/Users/82109/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe`.
5. **Broad `search_files` under `~/AppData/Local/hermes/...` may under-report on Windows.** Fall back to `terminal` with `ls` or `find` for absolute-path verification.

## Verification Checklist

- [ ] Run `hermes doctor` for healthy config/dependency state.
- [ ] Use `hermes cron list` / `hermes gateway status` to verify automation substrate.
- [ ] Inspect logs under `~/.hermes/logs/` for gateway, cron, or model errors.
- [ ] Confirm skill/profile changes with `hermes skills list` / `hermes profile list`.
- [ ] On Windows, verify `config.yaml` is UTF-8 without BOM if parsing errors appear.
