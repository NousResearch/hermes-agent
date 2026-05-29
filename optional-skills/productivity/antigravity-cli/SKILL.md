---
name: antigravity-cli
description: "Operate the Antigravity CLI (agy): plugins, auth, sandbox."
version: 0.1.0
author: Tony Simons (asimons81), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Antigravity, CLI, Productivity, Auth, Plugins, Sandbox]
    related_skills: []
---

# Antigravity CLI (`agy`)

Operator guide for the Antigravity CLI, invoked as `agy`. Run all `agy`
commands through the Hermes `terminal` tool; inspect its config and logs with
`read_file`. This skill is reference + procedure — it does not wrap a network
API, so there is nothing to authenticate from Hermes itself.

## When to Use

- Installing, updating, or smoke-testing the `agy` binary
- Driving non-interactive `agy --print` / `agy -p` one-shots
- Debugging Antigravity auth, sandbox, permissions, or plugin state
- Reading Antigravity settings, keybindings, conversations, or logs

## Mental model

Antigravity has two layers — keep them distinct or the guidance will be wrong:

1. **Shell wrapper commands** — `agy help`, `agy install`, `agy plugin`,
   `agy update`, `agy changelog`. Run these through the `terminal` tool.
2. **Interactive in-session slash commands** — `/config`, `/permissions`,
   `/skills`, `/agents`, etc. These only exist inside a running `agy` TUI
   session, not on the shell wrapper.

`agy help` shows the shell wrapper surface, NOT the in-session slash commands.

## Prerequisites

- The `agy` binary on PATH. Verify through the `terminal` tool:
  `command -v agy && agy --version`.
- No env vars or API keys required by this skill — Antigravity manages its own
  auth via the OS keyring / browser sign-in (see Authentication below).

## How to Run

Invoke every `agy` command through the `terminal` tool. Examples:

```
terminal(command="agy --version")
terminal(command="agy help")
terminal(command="agy plugin list")
terminal(command="agy --print 'Summarize the repo in 3 bullets'", workdir="/path/to/project")
```

For an interactive multi-turn TUI session, launch `agy` with `pty=true` (and
tmux for capture/monitoring), the same pattern the `codex` / `claude-code`
skills use. For one-shot smoke tests and scripted prompts, prefer
`agy --print` (non-interactive).

To inspect Antigravity's own files, use `read_file` on the paths under Core
paths below — do not `cat` them through the terminal.

## Core paths

- Binary / entrypoint: `agy`
- App data dir: `~/.gemini/antigravity-cli/`
- Settings file: `~/.gemini/antigravity-cli/settings.json`
- Keybindings file: `~/.gemini/antigravity-cli/keybindings.json`
- Logs: `~/.gemini/antigravity-cli/log/cli-*.log`
- Conversations: `~/.gemini/antigravity-cli/conversations/`
- Brain artifacts: `~/.gemini/antigravity-cli/brain/`
- History: `~/.gemini/antigravity-cli/history.jsonl`
- Plugin staging: `~/.gemini/antigravity-cli/plugins/<plugin_name>/`

## Quick Reference

### Wrapper commands
- `agy changelog`
- `agy help`
- `agy install`
- `agy plugin` / `agy plugins`
- `agy update`

### Useful flags
- `--add-dir`
- `--continue` / `-c`
- `--conversation`
- `--dangerously-skip-permissions`
- `--print` / `-p`
- `--print-timeout`
- `--prompt`
- `--prompt-interactive` / `-i`
- `--sandbox`
- `--log-file`
- `--version`

### Plugin subcommands (`agy plugin --help`)
- `list`, `import [source]`, `install <target>`, `uninstall <name>`,
  `enable <name>`, `disable <name>`, `validate [path]`, `link <mp> <target>`,
  `help`

### Install flags (`agy install --help`)
- `--dir`, `--skip-aliases`, `--skip-path`

### In-session slash commands
- **Conversation control:** `/resume` (`/switch`), `/rewind` (`/undo`),
  `/rename <name>`, `/clear`, `/fork`, `/reset`, `/new`
- **Settings & tools:** `/config`, `/settings`, `/permissions`, `/model`,
  `/keybindings`, `/statusline`, `/tasks`, `/skills`, `/mcp`, `/open <path>`,
  `/usage`, `/logout`, `/agents`
- **Prompt helpers:** `@` path autocomplete, `esc esc` clears the prompt (when
  not streaming), `!` runs a terminal command directly, `?` opens help

## Settings and permissions

### Common settings keys (`settings.json`)
- `allowNonWorkspaceAccess`
- `colorScheme`
- `permissions.allow`
- `trustedWorkspaces`

### Permission modes
`request-review`, `always-proceed`, `strict`, `proceed-in-sandbox`.

### Sandbox behavior
- `enableTerminalSandbox` is a boolean in `settings.json`; default `false`.
- Launch-time overrides (`--sandbox`, `--dangerously-skip-permissions`) can
  supersede persistent settings for the current session.

## Authentication behavior

- The CLI tries the OS secure keyring first.
- With no saved session, it falls back to browser-based Google sign-in.
- Locally it opens the default browser; over SSH it prints an authorization URL
  and expects the auth code pasted back.
- `/logout` removes saved credentials.

## Plugins

- Plugins stage under `~/.gemini/antigravity-cli/plugins/<plugin_name>/`.
- They can bundle skills, agents, rules, MCP servers, and hooks.
- `agy plugin list` returning no imported plugins is a valid empty state.

## Pitfalls

- `agy help` shows wrapper commands, not interactive slash commands.
- `agy --version` is the safe non-interactive version check; `agy version` is
  interactive and can fail without a real TTY.
- First place to look for failures: `~/.gemini/antigravity-cli/log/cli-*.log`
  (read with `read_file`).
- Don't confuse persistent JSON settings with launch-time overrides.
- `~/.gemini/antigravity-cli/bin/agentapi` is a thin wrapper to `agy agentapi`.
- On WSL, token storage is file-based, so auth issues are usually local-file /
  session-state problems, not browser-only problems.
- Workspace identity can depend on launch directory and the `.antigravitycli`
  project marker.

## Verification

Confirm the install is real and usable, all through the `terminal` tool (read
files with `read_file`):

1. `terminal(command="command -v agy")`
2. `terminal(command="agy --version")`
3. `terminal(command="agy help")`
4. `terminal(command="agy plugin list")`
5. `read_file` on `~/.gemini/antigravity-cli/settings.json`
6. `read_file` on the latest `~/.gemini/antigravity-cli/log/cli-*.log`
7. If needed, `read_file` on `~/.gemini/antigravity-cli/keybindings.json`

## Support files

- `references/cli-docs.md` — condensed notes from the getting-started, usage,
  and features docs.
