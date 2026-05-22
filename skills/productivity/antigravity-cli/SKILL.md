---
name: antigravity-cli
description: Antigravity CLI (`agy`) usage, configuration, authentication, plugins, permissions, sandboxing, slash commands, and troubleshooting.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [antigravity, agy, cli, productivity, auth, plugins, sandbox, permissions]
---

# Antigravity CLI (`agy`)

Use this skill when working with the Antigravity CLI, invoked as `agy`.

## Mental model

Antigravity has two layers:

1. **Shell wrapper commands** like `agy help`, `agy install`, `agy plugin`, `agy update`, and `agy changelog`
2. **Interactive in-session slash commands** like `/config`, `/permissions`, `/skills`, and `/agents`

Do not blur those together. `agy help` shows the shell wrapper surface, not the in-session slash command list.

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

## Shell surface

### Verified wrapper commands

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

### Plugin subcommands

`agy plugin --help` shows:

- `list`
- `import [source]`
- `install <target>`
- `uninstall <name>`
- `enable <name>`
- `disable <name>`
- `validate [path]`
- `link <mp> <target>`
- `help`

### Install flags

`agy install --help` shows:

- `--dir`
- `--skip-aliases`
- `--skip-path`

### Version check gotcha

- `agy --version` is the safe non-interactive version check.
- `agy version` is interactive and can fail without a real TTY.

## In-session slash commands

### Conversation control

- `/resume` or `/switch`
- `/rewind` or `/undo`
- `/rename <name>`
- `/clear`
- `/fork`
- `/reset`
- `/new`

### Settings and tools

- `/config`
- `/settings`
- `/permissions`
- `/model`
- `/keybindings`
- `/statusline`
- `/tasks`
- `/skills`
- `/mcp`
- `/open <path>`
- `/usage`
- `/logout`
- `/agents`

### Prompt helpers

- `@` starts path autocomplete
- `esc esc` clears the prompt when nothing is streaming
- `!` at the start runs a terminal command directly
- `?` opens help and the slash command list

## Settings and permissions

### Common settings keys

The local `settings.json` typically contains keys such as:

- `allowNonWorkspaceAccess`
- `colorScheme`
- `permissions.allow`
- `trustedWorkspaces`

### Permission modes

Docs and runtime logs show these permission modes / concepts:

- `request-review`
- `always-proceed`
- `strict`
- `proceed-in-sandbox`

### Sandbox behavior

- `enableTerminalSandbox` is a boolean in `settings.json`
- Default is `false`
- Launch-time overrides such as `--sandbox` and `--dangerously-skip-permissions` can supersede persistent settings for the current session

## Authentication behavior

- The CLI tries the OS secure keyring first.
- If no saved session exists, it falls back to browser-based Google sign-in.
- On a local machine, it opens the default browser.
- Over SSH, it prints a secure authorization URL and expects the auth code to be pasted back.
- `/logout` removes saved credentials.

## Plugins

- Plugins are staged under `~/.gemini/antigravity-cli/plugins/<plugin_name>/`.
- Plugins can bundle skills, agents, rules, MCP servers, and hooks.
- `agy plugin list` returning no imported plugins is a valid empty state.

## Prompt-mode verification

`agy --print` is useful for non-interactive smoke tests and one-shot prompts.
Use it when you want the CLI to answer without opening the full TUI.

## Troubleshooting and gotchas

- `agy help` shows wrapper commands, not the interactive slash commands.
- The first place to look for failures is `~/.gemini/antigravity-cli/log/cli-*.log`.
- Do not confuse persistent JSON settings with launch-time overrides.
- `~/.gemini/antigravity-cli/bin/agentapi` is only a thin wrapper to `agy agentapi`.
- On WSL, token storage is file-based, so auth issues are often local-file or session-state problems, not browser-only problems.
- Workspace identity can depend on launch directory and the `.antigravitycli` project marker.

## Practical verification checklist

When you need to confirm the install is real and usable:

1. `command -v agy`
2. `agy --version`
3. `agy help`
4. `agy plugin list`
5. Read `~/.gemini/antigravity-cli/settings.json`
6. Read the latest `~/.gemini/antigravity-cli/log/cli-*.log`
7. If needed, inspect `~/.gemini/antigravity-cli/keybindings.json`

## Good support posture

Be explicit about the distinction between:

- shell-level `agy` commands
- in-session slash commands
- settings-file config
- plugin staging
- auth state
- WSL token storage
- workspace/project discovery

If you blur those, the guidance will be wrong.

## Support files

- `references/cli-docs.md` — condensed notes from the getting-started, usage, and features docs.
