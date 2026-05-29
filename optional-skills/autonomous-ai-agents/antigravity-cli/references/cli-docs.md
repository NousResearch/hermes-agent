# Antigravity CLI docs, condensed

Source pages reviewed:
- `/docs/cli-getting-started`
- `/docs/cli-using`
- `/docs/cli-features`

## Install
- macOS/Linux: `curl -fsSL https://antigravity.google/cli/install.sh | bash`
- Windows PowerShell: `irm https://antigravity.google/cli/install.ps1 | iex`
- Windows CMD: `curl -fsSL https://antigravity.google/cli/install.cmd -o install.cmd && install.cmd && del install.cmd`

## Authentication
- Tries secure keyring first.
- If no saved session exists, falls back to browser-based Google sign-in.
- Local machine: opens the default browser.
- SSH/remote: prints a secure authorization URL, then expects the auth code to be pasted back.
- `/logout` removes saved credentials.

## Config and files
- Settings: `~/.gemini/antigravity-cli/settings.json`
- Keybindings: `~/.gemini/antigravity-cli/keybindings.json`
- Plugins: `~/.gemini/antigravity-cli/plugins/<plugin_name>/`

## Useful slash commands
- `/config`, `/settings`
- `/permissions`
- `/resume` / `/switch`
- `/rewind` / `/undo`
- `/rename <name>`
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

## Prompt helpers
- `@` path autocomplete
- `esc esc` clears prompt when not streaming
- `!` runs a terminal command
- `?` opens help / slash command list

## Permissions and sandbox
- Permission modes: `request-review`, `always-proceed`, `strict`, `proceed-in-sandbox`
- Launch overrides: `--sandbox`, `--dangerously-skip-permissions`
- Sandbox setting: `enableTerminalSandbox` in `settings.json` (default `false`)

## Plugins
- Plugins can bundle skills, agents, rules, MCP servers, and hooks.
- They are staged locally and auto-discovered once installed.

## Subagents
- `/agents` opens the panel for active/completed subagents.
- Subagents can run in parallel and request approvals.

## Keybindings
- `~/.gemini/antigravity-cli/keybindings.json`
- Malformed JSON falls back to defaults for broken actions.
- Docs list default bindings for clear, submit, cancel, exit, suspend, editor, approval yes/no, navigation, clipboard, undo/redo, and newline insertion.
