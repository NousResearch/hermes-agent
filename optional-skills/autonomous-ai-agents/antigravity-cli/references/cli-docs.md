# Antigravity CLI docs, condensed

Source pages and CLI output reviewed (`agy 1.5.7`):
- `/docs/cli-getting-started`
- `/docs/cli-using`
- `/docs/cli-features`
- `agy help`, `agy plugin --help`, `agy models`

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
- Logs under `~/.gemini/antigravity-cli/log/` may contain `"You are not logged into Antigravity"` errors from sessions started with a fresh or isolated config dir (e.g. automated or sandboxed runs with their own config dir); verify that the current run failed before treating it as an auth issue.

## Config and files
- Settings: `~/.gemini/antigravity-cli/settings.json` (top-level keys seen: `model`, `permissions`, `trustedWorkspaces`; also `allowNonWorkspaceAccess`, `colorScheme`, `enableTerminalSandbox`)
- Keybindings: `~/.gemini/antigravity-cli/keybindings.json`
- Logs: `~/.gemini/antigravity-cli/log/cli-*.log`
- Plugins: `~/.gemini/antigravity-cli/plugins/<plugin_name>/`
- Inspect config and logs with `read_file` and never print secrets.

## Print mode, models, and structured I/O
- `--print` / `-p` / `--prompt`: run a single prompt non-interactively.
- `--print-timeout`: time limit for print mode; defaults to `0s` (waits until the turn completes). Always pass `--print-timeout` on scripted runs and pair it with the `terminal` tool timeout. No `--max-turns`.
- `--output-format text|json|stream-json` (default `text`): `json` returns `{"conversation_id":"...","status":"SUCCESS","response":"...","duration_seconds":...,"num_turns":1,"usage":{"input_tokens":...,"output_tokens":...,"thinking_tokens":...,"cache_read_tokens":...,"total_tokens":...}}`.
- `--input-format text|stream-json` (default `text`): `stream-json` reads one NDJSON message per stdin line and requires `--output-format stream-json`.
- `--json-schema <string|path>`: enforce structured output against a JSON schema string or file path.
- `--model <id>`: takes the model ID from `agy models` (`<id>\t<display name>`, e.g. `gemini-4-argon-low`, `gemini-4-hydrogen-high`, `gemini-3.8-flash-low`, `claude-opus-5-5-medium`, `claude-sonnet-5-5-low`). Run `agy models` for the live list rather than hardcoding.
- `--effort low|medium|high|xhigh|max`: reasoning effort for the session.
- `--mode accept-edits|auto|plan`: agent execution mode for the session.
- `--agent`, `--project`, `--new-project`, `--remote-agent`, `--remote-control`, `--disable-slash-commands`, `--settings`.

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
- Session execution modes (`--mode`): `accept-edits`, `auto`, `plan`
- Launch overrides: `--sandbox`, `--dangerously-skip-permissions`, `--mode`, `--settings`
- Sandbox setting: `enableTerminalSandbox` in `settings.json` (default `false`)

## Plugins
- Plugins can bundle skills, agents, rules, MCP servers, and hooks.
- They are staged locally and auto-discovered once installed.
- `agy plugin list` (or `list --json`) shows installed plugins, e.g.:
  ```
  Installed plugins:
  - superpowers (enabled, 7.0.2)
      Source:     antigravity
      Components: 15 skills
  ```

## Subagents
- `/agents` opens the panel for active/completed subagents (`agy agent` / `agy agents` lists available agents from the shell).
- Subagents can run in parallel and request approvals.

## Keybindings
- `~/.gemini/antigravity-cli/keybindings.json`
- Malformed JSON falls back to defaults for broken actions.
- Docs list default bindings for clear, submit, cancel, exit, suspend, editor, approval yes/no, navigation, clipboard, undo/redo, and newline insertion.
