# Kiro CLI Auth Setup & Reference

## Auth Methods

### 1. Interactive Login (Browser-based)

```bash
kiro-cli login
```

Opens browser → choose auth provider → redirects back to terminal.

**Supported providers:**
- GitHub
- Google
- AWS Builder ID (quick setup for individual developers)
- AWS IAM Identity Center (enterprise)
- External IdP (Microsoft Entra ID, Okta, etc.)

### 2. API Key (Headless Mode)

For CI/CD pipelines and automated tasks. **Requires Pro, Pro+, or Power tier.**

```bash
export KIRO_API_KEY=ksk_xxxxxxxx
kiro-cli chat --no-interactive "your prompt here"
```

### 3. Remote / Device Flow (SSH / containers)

No port forwarding needed. CLI shows a URL + one-time code.

### Auth Precedence

1. Active browser session (from `kiro-cli login`)
2. `KIRO_API_KEY` environment variable
3. No credentials → CLI prompts sign-in

### Check Auth / Logout

```bash
kiro-cli whoami    # check auth status
kiro-cli logout    # sign out
```

## Headless Mode Usage

```bash
# Non-interactive one-shot
kiro-cli chat --no-interactive "Write a Python script that..."

# With specific model
kiro-cli chat --no-interactive --model claude-sonnet-4.5 "Write a Python script that..."
```

## Configuration

Settings stored in `~/.kiro/settings/cli.json`

```bash
# List all settings
kiro-cli settings list

# Set default model
kiro-cli settings chat.defaultModel claude-opus-4.7

# Disable auto-updates
kiro-cli settings "app.disableAutoupdates" "true"
```

### Environment Variables

| Variable | Effect |
|----------|--------|
| `KIRO_API_KEY` | API key for headless mode |
| `KIRO_NO_HYPERLINKS=1` | Disable hyperlinks |
| `KIRO_NO_PROGRESS=1` | Disable progress indicators |
| `KIRO_NO_SYNCHRONIZED=1` | Disable synchronized output |
| `KIRO_CHAT_UI` | Override UI engine |
| `KIRO_CHAT_LOG_FILE` | Override log file location |

### Log File Locations

- macOS: `$TMPDIR/kiro-log/kiro-chat.log`
- Linux: `$XDG_RUNTIME_DIR/kiro-log/kiro-chat.log`
- Windows: `%TEMP%\kiro-log\logs\kiro-chat.log`

## Proxy Support

Standard `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY` environment variables.

## Keyboard Shortcuts (in-session)

| Shortcut | Action |
|----------|--------|
| Shift+Tab | Plan mode |
| Ctrl+G | Crew monitor (subagents) |
| Ctrl+X | Activity tray |
| Ctrl+T | Conversation transcript |
| Ctrl+O | Toggle tool output |
| Ctrl+R | Reverse history search |
| Esc | Close panels, cancel agent |
| Shift+Enter / Ctrl+J / Alt+Enter | Newline |
| Tab | Drill into approval options / autocomplete |
| Ctrl+C / Ctrl+D | Exit session |

## Uninstall

```bash
kiro-cli uninstall
```

## Relationship to Amazon Q Developer

Kiro CLI is the **successor to Amazon Q CLI** (the `q` command). The install script automatically detects existing `q` installations and migrates them.
