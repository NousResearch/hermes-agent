# Hermes on Windows

Native Windows support for Hermes Agent. Works on Windows 10 (build 17134+)
and Windows 11 with Python 3.10+.

## Installation

### Quick Install (PowerShell)

```powershell
irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1 | iex
```

This installs Python (via uv), Git, Node.js, and sets up Hermes in
`%LOCALAPPDATA%\hermes-agent`.

If you're testing a branch or fork that carries newer Windows installer work,
use that branch's raw script URL instead of the upstream `main` one. Example:

```powershell
irm https://raw.githubusercontent.com/<owner>/<branch>/scripts/install-windows.ps1 | iex
```

For the one-command custom AF_UNIX Python + Hermes bootstrap on a branch/fork:

```powershell
irm https://raw.githubusercontent.com/<owner>/<branch>/scripts/bootstrap-windows-afunix-hermes.ps1 | iex
```

### Manual Install

```powershell
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
pip install -e ".[windows,pty]"
hermes setup
```

The `[windows]` extra installs `keyring` for secure credential storage
via Windows Credential Manager.

### One-command AF_UNIX bootstrap

If you want the custom Windows Python build with AF_UNIX support and Hermes
installed in one flow, run:

```powershell
.\scripts\bootstrap-windows-afunix-hermes.ps1
```

That pinned bootstrap does this for you:
- checks for Visual Studio Build Tools and bootstraps them if missing
- downloads the CPython `v3.13.12` source zip
- applies the AF_UNIX patch
- builds `PCbuild\amd64\python.exe`
- verifies `socket.AF_UNIX`
- reuses an already verified custom CPython on reruns instead of rebuilding
- falls back to a stock-Python Hermes install if the custom build path fails
- writes logs under `%LOCALAPPDATA%\hermes-agent-bootstrap\logs`

### Custom Python Install (manual path)

If you want to build first and install separately, this repo also includes the
lower-level helper path:

```powershell
.\scripts\build-cpython-windows-afunix.ps1
.\scripts\install-windows.ps1 -PythonExe C:\Users\you\cpython-3.13.12-afunix\PCbuild\amd64\python.exe
```

## Features

### What Works

- **CLI chat** — full interactive prompt with multiline, autocomplete, history
- **Clipboard paste** — Ctrl+V to paste screenshots directly into Hermes
  (Ctrl+PrtScn → Ctrl+V workflow). Also supports `/paste` command and Alt+V.
- **All tools** — terminal, file, search, browser, vision, code execution,
  MCP servers, voice mode, cron jobs
- **Code execution sandbox** — uses AF_UNIX when available, otherwise
  falls back to TCP localhost on stock Windows Python
- **Gateway** — messaging gateway with Telegram, Discord, Slack, etc.
- **Gateway service** — auto-start via Windows Task Scheduler
- **Credential security** — API keys stored in Windows Credential Manager
  (via keyring) instead of plaintext files
- **File protection** — config files secured via icacls (owner-only ACLs)
- **Git Bash shell** — terminal tool finds Git Bash automatically

### Windows-Specific Notes

**Clipboard image paste:**
Copy a screenshot (Ctrl+PrtScn or Win+Shift+S), then Ctrl+V in Hermes.
You'll see `📎 Image #1 attached from clipboard`. Type your message and
press Enter to send with the image.

**Terminal tool:**
Uses Git Bash under the hood. Install Git for Windows if you haven't:
https://git-scm.com/download/win

Override the path if needed:
```
set HERMES_GIT_BASH_PATH=C:\Program Files\Git\bin\bash.exe
```

**Gateway service:**
```powershell
hermes gateway install   # Creates Windows scheduled task (runs at logon)
hermes gateway start     # Start the task
hermes gateway stop      # Stop the task + kill processes
hermes gateway uninstall # Remove the scheduled task
```

The task appears as "HermesGateway" in Task Scheduler.

**Secrets in config.yaml:**
Use `${ENV_VAR}` syntax to reference environment variables instead of embedding
secrets directly in config.yaml:
```yaml
mcpServers:
  myserver:
    env:
      API_KEY: ${MY_SERVICE_API_KEY}
```
Set the variable in your shell profile or Windows environment variables.
Unresolved references are left as-is (no silent empty substitution).

**Credential storage:**
With keyring installed (`pip install "hermes-agent[keyring]"`), API keys are
stored in Windows Credential Manager (DPAPI-encrypted, tied to your user
account) instead of `~/.hermes/.env`. View stored credentials in
Control Panel → Credential Manager → Windows Credentials.

## Troubleshooting

### "No module named fcntl"
Old issue, fixed. All file locking now uses `msvcrt` on Windows with
`fcntl` fallback on Unix.

### Code execution tool shows as disabled
Hermes prefers AF_UNIX when the interpreter supports it, but stock Windows
Python may not expose `socket.AF_UNIX`. This branch includes a TCP localhost
fallback, so `execute_code` should still work on normal Windows installs.

If you want the custom AF_UNIX Python path too, this repo includes:
- `scripts/patches/cpython-3.13-windows-afunix.patch`
- `scripts/build-cpython-windows-afunix.ps1`

Build the custom interpreter, then install Hermes with:
```powershell
.\scripts\install-windows.ps1 -PythonExe C:\path\to\python.exe
```

If the bootstrap needs to install Visual Studio Build Tools, expect a UAC prompt
and a longer first run. The bootstrap also performs a disk-space preflight,
reuses a previously verified custom CPython when possible, and falls back to a
stock-Python Hermes install if the custom build path fails.

When something fails, check the logs in:

```text
%LOCALAPPDATA%\hermes-agent-bootstrap\logs
```

### Git Bash not found
Install Git for Windows, or set the path explicitly:
```
set HERMES_GIT_BASH_PATH=C:\path\to\bash.exe
```

### Gateway won't start as a service
Task Scheduler requires the user to be logged in (ONLOGON trigger).
If you need it to run as a background service regardless of login,
consider NSSM (Non-Sucking Service Manager):
```
nssm install HermesGateway "C:\path\to\python.exe" "-m hermes_cli.main gateway run"
```

### Slow clipboard paste
First clipboard operation may take 5-15 seconds while PowerShell cold-starts.
Subsequent operations are fast (PowerShell stays cached).

### Voice mode issues
Install espeak-ng for TTS: `choco install espeak-ng` (requires Chocolatey).
For STT, set `VOICE_TOOLS_OPENAI_KEY` or install faster-whisper locally.

## Architecture Notes

- File locking: `msvcrt.locking()` (Windows) / `fcntl.flock()` (Unix)
- Process management: `taskkill /F /T /PID` (Windows) / `pkill -P` (Unix)
- Terminal device: `CON` (Windows) / `/dev/tty` (Unix)
- Temp files: `%TEMP%\hermes-*` (Windows) / `/tmp/hermes-*` (Unix)
- Credential store: Windows Credential Manager via keyring / `.env` fallback
- File permissions: `icacls` (Windows) / `chmod` (Unix)
- Service management: Task Scheduler (Windows) / systemd (Linux) / launchd (macOS)
- Shell quoting: cmd.exe double-quote escaping / shlex.quote (Unix)
