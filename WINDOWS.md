# Hermes on Windows

Native Windows support for Hermes Agent. Works on Windows 10 (build 17134+)
and Windows 11 with Python 3.11+.

## Installation

### Quick Install (PowerShell)

```powershell
irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1 | iex
```

This installs Python (via uv), Git, Node.js, and sets up Hermes in
`%LOCALAPPDATA%\hermes`.

### Manual Install

```powershell
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
pip install -e ".[windows,pty]"
hermes setup
```

The `[windows]` extra installs `keyring` for secure credential storage
via Windows Credential Manager.

## Features

### What Works

- **CLI chat** — full interactive prompt with multiline, autocomplete, history
- **Clipboard paste** — Ctrl+V to paste screenshots directly into Hermes
  (Ctrl+PrtScn → Ctrl+V workflow). Also supports `/paste` command and Alt+V.
- **All tools** — terminal, file, search, browser, vision, code execution,
  MCP servers, voice mode, cron jobs
- **Code execution sandbox** — uses AF_UNIX sockets (available on Windows 10+)
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

**Credential storage:**
With keyring installed (`pip install keyring`), API keys are stored in
Windows Credential Manager instead of `~/.hermes/.env`. View stored
credentials in Control Panel → Credential Manager → Windows Credentials.

## Troubleshooting

### "No module named fcntl"
Old issue, fixed. All file locking now uses `msvcrt` on Windows with
`fcntl` fallback on Unix.

### Code execution tool shows as disabled
The sandbox requires AF_UNIX sockets. These are available on Windows 10
build 17134+ (April 2018 Update). If you're on an older build, the
sandbox will be disabled but all other tools work normally.

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
