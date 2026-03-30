# Install Hermes on Windows

Three ways to install, from easiest to most advanced.

## Option 1: Quick Install (recommended)

Open PowerShell and paste:

```powershell
irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1 | iex
```

That's it. Open a new terminal and run `hermes setup` to configure your API key.

A desktop shortcut with the NousResearch icon is created automatically.
To use the golden Hermes staff icon instead:

```powershell
.\scripts\install-windows.ps1 -DesktopIcon staff
```

To skip the desktop shortcut:

```powershell
.\scripts\install-windows.ps1 -DesktopIcon none
```

## Option 2: Full AF_UNIX Bootstrap (one command)

This builds a custom Python with AF_UNIX socket support and installs Hermes
with it. AF_UNIX gives the code execution sandbox better isolation.

```powershell
irm https://raw.githubusercontent.com/claudlos/hermes-agent/windows-qol-v2/scripts/bootstrap-windows-afunix-hermes.ps1 | iex
```

What it does automatically:
- checks for Visual Studio Build Tools (installs them if missing)
- downloads CPython 3.13.12 source
- patches it for AF_UNIX support
- compiles it
- installs Hermes with that interpreter
- creates a desktop shortcut
- on reruns, reuses the already-built Python (fast)

First run takes a few minutes (compile time). Reruns take seconds.

If the custom build fails for any reason, Hermes still gets installed
using your regular Python.

## Option 3: Manual Install

```powershell
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
pip install -e ".[keyring,pty]"
hermes setup
```

## After Install

1. Open a **new** terminal (so PATH takes effect)
2. Run `hermes setup` to configure your API key
3. Run `hermes` to start chatting

## Desktop Icon Options

The installer creates a desktop shortcut by default. Two icon choices:

- **nous** (default) — the NousResearch girl
- **staff** — golden Hermes caduceus

Change it anytime by rerunning:

```powershell
.\scripts\install-windows.ps1 -DesktopIcon staff
```

## Requirements

- Windows 10 (build 17134+) or Windows 11
- Python 3.10+
- Git for Windows (for terminal tool)

## Troubleshooting

If something goes wrong, check the logs:

```text
%LOCALAPPDATA%\hermes-agent-bootstrap\logs
```

For more details see [WINDOWS.md](WINDOWS.md).
