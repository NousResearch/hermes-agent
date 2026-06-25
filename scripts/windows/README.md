# Windows Helper Scripts

Helper scripts for running Hermes Agent on Windows. These complement the
cross-platform Python codebase with Windows-specific deployment conveniences.

## Scripts

| Script | Purpose |
|--------|---------|
| `startup-hermes.cmd` | Boot-time launcher: waits for network, starts gateway + web UI |
| `recover-hermes.cmd` | Emergency recovery: kills zombies, cleans locks, restarts everything |
| `hermes-watchdog.ps1` | Background health monitor: checks port every 5 min, auto-recovers |
| `health-check-hermes.cmd` | One-shot port check: can be wired into Task Scheduler |

## Setup

### 1. Auto-start on boot

Copy `startup-hermes.cmd` to your Windows Startup folder:

```
copy startup-hermes.cmd "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\hermes-startup.cmd"
```

### 2. Desktop recovery shortcut

Copy `recover-hermes.cmd` to your Desktop for one-click recovery if the
gateway becomes unresponsive.

### 3. Background watchdog

The startup script launches `hermes-watchdog.ps1` automatically. If you need
to start it manually:

```powershell
powershell -WindowStyle Hidden -ExecutionPolicy Bypass -File hermes-watchdog.ps1
```

## Customization

These scripts contain hardcoded paths (Python venv, Node.js, HERMES_HOME).
Edit them to match your installation before use. Search for:

- `C:\Users\aliyf\hermes-agent` — Hermes install directory
- `C:\Users\aliyf\AppData\Local\hermes` — HERMES_HOME
- `C:\nodejs\node_global\hermes-web-ui.cmd` — Web UI launcher

## Note

These scripts are Windows-specific deployment helpers. The Hermes core is
cross-platform — see `hermes_cli/_subprocess_compat.py` for the centralized
Windows compatibility layer used by the Python codebase.
