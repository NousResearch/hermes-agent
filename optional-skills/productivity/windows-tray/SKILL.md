---
name: windows-tray
description: "Windows system tray icon with live agent status for Hermes."
version: 1.0.0
author: "Raven (yuruiwen) + Hermes Agent"
license: MIT
platforms: [windows]
metadata:
  hermes:
    category: productivity
    tags: [windows, tray, system-tray, notifications, desktop, autostart]
    related_skills: [hermes-agent]
---

# Windows Tray Skill

A resident Windows system-tray icon for the Hermes desktop app: a status dot
that mirrors live agent state, minimize-to-tray, click-to-restore, and
boot-time autostart. It runs entirely OUTSIDE the Hermes core — three small
Python scripts plus one observer plugin dropped into the user's Hermes home —
so it needs no app rebuild and survives `hermes update`.

## When to Use

- A Windows user wants a tray icon that shows whether a Hermes session is
  running, idle, waiting for their input, or failed
- The user wants the desktop window to vanish from the taskbar when minimized
  and come back by double-clicking the tray icon
- Do NOT use to patch the Electron app itself — this skill deliberately stays
  out-of-tree

## Prerequisites

- Windows with the Hermes desktop app installed
- `uv` (preferred) or any Python ≥3.11 on PATH — the installer creates an
  isolated venv under `%LOCALAPPDATA%\hermes\tray\venv` and installs
  `pystray` + `pillow` into it. Never install into Hermes' own venv: the
  runtime's dependency sync silently uninstalls extras.

## How to Run

The agent installs and launches everything; the user only clicks the icon.

```
pwsh -NoProfile -File ${HERMES_SKILL_DIR}/scripts/install_tray.ps1   # create venv, scripts, autostart, start watchdog
pwsh -NoProfile -File ${HERMES_SKILL_DIR}/scripts/install_tray.ps1 -Uninstall
```

After install, enable the needs-input observer plugin (write marker for
"waiting for you"): `hermes config set plugins.enabled "[...existing...,
tray-needs-input]"`, then restart the desktop app once so the backend loads it.

## Quick Reference

| File (under `%LOCALAPPDATA%\hermes\tray\`) | Role |
|---|---|
| `hermes_tray.py` | Tray icon: 4-state dot, menu, minimize-to-tray, single-instance lock :45173 |
| `tray_watchdog.py` | Orphan sentinel: starts tray when a `Hermes*` window appears, stops it when the desktop session ends; lock :45174 |
| `start_watchdog.js` | `WScript.Shell.Run(...,0,false)` — launches the watchdog detached from any parent process tree |
| `%LOCALAPPDATA%\hermes\tray-needs-input.json` | Written by the bundled observer plugin; `{"pending","kind","ts"}` |

Dot states (polled every 2s): 🔵 active · 🟠 needs-input · ⚪ idle · 🔴 error.

## Procedure

1. Copy `scripts/hermes_tray.py`, `scripts/tray_watchdog.py`,
   `scripts/start_watchdog.js` and `scripts/tray-needs-input/*` into
   `%LOCALAPPDATA%\hermes\tray\` and `%LOCALAPPDATA%\hermes\plugins\tray-needs-input\`
   (the installer does this — run it, don't hand-copy).
2. `install_tray.ps1` creates the dedicated venv, installs `pystray pillow`,
   writes `HermesTray.lnk` into `shell:startup` pointing at the watchdog, and
   starts it hidden.
3. Add `tray-needs-input` to `plugins.enabled` via `hermes config set`
   (NEVER hand-edit config.yaml), then ask the user to restart the desktop app
   once. Until that restart, needs-input falls back to blue.
4. Verify per `## Verification`.

## Pitfalls

- `pystray` 3.x: menu-item text is a read-only property — dynamic labels need
  a callable passed as `MenuItem(lambda item: ...)`. Assigning `item.title`
  silently no-ops.
- `Icon.run()` has no `detach` kwarg in 3.x; background the icon by running
  under `pythonw.exe` instead.
- Spawning `tasklist.exe` from a `pythonw` process flashes a console window
  every poll unless `creationflags=CREATE_NO_WINDOW`; prefer EnumWindows for
  desktop presence, and key it by window PID so desktop restarts count as new
  sessions (that is also what clears the quit-flag).
- Autostart must survive app restarts: launch the watchdog through
  `cscript start_watchdog.js` (or the Startup shortcut), never from a shell
  that is itself a child of the desktop app — the whole process tree dies on
  app restart otherwise.
- The single-instance locks bind `127.0.0.1:45173/45174`; if import exits with
  code 0 and no output, an instance is already running — check before
  "debugging" the script.
- `hermes update` resyncs Hermes' OWN venv only; the tray venv is immune, but
  a wipe of `%LOCALAPPDATA%\hermes\tray\` needs just a re-run of the installer.

## Verification

- Tray icon appears within ~5s of the desktop window existing; `tray.log`
  shows `tray started`
- Minimize the window → it disappears from the taskbar; double-click the icon
  → restored (`tray.log`: `hid minimized window`)
- Trigger a clarify question → dot turns amber within the poll interval (after
  the plugin-enabled restart)
- Close the desktop app → tray disappears; reopen → tray returns without
  logging out
- Reboot → tray returns automatically (Startup shortcut)
