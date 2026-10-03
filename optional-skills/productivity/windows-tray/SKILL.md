---
name: windows-tray
description: "Windows system tray icon with live agent status for Hermes."
version: 2.0.0
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
boot-time autostart. It runs entirely OUTSIDE the Hermes core — two small
Python scripts plus one observer plugin dropped into the user's Hermes home —
so it needs no app rebuild and survives `hermes update`.

One resident process: the ICON (not the process) tracks the desktop session —
it lights up when a `Hermes.exe` process appears and hides when that process
exits. An earlier tray+watchdog two-process design cost ~58 MB across four
pythonw images (uv ships venv stubs as launcher+interpreter pairs) and its
watchdog could still freeze; the single ~32 MB process trades a crash
supervisor for a Startup shortcut plus a 10-minute heartbeat log.

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
pwsh -NoProfile -File ${HERMES_SKILL_DIR}/scripts/install_tray.ps1   # create venv, scripts, autostart, start tray
pwsh -NoProfile -File ${HERMES_SKILL_DIR}/scripts/install_tray.ps1 -Uninstall
```

After install, enable the needs-input observer plugin (write marker for
"waiting for you"): `hermes config set plugins.enabled "[...existing...,
tray-needs-input]"`, then restart the desktop app once so the backend loads it.

## Quick Reference

| File (under `%LOCALAPPDATA%\hermes\tray\`) | Role |
|---|---|
| `hermes_tray.py` | Tray icon: 4-state dot, menu, minimize-to-tray, icon follows the desktop session, single-instance lock :45173 |
| `windows_tray_state.py` | Pure-stdlib state machine (no UI imports): dot state from leases/logs/marker + `icon_visibility()` policy |
| `%LOCALAPPDATA%\hermes\tray-needs-input.json` | Written by the bundled observer plugin; `{"pending","kind","ts"}` |

Dot states (polled every 2s): 🔵 active · 🟠 needs-input · ⚪ idle · 🔴 error.

## Procedure

1. Copy `scripts/hermes_tray.py`, `scripts/windows_tray_state.py` and
   `scripts/tray-needs-input/*` into
   `%LOCALAPPDATA%\hermes\tray\` and `%LOCALAPPDATA%\hermes\plugins\tray-needs-input\`
   (the installer does this — run it, don't hand-copy).
2. `install_tray.ps1` creates the dedicated venv, installs `pystray pillow`,
   writes `HermesTray.lnk` into `shell:startup` pointing straight at
   `hermes_tray.py`, and starts it hidden.
3. Add `tray-needs-input` to `plugins.enabled` via `hermes config set`
   (NEVER hand-edit config.yaml), then ask the user to restart the desktop app
   once. Until that restart, needs-input falls back to blue.
4. Verify per `## Verification`.

## Pitfalls

- `pystray` 3.x: menu-item text is a read-only property — dynamic labels need
  a callable passed as `MenuItem(lambda item: ...)`. Assigning `item.title`
  silently no-ops.
- `Icon.run()` has no `detach` kwarg in 3.x; background the icon by running
  under `pythonw.exe` instead. pystray icons start invisible — set
  `icon.visible` from your own loop, and expect `visible = True` to throw if
  called before the Win32 message window exists (retry next beat).
- Never use `EnumWindows` for desktop PRESENCE: it messages every window
  thread and blocks forever behind one hung third-party thread (a watchdog
  built on it froze in the field and silently abandoned the tray). Use a
  kernel32 Toolhelp snapshot of `Hermes.exe`; declare `restype=c_void_p` on
  `CreateToolhelp32Snapshot` or the handle truncates to int32 on x64 and every
  lookup silently fails. EnumWindows is still OK for finding the window to
  hide/restore — behind a short cache.
- `tasklist.exe` spawned from a `pythonw` process flashes a console window
  unless `creationflags=CREATE_NO_WINDOW`.
- A quit/hide request must be scoped to the desktop session, not to time:
  key it on the desktop PID changing (fast relaunch or update-restart gaps
  can be shorter than any poll interval), and clear it on session swap.
- uv-created venvs ship `pythonw.exe` as a launcher stub that re-execs a base
  interpreter as a CHILD — double the process count (each ~5.5 MB) and the
  base path may be blocked by app-control policy on locked-down machines
  ("app can't start" with no error under pythonw). Replace the stubs with
  hardlinks to the base `python.exe`/`pythonw.exe` plus its root DLLs
  (`New-Item -ItemType HardLink`) when that bites.
- The single-instance lock binds `127.0.0.1:45173`; if import exits with
  code 0 and no output, an instance is already running — check before
  "debugging" the script.
- `hermes update` resyncs Hermes' OWN venv only; the tray venv is immune, but
  a wipe of `%LOCALAPPDATA%\hermes\tray\` needs just a re-run of the installer.
- Log every state change and a ~10 min heartbeat: an un-logged resident
  process that freezes is indistinguishable from one that exited.

## Verification

- Tray icon appears within ~5s of the desktop app process existing; `tray.log`
  shows `tray started` then `desktop up (pid=...)`
- Minimize the window → it disappears from the taskbar; double-click the icon
  → restored (`tray.log`: `hid minimized window`)
- Trigger a clarify question → dot turns amber within the poll interval (after
  the plugin-enabled restart)
- Close the desktop app → icon disappears (process stays resident); reopen →
  icon returns without logging out
- Kill the tray process (no supervisor any more) → relaunch via Startup
  shortcut target or the installer; heartbeat lines in `tray.log` stop where a
  freeze began
- Reboot → tray returns automatically (Startup shortcut)
