# Hermes Terminal Helper

A small, signed macOS app bundle (`com.hermes.terminal-helper`) that sends
the Apple Events Hermes needs to manage terminal windows and application
focus. Source: `HermesTerminalHelper.swift`. Build + install: `./build.sh`.

## Why it exists

Hermes' macOS raise-ladder (`tools/terminal_tool_macos_open.py`) and the
profile "open terminal" endpoint (`hermes_cli/web_routers/profiles.py`)
previously drove UI through bare `osascript`. macOS TCC attributes an
automation prompt to the process that **sends** the Apple Event, so the
prompts read:

> "osascript" wants to control "System Events.app"

— generic, unexplained, and indistinguishable from malware.

The helper sends the same events **in-process** via `NSAppleScript`, so the
prompts now read:

> "Hermes Terminal Helper" wants to control "System Events.app"

with the purpose string from `Info.plist`:

> Hermes (Herbie) uses this helper to raise terminal windows, activate
> applications, and query which application is frontmost — so documents the
> agent opens appear in front instead of behind the Hermes window.

## What it does (and does not do)

Actions (CLI): `activate <app>`, `frontmost <app>`, `unhide-all`, `getName`,
`stdin` (arbitrary AppleScript on stdin, used by the profile terminal
endpoint).

It **only** sends Apple Events for window focus/activation. It does not
capture the screen, read files, access the network, or run shell commands.
Every caller keeps a plain `osascript` fallback for hosts where the helper
is not installed, so nothing breaks if it is absent.

## Installation

`build.sh` installs to both `<repo>/.hermes/bin/` (next to the `hermes`
launcher) and `~/.hermes/bin/` (the shared Hermes install bin, same family
as `uv`/`tirith`). Ad-hoc signed; TCC persists the grant keyed on the
signature, so rebuilding the binary resets the grant and re-prompts once.

## Granting permission

First use shows a one-time macOS prompt per target application
(System Events, Terminal, etc.). Grant it in the dialog or later under
System Settings → Privacy & Security → Automation. Denying it degrades the
ladder to its `osascript` fallback — the original behaviour.
