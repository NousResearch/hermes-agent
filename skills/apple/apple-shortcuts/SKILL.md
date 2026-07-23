---
name: apple-shortcuts
description: "Run and automate macOS Shortcuts via the shortcuts CLI."
version: 1.0.0
author: Nick Coleman, Hermes Agent
license: MIT
platforms: [macos]
metadata:
  hermes:
    tags: [Shortcuts, automation, macOS, Apple, HomeKit, Focus]
prerequisites:
  commands: [shortcuts]
---

# Apple Shortcuts Skill

Run the user's macOS Shortcuts from the `terminal` tool via the first-party `shortcuts` CLI, and use Shortcuts personal automations to trigger Hermes on real-world events. This skill does not create or edit shortcuts — the user builds those in Shortcuts.app; it runs them.

## When to Use

- User asks to run a shortcut by name, or to do something one of their shortcuts already does
- Controlling apps that expose Shortcuts actions (HomeKit scenes, Focus modes, Music, third-party apps)
- A macOS capability has no CLI or dedicated skill — a user-built Shortcut is often the cleanest bridge
- NOT for Reminders, Notes, or iMessage (dedicated `apple-*` skills exist), and NOT for agent-side scheduling (use the cronjob tool)

## Prerequisites

- macOS 12+ (the `shortcuts` binary ships with the OS — nothing to install)
- Shortcuts that control other apps trigger a one-time Automation permission prompt on first run; the user must click Allow on the Mac
- Shortcuts sync from iPhone/iPad via iCloud, so phone-built shortcuts are runnable here

## How to Run

```bash
shortcuts list                          # all shortcut names, one per line
shortcuts list --folders                # folder names only
shortcuts run "Shortcut Name"           # run by exact name (quote it)
```

## Quick Reference

```bash
# Pass input from a file, capture output to a file
shortcuts run "Resize Image" --input-path photo.jpg --output-path resized.jpg

# Pipe text in via stdin, read result on stdout
echo "hello" | shortcuts run "Uppercase Text" | cat

# View a shortcut in Shortcuts.app (for the user, not headless)
shortcuts view "Shortcut Name"
```

Names are matched exactly, including spaces and emoji — always quote. Use `shortcuts list` first rather than guessing a name.

## Procedure

1. Discover: `shortcuts list` and pick the exact name.
2. Check interactivity: prefer shortcuts that take defined input and produce output. If a shortcut shows alerts, menus, or "Ask Each Time" fields, it will block or fail when run headless — ask the user to make an agent-friendly variant that reads input and returns text.
3. Run with `--input-path`/stdin and capture stdout or `--output-path`.
4. Verify the result (see Verification) — do not assume success from a silent exit.

### Hermes as an automation target

Shortcuts **personal automations** (Focus change, arriving at a location, time of day, charger connected, NFC tag — built in the Shortcuts app on the user's iPhone/iPad; macOS has no personal-automations surface) can trigger Hermes with zero new code: add a "Send Message" action to the automation that messages the line the user's Hermes gateway listens on (iMessage or Telegram). Example: an automation on "Work Focus turned on" sends "I just started work focus — silence non-urgent alerts and give me today's agenda" to the Hermes number. The gateway treats it as a normal inbound message and the agent reacts. Set the automation to "Run Immediately" so it fires without confirmation.

### Bridging missing capabilities

When an Apple app has no CLI (one-off Calendar writes, grabbing a recent photo), prefer asking the user to build a small Shortcut exposing that action, then run it here — more reliable than UI scripting.

## Pitfalls

- Interactive shortcuts hang the `terminal` call — there is no headless flag; recognize blocking runs and stop them rather than waiting
- Exit codes are unreliable for failures *inside* a shortcut; prefer shortcuts that emit checkable output text
- First run against a new app blocks on the Automation permission dialog — tell the user to expect it
- `shortcuts run` gives no progress output; long-running shortcuts look identical to hung ones — know the expected duration before running

## Verification

- After a run, check the expected side effect (output file exists, `--output-path` file is non-empty, stdout matches) instead of trusting exit code 0
- `shortcuts list | grep -c .` confirms the CLI and iCloud sync are working at all
