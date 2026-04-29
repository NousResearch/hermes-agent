---
name: wsl-windows-printer-invocation
description: Use when WSL or a Hermes runtime must invoke Windows printing tools, especially when `powershell.exe` or `cmd.exe` return `Invalid argument`, or when printing works in an interactive shell but not in a bot, gateway, or background process.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [windows, printer, printing, wsl, interop, troubleshooting, runtime]
    related_skills: [systematic-debugging, writing-plans, requesting-code-review]
---

# WSL-to-Windows Printer Invocation

## Overview

This skill covers a narrow but annoying class of failures: a process running in WSL can reach Windows tools in one context, but the same call fails in a bot, gateway, cron, or other non-interactive runtime.

The usual symptom is boring and useful: `powershell.exe` or `cmd.exe` throws `Invalid argument`, or the job prepares successfully but never reaches the Windows printing step. The fix is not to guess harder. It is to verify the runtime bridge, confirm the printer settings, and only then print.

This skill is intentionally generic. It is for invoking Windows-side printing from WSL/Hermes runtime, not for any single project or business flow.

## When to Use

Use this skill when:
- `powershell.exe` / `cmd.exe` fail from WSL with `Invalid argument`
- Printing works in an interactive WSL shell but fails from a bot, gateway, or scheduled task
- A workflow can prepare a print job but cannot complete it
- You need to call a Windows printer path, Windows shell, or Windows-side print helper from WSL
- Paper size, orientation, tray, scale, or destination printer is unclear and must be confirmed before printing

Do **not** use this skill when:
- The task is purely Linux-native printing
- The user already knows the exact printer and settings and only wants a one-off command
- You are troubleshooting document content rather than the Windows/WSL bridge

## Core Rules

1. **Do not assume Windows shell access exists.** A valid PATH does not prove `powershell.exe` is callable from the current runtime.
2. **Check the actual runtime.** Interactive terminals, bot processes, and cron jobs can have different environment state.
3. **Verify `WSL_INTEROP` before any Windows call.** If the bridge is missing or wrong, fix that first.
4. **Do not blindly execute `/mnt/c/.../powershell.exe`.** A resolved absolute path is not the same thing as a working interop bridge.
5. **Do not guess printer settings.** If paper size, orientation, tray, scaling, or printer target is ambiguous, ask the user.
6. **Prefer the smallest test that proves the bridge.** A harmless version check or trivial command is enough to prove Windows invocation.

## Diagnostic Flow

### 1) Identify the exact runtime

Write down where the failing command runs:
- interactive WSL shell
- Hermes CLI
- Telegram/Discord/Feishu gateway
- cron job
- other background runner

If the failure only happens in one context, treat it as an environment issue until proven otherwise.

### 2) Check the bridge

In the same runtime, inspect the interop state:

```bash
env | grep '^WSL_INTEROP=' || true
```

If the variable is missing, the Windows bridge is not established in that process. Do not move on and hope.

### 3) Test Windows shell access

Run a minimal Windows command from the same runtime:

```bash
powershell.exe -NoProfile -Command '$PSVersionTable.PSVersion.ToString()'
```

If that fails with `Invalid argument`, the problem is still the runtime bridge, not the printer.

### 4) Confirm printer intent

Before sending the real job, confirm with the user:
- target printer
- paper size
- orientation
- scale / fit-to-page behavior
- whether a specific tray or roll is required

If any of those are missing, ask. Do not invent defaults and then act surprised when paper comes out wrong.

### 5) Prepare, then print

Use the project or helper tool’s normal prepare step first, then print only after the bridge and settings are confirmed.

### 6) Verify the print outcome

Confirm the expected success signal for the workflow. Depending on the tool, that may be:
- a success exit code
- a queue-empty or spooler-cleared signal
- a document-specific confirmation such as page size or page count
- a downstream artifact update

## Common Pitfalls

1. **Assuming interactive success means runtime success.**
   The shell you used manually is not proof the bot process has the same environment.

2. **Hardcoding printer settings.**
   Paper size, tray, and orientation are user choices unless the document or project rules already pin them down.

3. **Treating `which powershell.exe` as permission to run it.**
   Discovery is not execution.

4. **Skipping the bridge check because the command “looks fine.”**
   WSL interop failures are environmental; the command text can be perfect and still die immediately.

5. **Debugging the printer before debugging the runtime.**
   Wrong order. The printer is usually innocent.

## Verification Checklist

- [ ] The exact runtime that will print has been identified
- [ ] `WSL_INTEROP` is present and valid in that runtime
- [ ] A minimal Windows shell command succeeds
- [ ] Printer target is known
- [ ] Paper size / orientation / scaling are confirmed or explicitly provided by the user
- [ ] The real print job completes successfully
- [ ] The workflow’s expected success signal is observed

## One-Shot Recipe

If you need a quick sanity check:

```bash
# 1. Confirm the runtime bridge
env | grep '^WSL_INTEROP=' || true

# 2. Verify Windows shell access
powershell.exe -NoProfile -Command '$PSVersionTable.PSVersion.ToString()'

# 3. Ask the user for any missing print settings
# 4. Run prepare
# 5. Run print
```

## Notes

- This skill is about the WSL-to-Windows bridge, not about a specific file format or business workflow.
- If a project needs its own printing pipeline, keep the project-specific steps in a separate skill and reference this one for the bridge and runtime checks.
