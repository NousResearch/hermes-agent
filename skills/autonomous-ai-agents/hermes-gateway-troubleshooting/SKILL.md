---
name: hermes-gateway-troubleshooting
description: "Use when the Hermes Gateway is down, the wrong profile is active on messaging platforms, platforms stop receiving messages, or desktop approval popups silently decline."
version: 1.0.0
author: Hermes Agent + Poncho
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [hermes, gateway, troubleshooting, approvals, profiles, messaging, desktop]
    related_skills: [hermes-agent]
---

# Hermes Gateway Troubleshooting

## Overview

Diagnose why messages from messaging platforms (Slack, Telegram, Discord, etc.)
aren't reaching the agent, why the agent shows the wrong identity on those
platforms, why the gateway dies at boot on Windows, or why desktop approval
popups silently decline. Covers the gateway process, profile identity,
platform connections, and the desktop approval wiring.

## When to Use

- User says Slack/Telegram/Discord messages aren't being received
- Agent responds on a platform with the wrong name/identity
- `hermes gateway status` shows no process, or the gateway crashes after reboot
- A gated tool call (SMS, dangerous command) returns `decline` in ~0.01s with no popup
- Any diagnosis starting from "the gateway is the problem"

## Quick Diagnosis

```bash
# 1. Check if gateway is running
hermes gateway status

# 2. Check gateway logs for active profile and platform connections
# NOTE: on Windows, logs live under the PROFILE dir: ~/AppData/Local/hermes/profiles/<profile>/logs/
tail ~/.hermes/logs/gateway.log
tail ~/AppData/Local/hermes/profiles/<profile>/logs/gateway.log   # Windows
# Also check the exit diagnostics log for crash traces:
tail -40 ~/AppData/Local/hermes/profiles/<profile>/logs/gateway-exit-diag.log
# Look for: "Active profile:" and "Authenticated as @" lines
```

## Common Issues

### 1. Gateway Not Running

**Symptom:** User sends messages on Slack/Telegram/Discord but the agent never receives them.

**Fix:**
```bash
# Run in foreground (for testing)
hermes gateway run

# Install as auto-start service (for production)
hermes gateway install
```

On **Windows:** `hermes gateway install` may trigger a UAC prompt for Scheduled Task creation. If denied, it falls back to a Startup folder entry (`%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\Hermes_Gateway_<profile>.vbs`).

**Verification:**
```bash
hermes gateway status
# Expected: ✓ Gateway process running (PID: ...)
```

### 2. Wrong Profile on Gateway

**Symptom:** Agent responds on Slack (or other platforms) but shows the wrong name/identity — e.g. "Hermes" instead of the expected agent name.

**Root cause:** The gateway is running under a different Hermes profile than expected. Each profile has its own SOUL.md (identity), sessions, and config. The Slack adapter authenticates with the display name configured in the active profile.

**Diagnosis:**
```bash
grep "Active profile" ~/AppData/Local/hermes/profiles/<profile>/logs/gateway.log | tail -3
grep "Authenticated as @" ~/AppData/Local/hermes/profiles/<profile>/logs/gateway.log | tail -3
```

**Fix:** Restart the gateway so it picks up the correct profile:
```bash
# Stop current gateway (Ctrl+C or kill the process)
# Then start from the correct profile
hermes gateway run
```

Or re-install so the auto-start uses the current profile:
```bash
hermes gateway install
```

**Verification:** After restart, logs should show:
```
Active profile: <correct-profile>
[Slack] Authenticated as @<correct-name> in workspace ...
```

### 3. Platform Not Connected

**Symptom:** Gateway is running but a specific platform (e.g. Slack) shows as disconnected.

**Diagnosis:**
```bash
grep -i "slack\|telegram\|discord\|connecting" ~/AppData/Local/hermes/profiles/<profile>/logs/gateway.log | tail -10
```

**Common causes:**
- Missing or expired API tokens in `.env`
- Slack Socket Mode disconnected (need `SLACK_APP_TOKEN` for socket mode)
- Platform not listed in `config.yaml` under `platforms`

**Check config:**
```bash
grep -A 2 "slack:" ~/AppData/Local/hermes/profiles/<profile>/config.yaml
```

### 4. Windows: Gateway dies at boot — pythonw.exe has no stderr

**Symptom:** Gateway runs fine when started manually (`hermes gateway start` / from a console) but is dead after every reboot/login. `hermes gateway status` shows "✗ No gateway process detected", and the log ends at "Starting Hermes Gateway..." with no "✓ slack connected" line after it.

**Root cause (confirmed 2026-07-31):** The auto-start chain (Startup-folder VBS → `profiles/<profile>/gateway-service/Hermes_Gateway_<profile>.vbs`) launches the gateway with **`pythonw.exe`** — the windowless Python. `pythonw` has NO stderr (`sys.stderr is None`), so `faulthandler.enable()` in `gateway/run.py` (in `start()`) throws `RuntimeError: sys.stderr is None` and the gateway exits instantly. Every boot-time start crashes; manual console starts work.

**Diagnosis:**
```bash
hermes gateway status                                                      # no process detected
tail ~/AppData/Local/hermes/profiles/<profile>/logs/gateway-exit-diag.log  # repeated RuntimeError: sys.stderr is None tracebacks
```

**Immediate fix (local, generated file — not in any git repo):** In `profiles/<profile>/gateway-service/Hermes_Gateway_<profile>.vbs`, change `pythonw.exe` → `python.exe` on the `sh.Run` line. Window style `0` already hides the console, so no window appears, but stderr exists and the gateway survives boot. Caveat: `hermes gateway install` regenerates this file, reverting the fix.

**Proper fix (upstream, documented in GitHub):** guard `faulthandler.enable()` when `sys.stderr is None`, and/or fix the `hermes gateway install` VBS generator in `hermes_cli/` to emit `python.exe` instead of `pythonw.exe`. See `references/pythonw-stderr-crash.md` for full evidence + a ready-made Claude Code brief to open a PR.

**Pitfall:** MSYS `ps -p <pid>` does NOT see native Windows processes (reports "not alive" for a running process). Use `tasklist | grep -i python` or `hermes gateway status` to verify a Windows process.

### 5. Desktop: Approval popup never appears — instant silent decline after app restart

**Symptom:** Every tool call that requires approval (SMS send, dangerous command) returns `not approved` / `decline` in **~0.01s** — far too fast for any human to have clicked. No popup ever appears in the desktop app. This starts right after the desktop app was restarted (approvals worked earlier the same day, taking 4–11s = user clicked them).

**Root cause (confirmed 2026-08-01):** The desktop app registers a per-session approval callback (`register_gateway_notify(key, cb)` in `tui_gateway/server.py`) that emits the `approval.request` popup. When that callback is missing for the session, `tools/approval.py` (gateway branch, `check_dangerous_command`/`check_execute_code_guard`) falls through to the **CLI `input()` fallback** — in a headless desktop context `input()` hits `EOFError` instantly and the approval is silently denied (~0.01s). The routing to the popup path depends on `HERMES_SESSION_PLATFORM` contextvars; after a desktop restart the wiring can be lost **while the gateway itself stays healthy** (`hermes gateway status` shows running).

**Diagnosis:**
- `hermes gateway status` → healthy. NOT a gateway-down problem.
- 4+ approvals ALL declined in 0.01s (no human could respond that fast).
- Approvals worked earlier the same day before the restart.
- MCP stderr shows `(declined/timeout)` lines for each attempt.

**Immediate fix:** Fully restart the desktop app (tray → Quit → reopen). Approval callbacks register per-boot; a fresh start re-wires them. No config change needed — the `*_REQUIRE_APPROVAL` gate stays ON (fails closed, so nothing slips out unapproved).

**Proper fix (upstream):** the bug class is (a) `except Exception: pass` swallowing the registration failure in `tui_gateway/server.py`, and (b) the gateway approval branch falling through to the `input()` path when `notify_cb is None` instead of returning a "no approval listener" error. See `references/approval-popup-unwired.md` for evidence + a ready-made Claude Code brief.

## Common Pitfalls

1. **`hermes gateway install` REGENERATES the VBS** — it will re-emit `pythonw.exe` and silently undo the local fix. Re-apply the one-line change after any reinstall until the upstream generator is fixed.
2. **Gateway profile ≠ chat profile.** The gateway can run under a profile different from the one you're chatting in. Always check `hermes gateway status` + logs, not just the active `/profile` in chat.
3. **Token expiry.** Slack tokens (`xoxb-...` and `xapp-...`) rarely expire but can be revoked if the Slack app is reinstalled or modified. If `Authenticated as @...` is missing from logs, regenerate tokens at api.slack.com/apps.
4. **Log retention.** Gateway logs accumulate under the profile dir — on Windows: `~/AppData/Local/hermes/profiles/<profile>/logs/gateway.log` (plus `gateway-exit-diag.log` for crash traces). Check the most recent entries (last 20-30 lines), not the first ones from a prior session.
5. **UAC on Windows.** If you decline the UAC prompt during `hermes gateway install`, it silently falls back to the Startup folder. The gateway starts on login but doesn't start immediately. Run `hermes gateway run` once to start it right now.
6. **Gateway restart required after config changes.** Changes to `.env`, `SOUL.md`, or `platforms` in `config.yaml` need a gateway restart (`Ctrl+C` then `hermes gateway run` again) — they are read at startup.
7. **Instant approval decline (0.01s) after a desktop restart ≠ user denying.** The desktop's approval popup wiring was lost; the request fell through to the CLI `input()` fallback and EOF-denied silently. Fix: fully restart the desktop app. Do NOT retry the action 4× before checking — one retry to confirm, then diagnose.

## Verification Checklist

- [ ] `hermes gateway status` shows ✓ process running
- [ ] Logs show `Active profile: <correct-profile>` and `Authenticated as @<correct-name>`
- [ ] After a desktop-app restart, a gated tool call shows the approval popup (not a 0.01s decline)
- [ ] After `hermes gateway install` on Windows, the VBS still uses `python.exe` (or the upstream generator fix is in place)
- [ ] Config changes were followed by a gateway restart

## References

- `references/pythonw-stderr-crash.md` — full evidence + Claude Code brief for the boot crash fix
- `references/approval-popup-unwired.md` — full evidence + Claude Code brief for the approval-popup fix
- `references/profile-mismatch-reproduction.md` — reproduction steps for the wrong-profile-on-gateway case
