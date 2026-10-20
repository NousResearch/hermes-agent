---
name: termux-gateway-keepalive
description: Supervise and resurrect Hermes gateway on Android Termux.
version: 0.2.0
author: Thamer (taljeri), Hermes Agent
license: MIT
platforms: [linux, android]
metadata:
  hermes:
    tags: [Termux, Android, Gateway, Keepalive, Watchdog, OOM]
    category: devops
    related_skills: [sdlc-review, systematic-debugging]
---

# Termux Gateway Keep-Alive & Phone Survival Package

Supervise, protect, and automatically resurrect the Hermes gateway and browser subsystem on Android Termux without requiring root or custom ROMs.

## The Four Death Vectors Solved

1. **Process-Group Reclamation**: When the Termux application is backgrounded or removed from recent tasks by the user/OS, child processes get terminated silently.
2. **OOM Cascades**: Unchecked browser and subprocess memory growth causes the Linux kernel Low Memory Killer (LMK) to terminate the entire Termux tree.
3. **Watchdog Treadmill**: Naive supervisors kill Chrome upon high RAM usage, runit immediately respawns it, and an endless kill-restart cycle causes an OOM crash anyway.
4. **Permanent Offline State**: Once dead, the agent cannot revive itself until manual operator intervention.

---

## ⚠️ Required User Action: Disable Child Process Killing

Before installing, prevent Android from killing Termux background subprocesses:

### 1. Disable Battery Optimization (per-OEM)
- **Stock Android**: Settings → Battery → Termux → Unrestricted
- **Motorola**: Settings → Battery → Optimization → Termux → Don't optimize
- **Samsung**: Settings → Battery → Background limits → Remove Termux from sleeping apps
- **Xiaomi / MIUI**: Settings → Apps → Termux → Autostart ON + No battery restrictions

### 2. Disable Phantom Process Killer (Android 12+)
```bash
adb shell device_config put activity_manager max_phantom_processes 2147483647
```
*(Or in Settings → Developer options → disable "Monitor phantom processes" if available).*

---

## The Six Defense Layers

```
Layer 1  gateway_watchdog.sh    Detached supervisor + wake-lock + auto-restart
Layer 2  ram_watchdog.sh        RAM monitor with treadmill prevention: stops browser before OOM
Layer 3  termux_priority.sh     OOM priority shaping: gateway (nice=-15), Chrome (oom_score_adj=500)
Layer 4  resurrect.sh           Android WorkManager job (runs outside Termux) — revives within ≤15 min
Layer 5  telegram_selfcheck.sh  Delivery health verification + Telegram revival alerts
Layer 6  ram_management.sh      Daily hygiene cron: cleans pip build leftovers and Chromium telemetry
```

---

## Key Operational Discoveries

- **Treadmill Prevention (`ram_watchdog.sh`)**: Breaks infinite kill-restart loops by stopping the runit service (`sv down chromium-headless`) before killing processes. If Chrome exceeds 8 kills in a single day, it remains offline until investigated.
- **Quiet Alert Discipline**: High-priority notifications fire ONLY when the gateway itself is dead or under PANIC-level memory pressure. Routine browser restarts are logged silently.
- **Non-Root OOM Priority Shaping**: While SELinux restricts lowering the gateway's own `oom_score_adj` below 200, users can safely raise child/browser processes to `oom_score_adj=500` and `nice=19` so the kernel always sacrifices browser tabs first.
- **Android WorkManager Resurrection**: `termux-job-scheduler` schedules tasks inside Android's system `WorkManager`. Even if the entire Termux app is terminated, Android wakes the revival script within ≤15 minutes.
- **Telegram Token Parsing**: Uses strict prefix matching (`grep "^TELEGRAM_BOT_TOKEN=."`) to avoid false matches against commented configuration lines.
- **Chromium Telemetry Prevention**: Offline headless Chromium generates unrotated `.pma` telemetry metrics. The daily hygiene script purges metrics exceeding 50 MB and launches with `--metrics-recording-only --disable-domain-reliability`.
- **Low-RAM (≤4GB) Browser Lifecycle**: Uses `browse_once.sh` for an on-demand START-USE-SHUTDOWN pattern rather than continuous idling.

---

## Quick Installation

Run the 1-command installer script inside Termux:

```bash
bash skills/devops/termux-gateway-keepalive/scripts/install_all.sh
```

Or manually:

```bash
# 1. Register Android WorkManager resurrection job
mkdir -p ~/.hermes/resurrect
cp skills/devops/termux-gateway-keepalive/scripts/resurrect.sh ~/.hermes/resurrect/
cp skills/devops/termux-gateway-keepalive/scripts/termux_priority.sh ~/.hermes/resurrect/
chmod +x ~/.hermes/resurrect/*.sh

termux-job-scheduler --job-id 1 \
  --script "$HOME/.hermes/resurrect/resurrect.sh" \
  --period-ms 900000 \
  --network any --battery-not-low false

# 2. Launch supervisor and RAM watchdog
bash skills/devops/termux-gateway-keepalive/scripts/gateway_watchdog.sh start
bash skills/devops/termux-gateway-keepalive/scripts/start_ram_watchdog.sh

# 3. Apply OOM priority shaping
bash skills/devops/termux-gateway-keepalive/scripts/termux_priority.sh

# 4. Schedule daily storage hygiene cron
(crontab -l 2>/dev/null; echo "0 4 * * * bash $HOME/skills/devops/termux-gateway-keepalive/scripts/ram_management.sh") | crontab -
```

---

## Script Inventory

| Script | Purpose |
|---|---|
| `gateway_watchdog.sh` | Detached supervisor, auto-restarts gateway with backoff |
| `gateway_monitor.sh` | Cron backstop for the supervisor |
| `ram_watchdog.sh` | Swap + RSS memory monitor with treadmill prevention |
| `termux_priority.sh` | Non-root OOM & CPU nice priority shaper |
| `resurrect.sh` | WorkManager survival script for full revival |
| `telegram_selfcheck.sh` | Gateway state and delivery validator |
| `hermes_presence.py` | Local ambient alert provider (vibrate, toast, notification) |
| `presence_notify.sh` | Budgeted ambient alert dispatcher (max 3/day) |
| `ram_management.sh` | Daily storage hygiene (tmp, build dirs, telemetry) |
| `tmp_inventory.sh` | Read-only `$PREFIX/tmp` directory analyzer |
| `wd_dedupe.sh` | Watchdog deduplication utility |
| `browse_once.sh` | START-USE-SHUTDOWN browser runner |
| `chrome_service.sh` | Chromium service runner and manager |
| `chrome_lowram.sh` | Single-process lean Chromium launcher |
| `chrome_zombie_hunt.sh` | Path-verified `/proc/$pid/exe` process cleaner |
| `start_browser_stack.sh` | Idempotent browser and runsvdir bootstrapper |
| `start_xvfb.sh` | Virtual framebuffer starter |
| `install_all.sh` | Automated 1-command installer script |

---

## Verification

Check status:
```bash
bash skills/devops/termux-gateway-keepalive/scripts/gateway_watchdog.sh status
cat ~/.hermes/ram_watchdog.state
```
