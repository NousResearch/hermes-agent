---
name: system-health-monitor
description: Monitor system health — disk usage, memory, CPU load, running services, and recent errors. Works on Linux and macOS without external dependencies.
version: 1.0.0
author: contributor
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [System, Monitoring, Disk, Memory, CPU, Services, Logs]
    related_skills: []
---

# System Health Monitor

Check the health of your system at a glance — disk space, memory, CPU load, running services, and recent errors. All commands use standard Unix tools available on every Linux and macOS system. No external dependencies required.

## When to Use

Load this skill when the user asks to:
- Check disk space or storage usage
- Monitor memory or RAM usage
- View CPU load or system uptime
- List or check the status of system services
- Inspect recent system errors or logs
- Get a full system health summary

## Quick Reference

| Goal | Command |
|------|---------|
| Disk usage (all mounts) | `df -h` |
| Largest directories | `du -sh /* 2>/dev/null \| sort -rh \| head -10` |
| Memory overview | `free -h` (Linux) / `vm_stat` (macOS) |
| CPU load + uptime | `uptime` |
| Top processes by CPU | `ps aux --sort=-%cpu \| head -15` |
| Top processes by memory | `ps aux --sort=-%mem \| head -15` |
| Active services (Linux) | `systemctl list-units --type=service --state=running` |
| Failed services (Linux) | `systemctl --failed` |
| Recent system errors (Linux) | `journalctl -p err -n 50 --no-pager` |
| Recent system errors (macOS) | `log show --last 1h --predicate 'messageType == 16' \| tail -50` |

## Procedure

### 1. Detect the operating system

Before running any command, detect the OS:

```bash
uname -s
```

Use the output to choose the correct commands for Linux vs macOS in the steps below.

### 2. Disk usage

Always start with disk — a full disk causes many downstream problems.

```bash
df -h
```

Flag any mount point where the `Use%` column is above **85%** and warn the user.

To find what is consuming the most space:

```bash
du -sh /home/* /var/* /tmp 2>/dev/null | sort -rh | head -15
```

### 3. Memory usage

**Linux:**
```bash
free -h
```

Parse the `available` column from the `Mem:` row. Warn if available memory is below 10% of total.

**macOS:**
```bash
vm_stat
sysctl hw.memsize
```

Calculate used memory from `vm_stat` page counts × 4096 bytes. Warn if free + inactive pages together represent less than 10% of total.

### 4. CPU load and uptime

```bash
uptime
```

The load averages shown are for 1, 5, and 15 minutes. Compare the 1-minute load to the number of CPU cores:

```bash
# Get CPU core count (works on both Linux and macOS)
nproc 2>/dev/null || sysctl -n hw.logicalcpu
```

Warn if the 1-minute load average exceeds the core count (system is overloaded).

### 5. Top resource-consuming processes

```bash
ps aux --sort=-%cpu | head -15    # Linux
ps aux -r | head -15              # macOS (sort by CPU)
```

Report the top 5 by CPU and top 5 by memory. If any single process uses more than 50% CPU or 30% memory, highlight it explicitly.

### 6. Service health (Linux only)

Check for failed services first — these are the most actionable findings:

```bash
systemctl --failed
```

If there are failed services, get details for each:

```bash
systemctl status <service-name> --no-pager -l
```

List all running services for an overview:

```bash
systemctl list-units --type=service --state=running --no-pager
```

### 7. Recent errors in system logs

**Linux:**
```bash
journalctl -p err -n 50 --no-pager
```

**macOS:**
```bash
log show --last 1h --predicate 'messageType == 16' | tail -50
```

Summarize any recurring error patterns. If the same error message appears more than 3 times, flag it as a pattern worth investigating.

### 8. Report findings

Present findings in this order:
1. **Critical** — disk full (>95%), OOM, failed services, kernel panics
2. **Warning** — disk high (>85%), memory low (<10%), overloaded CPU, recent repeated errors
3. **OK** — everything that looks healthy

Always end with a plain-language summary sentence such as:
> "Disk usage is healthy, memory is fine, but 2 services have failed and CPU load is elevated — worth investigating `myservice.service`."

## Pitfalls

- **`du` on `/proc` or `/sys`** — these virtual filesystems have no real size. Always add `2>/dev/null` to suppress errors.
- **macOS memory reporting** — `free` does not exist on macOS. Use `vm_stat` and `sysctl hw.memsize` instead.
- **`systemctl` not found** — older Linux systems or non-systemd distros (Alpine, Void) use `service --status-all` or `rc-status` instead.
- **`journalctl` not found** — fall back to `tail -n 100 /var/log/syslog` or `/var/log/messages`.
- **Permission errors on `du`** — some directories (e.g. `/root`) require sudo. Ask the user before escalating.
- **Load average interpretation** — on a 16-core machine, a load of 8.0 is fine. Always divide by core count before alarming the user.

## Verification

After running checks, confirm:
- `df -h` completed without errors
- At least one memory metric was retrieved successfully
- `uptime` returned a load average
- On Linux: `systemctl --failed` was checked

If any check fails (command not found, permission denied), note it clearly rather than silently skipping it.
