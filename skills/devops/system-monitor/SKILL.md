---
name: system-monitor
description: "Diagnose Linux CPU, memory, disk, process pressure."
version: 1.1.0
author: Alex Chen (@l46983284-cpu)
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [monitoring, diagnostics, linux, devops, system]
    category: devops
    requires_toolsets: [terminal]
    related_skills: [docker-management, systematic-debugging]
---

# System Monitor Skill

Quick Linux host diagnostics for CPU, memory, disk, processes, and basic network pressure.
This skill does not replace long-running observability stacks; it is for short operator checks via `terminal`.

## When to Use

- Operator asks about host health, load, free RAM, disk pressure, or "what's eating CPU"
- Writes fail despite free space (inode / overlay / journal bloat suspects)
- Need top process consumers or socket summary before deeper debugging
- Docker host resource snapshot is useful alongside container checks

Do not use for:

- Continuous monitoring / alerting product design
- Non-Linux hosts (this skill is `platforms: [linux]`)
- Application-level APM or distributed tracing

## Prerequisites

- Linux host with shell access through Hermes `terminal`
- Common CLI: `top`, `ps`, `df`, `free`, `ss` (iproute2), optional `lsof`, `docker`, `journalctl`

## How to Run

Run the commands below with `terminal`. Prefer read-only probes first; avoid destructive cleanup unless the operator asks.

## Quick Reference

| Goal | Command |
|------|---------|
| Load / CPU snapshot | `top -bn1 \| head -5` |
| Memory available | `free -h` |
| Disk free | `df -h / /home 2>/dev/null` |
| Inodes | `df -i / /home 2>/dev/null` |
| Top CPU | `ps aux --sort=-%cpu \| head -15` |
| Top memory | `ps aux --sort=-%mem \| head -15` |
| Socket summary | `ss -s` |
| Journal size | `journalctl --disk-usage` |
| Docker stats | `docker stats --no-stream` |

## Procedure

### 1. Quick health check

```bash
top -bn1 | head -5
uptime
free -h
df -h / /home 2>/dev/null
```

Read `load average` against CPU count, prefer `free`'s **available** column, and note filesystem free space before digging further.

### 2. Disk pressure

```bash
df -h -x overlay -x tmpfs -x devtmpfs
df -i / /home 2>/dev/null
du -xhd1 / 2>/dev/null | sort -h | tail -20
find /var /home /tmp -xdev -type f -size +100M 2>/dev/null | head -40
```

Use `-xdev` / filesystem filters so overlay/container layers do not dominate the picture.

### 3. Process diagnostics

```bash
ps aux --sort=-%cpu | head -15
ps aux --sort=-%mem | head -15
ss -s
```

If a PID is suspicious and `lsof` is installed:

```bash
lsof -p <PID> | wc -l
```

### 4. Logs and containers (optional)

```bash
journalctl --disk-usage
docker stats --no-stream
```

For broader Docker lifecycle work, load `docker-management`. For deeper host failure diagnosis after the snapshot, load `systematic-debugging`.

## Pitfalls

1. **`df -h` on overlay roots is misleading** — exclude overlay/tmpfs or inspect the real backing mount.
2. **`free` "used" includes cache** — operator-relevant number is **available**.
3. **Space free but writes fail** — check inodes (`df -i`) and read-only remounts.
4. **`du /*` without `-x`** walks other mounts and can look like disk explosion.
5. **Journal bloat is easy to miss** — `journalctl --disk-usage` before deleting app data.
6. **`docker stats` without Docker** fails noisily — treat as optional.

## Verification

- [ ] Load, memory available, and root disk free reported
- [ ] Top CPU and memory PIDs identified when load is high
- [ ] Inodes checked when free space looks fine but writes fail
- [ ] Overlay/tmpfs excluded when interpreting disk usage
- [ ] No destructive cleanup run unless operator explicitly asked
