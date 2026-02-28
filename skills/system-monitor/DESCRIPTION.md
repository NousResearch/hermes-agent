---
name: system-monitor
description: System resource monitoring and diagnostics using Python stdlib. Use this skill for CPU usage, memory/RAM analysis, disk space, running processes, network connections, and overall system health checks. No API keys required. Triggers on requests like "check cpu usage", "how much memory is free", "show disk space", "list running processes", "check network connections", "system health report".
license: MIT
---

Real-time system monitoring and diagnostics using only Python stdlib.
Zero dependencies. Zero API keys. Works on Linux, macOS, and Windows.

## Capabilities

- CPU usage monitoring (per-core and overall, load averages)
- Memory and swap analysis (used, free, cached, percent)
- Disk space reporting across all mounted filesystems
- Running process listing with CPU/memory per process
- Active network connections and listening ports
- Full system health snapshot in one command

## Data Sources

- `/proc` filesystem — Linux kernel real-time stats
- `psutil`-free: uses only `os`, `subprocess`, `socket`, `struct`, `time`
- Cross-platform fallbacks for macOS via `sysctl` and `vm_stat`
- 
