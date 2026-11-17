---
name: system-health-check
description: Run comprehensive system health checks — disk usage, memory, CPU, load average, uptime, running services, and network status. Works on Linux and macOS with standard CLI tools.
version: 1.0.0
author: sprmn24
license: MIT
metadata:
  hermes:
    tags: [system, devops, monitoring, health-check, diagnostics]
    requires_toolsets: [terminal]
    related_skills: []
---
# System Health Check

Run a comprehensive system health report using standard CLI tools. No dependencies required — uses only built-in commands available on Linux and macOS.

## Prerequisites
- Linux or macOS system
- Standard CLI tools: `df`, `free` (Linux) / `vm_stat` (macOS), `uptime`, `ps`, `ss`/`netstat`

---

## 1. Quick Health Summary

Run all checks and produce a single summary:
```bash
echo "===== SYSTEM HEALTH REPORT ====="
echo "Generated: $(date)"
echo ""

# OS Info
echo "--- OS Info ---"
if [ -f /etc/os-release ]; then
  . /etc/os-release && echo "$PRETTY_NAME"
else
  sw_vers 2>/dev/null || uname -a
fi

# Uptime & Load
echo ""
echo "--- Uptime & Load ---"
uptime

# CPU
echo ""
echo "--- CPU Info ---"
if command -v nproc &>/dev/null; then
  echo "Cores: $(nproc)"
else
  echo "Cores: $(sysctl -n hw.ncpu 2>/dev/null || echo 'unknown')"
fi

# Memory
echo ""
echo "--- Memory Usage ---"
if command -v free &>/dev/null; then
  free -h
else
  vm_stat 2>/dev/null | head -10
fi

# Disk
echo ""
echo "--- Disk Usage ---"
df -h | grep -E '^/|^Filesystem'

# Top Processes by CPU
echo ""
echo "--- Top 5 Processes (CPU) ---"
ps aux --sort=-%cpu 2>/dev/null | head -6 || ps aux -r | head -6

# Top Processes by Memory
echo ""
echo "--- Top 5 Processes (Memory) ---"
ps aux --sort=-%mem 2>/dev/null | head -6 || ps aux -m | head -6

echo ""
echo "===== END OF REPORT ====="
```

---

## 2. Disk Usage Analysis

Check disk usage with warnings for partitions above 80%:
```bash
echo "--- Disk Usage Analysis ---"
df -h | awk 'NR==1 {print; next} /^\//{
  used = int($5)
  if (used >= 90) status = "🔴 CRITICAL"
  else if (used >= 80) status = "🟡 WARNING"
  else status = "🟢 OK"
  print $0, status
}'
```

Find the largest directories:
```bash
echo "--- Top 10 Largest Directories ---"
du -h --max-depth=1 / 2>/dev/null | sort -rh | head -10
```

---

## 3. Memory Details

Detailed memory breakdown:
```bash
echo "--- Memory Details ---"
if command -v free &>/dev/null; then
  free -h
  echo ""
  echo "Swap usage:"
  swapon --show 2>/dev/null || echo "No swap configured"
else
  echo "Physical Memory:"
  sysctl -n hw.memsize 2>/dev/null | awk '{printf "Total: %.1f GB\n", $1/1073741824}'
  vm_stat
fi
```

---

## 4. CPU & Load Average
```bash
echo "--- CPU & Load ---"
echo "Load Average: $(uptime | awk -F'load average:' '{print $2}')"
echo ""
if command -v nproc &>/dev/null; then
  CORES=$(nproc)
else
  CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo 1)
fi
echo "CPU Cores: $CORES"
LOAD=$(uptime | awk -F'load average:' '{print $2}' | cut -d, -f1 | xargs)
echo ""
if command -v bc &>/dev/null; then
  RATIO=$(echo "$LOAD / $CORES" | bc -l 2>/dev/null)
  if [ "$(echo "$RATIO > 1.0" | bc -l 2>/dev/null)" = "1" ]; then
    echo "⚠️  Load is above CPU core count — system may be overloaded"
  else
    echo "✅ Load is within normal range"
  fi
fi
```

---

## 5. Running Services
```bash
echo "--- Active Services ---"
if command -v systemctl &>/dev/null; then
  systemctl list-units --type=service --state=running --no-pager | head -20
elif command -v launchctl &>/dev/null; then
  launchctl list | head -20
else
  echo "Service manager not detected"
fi
```

Check for failed services (Linux):
```bash
if command -v systemctl &>/dev/null; then
  echo "--- Failed Services ---"
  FAILED=$(systemctl --failed --no-pager 2>/dev/null)
  if echo "$FAILED" | grep -q "0 loaded"; then
    echo "✅ No failed services"
  else
    echo "$FAILED"
  fi
fi
```

---

## 6. Network Status
```bash
echo "--- Network Interfaces ---"
if command -v ip &>/dev/null; then
  ip -brief addr show
else
  ifconfig | grep -E "^[a-z]|inet "
fi

echo ""
echo "--- Listening Ports ---"
if command -v ss &>/dev/null; then
  ss -tlnp 2>/dev/null | head -15
else
  netstat -tlnp 2>/dev/null | head -15 || netstat -an | grep LISTEN | head -15
fi
```

---

## 7. Recent Errors (System Log)
```bash
echo "--- Recent System Errors (last 20) ---"
if command -v journalctl &>/dev/null; then
  journalctl -p err --since "1 hour ago" --no-pager | tail -20
elif [ -f /var/log/syslog ]; then
  grep -i "error\|critical\|fatal" /var/log/syslog | tail -20
elif [ -f /var/log/system.log ]; then
  grep -i "error\|critical\|fatal" /var/log/system.log | tail -20
fi
```

---

## Usage Notes

- All commands use standard tools — no installation required
- Works on both Linux (Ubuntu, Debian, CentOS, Arch) and macOS
- Memory commands auto-detect: `free` on Linux, `vm_stat` on macOS
- Service commands auto-detect: `systemctl` on systemd, `launchctl` on macOS
- Run individual sections for targeted diagnostics, or Section 1 for a full overview
