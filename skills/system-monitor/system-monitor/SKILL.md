---
name: system-monitor
description: System resource monitoring and diagnostics using Python stdlib. CPU usage, memory/RAM, disk space, running processes, network connections, and system health reports. No API keys or external packages required.
---

# System Monitor — Real-Time Diagnostics

System resource monitoring using only Python stdlib.
**Zero dependencies. Zero API keys. Works on Linux, macOS, and Windows.**

## Helper script

This skill includes `scripts/system_monitor.py` — a complete CLI tool for all system monitoring operations.

```bash
# CPU usage (overall + per-core + load averages)
python3 SKILL_DIR/scripts/system_monitor.py cpu

# Memory and swap analysis
python3 SKILL_DIR/scripts/system_monitor.py memory

# Disk space across all filesystems
python3 SKILL_DIR/scripts/system_monitor.py disk

# Top processes by CPU or memory
python3 SKILL_DIR/scripts/system_monitor.py processes
python3 SKILL_DIR/scripts/system_monitor.py processes --sort memory --limit 20

# Active network connections and listening ports
python3 SKILL_DIR/scripts/system_monitor.py network

# Full system health snapshot (all of the above)
python3 SKILL_DIR/scripts/system_monitor.py health
```

All commands output structured JSON. Parse and summarize results for the user.

---

## Usage

When the user asks about system resources, use the `terminal` tool to run the appropriate snippet below, or use the helper script with `SKILL_DIR` replaced by the actual path.

---

## 1. CPU Usage

```python
import json, os, time

def cpu_usage(interval=1.0):
    def read_stat():
        with open("/proc/stat") as f:
            lines = f.readlines()
        cpus = {}
        for line in lines:
            if not line.startswith("cpu"): break
            parts = line.split()
            name = parts[0]
            vals = list(map(int, parts[1:]))
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
            total = sum(vals)
            cpus[name] = (idle, total)
        return cpus

    def load_avg():
        try:
            with open("/proc/loadavg") as f:
                parts = f.read().split()
            return {"1min": float(parts[0]), "5min": float(parts[1]), "15min": float(parts[2])}
        except:
            import subprocess
            try:
                out = subprocess.check_output(["sysctl", "-n", "vm.loadavg"], text=True).strip()
                nums = [float(x) for x in out.strip("{}").split() if x.replace(".","").isdigit()]
                return {"1min": nums[0], "5min": nums[1], "15min": nums[2]}
            except:
                return {}

    try:
        s1 = read_stat()
        time.sleep(interval)
        s2 = read_stat()
        result = {}
        for name in s1:
            idle1, total1 = s1[name]
            idle2, total2 = s2[name]
            dt = total2 - total1
            di = idle2 - idle1
            pct = round(100 * (1 - di / dt), 1) if dt else 0.0
            result[name] = pct
        overall = result.pop("cpu", None)
        cores = {k: v for k, v in result.items()}
        print(json.dumps({
            "overall_percent": overall,
            "per_core_percent": cores,
            "core_count": len(cores),
            "load_avg": load_avg(),
        }, indent=2))
    except FileNotFoundError:
        # macOS fallback
        import subprocess
        out = subprocess.check_output(["top", "-l", "1", "-n", "0"], text=True)
        for line in out.splitlines():
            if "CPU usage" in line:
                print(json.dumps({"raw": line.strip(), "note": "macOS — use Activity Monitor for per-core"}, indent=2))
                return
        print(json.dumps({"error": "Could not read CPU stats"}, indent=2))

cpu_usage()
```

---

## 2. Memory & Swap

```python
import json

def memory():
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k.strip()] = int(v.split()[0])  # kB

        def kb(key): return info.get(key, 0)
        def mb(key): return round(kb(key) / 1024, 1)

        total = kb("MemTotal")
        free  = kb("MemFree")
        avail = kb("MemAvailable")
        buf   = kb("Buffers")
        cache = kb("Cached") + kb("SReclaimable")
        used  = total - free - buf - cache

        swap_total = kb("SwapTotal")
        swap_free  = kb("SwapFree")
        swap_used  = swap_total - swap_free

        def pct(a, b): return round(100 * a / b, 1) if b else 0.0

        print(json.dumps({
            "ram": {
                "total_mb":     round(total / 1024, 1),
                "used_mb":      round(used / 1024, 1),
                "available_mb": round(avail / 1024, 1),
                "buffers_mb":   round(buf / 1024, 1),
                "cached_mb":    round(cache / 1024, 1),
                "used_percent": pct(total - avail, total),
            },
            "swap": {
                "total_mb": round(swap_total / 1024, 1),
                "used_mb":  round(swap_used / 1024, 1),
                "free_mb":  round(swap_free / 1024, 1),
                "used_percent": pct(swap_used, swap_total),
            }
        }, indent=2))
    except FileNotFoundError:
        import subprocess, re
        out = subprocess.check_output(["vm_stat"], text=True)
        page = 4096
        stats = {}
        for line in out.splitlines():
            m = re.match(r"Pages (.+?):\s+(\d+)", line)
            if m: stats[m.group(1)] = int(m.group(2)) * page
        free_b  = stats.get("free", 0) + stats.get("speculative", 0)
        wired_b = stats.get("wired down", 0)
        active  = stats.get("active", 0)
        inactive= stats.get("inactive", 0)
        total_b = wired_b + active + inactive + free_b
        print(json.dumps({
            "ram": {
                "total_mb":     round(total_b / 1024**2, 1),
                "wired_mb":     round(wired_b / 1024**2, 1),
                "active_mb":    round(active  / 1024**2, 1),
                "inactive_mb":  round(inactive/ 1024**2, 1),
                "free_mb":      round(free_b  / 1024**2, 1),
            },
            "note": "macOS — swap via sysctl not included"
        }, indent=2))

memory()
```

---

## 3. Disk Space

```python
import json, os, subprocess

def disk():
    results = []
    try:
        with open("/proc/mounts") as f:
            mounts = [line.split() for line in f if len(line.split()) >= 3]
        seen = set()
        for parts in mounts:
            dev, mp, fstype = parts[0], parts[1], parts[2]
            if fstype in ("tmpfs","devtmpfs","sysfs","proc","devpts","cgroup","cgroup2",
                          "pstore","securityfs","configfs","debugfs","tracefs","hugetlbfs",
                          "mqueue","fusectl","binfmt_misc","overlay"):
                continue
            if mp in seen: continue
            seen.add(mp)
            try:
                st = os.statvfs(mp)
                total = st.f_blocks * st.f_frsize
                free  = st.f_bfree  * st.f_frsize
                avail = st.f_bavail * st.f_frsize
                used  = total - free
                pct   = round(100 * used / total, 1) if total else 0
                results.append({
                    "mount":      mp,
                    "device":     dev,
                    "fstype":     fstype,
                    "total_gb":   round(total / 1024**3, 2),
                    "used_gb":    round(used  / 1024**3, 2),
                    "free_gb":    round(free  / 1024**3, 2),
                    "used_percent": pct,
                    "status": "WARNING" if pct >= 90 else "OK",
                })
            except (PermissionError, OSError):
                pass
    except FileNotFoundError:
        # macOS/BSD fallback
        out = subprocess.check_output(["df", "-k"], text=True)
        for line in out.splitlines()[1:]:
            parts = line.split()
            if len(parts) < 6: continue
            try:
                total = int(parts[1]) * 1024
                used  = int(parts[2]) * 1024
                avail = int(parts[3]) * 1024
                pct   = round(100 * used / total, 1) if total else 0
                results.append({
                    "mount": parts[5], "device": parts[0],
                    "total_gb": round(total/1024**3,2),
                    "used_gb":  round(used /1024**3,2),
                    "free_gb":  round(avail/1024**3,2),
                    "used_percent": pct,
                    "status": "WARNING" if pct >= 90 else "OK",
                })
            except: pass

    print(json.dumps({"filesystems": results, "count": len(results)}, indent=2))

disk()
```

---

## 4. Top Processes

```python
import json, os, subprocess

def processes(sort_by="cpu", limit=15):
    try:
        # Linux: read from /proc
        pids = [p for p in os.listdir("/proc") if p.isdigit()]
        
        def read_proc(pid):
            try:
                with open(f"/proc/{pid}/stat") as f:
                    stat = f.read().split()
                with open(f"/proc/{pid}/status") as f:
                    status = {}
                    for line in f:
                        if ":" in line:
                            k, v = line.split(":", 1)
                            status[k.strip()] = v.strip()
                name = status.get("Name", stat[1].strip("()"))
                vm_rss = int(status.get("VmRSS", "0 kB").split()[0]) * 1024
                utime = int(stat[13])
                stime = int(stat[14])
                return {"pid": int(pid), "name": name, "cpu_ticks": utime + stime, "rss_mb": round(vm_rss / 1024**2, 1)}
            except: return None

        procs = [p for p in (read_proc(pid) for pid in pids) if p]

        import time
        tick = os.sysconf("SC_CLK_TCK")
        uptime = float(open("/proc/uptime").read().split()[0])
        for p in procs:
            p["cpu_percent"] = round(p["cpu_ticks"] / tick / uptime * 100, 2)

        key = "cpu_percent" if sort_by == "cpu" else "rss_mb"
        procs.sort(key=lambda x: x[key], reverse=True)
        procs = procs[:limit]
        for p in procs: del p["cpu_ticks"]

    except (FileNotFoundError, OSError):
        # macOS fallback
        out = subprocess.check_output(
            ["ps", "aux", "--sort=-pcpu"] if sort_by == "cpu" else ["ps", "aux", "--sort=-rss"],
            text=True
        )
        procs = []
        for line in out.splitlines()[1:limit+1]:
            parts = line.split(None, 10)
            if len(parts) < 11: continue
            procs.append({
                "pid": int(parts[1]),
                "name": parts[10].strip()[:40],
                "cpu_percent": float(parts[2]),
                "rss_mb": round(int(parts[5]) / 1024, 1),
            })

    print(json.dumps({"sort_by": sort_by, "limit": limit, "processes": procs}, indent=2))

processes()
```

---

## 5. Network Connections

```python
import json, socket, struct

def network():
    connections = []

    def hex_to_ip(h):
        try:
            ip = socket.inet_ntoa(struct.pack("<I", int(h, 16)))
            return ip
        except: return h

    def hex_to_port(h):
        try: return int(h, 16)
        except: return 0

    state_map = {
        "01":"ESTABLISHED","02":"SYN_SENT","03":"SYN_RECV","04":"FIN_WAIT1",
        "05":"FIN_WAIT2","06":"TIME_WAIT","07":"CLOSE","08":"CLOSE_WAIT",
        "09":"LAST_ACK","0A":"LISTEN","0B":"CLOSING",
    }

    for proto, path in [("tcp","/proc/net/tcp"), ("tcp6","/proc/net/tcp6"), ("udp","/proc/net/udp")]:
        try:
            with open(path) as f:
                lines = f.readlines()[1:]
            for line in lines:
                parts = line.split()
                if len(parts) < 4: continue
                local_addr, local_port = parts[1].split(":")
                rem_addr,   rem_port   = parts[2].split(":")
                state_hex = parts[3].upper()
                state = state_map.get(state_hex, state_hex)
                lp = hex_to_port(local_port)
                rp = hex_to_port(rem_port)
                la = hex_to_ip(local_addr) if len(local_addr) == 8 else local_addr
                ra = hex_to_ip(rem_addr)   if len(rem_addr)  == 8 else rem_addr
                if state == "LISTEN" or state == "ESTABLISHED":
                    connections.append({
                        "proto": proto,
                        "local":  f"{la}:{lp}",
                        "remote": f"{ra}:{rp}",
                        "state":  state,
                    })
        except FileNotFoundError:
            import subprocess
            try:
                out = subprocess.check_output(["ss", "-tunp"], text=True)
                for line in out.splitlines()[1:]:
                    parts = line.split()
                    if len(parts) < 5: continue
                    connections.append({
                        "proto": parts[0], "state": parts[1],
                        "local": parts[4], "remote": parts[5] if len(parts) > 5 else "",
                    })
            except: pass
            break

    listening = [c for c in connections if c["state"] == "LISTEN"]
    established = [c for c in connections if c["state"] == "ESTABLISHED"]
    print(json.dumps({
        "listening_count": len(listening),
        "established_count": len(established),
        "listening": listening[:30],
        "established": established[:30],
    }, indent=2))

network()
```

---

## 6. Full Health Snapshot

Run all checks together for a comprehensive report. Use the helper script:

```bash
python3 SKILL_DIR/scripts/system_monitor.py health
```

Or inline:

```python
import json, subprocess, sys

def health(script_path):
    report = {}
    for cmd in ["cpu", "memory", "disk", "network"]:
        try:
            out = subprocess.check_output([sys.executable, script_path, cmd], text=True, timeout=10)
            report[cmd] = json.loads(out)
        except Exception as e:
            report[cmd] = {"error": str(e)}
    print(json.dumps(report, indent=2))

# Replace with actual path:
health("SKILL_DIR/scripts/system_monitor.py")
```

---

## Quick Reference

| Task | Command / Snippet |
|------|-------------------|
| CPU usage % | Snippet 1 |
| Memory & swap | Snippet 2 |
| Disk space | Snippet 3 |
| Top processes by CPU | Snippet 4 |
| Network connections | Snippet 5 |
| Full health report | `system_monitor.py health` |

## Notes

- All reads are **passive** — no writes, no network calls to target hosts
- Linux primary: reads from `/proc` filesystem (no root needed)
- macOS fallback: uses `vm_stat`, `df`, `ps`, `sysctl`
- Results are structured JSON — summarize key findings for the user
- Highlight WARNING status on disk (≥90% full) or swap usage
- 
