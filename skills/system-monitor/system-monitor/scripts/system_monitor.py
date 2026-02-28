#!/usr/bin/env python3
"""
system_monitor.py — Real-time system diagnostics using only Python stdlib.

Usage:
    python3 system_monitor.py cpu
    python3 system_monitor.py memory
    python3 system_monitor.py disk
    python3 system_monitor.py processes [--sort cpu|memory] [--limit N]
    python3 system_monitor.py network
    python3 system_monitor.py health

Output: structured JSON printed to stdout.
"""

import json
import os
import socket
import struct
import subprocess
import sys
import time


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------

def cmd_cpu(args=None):
    interval = 1.0

    def read_stat():
        with open("/proc/stat") as f:
            lines = f.readlines()
        cpus = {}
        for line in lines:
            if not line.startswith("cpu"):
                break
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
            return {
                "1min": float(parts[0]),
                "5min": float(parts[1]),
                "15min": float(parts[2]),
            }
        except Exception:
            try:
                out = subprocess.check_output(
                    ["sysctl", "-n", "vm.loadavg"], text=True
                ).strip()
                nums = [
                    float(x)
                    for x in out.strip("{}").split()
                    if x.replace(".", "").isdigit()
                ]
                return {"1min": nums[0], "5min": nums[1], "15min": nums[2]}
            except Exception:
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
        cores = dict(sorted(result.items()))
        return {
            "overall_percent": overall,
            "per_core_percent": cores,
            "core_count": len(cores),
            "load_avg": load_avg(),
        }
    except FileNotFoundError:
        # macOS fallback
        out = subprocess.check_output(["top", "-l", "1", "-n", "0"], text=True)
        for line in out.splitlines():
            if "CPU usage" in line:
                return {"raw": line.strip(), "platform": "macOS"}
        return {"error": "Could not read CPU stats"}


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def cmd_memory(args=None):
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k.strip()] = int(v.split()[0])  # kB

        def kb(key):
            return info.get(key, 0)

        total = kb("MemTotal")
        free = kb("MemFree")
        avail = kb("MemAvailable")
        buf = kb("Buffers")
        cache = kb("Cached") + kb("SReclaimable")
        used = total - free - buf - cache

        swap_total = kb("SwapTotal")
        swap_free = kb("SwapFree")
        swap_used = swap_total - swap_free

        def pct(a, b):
            return round(100 * a / b, 1) if b else 0.0

        return {
            "ram": {
                "total_mb": round(total / 1024, 1),
                "used_mb": round(used / 1024, 1),
                "available_mb": round(avail / 1024, 1),
                "buffers_mb": round(buf / 1024, 1),
                "cached_mb": round(cache / 1024, 1),
                "used_percent": pct(total - avail, total),
                "status": "WARNING" if pct(total - avail, total) >= 90 else "OK",
            },
            "swap": {
                "total_mb": round(swap_total / 1024, 1),
                "used_mb": round(swap_used / 1024, 1),
                "free_mb": round(swap_free / 1024, 1),
                "used_percent": pct(swap_used, swap_total),
                "status": "WARNING" if pct(swap_used, swap_total) >= 80 else "OK",
            },
        }
    except FileNotFoundError:
        import re

        out = subprocess.check_output(["vm_stat"], text=True)
        page = 4096
        stats = {}
        for line in out.splitlines():
            m = re.match(r"Pages (.+?):\s+(\d+)", line)
            if m:
                stats[m.group(1)] = int(m.group(2)) * page
        free_b = stats.get("free", 0) + stats.get("speculative", 0)
        wired_b = stats.get("wired down", 0)
        active = stats.get("active", 0)
        inactive = stats.get("inactive", 0)
        total_b = wired_b + active + inactive + free_b
        return {
            "ram": {
                "total_mb": round(total_b / 1024 ** 2, 1),
                "wired_mb": round(wired_b / 1024 ** 2, 1),
                "active_mb": round(active / 1024 ** 2, 1),
                "inactive_mb": round(inactive / 1024 ** 2, 1),
                "free_mb": round(free_b / 1024 ** 2, 1),
            },
            "platform": "macOS",
        }


# ---------------------------------------------------------------------------
# Disk
# ---------------------------------------------------------------------------

SKIP_FSTYPES = {
    "tmpfs", "devtmpfs", "sysfs", "proc", "devpts", "cgroup", "cgroup2",
    "pstore", "securityfs", "configfs", "debugfs", "tracefs", "hugetlbfs",
    "mqueue", "fusectl", "binfmt_misc", "overlay", "squashfs", "ramfs",
    "autofs", "nsfs", "efivarfs",
}


def cmd_disk(args=None):
    results = []
    try:
        with open("/proc/mounts") as f:
            mounts = [line.split() for line in f if len(line.split()) >= 3]
        seen = set()
        for parts in mounts:
            dev, mp, fstype = parts[0], parts[1], parts[2]
            if fstype in SKIP_FSTYPES:
                continue
            if mp in seen:
                continue
            seen.add(mp)
            try:
                st = os.statvfs(mp)
                total = st.f_blocks * st.f_frsize
                free = st.f_bfree * st.f_frsize
                avail = st.f_bavail * st.f_frsize
                used = total - free
                pct = round(100 * used / total, 1) if total else 0
                results.append(
                    {
                        "mount": mp,
                        "device": dev,
                        "fstype": fstype,
                        "total_gb": round(total / 1024 ** 3, 2),
                        "used_gb": round(used / 1024 ** 3, 2),
                        "free_gb": round(free / 1024 ** 3, 2),
                        "used_percent": pct,
                        "status": "WARNING" if pct >= 90 else "OK",
                    }
                )
            except (PermissionError, OSError):
                pass
    except FileNotFoundError:
        out = subprocess.check_output(["df", "-k"], text=True)
        for line in out.splitlines()[1:]:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                total = int(parts[1]) * 1024
                used = int(parts[2]) * 1024
                avail = int(parts[3]) * 1024
                pct = round(100 * used / total, 1) if total else 0
                results.append(
                    {
                        "mount": parts[5],
                        "device": parts[0],
                        "total_gb": round(total / 1024 ** 3, 2),
                        "used_gb": round(used / 1024 ** 3, 2),
                        "free_gb": round(avail / 1024 ** 3, 2),
                        "used_percent": pct,
                        "status": "WARNING" if pct >= 90 else "OK",
                    }
                )
            except Exception:
                pass

    warnings = [r for r in results if r["status"] == "WARNING"]
    return {
        "filesystems": results,
        "count": len(results),
        "warnings": len(warnings),
    }


# ---------------------------------------------------------------------------
# Processes
# ---------------------------------------------------------------------------

def cmd_processes(args=None):
    sort_by = "cpu"
    limit = 15
    if args:
        for i, a in enumerate(args):
            if a == "--sort" and i + 1 < len(args):
                sort_by = args[i + 1]
            if a == "--limit" and i + 1 < len(args):
                try:
                    limit = int(args[i + 1])
                except ValueError:
                    pass

    try:
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
                return {
                    "pid": int(pid),
                    "name": name,
                    "cpu_ticks": utime + stime,
                    "rss_mb": round(vm_rss / 1024 ** 2, 1),
                }
            except Exception:
                return None

        procs = [p for p in (read_proc(pid) for pid in pids) if p]

        tick = os.sysconf("SC_CLK_TCK")
        uptime = float(open("/proc/uptime").read().split()[0])
        for p in procs:
            p["cpu_percent"] = round(p["cpu_ticks"] / tick / uptime * 100, 2)

        key = "cpu_percent" if sort_by == "cpu" else "rss_mb"
        procs.sort(key=lambda x: x[key], reverse=True)
        procs = procs[:limit]
        for p in procs:
            del p["cpu_ticks"]

    except (FileNotFoundError, OSError):
        sort_flag = "-pcpu" if sort_by == "cpu" else "-rss"
        try:
            out = subprocess.check_output(
                ["ps", "aux", f"--sort={sort_flag}"], text=True
            )
        except Exception:
            out = subprocess.check_output(["ps", "aux"], text=True)
        procs = []
        for line in out.splitlines()[1: limit + 1]:
            parts = line.split(None, 10)
            if len(parts) < 11:
                continue
            procs.append(
                {
                    "pid": int(parts[1]),
                    "name": parts[10].strip()[:50],
                    "cpu_percent": float(parts[2]),
                    "rss_mb": round(int(parts[5]) / 1024, 1),
                }
            )

    return {"sort_by": sort_by, "limit": limit, "processes": procs}


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

def cmd_network(args=None):
    connections = []

    def hex_to_ip(h):
        try:
            return socket.inet_ntoa(struct.pack("<I", int(h, 16)))
        except Exception:
            return h

    def hex_to_port(h):
        try:
            return int(h, 16)
        except Exception:
            return 0

    state_map = {
        "01": "ESTABLISHED", "02": "SYN_SENT", "03": "SYN_RECV",
        "04": "FIN_WAIT1",   "05": "FIN_WAIT2", "06": "TIME_WAIT",
        "07": "CLOSE",       "08": "CLOSE_WAIT", "09": "LAST_ACK",
        "0A": "LISTEN",      "0B": "CLOSING",
    }

    linux_ok = False
    for proto, path in [
        ("tcp", "/proc/net/tcp"),
        ("tcp6", "/proc/net/tcp6"),
        ("udp", "/proc/net/udp"),
    ]:
        try:
            with open(path) as f:
                lines = f.readlines()[1:]
            linux_ok = True
            for line in lines:
                parts = line.split()
                if len(parts) < 4:
                    continue
                local_addr, local_port = parts[1].split(":")
                rem_addr, rem_port = parts[2].split(":")
                state_hex = parts[3].upper()
                state = state_map.get(state_hex, state_hex)
                lp = hex_to_port(local_port)
                rp = hex_to_port(rem_port)
                la = hex_to_ip(local_addr) if len(local_addr) == 8 else local_addr
                ra = hex_to_ip(rem_addr) if len(rem_addr) == 8 else rem_addr
                if state in ("LISTEN", "ESTABLISHED"):
                    connections.append(
                        {
                            "proto": proto,
                            "local": f"{la}:{lp}",
                            "remote": f"{ra}:{rp}",
                            "state": state,
                        }
                    )
        except FileNotFoundError:
            pass

    if not linux_ok:
        try:
            out = subprocess.check_output(["ss", "-tunp"], text=True)
            for line in out.splitlines()[1:]:
                parts = line.split()
                if len(parts) < 5:
                    continue
                connections.append(
                    {
                        "proto": parts[0],
                        "state": parts[1],
                        "local": parts[4],
                        "remote": parts[5] if len(parts) > 5 else "",
                    }
                )
        except Exception:
            try:
                out = subprocess.check_output(
                    ["netstat", "-an"], text=True, stderr=subprocess.DEVNULL
                )
                for line in out.splitlines():
                    parts = line.split()
                    if len(parts) < 6:
                        continue
                    if parts[0] in ("tcp", "tcp6", "udp", "udp6"):
                        connections.append(
                            {
                                "proto": parts[0],
                                "local": parts[3],
                                "remote": parts[4],
                                "state": parts[5] if len(parts) > 5 else "",
                            }
                        )
            except Exception:
                pass

    listening = [c for c in connections if c["state"] == "LISTEN"]
    established = [c for c in connections if c["state"] == "ESTABLISHED"]
    return {
        "listening_count": len(listening),
        "established_count": len(established),
        "listening": listening[:30],
        "established": established[:30],
    }


# ---------------------------------------------------------------------------
# Health (all-in-one)
# ---------------------------------------------------------------------------

def cmd_health(args=None):
    report = {}
    for name, fn in [
        ("cpu", cmd_cpu),
        ("memory", cmd_memory),
        ("disk", cmd_disk),
        ("network", cmd_network),
    ]:
        try:
            report[name] = fn()
        except Exception as e:
            report[name] = {"error": str(e)}

    # summary flags
    warnings = []
    try:
        if report["cpu"].get("overall_percent", 0) >= 90:
            warnings.append("HIGH CPU")
        if report["memory"]["ram"].get("used_percent", 0) >= 90:
            warnings.append("HIGH RAM")
        if report["memory"]["swap"].get("used_percent", 0) >= 80:
            warnings.append("HIGH SWAP")
        for fs in report["disk"].get("filesystems", []):
            if fs["status"] == "WARNING":
                warnings.append(f"DISK {fs['mount']} {fs['used_percent']}%")
    except Exception:
        pass

    report["_summary"] = {
        "status": "WARNING" if warnings else "OK",
        "warnings": warnings,
    }
    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

COMMANDS = {
    "cpu": cmd_cpu,
    "memory": cmd_memory,
    "disk": cmd_disk,
    "processes": cmd_processes,
    "network": cmd_network,
    "health": cmd_health,
}


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    cmd = args[0].lower()
    if cmd not in COMMANDS:
        print(
            json.dumps(
                {"error": f"Unknown command '{cmd}'", "available": list(COMMANDS.keys())}
            )
        )
        sys.exit(1)

    result = COMMANDS[cmd](args[1:])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
