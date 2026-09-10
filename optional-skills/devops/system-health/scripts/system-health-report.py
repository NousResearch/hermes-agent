#!/usr/bin/env python3
"""VPS system health report — collects stats and emails report via AgentMail API.

CONFIGURATION — edit these values at the top of the file before first use.
"""

import json
import os
import subprocess
import urllib.request

# --- Configuration: set these before running ---
TO_EMAIL = "you@example.com"                     # Where to send reports
FROM_EMAIL = "your-agent@agentmail.to"           # Your AgentMail inbox address
API_KEY_FILE="~/.hermes/agentmail_key.txt"  # Path to AgentMail API key
THRESHOLDS = {
    "mem_pct": 85,
    "swap_pct": 20,
    "disk_pct": 80,
    "cpu_load": 0.8,
}


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()


def get_nproc():
    return int(run(["nproc"]) or "1")


def get_memory():
    out = run(["free", "-m"])
    total = used = pct = swap_total = swap_used = swap_pct = 0
    for line in out.splitlines():
        if line.startswith("Mem:"):
            parts = line.split()
            total, used = int(parts[1]), int(parts[2])
            pct = round(used / total * 100, 1) if total else 0
        elif line.startswith("Swap:"):
            parts = line.split()
            swap_total, swap_used = int(parts[1]), int(parts[2])
            swap_pct = round(swap_used / swap_total * 100, 1) if swap_total else 0
    return total, used, pct, swap_total, swap_used, swap_pct


def get_cpu_load():
    loadavg = open("/proc/loadavg").read().strip().split()
    return float(loadavg[0]), float(loadavg[1]), float(loadavg[2])


def get_disk():
    parts = run(["df", "-h", "/"]).splitlines()[-1].split()
    return parts[1], parts[2], parts[3], int(parts[4].rstrip("%"))


def get_uptime():
    return run(["uptime", "-p"])


def get_top_procs(n=5):
    lines = run(["ps", "aux", "--sort=-%mem"]).splitlines()
    if len(lines) < 2:
        return "No data"
    return "\n".join(lines[: n + 1])


def send_email(subject, body):
    api_key_path = os.path.expanduser(API_KEY_FILE)
    with open(api_key_path) as f:
        api_key = f.read().strip()
    payload = json.dumps({"to": [TO_EMAIL], "subject": subject, "text": body}).encode()
    req = urllib.request.Request(
        f"https://api.agentmail.to/v0/inboxes/{FROM_EMAIL}/messages/send",
        data=payload,
        method="POST",
    )
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read())
        print(f"Email sent: {result.get('id', 'unknown')}")
    except urllib.error.HTTPError as e:
        print(f"FAILED: HTTP {e.code} {e.read().decode()}")
        raise


def main():

    nproc = get_nproc()
    load_1, load_5, load_15 = get_cpu_load()
    mem_total, mem_used, mem_pct, swap_total, swap_used, swap_pct = get_memory()
    disk_total, disk_used, disk_avail, disk_pct = get_disk()
    uptime = get_uptime()
    top_procs = get_top_procs()

    warnings = []
    if mem_pct > THRESHOLDS["mem_pct"]:
        warnings.append(f"Memory at {mem_pct}%")
    if swap_pct > THRESHOLDS["swap_pct"]:
        warnings.append(f"Swap at {swap_pct}%")
    if load_1 > nproc * THRESHOLDS["cpu_load"]:
        warnings.append(f"CPU load {load_1} (limit {nproc*0.8:.1f}, {nproc} cores)")
    if disk_pct > THRESHOLDS["disk_pct"]:
        warnings.append(f"Disk at {disk_pct}%")

    hostname = run(["hostname"])
    prefix = "⚠️ " if warnings else "✅ "
    subject = f"{prefix}System Health — {hostname}"

    body = f"""System Health Report — {hostname}
================================
Uptime: {uptime}

Memory:  {mem_used}MB / {mem_total}MB ({mem_pct}%)
Swap:    {swap_used}MB / {swap_total}MB ({swap_pct}%)
CPU Load: {load_1} (1m) / {load_5} (5m) / {load_15} (15m) — {nproc} cores
Disk /:  {disk_used} / {disk_total} used ({disk_pct}%) — {disk_avail} avail

"""

    if warnings:
        body += "⚠️  Warnings:\n" + "\n".join(f"  • {w}" for w in warnings) + "\n\n"
    else:
        body += "All metrics within thresholds.\n\n"

    body += f"Top 5 memory consumers:\n{top_procs}\n"

    send_email(subject, body)
    print(f"Subject: {subject}")


if __name__ == "__main__":
    main()
