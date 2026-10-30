#!/usr/bin/env python3
"""
fleet_status.py — SSH health check across a fleet of hosts.

Reads ~/.hermes/fleet/hosts.yaml (or $FLEET_HOSTS_FILE) and reports
uptime, load average, disk usage, and memory for each host in parallel.

Usage:
    python3 fleet_status.py
    FLEET_HOSTS_FILE=/path/to/hosts.yaml python3 fleet_status.py
"""

import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("Error: pyyaml is required. Install with: pip install pyyaml")

DEFAULT_HOSTS_FILE = Path.home() / ".hermes" / "fleet" / "hosts.yaml"
CONNECT_TIMEOUT = 10  # seconds per host

# Collects one line of stats from a remote host via SSH.
REMOTE_CMD = (
    "uptime_str=$(uptime -p 2>/dev/null || uptime); "
    "load=$(cat /proc/loadavg 2>/dev/null | awk '{print $1}' || sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}'); "
    "disk=$(df -h / | awk 'NR==2{print $5}'); "
    "mem=$(free 2>/dev/null | awk '/^Mem:/{printf \"%.0f%%\", $3/$2*100}' || "
    "      vm_stat 2>/dev/null | python3 -c \""
    "import sys; lines=sys.stdin.read().splitlines(); "
    "pages={l.split(':')[0].strip(): int(l.split(':')[1].strip().rstrip('.')) "
    "       for l in lines if ':' in l}; "
    "total=pages.get('Pages free',0)+pages.get('Pages active',0)+pages.get('Pages inactive',0)+pages.get('Pages wired down',0); "
    "used=pages.get('Pages active',0)+pages.get('Pages wired down',0); "
    "print(f'{used*100//total}%') if total else print('n/a')"
    "\"); "
    "echo \"${uptime_str}|${load}|${disk}|${mem}\""
)


def check_host(host: dict, results: dict) -> None:
    name = host["name"]
    address = host["host"]
    user = host["user"]
    port = str(host.get("port", 22))

    ssh_cmd = [
        "ssh",
        "-p", port,
        "-o", f"ConnectTimeout={CONNECT_TIMEOUT}",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        f"{user}@{address}",
        REMOTE_CMD,
    ]

    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=CONNECT_TIMEOUT + 5,
        )
        if result.returncode == 0 and "|" in result.stdout:
            parts = result.stdout.strip().split("|")
            uptime = parts[0].strip() if len(parts) > 0 else "n/a"
            load = parts[1].strip() if len(parts) > 1 else "n/a"
            disk = parts[2].strip() if len(parts) > 2 else "n/a"
            mem = parts[3].strip() if len(parts) > 3 else "n/a"
            # Shorten "up X days, Y hours, Z minutes" style output
            uptime = uptime.replace("up ", "").strip()
            results[name] = {
                "status": "UP",
                "address": address,
                "uptime": uptime[:20],
                "load": load,
                "disk": disk,
                "mem": mem,
                "error": None,
            }
        else:
            results[name] = {
                "status": "DOWN",
                "address": address,
                "uptime": "—",
                "load": "—",
                "disk": "—",
                "mem": "—",
                "error": result.stderr.strip()[:60] if result.stderr else "no output",
            }
    except subprocess.TimeoutExpired:
        results[name] = {
            "status": "TIMEOUT",
            "address": address,
            "uptime": "—",
            "load": "—",
            "disk": "—",
            "mem": "—",
            "error": "connection timed out",
        }
    except Exception as exc:
        results[name] = {
            "status": "ERROR",
            "address": address,
            "uptime": "—",
            "load": "—",
            "disk": "—",
            "mem": "—",
            "error": str(exc)[:60],
        }


def load_hosts(hosts_file: Path) -> list:
    if not hosts_file.exists():
        sys.exit(
            f"Hosts file not found: {hosts_file}\n"
            f"Create it or set FLEET_HOSTS_FILE env var.\n"
            f"See skills/devops/fleet-ssh/templates/hosts.yaml for an example."
        )
    with open(hosts_file) as f:
        data = yaml.safe_load(f)
    hosts = data.get("hosts", [])
    if not hosts:
        sys.exit(f"No hosts defined in {hosts_file}")
    return hosts


def print_report(hosts: list, results: dict) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    width = 75
    header = f"Fleet Health Report — {now}"

    print()
    print(header)
    print("═" * width)
    print(
        f"{'Host':<12} {'Address':<16} {'Status':<8} {'Uptime':<22} {'Load':<7} {'Disk':>5}  {'Mem':>5}"
    )
    print("─" * width)

    up_count = 0
    for host in hosts:
        name = host["name"]
        r = results.get(name, {"status": "UNKNOWN", "address": host["host"],
                                "uptime": "—", "load": "—", "disk": "—", "mem": "—"})
        if r["status"] == "UP":
            up_count += 1
        print(
            f"{name:<12} {r['address']:<16} {r['status']:<8} {r['uptime']:<22} "
            f"{r['load']:<7} {r['disk']:>5}  {r['mem']:>5}"
        )
        if r.get("error") and r["status"] != "UP":
            print(f"  {'':12} {'':16} └─ {r['error']}")

    print("═" * width)
    total = len(hosts)
    print(f"{up_count}/{total} hosts reachable")
    print()


def main() -> None:
    hosts_file = Path(os.environ.get("FLEET_HOSTS_FILE", DEFAULT_HOSTS_FILE)).expanduser()
    hosts = load_hosts(hosts_file)

    print(f"Checking {len(hosts)} host(s) from {hosts_file} ...")

    results: dict = {}
    threads = []
    for host in hosts:
        t = threading.Thread(target=check_host, args=(host, results), daemon=True)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    print_report(hosts, results)

    # Exit with non-zero if any host is down
    down = [h["name"] for h in hosts if results.get(h["name"], {}).get("status") != "UP"]
    if down:
        sys.exit(1)


if __name__ == "__main__":
    main()
