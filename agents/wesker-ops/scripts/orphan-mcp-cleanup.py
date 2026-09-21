#!/usr/bin/env python3
"""
Orphan MCP Cleanup — periodic sweep of leftover MCP children from disconnected
interactive sessions.

Hermes bug #15275: when an interactive `hermes chat` session disconnects,
its MCP children (gitnexus, workspace-mcp, github-mcp, mailbox_cleaner, ms-365)
become orphans under PID 1. They duplicate the gateway's own set.

no_agent=True — output is delivered verbatim as cron message.

DRY_RUN=true by default. Set DRY_RUN=false to actually kill.
"""

import os
import signal
import subprocess
import time


MCP_PATTERNS = [
    "gitnexus",
    "workspace-mcp",
    "github-mcp",
    "mailbox_cleaner",
    "ms-365",
    "adaptive_rate_limiter",
]

GATEWAY_CMDS = ["hermes_cli.main gateway", "hermes gateway run"]
CHAT_CMDS = ["hermes chat", "hermes_cli.main chat"]

# Known specialist bot systemd gateways — these are legitimate PPID=1 gateways
# that don't host MCPs. Never flag them.
KNOWN_SPECIALIST_SERVICES = {
    "hermes-gateway-ceecee", "hermes-gateway-denji", "hermes-gateway-dezzy",
    "hermes-gateway-gojo", "hermes-gateway-light", "hermes-gateway-misa-misa",
    "hermes-gateway-mrhermagi", "hermes-gateway-octacon", "hermes-gateway-quan",
    "hermes-gateway-remii", "hermes-gateway-wesker",
}


def read_cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read()
        return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def get_ppid(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("PPid:"):
                    return int(line.split()[1])
    except OSError:
        return None


def get_rss_kb(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        return 0


def is_alive(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def get_systemd_unit(pid):
    """Return the systemd unit name for a PID, or None."""
    try:
        result = subprocess.run(
            ["systemctl", "status", str(pid)],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if "CGroup: /system.slice/" in line:
                unit = line.split("/system.slice/")[1].split(".service")[0]
                return f"{unit}.service"
            if "/system.slice/" in line:
                unit = line.split("/system.slice/")[1].split(".service")[0]
                return f"{unit}.service"
    except Exception:
        pass
    return None


def scan_all():
    """Return dict of pid -> {cmdline, ppid, rss_kb} for all processes."""
    procs = {}
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            cmdline = read_cmdline(pid)
            if cmdline:
                procs[pid] = {
                    "cmdline": cmdline,
                    "ppid": get_ppid(pid),
                    "rss_kb": get_rss_kb(pid),
                }
    except OSError:
        pass
    return procs


def is_specialist_gateway(pid, gateways):
    """Check if a PID is a known specialist bot gateway (systemd service)."""
    if pid not in gateways:
        return False
    unit = get_systemd_unit(pid)
    if unit:
        service_name = unit.replace(".service", "")
        return service_name in KNOWN_SPECIALIST_SERVICES
    return False


def classify(procs):
    """
    Identify MCP processes and classify as legitimate or orphan.
    Returns (gateways, chats, mcps, legitimate_mcps, orphans).
    """
    gateways = {}
    chats = {}
    mcps = {}

    for pid, info in procs.items():
        cmd = info["cmdline"]
        if any(p in cmd for p in GATEWAY_CMDS):
            gateways[pid] = info
        if any(p in cmd for p in CHAT_CMDS):
            chats[pid] = info
        if any(p in cmd for p in MCP_PATTERNS):
            mcps[pid] = info

    # Build legitimate parent set: primary gateway + all chat sessions
    # Exclude specialist bot gateways (they don't host MCPs)
    legit_parents = set(chats.keys())
    for gpid in gateways:
        if not is_specialist_gateway(gpid, gateways):
            legit_parents.add(gpid)

    orphans = []
    legitimate_mcps = []

    for pid, info in mcps.items():
        ppid = info["ppid"]
        if ppid in legit_parents:
            legitimate_mcps.append((pid, info))
            continue

        # Check if parent is an intermediate process in a known chain
        # Pattern: chat → uv uvx workspace-mcp → python workspace-mcp
        # The workspace-mcp Python process has PPID=uv, not PPID=chat
        if ppid and ppid > 1 and is_alive(ppid):
            parent_cmdline = read_cmdline(ppid)
            if parent_cmdline:
                # Check if grandparent is legitimate
                grandparent_ppid = get_ppid(ppid)
                if grandparent_ppid in legit_parents:
                    legitimate_mcps.append((pid, info))
                    continue
                # Also check if parent is itself an MCP process (chained)
                if any(p in parent_cmdline for p in MCP_PATTERNS):
                    legitimate_mcps.append((pid, info))
                    continue

        # PPID=1 or parent dead → orphan
        if ppid == 1 or not is_alive(ppid):
            orphans.append((pid, info))
        else:
            # Parent alive but unknown → keep as legitimate (safe default)
            legitimate_mcps.append((pid, info))

    return gateways, chats, mcps, legitimate_mcps, orphans


def kill_procs(targets, dry_run):
    """Kill a list of (pid, info) tuples. Returns (count, rss_kb, details)."""
    count = 0
    rss = 0
    details = []
    for pid, info in targets:
        rss_kb = info.get("rss_kb", 0)
        count += 1
        rss += rss_kb

        if not dry_run:
            try:
                os.kill(pid, signal.SIGTERM)
                details.append({"pid": pid, "rss_kb": rss_kb, "action": "SIGTERM",
                                "cmdline": info["cmdline"][:100]})
            except OSError as e:
                details.append({"pid": pid, "rss_kb": rss_kb, "action": f"FAILED: {e}",
                                "cmdline": info["cmdline"][:100]})
        else:
            details.append({"pid": pid, "rss_kb": rss_kb, "action": "would SIGTERM",
                            "cmdline": info["cmdline"][:100]})
    return count, rss, details


def main():
    dry_run = os.environ.get("DRY_RUN", "true").lower() == "true"

    procs = scan_all()
    gateways, chats, mcps, legitimate_mcps, orphans = classify(procs)

    orphan_count, orphan_rss, orphan_details = kill_procs(orphans, dry_run)

    # --- Output ---
    ts = time.strftime("%d/%m/%y %H:%M")
    print(f"Orphan MCP Cleanup — {ts}")
    print(f"Mode: {'DRY-RUN' if dry_run else 'LIVE'}")
    print(f"Gateways (total): {len(gateways)}  |  Active chats: {len(chats)}")
    print(f"Total MCPs: {len(mcps)}  |  Legitimate: {len(legitimate_mcps)}  |  Orphans: {orphan_count}")

    if orphan_count:
        total_mb = orphan_rss / 1024
        print(f"\nOrphans killed: {orphan_count}  (~{total_mb:.0f}MB reclaimed)")
        for d in orphan_details:
            print(f"  [{d['action']}] PID {d['pid']}  ({d['rss_kb']}KB)  {d['cmdline'][:80]}")
        if dry_run:
            print(f"\nSet DRY_RUN=false to execute.")
    else:
        print("\nNo orphan MCPs found.")

    if dry_run and orphan_count == 0:
        print("All MCP processes are properly parented. System clean.")


if __name__ == "__main__":
    main()
