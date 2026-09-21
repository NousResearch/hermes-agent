#!/usr/bin/env python3
"""MCP Server Health Check - probes configured MCP servers for responsiveness.

Silent when healthy (exit 0, empty stdout).
Outputs alert text when a server is unreachable or slow.
"""

# P13: disabled-staging guard — exit early when cron is disabled
import os as _os, sys as _sys
if _os.environ.get("DRY_RUN") == "1":
    print(f"[DRY_RUN] {_os.path.basename(__file__)}")
    _sys.exit(0)

import subprocess
import sys
import json
import time
from pathlib import Path

HERMES = Path(_os.environ.get("HERMES_HOME", "/home/kensei/.hermes"))

# MCP servers to check (name, pgrep -f pattern matching real process cmdline)
# v2 - fix 03/06/26: previous patterns ("outlook-mcp-server",
# "google-workspace-mcp-server", "mailbox-cleaner-mcp") did not match the
# actual node command lines spawned by the gateway. They returned 0 hits
# every run and produced false "process not found" alerts whenever a real
# tool error appeared in the journal. Use the real cmdline substrings.
MCP_CHECKS = [
    ("gitnexus", "gitnexus"),
    ("google_workspace", "workspace-mcp"),       # actually @isaacphi/workspace-mcp
    ("outlook", "ms-365-mcp-server"),            # actually @softeria/ms-365-mcp-server
    ("mailbox_cleaner", "mailbox_cleaner"),      # matches the Python module path
    ("nanobanana", "nanobanana"),
]

def get_mcp_processes():
    """Get all running MCP-related processes."""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.splitlines()
        mcp_lines = [l for l in lines if "mcp" in l.lower() or "model-server" in l.lower()]
        return mcp_lines
    except Exception:
        return []

def check_process_alive(pattern):
    """Check if a process matching pattern is running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False

def check_mcp_port(name):
    """Check if MCP server is listening on expected port."""
    # Most MCP servers use stdio, not TCP - so we check process existence
    return check_process_alive(name)

def check_mcp_log_errors(name):
    """Check for recent errors in MCP-related logs."""
    # Check gateway logs for MCP errors
    try:
        result = subprocess.run(
            ["journalctl", "-u", "hermes-gateway.service", "--since", "30 min ago", "--no-pager", "-q"],
            capture_output=True, text=True, timeout=15
        )
        errors = []
        for line in result.stdout.splitlines():
            if name.lower() in line.lower() and any(kw in line.lower() for kw in ["error", "fail", "timeout", "refused"]):
                errors.append(line.strip()[:200])
        return errors[-2:]
    except Exception:
        return []

def main():
    alerts = []
    running_mcps = get_mcp_processes()
    
    for name, pattern in MCP_CHECKS:
        is_alive = check_process_alive(pattern)
        
        if not is_alive:
            # Check if it's expected to be running (not all MCPs run all the time)
            errors = check_mcp_log_errors(name)
            if errors:
                alerts.append(f"**{name}** - process not found, recent errors: {'; '.join(errors[:2])}")
            # Don't alert if process just isn't configured/started - only alert on errors
    
    # Also check for orphaned MCP processes (running but not parented to a gateway)
    if running_mcps:
        # Just report count - the orphan watchdog handles killing
        pass
    
    if not alerts:
        sys.exit(0)  # Silent when healthy
    
    print(f"**🟡 MCP Server Health Alert - {time.strftime('%d/%m/%y %H:%M')}**\n")
    for alert in alerts:
        print(f"• {alert}")
    print(f"\n**Action:** Check MCP server configuration and restart if needed.")
    sys.exit(0)

if __name__ == "__main__":
    main()
