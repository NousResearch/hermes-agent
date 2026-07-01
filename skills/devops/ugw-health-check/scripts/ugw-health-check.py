#!/usr/bin/env python3
"""
Unified Gateway (UGW) Health Check
====================================
Standalone script — no dependency on the hermes agent codebase.

Reads ~/.hermes/gateway_state.json and reports the gateway health status.

Exit codes:
  0 = running (healthy)
  1 = critical / not running
  2 = degraded (warning)

Install:
  cp ugw-health-check.py ~/.hermes/scripts/ugw-health-check.py
  chmod +x ~/.hermes/scripts/ugw-health-check.py
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
_STATE_FILE = _HERMES_HOME / "gateway_state.json"

# Valid gateway_state values as defined in gateway/status.py
_HEALTHY_STATES = {"running"}
_DEGRADED_STATES = {"degraded"}
# Everything else (starting, draining, stopped, startup_failed) is CRITICAL


def _read_state() -> Optional[dict]:
    """Read and parse gateway_state.json, returning None on any failure."""
    if not _STATE_FILE.exists():
        return None
    try:
        raw = _STATE_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _format_uptime(start_time: Optional[int]) -> str:
    """Return a human-readable uptime string, or 'unknown' if unavailable."""
    if start_time is None:
        return "unknown"
    # start_time is a kernel clock-tick value from /proc/<pid>/stat — not a
    # Unix timestamp. We cannot convert it to a wall-clock duration without
    # knowing the boot time, so we fall back to the file's updated_at field
    # (which is a real ISO timestamp) for a rough "last updated" indicator.
    return f"start_time tick={start_time}"


def _format_uptime_from_updated(updated_at: Optional[str]) -> str:
    """Return a human-readable age from an ISO timestamp string."""
    if not updated_at:
        return "unknown"
    try:
        dt = datetime.fromisoformat(updated_at)
        age_s = int((datetime.now(timezone.utc) - dt).total_seconds())
        if age_s < 0:
            age_s = 0
        hours, rem = divmod(age_s, 3600)
        minutes, seconds = divmod(rem, 60)
        if hours:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    except (ValueError, TypeError):
        return "unknown"


def _pid_is_alive(pid: int) -> bool:
    """Check if a process with the given PID is alive using signal 0."""
    try:
        os.kill(pid, 0)  # signal 0 = check existence without sending a signal
        return True
    except ProcessLookupError:
        return False  # PID does not exist
    except PermissionError:
        return True  # PID exists but we lack permission to signal it


_HERMES_CMDLINE_PATTERNS = ["hermes_cli.main", "hermes_cli/main", "hermes-agent", "hermes"]


def _pid_is_gateway(pid: int, argv: list, start_time: Optional[int]) -> bool:
    """Check PID exists and belongs to the gateway process (not a reused PID).

    Uses /proc/{pid}/cmdline on Linux to verify the running process matches the
    recorded argv[0] from the state file.  Falls back gracefully on macOS or
    other platforms where /proc is not available.
    """
    # First check it's alive
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # alive but can't signal — proceed to identity check

    # On Linux, verify via /proc/{pid}/cmdline
    cmdline_path = f"/proc/{pid}/cmdline"
    if os.path.exists(cmdline_path):
        try:
            with open(cmdline_path, "rb") as f:
                cmdline = f.read().decode("utf-8", errors="replace").replace("\x00", " ").strip()
            # Check if any hermes-identifying pattern appears in the cmdline.
            # When the gateway runs as a systemd service, argv[0] in the state
            # file may be the resolved path (e.g. /usr/lib/python3.x/.../main.py)
            # while /proc/<pid>/cmdline contains "python -m hermes_cli.main
            # gateway run". A direct argv[0] substring match would fail, so we
            # check for known hermes patterns instead.
            cmdline_lower = cmdline.lower()
            is_hermes_process = any(p in cmdline_lower for p in _HERMES_CMDLINE_PATTERNS)
            if not is_hermes_process:
                return False  # PID reuse — unrelated process
        except OSError:
            pass

    # Check process state — reject stopped/traced processes.
    # os.kill(pid, 0) succeeds even for SIGSTOP'd processes, and a stopped
    # gateway cannot process messages despite appearing alive.
    status_path = f"/proc/{pid}/status"
    if os.path.exists(status_path):
        try:
            with open(status_path) as f:
                for line in f:
                    if line.startswith("State:"):
                        state_char = line.split()[1]  # e.g. "S", "R", "T", "t", "Z"
                        if state_char in ("T", "t"):
                            return False  # Stopped or traced — cannot process messages
                        break
        except OSError:
            pass  # Can't read status, assume alive

    # On macOS/other: if start_time is available, try to compare uptime
    # (This is a best-effort check; /proc is Linux-only)

    return True


def _format_platforms(platforms: dict) -> str:
    """Return a compact platform summary string."""
    if not platforms:
        return "(none)"
    parts = []
    for name, info in platforms.items():
        state = info.get("state", "unknown")
        parts.append(f"{name}:{state}")
    return ", ".join(parts)


def main() -> int:
    data = _read_state()

    if data is None:
        print("Unified Gateway CRITICAL")
        print(f"Status: CRITICAL (gateway_state_check: state file not found or unreadable)")
        print(f"State file: {_STATE_FILE}")
        return 1

    gateway_state = data.get("gateway_state", "unknown")
    pid = data.get("pid")
    active_agents = data.get("active_agents", 0)
    platforms = data.get("platforms", {})
    start_time = data.get("start_time")
    updated_at = data.get("updated_at")
    exit_reason = data.get("exit_reason")

    platform_summary = _format_platforms(platforms)
    uptime = _format_uptime_from_updated(updated_at)

    # Verify the recorded PID is actually alive and is the gateway process.
    # A SIGKILL or OOM kill leaves a stale gateway_state file reporting
    # "running" even though the process is gone. After a crash, the PID may
    # also be reused by an unrelated process — checking os.kill(pid, 0) alone
    # would return healthy for a reused PID.  _pid_is_gateway() additionally
    # verifies the process identity via /proc/{pid}/cmdline on Linux.
    if pid and not _pid_is_gateway(pid, data.get("argv", []), data.get("start_time")):
        print("Unified Gateway CRITICAL")
        print(f"Status: CRITICAL (PID {pid} is not the gateway process — possible PID reuse after crash)")
        print(f"Last updated: {uptime} ago")
        print(f"Platforms: {platform_summary}")
        return 1

    if gateway_state in _HEALTHY_STATES:
        print("Unified Gateway OK")
        print(f"Status: RUNNING (report_type: gateway_state_check)")
        print(f"Active agents: {active_agents}")
        print(f"Platforms: {platform_summary}")
        print(f"PID: {pid} | Last updated: {uptime} ago")
        return 0

    elif gateway_state in _DEGRADED_STATES:
        print("Unified Gateway DEGRADED")
        print(f"Status: DEGRADED (gateway_state: {gateway_state})")
        print(f"Active agents: {active_agents}")
        print(f"Platforms: {platform_summary}")
        print(f"PID: {pid} | Last updated: {uptime} ago")
        if exit_reason:
            print(f"Exit reason: {exit_reason}")
        return 2

    else:
        print("Unified Gateway CRITICAL")
        print(f"Status: CRITICAL (gateway_state: {gateway_state})")
        if exit_reason:
            print(f"Exit reason: {exit_reason}")
        print(f"PID: {pid} | Last updated: {uptime} ago")
        print(f"Platforms: {platform_summary}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
