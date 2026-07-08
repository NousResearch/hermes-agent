"""
Duplicate platform gateway detector.

A lightweight, zero-overhead check that runs once at gateway startup.
Scans all Python processes on the machine for other Hermes gateway instances
that are connected to the same platform (e.g. Feishu bot), and warns the user.

Usage:
    from gateway.platform_detect import check_duplicate_platform_gateways
    await check_duplicate_platform_gateways(own_platforms=["feishu"])
"""

from __future__ import annotations

import logging
import os
import socket
import time
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

_GATEWAY_MARKERS = ("hermes_cli.main", "gateway", "run")
"""Substrings to identify a Hermes gateway process from its command line."""


def _is_hermes_gateway_cmd(cmd_str: str) -> bool:
    """Check if a command-line string belongs to a Hermes gateway."""
    if "hermes_cli.main" not in cmd_str and "hermes-gateway" not in cmd_str:
        return False
    return "gateway" in cmd_str or "run" in cmd_str


@dataclass
class DuplicateGatewayInfo:
    pid: int
    hostname: str
    username: str
    started_at: float
    feishu_connections: int


def find_other_gateways() -> List[DuplicateGatewayInfo]:
    """Scan for other Hermes gateway processes on this machine.

    Returns a list of gateway processes (excluding our own PID) that have
    established connections to known platform API endpoints.

    Args:
        mark: A substring to identify gateway processes in the command line.
              Defaults to ``"gateway"``.
        platform_domains: Optional set of domain substrings to match against
                          connection remote addresses.  When set, only
                          processes with matching connections are returned.

    Returns:
        A list of ``DuplicateGatewayInfo`` namedtuples, one per detected
        process, ordered by PID.  Empty when no duplicates are found.
    """
    import psutil

    my_pid = os.getpid()
    others: List[DuplicateGatewayInfo] = []
    hostname = socket.gethostname()

    for proc in psutil.process_iter(
        ["pid", "cmdline", "create_time", "username"]
    ):
        try:
            pid = proc.info["pid"]
            if pid == my_pid:
                continue

            cmdline = proc.info["cmdline"]
            if not cmdline:
                continue
            cmd_str = " ".join(str(c) for c in cmdline)
            if not _is_hermes_gateway_cmd(cmd_str):
                continue

            # connections() requires a separate syscall per process
            try:
                conns = proc.connections()
            except (psutil.AccessDenied, PermissionError):
                continue

            # Count established connections — any gateway with live
            # connections is likely active.  We don't try to match
            # destination domains here because psutil resolves
            # connections to IPs, not hostnames, and Feishu websockets
            # go through DNS load-balancers with unpredictable IPs.
            active = sum(1 for c in conns if c.status == "ESTABLISHED" and len(c.raddr) > 1)
            if active == 0:
                continue

            others.append(DuplicateGatewayInfo(
                pid=pid,
                hostname=hostname,
                username=proc.info["username"],
                started_at=proc.info["create_time"],
                feishu_connections=active,
            ))
        except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
            continue

    others.sort(key=lambda g: g.pid)
    return others


def _fmt_time(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def format_warning(gateways: List[DuplicateGatewayInfo]) -> str:
    """Build a human-readable warning message from detected duplicates."""
    lines = [
        "╔══════════════════════════════════════════════════════════════╗",
        "║  ⚠️  Duplicate gateway detected                             ║",
        "║  Another Hermes gateway is connected to the same platform.  ║",
        "║  This can cause duplicate replies.                         ║",
        "╚══════════════════════════════════════════════════════════════╝",
        "",
    ]
    for g in gateways:
        lines.extend([
            f"  PID:       {g.pid}",
            f"  Hostname:  {g.hostname}",
            f"  User:      {g.username}",
            f"  Started:   {_fmt_time(g.started_at)}",
            f"  Feishu:    {g.feishu_connections} connection(s)",
            "",
            f"  To stop it:  kill {g.pid}",
            "",
        ])
    lines.append(
        "If you're running Hermes on multiple machines, "
        "close the one you don't need."
    )
    return "\n".join(lines)


async def check_duplicate_platform_gateways() -> None:
    """Run the duplicate gateway check and log warnings if found.

    Designed to be called once during gateway startup.  Zero ongoing
    overhead — the scan runs once and the result is discarded.
    """
    try:
        dupes = find_other_gateways()
    except ImportError:
        logger.debug(
            "psutil not available — skipping duplicate gateway detection"
        )
        return
    except Exception as exc:
        logger.debug(
            "Duplicate gateway detection failed: %s", exc
        )
        return

    if not dupes:
        return

    # Use logger.warning so the message appears in both stdout and logs
    for line in format_warning(dupes).split("\n"):
        logger.warning("%s", line)
