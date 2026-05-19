#!/usr/bin/env python3
"""Emit a redacted JSON MCP runtime-depth probe report."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.mcp_runtime_probe import probe_default_sources  # noqa: E402


def _hard_timeout_exit() -> None:
    report = {
        "status": "timeout",
        "status_counts": {"timeout": 1},
        "check_status_counts": {"timeout": 1},
        "servers": [],
        "status_classes": ["timeout"],
        "status_samples": {},
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    os._exit(124)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=["codex", "claude", "hermes", "all"], default="all")
    parser.add_argument("--server", help="probe only one server name or source:name id")
    parser.add_argument("--json", action="store_true", help="emit JSON; retained for proof command symmetry")
    parser.add_argument("--redact", action="store_true", help="force redacted output; all output is always redacted")
    parser.add_argument("--timeout", type=int, default=120, help="overall probe timeout in seconds")
    parser.add_argument(
        "--skip-auth-needed",
        action="store_true",
        help="skip entries known to need interactive auth, including Google Drive",
    )
    parser.add_argument(
        "--include-google-drive-auth-needed",
        action="store_true",
        help="Include Google Drive MCP entries that are only missing auth.",
    )
    args = parser.parse_args()

    timer = None
    if args.timeout > 0:
        # MCP SDK/client shutdown can block while child stdio servers are wedged.
        # Use a process-level watchdog so proof runs have a hard wall clock.
        timer = threading.Timer(args.timeout + 5, _hard_timeout_exit)
        timer.daemon = True
        timer.start()
    try:
        report = asyncio.run(
            asyncio.wait_for(
                probe_default_sources(
                    runtime=args.runtime,
                    server_name=args.server,
                    skip_google_drive_auth_needed=args.skip_auth_needed or not args.include_google_drive_auth_needed,
                ),
                timeout=args.timeout if args.timeout > 0 else None,
            )
        )
    except asyncio.TimeoutError:
        report = {
            "status": "timeout",
            "status_counts": {"timeout": 1},
            "check_status_counts": {"timeout": 1},
            "servers": [],
            "status_classes": ["timeout"],
            "status_samples": {},
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 124
    finally:
        if timer is not None:
            timer.cancel()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
