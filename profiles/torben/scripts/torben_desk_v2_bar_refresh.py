#!/usr/bin/env python3
"""Hermes wrapper for torben-desk-v2-bar-refresh."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(os.getenv("RATATOSK_DESK_V2_ROOT", '/Users/ericfreeman/ratatosk'))
ENV = os.environ.copy()
ENV.update({'RATATOSK_LIVE_TRADING': '0', 'ROBINHOOD_LIVE': '0', 'ROBINHOOD_EQUITY_LIVE': '0'})
ENV.setdefault("NO_COLOR", "1")
ENV.setdefault("TERM", "dumb")


def main() -> int:
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "ratatosk.desk.orchestration.runner",
        'bar_store_refresh',
        "--repo-root",
        str(REPO_ROOT),
    ]
    return subprocess.run(cmd, cwd=REPO_ROOT, env=ENV, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
