"""The READY-sentinel contract of ``hermes serve`` (whatever spawns it learns the port from stdout).

The Desktop's TypeScript parser that used to read this sentinel was removed with the desktop app;
``desktop_parse`` keeps its rule in Python: ``^HERMES_(BACKEND|DASHBOARD)_READY port=<digits>`` on a
line of the stdout stream seen so far.
"""

from __future__ import annotations

import re

_READY_RE = re.compile(r"^HERMES_(?:BACKEND|DASHBOARD)_READY port=(\d+)", re.M)


def desktop_parse(stdout_bytes: str) -> dict:
    """The port a spawner learns from this exact stdout stream, or why it learns none."""
    match = _READY_RE.search(stdout_bytes)
    return {"port": int(match.group(1))} if match else {"error": "no READY sentinel"}
