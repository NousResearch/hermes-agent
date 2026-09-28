"""Bridge to the Desktop's real READY-sentinel parser (``apps/desktop/electron/backend-ready.ts``).

There is no shared constant for the sentinel format, and regexing the TS source
would test a copy of the contract. Instead the captured backend stdout is fed
through the parser itself under Node (``desktop_ready_parse.mjs``), so the Python
side asserts exactly what the Desktop would conclude from those bytes.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.e2e.core.parity._helpers import REPO_ROOT

PARSER_TS = REPO_ROOT / "apps" / "desktop" / "electron" / "backend-ready.ts"
BRIDGE_MJS = Path(__file__).with_name("desktop_ready_parse.mjs")
# Native TypeScript type-stripping landed unflagged in Node 22.18 / 23.6.
_MIN_NODE = (22, 18)


def node_binary() -> str | None:
    node = shutil.which("node")
    if not node:
        return None
    out = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=30,
                         stdin=subprocess.DEVNULL).stdout.strip()
    m = re.match(r"v(\d+)\.(\d+)", out)
    if not m or (int(m.group(1)), int(m.group(2))) < _MIN_NODE:
        return None
    return node


def require_node() -> str:
    node = node_binary()
    if node is None:
        pytest.skip(f"needs node >= {'.'.join(map(str, _MIN_NODE))} to run the Desktop READY parser")
    return node


def desktop_parse(stdout_bytes: str) -> dict:
    """What ``waitForDashboardPort`` resolves/rejects with for this exact stdout stream."""
    proc = subprocess.run(
        [require_node(), str(BRIDGE_MJS), str(PARSER_TS), "5000"], input=stdout_bytes,
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, f"desktop parser bridge crashed: {proc.stderr[-2000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])
