"""Regression test for issue #21188.

`Gateway._notify_active_sessions_of_shutdown` must bound every adapter
send with ``asyncio.wait_for`` so a slow / unreachable platform (e.g.
Discord returning 403 Forbidden after a long network wait) can never
delay the shutdown sequence past systemd's ``TimeoutStopSec``.
"""

import asyncio
import re
from pathlib import Path

import pytest


_RUN_PY = Path(__file__).resolve().parents[2] / "gateway" / "run.py"


def test_shutdown_notify_uses_wait_for_on_active_session_send():
    """Static check: the active-session send is wrapped in asyncio.wait_for."""
    src = _RUN_PY.read_text(encoding="utf-8")
    # Locate the function body.
    match = re.search(
        r"async def _notify_active_sessions_of_shutdown\(self\).*?(?=\n    async def |\n    def )",
        src,
        re.DOTALL,
    )
    assert match, "_notify_active_sessions_of_shutdown not found in gateway/run.py"
    body = match.group(0)
    assert "asyncio.wait_for(" in body, (
        "Shutdown notification sends must be bounded by asyncio.wait_for "
        "(issue #21188)."
    )
    assert "_SHUTDOWN_NOTIFY_TIMEOUT_S" in body, (
        "Use the _SHUTDOWN_NOTIFY_TIMEOUT_S module constant for the timeout."
    )


def test_shutdown_notify_timeout_constant_defined():
    """Static check: the module exposes the timeout constant."""
    src = _RUN_PY.read_text(encoding="utf-8")
    assert re.search(r"^_SHUTDOWN_NOTIFY_TIMEOUT_S\s*=\s*\d", src, re.MULTILINE), (
        "gateway.run must define _SHUTDOWN_NOTIFY_TIMEOUT_S at module scope."
    )