"""SII Worker AI1 pre-dispatch readiness gate.

The gate is assignee-scoped to ``sii-worker`` by default (overridable through
``HERMES_SII_AI1_ASSIGNEE``). It probes the configured AI1 OpenAI-compatible
sends one Wake-on-LAN packet when configured and needed, waits for readiness,
and raises instead of allowing a Worker spawn on an unavailable or substituted
host.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes_cli.kanban_db import Task


class AI1Unavailable(RuntimeError):
    """The dedicated SII Worker inference host did not become ready."""


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise AI1Unavailable(f"invalid {name}={raw!r}") from exc
    if value <= 0:
        raise AI1Unavailable(f"invalid {name}={raw!r}")
    return value


def _probe(url: str, model: str, *, timeout: int) -> bool:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        return False
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    return any(isinstance(row, dict) and row.get("id") == model for row in rows)


def _send_wake(mac: str) -> None:
    try:
        result = subprocess.run(
            ["wakeonlan", mac],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AI1Unavailable(f"AI1 wake command failed: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown error").strip()
        raise AI1Unavailable(f"AI1 wake command failed: {detail}")


def ensure_ai1_available(task: Task) -> None:
    """Ensure AI1 is ready before spawning the explicitly configured SII Worker.

    No other assignee is affected. Failure raises with an explicit no-fallback
    message so the dispatcher records the attempt and blocks at its configured
    failure limit instead of invoking the Worker against another host.
    """
    assignee = os.environ.get("HERMES_SII_AI1_ASSIGNEE", "sii-worker").strip()
    if not assignee or task.assignee != assignee:
        return

    models_url = os.environ.get(
        "HERMES_SII_AI1_MODELS_URL", "http://100.67.5.70:11436/v1/models"
    ).strip()
    model = os.environ.get(
        "HERMES_SII_AI1_MODEL", "qwen38-27b-mtp-fullctx"
    ).strip()
    probe_timeout = _positive_int("HERMES_SII_AI1_PROBE_TIMEOUT_SECONDS", 5)
    if _probe(models_url, model, timeout=probe_timeout):
        return

    mac = os.environ.get("HERMES_SII_AI1_WOL_MAC", "b0:82:e2:ac:d2:21").strip()
    if not mac:
        raise AI1Unavailable(
            "AI1 is unavailable and no Wake-on-LAN MAC is configured; "
            "no alternate host or model is permitted"
        )
    _send_wake(mac)

    deadline = time.monotonic() + _positive_int(
        "HERMES_SII_AI1_WAKE_TIMEOUT_SECONDS", 300
    )
    while time.monotonic() < deadline:
        time.sleep(5)
        if _probe(models_url, model, timeout=probe_timeout):
            return
    raise AI1Unavailable(
        "AI1 did not become ready after Wake-on-LAN; "
        "no alternate host or model is permitted"
    )
