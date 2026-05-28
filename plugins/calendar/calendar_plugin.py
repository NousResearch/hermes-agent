"""Calendar plugin — thin subprocess wrapper over `gws calendar`.

OAuth and token storage are owned entirely by gws. This module shells out to
the gws binary resolved from ``AGENTE_GWS_BIN`` env var. Each tool is a
single-function wrapper that builds the correct gws subcommand and returns
JSON-parsed output.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def _gws_bin() -> str:
    bin_path = os.environ.get("AGENTE_GWS_BIN")
    if not bin_path:
        raise RuntimeError(
            "AGENTE_GWS_BIN is not set — gws binary path required for calendar plugin"
        )
    return bin_path


def _gws_json(args: list[str]) -> list[dict] | dict:
    """Run a gws subcommand and return the parsed JSON output.

    All gws calendar commands produce line-delimited or single-object JSON
    when invoked with ``--json``. Output streams are captured and logged
    only at debug level to avoid leaking event content to logs.
    """
    cmd = [_gws_bin()] + args
    logger.debug("gws: %s", cmd)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        stderr_snippet = result.stderr.strip()[:500]
        raise RuntimeError(f"gws exited {result.returncode}: {stderr_snippet}")

    if not result.stdout.strip():
        return []

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        lines = [json.loads(line) for line in result.stdout.strip().split("\n") if line.strip()]
        return lines[0] if len(lines) == 1 else lines

    return data


def list_events(start: str, end: str, calendar_id: str | None = None) -> list[dict]:
    args = ["calendar", "list", "--from", start, "--to", end, "--json"]
    if calendar_id:
        args += ["--calendar", calendar_id]
    return _gws_json(args)


def create_event(
    start: str,
    end: str,
    title: str,
    description: str | None = None,
    location: str | None = None,
    calendar_id: str | None = None,
) -> dict:
    args = [
        "calendar", "create",
        "--from", start,
        "--to", end,
        "--title", title,
        "--json",
    ]
    if description:
        args += ["--description", description]
    if location:
        args += ["--location", location]
    if calendar_id:
        args += ["--calendar", calendar_id]
    return _gws_json(args)


def update_event(
    event_id: str,
    start: str | None = None,
    end: str | None = None,
    title: str | None = None,
    description: str | None = None,
    location: str | None = None,
    calendar_id: str | None = None,
) -> dict:
    args = ["calendar", "update", event_id, "--json"]
    if start:
        args += ["--from", start]
    if end:
        args += ["--to", end]
    if title:
        args += ["--title", title]
    if description:
        args += ["--description", description]
    if location:
        args += ["--location", location]
    if calendar_id:
        args += ["--calendar", calendar_id]
    return _gws_json(args)


def cancel_event(event_id: str, calendar_id: str | None = None) -> dict:
    args = ["calendar", "delete", event_id, "--json"]
    if calendar_id:
        args += ["--calendar", calendar_id]
    return _gws_json(args)


def find_free_slots(
    duration_minutes: int = 30,
    within: str = "",
    calendar_id: str | None = None,
) -> list[dict]:
    args = [
        "calendar", "freebusy",
        "--duration", str(duration_minutes),
        "--within", within,
        "--json",
    ]
    if calendar_id:
        args += ["--calendar", calendar_id]
    return _gws_json(args)
