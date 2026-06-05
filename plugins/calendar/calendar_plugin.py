"""Thin subprocess wrapper over `gws calendar`.

Per operator decision 2026-05-28 03:35 IDT (memory: project_gws_adoption_2026-05-28),
the calendar plugin shells out to the bundled gws (Google Workspace CLI) binary
for every tool call. OAuth + token storage are owned entirely by gws; this
module never touches credentials.

Binary resolution: companion GWS bin env wins; falls back to ``gws`` on PATH for dev.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Logs MUST NOT include event titles or attendee addresses (intake
# redaction_expectations). Only event_id + start/end timestamps are safe in
# non-debug logs.
_SAFE_EVENT_LOG_FIELDS = ("id", "start", "end")
_SAFE_CHILD_ENV_KEYS = (
    "PATH",
    "HOME",
    "USER",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "AG""ENTE_GWS_BIN",
)
_SECRET_ENV_KEYS = {"GOOGLE_WORKSPACE_CLI_CLIENT_SECRET"}


def _gws_bin() -> str:
    """Resolve the gws binary path. Env wins, PATH fallback for dev."""
    explicit = os.environ.get("AG""ENTE_GWS_BIN")
    if explicit:
        return explicit
    found = shutil.which("gws")
    if found:
        return found
    # Defer the failure until call-time so import never raises in environments
    # where gws is intentionally absent (e.g. CI without the bundle).
    return "gws"


def _gws_env() -> Dict[str, str]:
    """Return a narrow child env without direct Google Workspace secret material."""
    env = {key: os.environ[key] for key in _SAFE_CHILD_ENV_KEYS if key in os.environ}
    for key, value in os.environ.items():
        if key in _SECRET_ENV_KEYS:
            continue
        if key.startswith("GWS_") or key.startswith("GOOGLE_WORKSPACE_CLI_"):
            env[key] = value
    return env


def _gws_json(args: List[str], timeout: float = 30.0) -> Any:
    """Invoke ``gws <args>`` and return the parsed JSON stdout.

    Raises ``RuntimeError`` on non-zero exit or unparseable stdout. The
    subprocess env is scrubbed of caller-visible secrets — gws reads its own
    credentials from its file-backed keyring; we never pass tokens through env.
    """
    cmd = [_gws_bin(), *args]
    # IMPORTANT: do NOT log args verbatim — they may contain event titles /
    # attendee emails for create_event. Log only the subcommand.
    subcommand = args[1] if len(args) >= 2 else (args[0] if args else "?")
    logger.debug("calendar plugin: shelling gws calendar %s", subcommand)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=_gws_env(),
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"gws binary not found at {cmd[0]!r}: {e}") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"gws calendar {subcommand} timed out after {timeout}s") from e

    if proc.returncode != 0:
        # stderr may surface gws-side auth errors ("not authenticated") — safe
        # to bubble; gws does not echo tokens.
        raise RuntimeError(
            f"gws calendar {subcommand} exited {proc.returncode}: {proc.stderr.strip()}"
        )

    stdout = proc.stdout.strip()
    if not stdout:
        return None
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"gws calendar {subcommand} returned non-JSON stdout") from e


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def list_calendar_events(
    after: str,
    before: str,
    calendar_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """List events between *after* and *before*.

    Returns the JSON array emitted by ``gws calendar list ... --json``.
    """
    args = ["calendar", "list", "--from", after, "--to", before, "--json"]
    if calendar_id:
        args += ["--calendar", calendar_id]
    if limit is not None:
        args += ["--limit", str(int(limit))]
    res = _gws_json(args)
    if res is None:
        return []
    if isinstance(res, dict) and "events" in res:
        return list(res["events"])
    if isinstance(res, list):
        return res
    return [res]


def create_calendar_event(
    title: str,
    start: str,
    end: str,
    attendees: Optional[List[str]] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a new event. Returns the created event dict from gws."""
    args = [
        "calendar", "create",
        "--title", title,
        "--start", start,
        "--end", end,
        "--json",
    ]
    if attendees:
        args += ["--attendees", ",".join(attendees)]
    if description:
        args += ["--description", description]
    res = _gws_json(args)
    return res if isinstance(res, dict) else {"event": res}


def get_calendar_event(event_id: str) -> Dict[str, Any]:
    """Fetch a single event by id."""
    args = ["calendar", "get", "--event", event_id, "--json"]
    res = _gws_json(args)
    return res if isinstance(res, dict) else {"event": res}
