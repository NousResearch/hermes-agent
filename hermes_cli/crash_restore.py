"""Startup crash-restore offers for interactive CLI sessions.

A cleanly exited interactive CLI always finalizes its session row
(``end_session(..., "cli_close")``).  A crash — SIGKILL, machine restart,
terminal window closed — skips that path and leaves the row open
(``ended_at IS NULL``) with no pointer back to it: the user has to dig the
id out of ``hermes sessions list`` by hand.

This module gives each interactive CLI session a tiny liveness marker file
``$HERMES_HOME/runtime/cli-live/<session_id>.json`` containing the owning
``pid`` + ``process_start_time``.  The marker is written when the
interactive loop starts (and re-pointed when the session id changes via
``/new`` / ``/resume``), and removed on clean exit.  At the next
interactive startup, markers whose owning process is gone but whose session
row is still open with content are offered for restore — mirroring GitHub
Copilot CLI's "restore sessions that were still open when their CLI went
away" startup flow (v1.0.81).

Everything is best-effort: no function raises, and the feature is gated by
``session.crash_restore`` in config.yaml (default true).

Pid liveness reuses :func:`hermes_cli.active_sessions._pid_alive`, which
pairs the pid with the process create time so a recycled pid cannot keep a
dead session looking alive.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Markers older than this are removed regardless of row state — a marker
# that survived 30 days belongs to a session the user has moved past.
_STALE_AFTER_SECONDS = 30 * 24 * 60 * 60

# Cap on how many crashed sessions one startup will offer.
DEFAULT_OFFER_LIMIT = 3


def _markers_dir() -> Path:
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home()) / "runtime" / "cli-live"


def is_enabled() -> bool:
    """Config gate: ``session.crash_restore`` (default true)."""
    try:
        from hermes_cli.config import load_config

        return bool((load_config().get("session") or {}).get("crash_restore", True))
    except Exception:
        return True


def _process_start_time(pid: int) -> Optional[float]:
    try:
        import psutil  # type: ignore

        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def write_live_marker(session_id: str) -> None:
    """Record that this process currently owns ``session_id``.

    Synchronous, best-effort, never raises.  No-op when the feature is
    disabled or the session id is empty.
    """
    try:
        if not session_id or not is_enabled():
            return
        directory = _markers_dir()
        directory.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        payload = {
            "session_id": str(session_id),
            "pid": pid,
            "process_start_time": _process_start_time(pid),
            "ts": time.time(),
        }
        path = directory / f"{session_id}.json"
        tmp = directory / f".{session_id}.{pid}.tmp"
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        logger.debug("crash-restore marker write failed", exc_info=True)


def remove_live_marker(session_id: str) -> None:
    """Drop the marker on clean exit (or session-id handoff). Never raises."""
    try:
        if not session_id:
            return
        (_markers_dir() / f"{session_id}.json").unlink(missing_ok=True)
    except Exception:
        logger.debug("crash-restore marker removal failed", exc_info=True)


def _read_marker(path: Path) -> Optional[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if not str(data.get("session_id") or "").strip():
        return None
    return data


def find_crashed_sessions(
    session_db,
    *,
    current_session_id: str = "",
    limit: int = DEFAULT_OFFER_LIMIT,
) -> list[dict[str, Any]]:
    """Return crashed-session rows worth offering for restore, newest first.

    A marker qualifies when ALL of:

    - its owning process is gone (pid dead, or alive-but-different process
      via the create-time pair check);
    - its session row still exists, is still open (``ended_at IS NULL``),
      and has at least one message to restore.

    Housekeeping happens inline: markers for cleanly-ended or deleted rows
    and markers past the staleness window are removed so the directory
    can't accumulate.  Never raises; returns ``[]`` on any failure.
    """
    try:
        if session_db is None or not is_enabled():
            return []
        directory = _markers_dir()
        if not directory.is_dir():
            return []
        from hermes_cli.active_sessions import _pid_alive

        now = time.time()
        offers: list[dict[str, Any]] = []
        for path in sorted(directory.glob("*.json")):
            marker = _read_marker(path)
            if marker is None:
                _unlink_quietly(path)
                continue
            session_id = str(marker["session_id"])
            ts = marker.get("ts")
            if isinstance(ts, (int, float)) and now - ts > _STALE_AFTER_SECONDS:
                _unlink_quietly(path)
                continue
            if session_id == current_session_id:
                continue
            if _pid_alive(marker.get("pid"), marker.get("process_start_time")):
                continue  # live in another terminal — not ours to offer
            row = None
            try:
                row = session_db.get_session(session_id)
            except Exception:
                continue
            if not row:
                _unlink_quietly(path)
                continue
            if row.get("ended_at") is not None:
                # Ended cleanly elsewhere (e.g. resumed + closed) — stale marker.
                _unlink_quietly(path)
                continue
            if not row.get("message_count"):
                # Nothing to restore; leave the empty-row pruning to the
                # existing exit-time discard logic.
                _unlink_quietly(path)
                continue
            offers.append(row)
        offers.sort(
            key=lambda r: r.get("last_activity_at") or r.get("started_at") or 0,
            reverse=True,
        )
        return offers[: max(1, int(limit))]
    except Exception:
        logger.debug("crash-restore scan failed", exc_info=True)
        return []


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
