"""Outage notice for a gateway the Windows hang watchdog restarted (CORP-836).

The watchdog (``hermes_cli/gateway_windows_watchdog.py``) runs outside the gateway and holds no
messaging credentials, so it never sends anything itself. Right before it starts a fresh gateway it
records the estimated outage window in the planned-restart marker (``.restart_pending.json``, the
file the gateway's in-band restart writes in ``run_shutdown._stop_persist_exit_state``). The
recovered gateway's existing boot/reconnect pass
(``run_notifications._replay_pending_planned_restart_notification``) then sends the home-channel
notice through its live adapters, with that pass's per-target delivery ledger (no duplicates across
replays or processes) and its ``gateway_restart_notification`` opt-out.

Only facts are stated: the start is the last check that saw the gateway healthy (unknown when there
was none), the detection time is the first failed check, and the end is stamped by the gateway the
first time it is back to send the notice. Nothing but these timestamps is recorded or sent.

Stdlib only: the watchdog imports this without the gateway's dependency set.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

PLANNED_RESTART_MARKER_FILENAME = ".restart_pending.json"
OUTAGE_KEY = "watchdog_outage"
NOTICE_KEY = "gateway.startup.watchdog_recovered"
NOTICE_KEY_START_UNKNOWN = "gateway.startup.watchdog_recovered_start_unknown"


def _number(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _read_text(path: Path) -> Optional[str]:
    try:
        # surrogateescape: a foreign or torn file still reads, and a retraction writes its bytes back.
        return path.read_text(encoding="utf-8", errors="surrogateescape")
    except FileNotFoundError:
        return None


def _parse_marker(text: Optional[str]) -> dict[str, Any]:
    try:
        data = json.loads((text or "{}").lstrip("﻿"))
    except ValueError:
        return {}  # a torn or foreign file: the outage record replaces it
    return data if isinstance(data, dict) else {}


def _write_marker(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", errors="surrogateescape", newline="") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def record_outage(
    home: Path, *, last_healthy_at: Optional[float], detected_at: float,
) -> Optional[tuple[Optional[str], str]]:
    """Merge the outage window into ``home``'s planned-restart marker; raises on I/O errors (the
    caller decides that a lost notice must not stop a restart).

    An outage still waiting for its notice keeps its earlier start (one notice for one long outage).
    A new outage resets ``delivered_targets``: homes that already got an earlier "online" notice
    still owe this one. Other fields of a pending planned restart are kept.

    Returns the receipt ``retract_outage`` needs — the marker text before and after this write —
    or None when nothing was written."""
    path = Path(home) / PLANNED_RESTART_MARKER_FILENAME
    previous = _read_text(path)
    data = _parse_marker(previous)
    pending = data.get(OUTAGE_KEY)
    if isinstance(pending, dict) and "recovered_at" not in pending and _number(pending.get("detected_at")):
        return None
    data[OUTAGE_KEY] = {"last_healthy_at": _number(last_healthy_at), "detected_at": float(detected_at)}
    data.pop("delivered_targets", None)
    written = json.dumps(data)
    _write_marker(path, written)
    return previous, written


def retract_outage(home: Path, receipt: Optional[tuple[Optional[str], str]]) -> None:
    """Undo ``record_outage`` when the automatic start it announced did not happen: put the marker
    back exactly as it was (or remove it when there was none). Only while it still holds this
    write — a marker someone rewrote since is theirs and is left alone. Raises on I/O errors."""
    if receipt is None:
        return
    previous, written = receipt
    path = Path(home) / PLANNED_RESTART_MARKER_FILENAME
    if _read_text(path) != written:
        return
    if previous is None:
        path.unlink(missing_ok=True)
    else:
        _write_marker(path, previous)


def carry_pending_outage(path: Path, marker: dict[str, Any]) -> dict[str, Any]:
    """``marker`` plus the outage record still pending in the marker file at ``path``: a planned
    restart that rewrites the file must not drop it, the people who wrote during that outage are
    still owed its resend notice. An unreadable file carries nothing."""
    try:
        previous = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        previous = None
    if isinstance(previous, dict) and isinstance(previous.get(OUTAGE_KEY), dict):
        marker[OUTAGE_KEY] = previous[OUTAGE_KEY]
    return marker


def stamp_recovered_at(data: dict[str, Any], now: float) -> bool:
    """Stamp the recovery time the first time the gateway is back to send; True when it stamped."""
    record = data.get(OUTAGE_KEY)
    if not isinstance(record, dict) or _number(record.get("recovered_at")) is not None:
        return False
    record["recovered_at"] = float(now)
    return True


def format_time(ts: float) -> str:
    """Host-local wall time with its UTC offset, e.g. ``2026-11-02 10:04 +0900``; UTC when the host
    cannot convert it (Windows refuses local conversion of some timestamps)."""
    moment = datetime.fromtimestamp(ts, UTC)
    try:
        moment = moment.astimezone()
    except (OSError, OverflowError, ValueError):
        pass
    return moment.strftime("%Y-%m-%d %H:%M %z")


def outage_notice(data: dict[str, Any]) -> Optional[tuple[str, dict[str, str]]]:
    """``(i18n key, format kwargs)`` for a recovered outage, or None (no record, not yet recovered,
    or inconsistent times): the caller then sends the plain "online" notice, never a guessed window."""
    record = data.get(OUTAGE_KEY)
    if not isinstance(record, dict):
        return None
    detected, recovered = _number(record.get("detected_at")), _number(record.get("recovered_at"))
    if detected is None or recovered is None or recovered < detected:
        return None
    start = _number(record.get("last_healthy_at"))
    if start is None or start > detected:
        return NOTICE_KEY_START_UNKNOWN, {"detected": format_time(detected), "end": format_time(recovered)}
    return NOTICE_KEY, {"start": format_time(start), "end": format_time(recovered)}
