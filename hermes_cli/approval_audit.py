"""Audit log for ``approvals.mode`` changes (issue #84547).

Profile-aware location: ``$HERMES_HOME/logs/approvals.log``.
Format: one JSON object per line.

Mirrors ``hermes_cli/dashboard_auth/audit.py``: a deliberately minimal
dependency surface (only the leaf ``hermes_constants`` module) so it can
be imported safely from config write paths and runtime observers, and it
never raises on write failure — auditing must not break the config write
that triggered it.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import threading
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)
_write_lock = threading.Lock()

# Config path -> last approvals.mode written by THIS process. Lets the
# runtime observer (tools.approval._observe_approval_mode_transition)
# distinguish a change this process just made and already audited on the
# write path from a SILENT one (hand edit, key dropped by a
# re-serialization) that still needs detection.
_LAST_WRITTEN_MODE: dict = {}


def note_mode_written(path: str, mode: str) -> None:
    """Record that this process just persisted ``mode`` for ``path``."""
    with _write_lock:
        _LAST_WRITTEN_MODE[path] = mode


def last_written_mode(path: str):
    """Return the last mode this process wrote for ``path``, or None."""
    with _write_lock:
        return _LAST_WRITTEN_MODE.get(path)


def _resolve_log_path() -> Path:
    """``$HERMES_HOME/logs/approvals.log``."""
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "logs" / "approvals.log"


def audit_approval_mode(
    *,
    old_mode: Optional[str],
    new_mode: str,
    source: str,
    actor: Optional[str] = None,
    detail: str = "",
) -> None:
    """Append one ``approvals.mode`` change event to the audit log.

    Args:
        old_mode: mode in effect before the change (``None`` when the key
            was absent / never set).
        new_mode: mode after the change.
        source: where the change was initiated — e.g. ``cli-config-set``,
            ``tui-config-set``, ``effective-mode-observer``.
        actor: human/process identity that made the change. Defaults to
            the OS user; the runtime observer passes ``runtime``.
        detail: optional free-form context (config path, session, ...).
    """
    if actor is None:
        try:
            import getpass

            actor = getpass.getuser()
        except Exception:
            actor = "unknown"
    entry = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "event": "approvals.mode_changed",
        "actor": actor,
        "source": source,
        "old_mode": old_mode,
        "new_mode": new_mode,
    }
    if detail:
        entry["detail"] = detail
    line = json.dumps(entry, separators=(",", ":")) + "\n"
    path = _resolve_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _write_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
    except OSError as e:
        # Auditing must never break the config write that triggered it.
        _log.debug("Could not write approvals audit log: %s", e)
