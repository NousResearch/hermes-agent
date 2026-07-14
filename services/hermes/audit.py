"""Audit log for Hermes RPE (Rate of Perceived Exertion) command invocations.

One JSONL entry per invocation, written to $HERMES_HOME/logs/rpe-audit.log.
Bearer tokens and credentials are never written to this log.
Write failures are logged at WARNING but never raised — a log failure must
not block the user-facing command response.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import threading
from pathlib import Path

_log = logging.getLogger(__name__)
_write_lock = threading.Lock()


def _resolve_log_path() -> Path:
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(home) / "logs" / "rpe-audit.log"


def _resolve_journal_log_path() -> Path:
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(home) / "logs" / "journal-approve-audit.log"


def log_rpe_invocation(
    *,
    user_id: str,
    rpe: int,
    has_notes: bool,
    target_date: str,
    http_outcome: str,
) -> None:
    """Append one RPE invocation entry to the audit log.

    Fields logged: user_id, rpe, has_notes, target_date, http_outcome, ts.
    The bearer token is never logged.
    """
    entry = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "user_id": user_id,
        "rpe": rpe,
        "has_notes": has_notes,
        "target_date": target_date,
        "http_outcome": http_outcome,
    }
    line = json.dumps(entry, separators=(",", ":")) + "\n"
    path = _resolve_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _write_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception as exc:
        _log.warning("RPE audit log write failed: %s", exc)


def log_journal_approve_attempt(
    *,
    todo_id: str,
    actor: str,
    outcome: str,
) -> None:
    """Append one journal-approve attempt to the audit log.

    Fields logged: ts, todo_id, actor, outcome.
    Write failures are logged at WARNING but never raised.
    """
    entry = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "todo_id": todo_id,
        "actor": actor,
        "outcome": outcome,
    }
    line = json.dumps(entry, separators=(",", ":")) + "\n"
    path = _resolve_journal_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _write_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception as exc:
        _log.warning("Journal approve audit log write failed: %s", exc)
