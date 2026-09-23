"""Whether the workforce can actually reach its model, from the runtime's own records.

Evidence, not a probe. The control plane holds no model credential and makes no model call
of its own: a status page that spent tokens on every refresh, or that needed a permission
the runtime role deliberately lacks, would be the wrong trade. What the runtime already
writes is enough to answer the question an operator asks first — "is the model working?":

* the most recent **successful** call: a session in some profile's ``state.db`` that
  recorded model usage with output tokens;
* the most recent **failed** call: the runtime's ``API call failed (attempt n/m) …
  summary=…`` line (agent/turn_recovery.py::log_api_error_attempt) in an ``errors.log``.

Whichever is newer decides the verdict. Neither means nothing has been tried, which is
reported as ``unknown`` rather than guessed.

Everything is opened read-only: ``state.db`` through a ``mode=ro`` URI, so a missing file
is reported absent instead of being created, and log files through a bounded tail read.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from nova.runtime.model_errors import classify_model_error

#: How much of each log is read. The newest failure is what matters, and these files grow.
TAIL_BYTES = 256 * 1024

_FAILURE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[,.]\d+ .*?API call failed \(attempt [^)]*\)"
    r"(?P<context>.*?) summary=(?P<summary>.*)$"
)
_FIELD = re.compile(r"\b(provider|model|base_url)=(\S+)")


def _tail_lines(path: Path) -> list[str]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - TAIL_BYTES))
            data = handle.read()
    except OSError:
        return []
    return data.decode("utf-8", errors="replace").splitlines()


def _log_files(home: Path) -> Iterable[Path]:
    yield home / "logs" / "errors.log"
    profiles = home / "profiles"
    if profiles.is_dir():
        for profile in sorted(profiles.iterdir()):
            yield profile / "logs" / "errors.log"


def last_failure(home: Path) -> Optional[dict[str, Any]]:
    """The newest failed model call across the gateway and every profile, or None."""
    newest: Optional[dict[str, Any]] = None
    for path in _log_files(home):
        for line in _tail_lines(path):
            match = _FAILURE.match(line)
            if not match:
                continue
            try:
                at = datetime.strptime(match["ts"], "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=timezone.utc
                ).timestamp()
            except ValueError:
                continue
            if newest is not None and at < newest["at"]:
                continue
            fields = dict(_FIELD.findall(match["context"]))
            profile = path.parent.parent.name if path.parent.parent != home else "default"
            newest = {
                "at": at,
                "profile": profile,
                "provider": fields.get("provider", ""),
                "model": fields.get("model", ""),
                "error": classify_model_error(match["summary"]).to_dict(),
            }
    return newest


def last_success(home: Path) -> Optional[dict[str, Any]]:
    """The newest session that got a model answer, across every profile, or None."""
    newest: Optional[dict[str, Any]] = None
    profiles = home / "profiles"
    candidates = [home] + (sorted(p for p in profiles.iterdir() if p.is_dir()) if profiles.is_dir() else [])
    for directory in candidates:
        db = directory / "state.db"
        if not db.is_file():
            continue
        try:
            connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2)
        except sqlite3.Error:
            continue
        try:
            row = connection.execute(
                "SELECT s.started_at, u.model FROM session_model_usage u "
                "JOIN sessions s ON s.id = u.session_id "
                "WHERE u.api_call_count > 0 AND u.output_tokens > 0 "
                "ORDER BY s.started_at DESC LIMIT 1"
            ).fetchone()
        except sqlite3.Error:
            row = None
        finally:
            connection.close()
        if row and (newest is None or row[0] > newest["at"]):
            newest = {
                "at": float(row[0]),
                "profile": directory.name if directory != home else "default",
                "model": row[1] or "",
            }
    return newest


def model_status(home: Path) -> dict[str, Any]:
    """``state`` is ``working``, ``failing`` or ``unknown``, with the evidence for it."""
    success = last_success(home)
    failure = last_failure(home)
    if failure and (not success or failure["at"] >= success["at"]):
        state = "failing"
    elif success:
        state = "working"
    else:
        state = "unknown"
    return {"state": state, "last_success": success, "last_failure": failure}
