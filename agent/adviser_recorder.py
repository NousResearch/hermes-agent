"""Append-only adviser transcript, one JSONL per adviser per session.

Port of oh-my-pi's ``advisor/transcript-recorder.ts``. Adviser context is not
part of the primary session, so without this a failed review or a later
investigation has no record of what the reviewer saw or spent. The file lives
under ``<HERMES_HOME>/adviser/<session_id>/<slug>.jsonl`` (never in the session
DB): it is telemetry, not conversation state, and a re-prime discards the
reviewer's context while the record survives.

Replay window: a failed review is retried with the *same* user deltas, so a user
row already written by the failed attempt is skipped instead of duplicated.
Assistant/tool rows are always written — that work was billed.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

from agent.adviser_delta import fingerprint_message
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

TRANSCRIPT_STEM = "__adviser"
DEFAULT_FILENAME = "__adviser.jsonl"


def adviser_transcript_filename(slug: str) -> str:
    return f"{TRANSCRIPT_STEM}.{slug}.jsonl" if slug else DEFAULT_FILENAME


def is_adviser_transcript_name(name: str) -> bool:
    return name == DEFAULT_FILENAME or (name.startswith(f"{TRANSCRIPT_STEM}.") and name.endswith(".jsonl"))


class AdviserRecorder:
    """Thread-safe JSONL writer with a user-row replay window."""

    def __init__(self, *, session_id: str, slug: str, cwd: str = "", home: Optional[str] = None) -> None:
        self.session_id = str(session_id or "")
        self.slug = str(slug or "")
        self.cwd = str(cwd or "")
        self._home = Path(home) if home else None
        self._lock = threading.Lock()
        self._window: list[Optional[str]] = []
        self._cursor = 0
        self._header_written = False

    @property
    def path(self) -> Optional[Path]:
        if not self.session_id:
            return None
        home = self._home or get_hermes_home()
        return Path(home) / "adviser" / self.session_id / adviser_transcript_filename(self.slug)

    def begin_review(self) -> None:
        with self._lock:
            self._cursor = 0

    def commit_review(self) -> None:
        with self._lock:
            self._window = []
            self._cursor = 0

    def abandon_review(self) -> None:
        with self._lock:
            self._window = []
            self._cursor = 0

    def record(self, message: dict[str, Any]) -> None:
        """Persist one adviser-visible message, skipping a replayed user batch row."""
        if not isinstance(message, dict):
            return
        role = str(message.get("role") or "")
        if role not in ("user", "assistant", "tool"):
            return
        with self._lock:
            payload = dict(message)
            if role == "user":
                payload.setdefault("synthetic", True)
                fingerprint = fingerprint_message(payload)
                if fingerprint is not None:
                    if self._cursor < len(self._window) and self._window[self._cursor] == fingerprint:
                        self._cursor += 1
                        return
                    if self._cursor < len(self._window):
                        self._window = self._window[: self._cursor]
                    self._window.append(fingerprint)
                    self._cursor = len(self._window)
            self._write_locked({"type": "message", "timestamp": time.time(), "message": payload})

    def _write_locked(self, record: dict[str, Any]) -> None:
        path = self.path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                if not self._header_written:
                    handle.write(json.dumps({
                        "type": "session", "version": 1, "id": self.session_id,
                        "timestamp": time.time(), "cwd": self.cwd, "adviser": self.slug,
                    }, ensure_ascii=False) + "\n")
                    self._header_written = True
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            logger.debug("adviser recorder write failed (%s): %s", path, exc)


def load_adviser_costs(directory: Optional[str]) -> dict[str, float]:
    """Sum ``message.usage.cost`` per adviser slug from a session's transcript directory."""
    costs: dict[str, float] = {}
    if not directory:
        return costs
    try:
        entries = [item for item in Path(directory).iterdir() if item.is_file()]
    except OSError:
        return costs
    for entry in entries:
        if not is_adviser_transcript_name(entry.name):
            continue
        slug = "" if entry.name == DEFAULT_FILENAME else entry.name[len(TRANSCRIPT_STEM) + 1:-len(".jsonl")]
        total = 0.0
        try:
            with entry.open("r", encoding="utf-8-sig", errors="replace") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if record.get("type") != "message":
                        continue
                    message = record.get("message") or {}
                    if message.get("role") != "assistant":
                        continue
                    usage = message.get("usage") or {}
                    cost = usage.get("cost") if isinstance(usage, dict) else None
                    value = cost.get("total") if isinstance(cost, dict) else cost
                    if isinstance(value, (int, float)) and value == value:
                        total += float(value)
        except OSError:
            continue
        if total > 0:
            costs[slug] = total
    return costs


__all__ = [
    "DEFAULT_FILENAME",
    "TRANSCRIPT_STEM",
    "AdviserRecorder",
    "adviser_transcript_filename",
    "is_adviser_transcript_name",
    "load_adviser_costs",
]
