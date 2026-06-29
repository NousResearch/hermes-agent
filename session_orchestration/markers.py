"""
Marker protocol module for session_orchestration (T001).

Architecture
------------
Each managed agent session writes structured JSON-line events ("markers")
to a dedicated file (``HERMES_MARKER_FILE``).  The watcher reads those lines
each tick and uses them as the *authoritative* signal for session lifecycle
state, falling back to pane-scraping only when no recent markers exist.

This file is the cross-repo contract shared between hermes-agent (reader)
and z-harness / omp (writers).  Changes to the schema must be backward-
compatible and version-gated.

Schema
------
Every line is a JSON object with the envelope fields::

    {
        "v": 1,           # schema version (int)
        "ts": "<iso8601>",# UTC timestamp when the marker was written
        "kind": "<kind>", # one of the MARKER_* constants below
        "task": "<str>",  # opaque task / session identifier
        "payload": {...}  # kind-specific payload (see PAYLOAD SHAPES)
    }

Payload shapes
--------------
status          {"phase": str, "detail": str}
heartbeat       {"note": str | null}
needs_input     {"question": str, "options": list[str] | null, "context": str | null}
handoff_continue {"handoff_text": str}
handoff_decision {"question": str, "handoff_text": str}
done            {"summary": str, "artifacts": list[str] | null}

Malformed lines (invalid JSON, missing envelope fields, wrong types) are
**skipped silently** — they must never raise.

Helpers
-------
``append_marker(path, kind, payload)``
    Atomically append one marker line to *path*.
``read_markers_since(path, offset) -> (markers, new_offset)``
    Return only lines written after *offset* bytes into the file.
``marker_kind_to_lifecycle(kind) -> SessionLifecycle | None``
    Map a marker kind to the canonical ``SessionLifecycle`` state.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from session_orchestration.types import SessionLifecycle

# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

MARKER_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Marker kind constants
# ---------------------------------------------------------------------------

MARKER_STATUS = "status"
MARKER_HEARTBEAT = "heartbeat"
MARKER_NEEDS_INPUT = "needs_input"
MARKER_HANDOFF_CONTINUE = "handoff_continue"
MARKER_HANDOFF_DECISION = "handoff_decision"
MARKER_DONE = "done"

# Complete vocabulary in a set for validation convenience.
MARKER_KINDS: frozenset[str] = frozenset(
    {
        MARKER_STATUS,
        MARKER_HEARTBEAT,
        MARKER_NEEDS_INPUT,
        MARKER_HANDOFF_CONTINUE,
        MARKER_HANDOFF_DECISION,
        MARKER_DONE,
    }
)

# ---------------------------------------------------------------------------
# kind → SessionLifecycle mapping
# ---------------------------------------------------------------------------

_KIND_TO_LIFECYCLE: dict[str, SessionLifecycle] = {
    MARKER_STATUS: SessionLifecycle.RUNNING,
    MARKER_HEARTBEAT: SessionLifecycle.RUNNING,
    MARKER_NEEDS_INPUT: SessionLifecycle.WAITING_USER,
    MARKER_HANDOFF_CONTINUE: SessionLifecycle.PAUSED_HANDOFF,
    MARKER_HANDOFF_DECISION: SessionLifecycle.PAUSED_HANDOFF,
    MARKER_DONE: SessionLifecycle.DONE,
}


def marker_kind_to_lifecycle(kind: str) -> Optional[SessionLifecycle]:
    """Map a marker *kind* string to the corresponding ``SessionLifecycle``.

    Returns ``None`` for unknown kinds so callers can decide whether to
    ignore or log the anomaly rather than raising.
    """
    return _KIND_TO_LIFECYCLE.get(kind)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _build_envelope(kind: str, task: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "v": MARKER_SCHEMA_VERSION,
        "ts": _utc_now_iso(),
        "kind": kind,
        "task": task,
        "payload": payload,
    }


def _is_valid_envelope(obj: Any) -> bool:
    """Return True iff *obj* has all required envelope fields with correct types."""
    if not isinstance(obj, dict):
        return False
    if not isinstance(obj.get("v"), int):
        return False
    if not isinstance(obj.get("ts"), str):
        return False
    if not isinstance(obj.get("kind"), str):
        return False
    if not isinstance(obj.get("task"), str):
        return False
    if "payload" not in obj or not isinstance(obj["payload"], dict):
        return False
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def append_marker(path: "str | Path", kind: str, payload: dict[str, Any], task: str = "") -> None:
    """Append a single marker line to *path*.

    The parent directory is created if it does not exist.  The write is a
    single ``os.write`` on an O_APPEND fd so that concurrent writers from
    different processes do not interleave partial lines (POSIX guarantees
    atomicity for writes <= PIPE_BUF; a compact JSON line easily fits).

    Parameters
    ----------
    path:
        Path to the ``.jsonl`` marker file.
    kind:
        One of the ``MARKER_*`` constants (e.g. ``MARKER_DONE``).
    payload:
        Kind-specific payload dict — caller is responsible for correctness.
    task:
        Opaque task/session identifier (defaults to empty string).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    envelope = _build_envelope(kind, task, payload)
    line = json.dumps(envelope, separators=(",", ":")) + "\n"
    encoded = line.encode()

    fd = os.open(str(path), os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
    try:
        os.write(fd, encoded)
    finally:
        os.close(fd)


def read_markers_since(
    path: "str | Path",
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Read marker lines from *path* starting at byte *offset*.

    Malformed lines (invalid JSON, missing envelope fields) are silently
    skipped.  The returned offset is the new byte position after the last
    line read; pass it back on the next call to get only new lines.

    Parameters
    ----------
    path:
        Path to the ``.jsonl`` marker file.
    offset:
        Byte offset to start reading from (0 means start of file).

    Returns
    -------
    tuple[list[dict], int]
        ``(markers, new_offset)`` where *markers* is a list of valid
        envelope dicts and *new_offset* is the byte position after the
        last byte consumed.
    """
    path = Path(path)
    if not path.exists():
        return [], offset

    markers: list[dict[str, Any]] = []
    new_offset = offset

    with path.open("rb") as fh:
        fh.seek(offset)
        for raw_line in fh:
            new_offset += len(raw_line)
            line = raw_line.decode(errors="replace").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # malformed — skip silently
            if not _is_valid_envelope(obj):
                continue  # missing fields — skip silently
            markers.append(obj)

    return markers, new_offset
