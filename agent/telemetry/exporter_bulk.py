"""Export telemetry (and optionally session content) to a file or stream.

Two data domains, both written to an operator-chosen destination:

  * Telemetry: the tel_* rows + events.jsonl (structural observability).
  * Content (opt-in via telemetry.trajectories): sessions + messages, with every
    content field (message body, reasoning, raw tool-call args) passed through the
    redaction pipeline (secrets always stripped; PII per content_redaction).

Formats: ndjson (default) and json. OTLP streaming export lives in otlp_exporter.py.

Content export is gated by ``redaction.content_export_enabled``.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, TextIO

from . import redaction

_TEL_TABLES = (
    "tel_runs", "tel_model_calls", "tel_tool_calls", "tel_error_events",
)


def _open(db_path: Optional[Path]) -> sqlite3.Connection:
    if db_path is None:
        from hermes_constants import get_hermes_home
        db_path = get_hermes_home() / "state.db"
    c = sqlite3.connect(str(db_path), timeout=5.0)
    c.row_factory = sqlite3.Row
    return c


def _iter_telemetry(conn: sqlite3.Connection, since_ns: Optional[int]) -> Iterator[Dict[str, Any]]:
    for table in _TEL_TABLES:
        # only tel_runs has start_ns; window the rest by run join when needed.
        if table == "tel_runs" and since_ns:
            rows = conn.execute(
                f"SELECT * FROM {table} WHERE start_ns >= ?", (int(since_ns),)
            ).fetchall()
        else:
            rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        for r in rows:
            d = dict(r)
            d["_kind"] = table
            yield d


def _iter_content(
    db_path: Optional[Path],
    *,
    config: Optional[Dict[str, Any]],
    include_content: bool,
) -> Iterator[Dict[str, Any]]:
    """Yield session records. Message bodies included only when trajectories on."""
    from hermes_state import SessionDB

    content_mode = redaction.content_mode_for(config)
    db = SessionDB(db_path=db_path) if db_path else SessionDB()
    try:
        for session in db.export_all():
            msgs = session.get("messages", []) or []
            red_msgs = [
                redaction.redact_message(
                    m, content_mode=content_mode, include_content=include_content
                )
                for m in msgs
            ]
            # Session-level metadata is structural; keep ids/model/counts, drop
            # any free-text title only when content is excluded.
            out = {
                "_kind": "session",
                "id": session.get("id"),
                "source": session.get("source"),
                "model": session.get("model"),
                "started_at": session.get("started_at"),
                "ended_at": session.get("ended_at"),
                "message_count": session.get("message_count"),
                "tool_call_count": session.get("tool_call_count"),
                "messages": red_msgs,
            }
            if include_content and session.get("title"):
                out["title"] = redaction.redact_for_export(
                    session["title"], content_mode=content_mode
                )
            yield out
    finally:
        db.close()


def export(
    out: TextIO,
    *,
    fmt: str = "ndjson",
    since_ns: Optional[int] = None,
    include_content: bool = False,
    config: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, int]:
    """Write telemetry (+ optional content) to ``out``. Returns counts.

    ``include_content`` is honored only when telemetry.trajectories is enabled in
    ``config``; otherwise content is forced off and only structural data is written.
    """
    # Trajectories gate: a flag cannot override the config setting.
    content_allowed = include_content and redaction.content_export_enabled(config)
    counts = {"telemetry": 0, "sessions": 0, "content_included": int(content_allowed)}

    conn = _open(db_path)
    records: List[Dict[str, Any]] = []
    try:
        for rec in _iter_telemetry(conn, since_ns):
            counts["telemetry"] += 1
            if fmt == "ndjson":
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                records.append(rec)
    finally:
        conn.close()

    # Content/session domain (separate connection via SessionDB).
    for rec in _iter_content(db_path, config=config, include_content=content_allowed):
        counts["sessions"] += 1
        if fmt == "ndjson":
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            records.append(rec)

    if fmt != "ndjson":
        json.dump({"records": records}, out, ensure_ascii=False, indent=2)

    return counts


__all__ = ["export"]
