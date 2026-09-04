"""Read-only session storage attribution report (#90719).

Answers "where is my state.db space actually going?" without touching a
byte: opens the database with ``PRAGMA query_only=1`` on a dedicated
``mode=ro`` connection (SessionDB's own machinery is never invoked on the
report path, so there is zero chance of a migration, FTS rebuild, or WAL
checkpoint firing under the diagnostician's feet).

Layers reported:

* Filesystem: state.db / -wal / -shm sizes.
* Storage engine: page size, page count, freelist pages + estimated free
  bytes.
* Physical attribution per table/index via the SQLite ``dbstat`` virtual
  table when the local build provides it, with FTS5 shadow tables
  (``messages_fts_data`` etc.) grouped under their parent virtual table.
  Without ``dbstat`` the physical section is explicitly marked
  unavailable — never estimated.
* Logical attribution: stored message payload bytes by role, and the
  largest sessions by payload, via ordinary SQL.

V1 deliberately does no remediation: no VACUUM, no prune, no FTS
migration. Commands that fix things stay responsible for fixing things.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_state import _default_db_path

_FTS_TABLE_PREFIXES = ("messages_fts",)

_KIB = 1024


def _fmt_mb(nbytes: Optional[int]) -> str:
    if nbytes is None:
        return "unavailable"
    return f"{nbytes / (_KIB * _KIB):.1f} MiB"


def _file_size(path: Path) -> Optional[int]:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _connect_query_only(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    return conn


def _database_layer(conn: sqlite3.Connection) -> Dict[str, Any]:
    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    page_count = conn.execute("PRAGMA page_count").fetchone()[0]
    freelist = conn.execute("PRAGMA freelist_count").fetchone()[0]
    return {
        "page_size": page_size,
        "page_count": page_count,
        "freelist_pages": freelist,
        "free_bytes_estimate": freelist * page_size,
    }


def _group_fts(name: str) -> str:
    for prefix in _FTS_TABLE_PREFIXES:
        if name == prefix or name.startswith(prefix + "_"):
            return f"{prefix} (FTS shadow tables)"
    return name


def _physical_layer(conn: sqlite3.Connection) -> Dict[str, Any]:
    try:
        rows = conn.execute(
            "SELECT name, sum(pgsize) FROM dbstat GROUP BY name ORDER BY 2 DESC"
        ).fetchall()
    except sqlite3.Error:
        return {"available": False, "entries": [], "total_bytes": None}

    entries: List[Dict[str, Any]] = []
    grouped: Dict[str, int] = {}
    for name, pgsize in rows:
        if name is None or pgsize is None:
            continue
        grouped[_group_fts(name)] = grouped.get(_group_fts(name), 0) + pgsize
    for name, total in sorted(grouped.items(), key=lambda kv: -kv[1]):
        entries.append({"name": name, "bytes": total})
    return {
        "available": True,
        "entries": entries,
        "total_bytes": sum(e["bytes"] for e in entries),
    }


def _logical_layer(conn: sqlite3.Connection) -> Dict[str, Any]:
    layer: Dict[str, Any] = {"by_role": [], "largest_sessions": []}
    try:
        rows = conn.execute(
            "SELECT role, count(*), sum(length(content)) "
            "FROM messages GROUP BY role ORDER BY 3 DESC"
        ).fetchall()
        layer["by_role"] = [
            {"role": r, "messages": c, "content_bytes": b or 0}
            for r, c, b in rows
        ]
    except sqlite3.Error:
        pass
    try:
        rows = conn.execute(
            "SELECT session_id, count(*), sum(length(content)) "
            "FROM messages GROUP BY session_id ORDER BY 3 DESC LIMIT 10"
        ).fetchall()
        layer["largest_sessions"] = [
            {"session_id": s, "messages": c, "content_bytes": b or 0}
            for s, c, b in rows
        ]
    except sqlite3.Error:
        pass
    return layer


def _fts_layout(conn: sqlite3.Connection) -> List[str]:
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name LIKE 'messages_fts%' "
            "ORDER BY name"
        ).fetchall()
    except sqlite3.Error:
        return []
    return [r[0] for r in rows]


def build_storage_report(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Assemble the full storage report dict. Never mutates the database."""
    path = db_path or _default_db_path()
    report: Dict[str, Any] = {
        "database_path": str(path),
        "files": {
            "state_db_bytes": _file_size(path),
            "wal_bytes": _file_size(Path(str(path) + "-wal")),
            "shm_bytes": _file_size(Path(str(path) + "-shm")),
        },
    }
    if not path.exists():
        report["error"] = "database file not found"
        return report

    conn = _connect_query_only(path)
    try:
        report["database"] = _database_layer(conn)
        report["physical"] = _physical_layer(conn)
        report["logical"] = _logical_layer(conn)
        report["fts_tables"] = _fts_layout(conn)
    finally:
        conn.close()
    return report


def format_storage_report(report: Dict[str, Any]) -> str:
    """Render the report dict as the human-readable console output."""
    if report.get("error"):
        return f"Storage report: {report['error']} ({report['database_path']})"

    files = report["files"]
    db = report["database"]
    lines = [
        f"Database: {report['database_path']}",
        f"  state.db: {_fmt_mb(files['state_db_bytes'])}",
        f"  WAL: {_fmt_mb(files['wal_bytes'])}",
        f"Pages: {db['page_count']} x {db['page_size']} B, "
        f"freelist {db['freelist_pages']} pages "
        f"(~{_fmt_mb(db['free_bytes_estimate'])} reclaimable by VACUUM)",
        "",
        "Physical attribution by table/index"
        " (dbstat):",
    ]
    phys = report["physical"]
    if not phys["available"]:
        lines.append(
            "  unavailable: this SQLite build lacks the dbstat virtual table."
        )
    else:
        for entry in phys["entries"]:
            lines.append(
                f"  {entry['name']}: {_fmt_mb(entry['bytes'])}"
            )
    lines.append("")
    lines.append("Logical message payload by role:")
    if report["logical"]["by_role"]:
        for row in report["logical"]["by_role"]:
            lines.append(
                f"  {row['role']}: {row['messages']} msgs, "
                f"{_fmt_mb(row['content_bytes'])}"
            )
    else:
        lines.append("  no messages")
    lines.append("")
    lines.append("Largest sessions by stored payload:")
    if report["logical"]["largest_sessions"]:
        for row in report["logical"]["largest_sessions"]:
            lines.append(
                f"  {row['session_id']}: {row['messages']} msgs, "
                f"{_fmt_mb(row['content_bytes'])}"
            )
    else:
        lines.append("  none")
    lines.append("")
    lines.append(f"FTS tables present: {len(report['fts_tables'])}")
    for name in report["fts_tables"]:
        lines.append(f"  {name}")
    lines.append("")
    lines.append(
        "Read-only report: no VACUUM, prune, or migration was performed."
    )
    return "\n".join(lines)
