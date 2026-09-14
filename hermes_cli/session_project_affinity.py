"""Read-only, profile-scoped Session Project affinity snapshots for Plugins and Core."""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import time
from types import MappingProxyType
from typing import Any, Mapping, Optional


_AFFINITY_COLUMNS = (
    "project_id",
    "project_root",
    "project_affinity_generation",
    "project_context_hash",
)


def _norm(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(os.path.expanduser(str(path or "")))))


def _query_project(projects_db_path: Path, project_id: str):
    """Read Project authority with the same bounded WAL IOERR retry class as state.db."""
    uri = "file:" + projects_db_path.resolve().as_posix() + "?mode=ro"
    for attempt in range(4):
        conn = None
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=1.0)
            conn.row_factory = sqlite3.Row
            return conn.execute(
                "SELECT id, primary_path, archived FROM projects WHERE id = ?",
                (project_id,),
            ).fetchone()
        except sqlite3.OperationalError as exc:
            if attempt >= 3 or "disk i/o error" not in str(exc).lower():
                raise
            time.sleep(0.05)
        finally:
            if conn is not None:
                conn.close()


def _base(status: str, session_id: str) -> dict[str, Any]:
    return {
        "status": status,
        "session_id": str(session_id or ""),
        "project_id": "",
        "project_root": "",
        "project_generation": 0,
        "project_context_hash": "",
    }


def validate_project_affinity_row(
    row: Mapping[str, Any] | None,
    *,
    session_id: str,
    projects_db_path: Optional[Path],
) -> dict[str, Any]:
    """Normalize one session row and validate that its Project owner still exists."""
    if row is None:
        return _base("missing_session", session_id)
    values = tuple(row.get(key) for key in _AFFINITY_COLUMNS)
    identity = tuple(str(value or "").strip() for value in values[:2] + values[3:])
    generation = int(values[2] or 0)
    if not any(identity):
        return _base("unbound", session_id)
    if not all(identity):
        result = _base("partial", session_id)
        result.update(
            project_id=identity[0], project_root=identity[1],
            project_generation=generation, project_context_hash=identity[2],
        )
        return result

    result = _base("bound", session_id)
    result.update(
        project_id=identity[0], project_root=identity[1],
        project_generation=generation, project_context_hash=identity[2],
    )
    if projects_db_path is None:
        result["status"] = "unavailable"
        return result
    if not projects_db_path.is_file():
        result["status"] = "unavailable"
        return result
    try:
        project = _query_project(projects_db_path, identity[0])
    except (OSError, sqlite3.Error):
        result["status"] = "unavailable"
        return result
    if project is None or bool(project["archived"]):
        result["status"] = "stale"
        return result
    primary_path = str(project["primary_path"] or "").strip()
    if not primary_path or _norm(primary_path) != _norm(identity[1]):
        result["status"] = "stale"
    return result


def read_session_project_affinity(home_path: Path, session_id: str) -> Mapping[str, Any]:
    """Read one affinity snapshot through Core's WAL-aware read-only state path."""
    sid = str(session_id or "").strip()
    if not sid:
        return MappingProxyType(_base("missing_session", sid))
    home = Path(home_path)
    state_path = home / "state.db"
    if not state_path.is_file():
        return MappingProxyType(_base("missing_session", sid))
    try:
        from hermes_state import SessionDB

        db = SessionDB(db_path=state_path, read_only=True)
        try:
            mapping = db.get_session(sid)
        finally:
            db.close()
    except (OSError, sqlite3.Error):
        return MappingProxyType(_base("unavailable", sid))
    return MappingProxyType(validate_project_affinity_row(
        mapping,
        session_id=sid,
        projects_db_path=home / "projects.db",
    ))
