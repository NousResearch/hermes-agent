"""Process-wide SQLite tracing for Kanban task-table corruption forensics.

Enabled by default in the gateway; set HERMES_KANBAN_SQL_TRACE=0 to disable.
The gateway wraps sqlite3.connect process-wide so user hooks/plugins that open
kanban.db directly are visible too. Only statements that WRITE the exact
`tasks` table are logged; SQL literal values are redacted before logging.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TRACE_ENV = "HERMES_KANBAN_SQL_TRACE"
_FALSEY = frozenset({"0", "false", "no", "off"})
_INSTALL_LOCK = threading.Lock()
_INSTALLED = False
_ORIGINAL_CONNECT = sqlite3.connect

_TASK_WRITE_RE = re.compile(
    r"""(?is)\b(?P<op>
        INSERT\s+(?:OR\s+\w+\s+)?INTO
      | REPLACE\s+(?:OR\s+\w+\s+)?INTO
      | UPDATE(?:\s+OR\s+\w+)?
      | DELETE\s+FROM
      | ALTER\s+TABLE
      | DROP\s+TABLE(?:\s+IF\s+EXISTS)?
      | CREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?
    )\s+(?:(?:["`\[]?\w+["`\]]?)\s*\.\s*)?["`\[]?tasks["`\]]?\b""",
    re.VERBOSE,
)
_BLOB_LITERAL_RE = re.compile(r"(?is)\b[xX]'(?:''|[^'])*'")
_STRING_LITERAL_RE = re.compile(r"(?s)'(?:''|[^'])*'")
_DOUBLE_QUOTED_TOKEN_RE = re.compile(r'(?s)"(?:""|[^"])*"')
_HEX_NUMBER_LITERAL_RE = re.compile(r"(?i)(?<![\w.])0x[0-9a-f]+(?![\w.])")
_NUMBER_LITERAL_RE = re.compile(
    r"(?<![\w.])-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?![\w.])"
)
_SQL_COMMENT_RE = re.compile(r"(?s)/\*.*?\*/|--[^\r\n]*")
_WS_RE = re.compile(r"\s+")


def _enabled() -> bool:
    """Default-on; only an explicit false value disables forensic tracing."""
    return os.environ.get(_TRACE_ENV, "").strip().lower() not in _FALSEY


def _redacted_sql_shape(statement: str, *, limit: int = 1200) -> str:
    """Return statement structure without runtime literal values."""
    shaped = _BLOB_LITERAL_RE.sub("?", statement)
    shaped = _STRING_LITERAL_RE.sub("?", shaped)
    shaped = _DOUBLE_QUOTED_TOKEN_RE.sub("?", shaped)
    shaped = _HEX_NUMBER_LITERAL_RE.sub("?", shaped)
    shaped = _NUMBER_LITERAL_RE.sub("?", shaped)
    shaped = _SQL_COMMENT_RE.sub("?", shaped)
    shaped = _WS_RE.sub(" ", shaped).strip()
    return shaped[:limit] + ("…" if len(shaped) > limit else "")


def _stack_summary() -> str:
    """Compact stack with no source-line text (which could contain literals)."""
    frames = traceback.extract_stack(limit=28)[:-1]
    own = str(Path(__file__).resolve())
    parts = [
        f"{frame.filename}:{frame.lineno}:{frame.name}"
        for frame in frames
        if str(Path(frame.filename).resolve()) != own
    ]
    return " <- ".join(parts[-14:])


def _runtime_context() -> dict[str, Any]:
    context = {
        "profile_env": os.environ.get("HERMES_PROFILE", ""),
        "home_env": os.environ.get("HERMES_HOME", ""),
        "board_env": os.environ.get("HERMES_KANBAN_BOARD", ""),
        "db_env": os.environ.get("HERMES_KANBAN_DB", ""),
        "task_env_set": bool(os.environ.get("HERMES_KANBAN_TASK")),
        "run_env": os.environ.get("HERMES_KANBAN_RUN_ID", ""),
        "home_override": "",
        "secret_scope_home": "",
    }
    try:
        from hermes_constants import get_hermes_home_override
        context["home_override"] = get_hermes_home_override() or ""
    except Exception:
        pass
    try:
        from agent.secret_scope import current_secret_scope_home
        context["secret_scope_home"] = current_secret_scope_home() or ""
    except Exception:
        pass
    return context


def _connection_label(conn: sqlite3.Connection, database: Any) -> str:
    try:
        row = conn.execute("PRAGMA database_list").fetchone()
        if row and len(row) >= 3 and row[2]:
            return str(Path(row[2]).resolve())
    except Exception:
        pass
    return str(database)


def _install_connection_trace(conn: sqlite3.Connection, database: Any) -> None:
    db_label = _connection_label(conn, database)

    def _trace(statement: str) -> None:
        match = _TASK_WRITE_RE.search(statement or "")
        if match is None:
            return
        try:
            ctx = _runtime_context()
            logger.warning(
                "[kanban-sql-trace] tasks write op=%s db=%s pid=%s thread=%s/%s "
                "in_txn=%s profile_env=%r home_env=%r home_override=%r "
                "secret_scope_home=%r board_env=%r db_env=%r task_env_set=%s run_env=%r "
                "sql=%s stack=%s",
                _WS_RE.sub(" ", match.group("op")).upper(),
                db_label,
                os.getpid(),
                threading.current_thread().name,
                threading.get_ident(),
                bool(conn.in_transaction),
                ctx["profile_env"],
                ctx["home_env"],
                ctx["home_override"],
                ctx["secret_scope_home"],
                ctx["board_env"],
                ctx["db_env"],
                ctx["task_env_set"],
                ctx["run_env"],
                _redacted_sql_shape(statement),
                _stack_summary(),
            )
        except Exception:
            # Diagnostics must never be able to break or roll back the write.
            logger.exception("[kanban-sql-trace] failed to record tasks write for db=%s", db_label)

    conn.set_trace_callback(_trace)


def _traced_connect(database, *args, **kwargs):
    conn = _ORIGINAL_CONNECT(database, *args, **kwargs)
    _install_connection_trace(conn, database)
    return conn


def install_if_enabled() -> bool:
    """Install process-wide sqlite3.connect tracing once; return whether active."""
    global _INSTALLED
    if not _enabled():
        return False
    with _INSTALL_LOCK:
        if _INSTALLED:
            return True
        sqlite3.connect = _traced_connect
        sqlite3.dbapi2.connect = _traced_connect
        _INSTALLED = True
    logger.warning(
        "[kanban-sql-trace] enabled process-wide SQLite tasks-write tracing "
        "(default-on; set HERMES_KANBAN_SQL_TRACE=0 to disable; SQL literal values are redacted)"
    )
    return True
