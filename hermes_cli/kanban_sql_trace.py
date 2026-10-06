"""Process-wide SQLite tracing for Kanban task-identity forensics (#119003).

Enabled by default in the gateway; set HERMES_KANBAN_SQL_TRACE=0 to disable.
The gateway wraps sqlite3.connect process-wide so user hooks/plugins that open
kanban.db directly are visible too. Each connection gets an SQLite authorizer:
the parser calls it while compiling a statement with the action, table, updated
column and the trigger the write comes from. It never sees statement text or
bound values, so there is nothing to redact or regex-match, and a statement
costs a few comparisons once per compile rather than a callback per execution.

Only writes that can create, remove or rename a task are logged: INSERT and
ALTER TABLE at INFO; DELETE, DROP TABLE, an UPDATE of ``tasks.id`` and any
``tasks`` write a trigger issues at WARNING (Hermes's own triggers never write).
Status, claim and heartbeat updates are never logged.

Known blind spot: SQLite reports ``INSERT OR REPLACE`` to the authorizer as a
single ``SQLITE_INSERT`` -- the implicit delete of the conflicting row emits no
``SQLITE_DELETE`` -- and the conflict clause is not visible here, so a REPLACE
is traced as a routine INSERT. The schema's ``kanban_guard_task_id_replace``
trigger is what refuses an insert over an existing task id; this trace only
records who issued it.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
import threading
import traceback
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_TRACE_ENV = "HERMES_KANBAN_SQL_TRACE"
_FALSEY = frozenset({"0", "false", "no", "off"})
_INSTALL_LOCK = threading.Lock()
_INSTALLED = False
_ORIGINAL_CONNECT = sqlite3.connect
_TASKS = "tasks"
_ROUTINE_OPS = frozenset({"INSERT", "ALTER TABLE"})


def _enabled() -> bool:
    """Default-on; only an explicit false value disables forensic tracing."""
    return os.environ.get(_TRACE_ENV, "").strip().lower() not in _FALSEY


def _identity_write(action: int, arg1: Optional[str], arg2: Optional[str]) -> Optional[str]:
    """Op name when an authorizer call is a ``tasks`` write that can create,
    remove or rename a task; None for everything else (reads, other tables,
    updates of any other column)."""
    table = (arg2 if action == sqlite3.SQLITE_ALTER_TABLE else arg1) or ""
    if table.lower() != _TASKS:
        return None
    if action == sqlite3.SQLITE_INSERT:
        return "INSERT"
    if action == sqlite3.SQLITE_DELETE:
        return "DELETE"
    if action == sqlite3.SQLITE_UPDATE and (arg2 or "").lower() == "id":
        return "UPDATE id"
    if action == sqlite3.SQLITE_DROP_TABLE:
        return "DROP TABLE"
    if action == sqlite3.SQLITE_ALTER_TABLE:
        return "ALTER TABLE"
    return None


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


def _log_identity_write(conn: sqlite3.Connection, db_label: str, op: str, trigger: Optional[str]) -> None:
    ctx = _runtime_context()
    level = logging.INFO if op in _ROUTINE_OPS and not trigger else logging.WARNING
    logger.log(
        level,
        "[kanban-sql-trace] tasks write op=%s via_trigger=%r db=%s pid=%s thread=%s/%s "
        "in_txn=%s profile_env=%r home_env=%r home_override=%r "
        "secret_scope_home=%r board_env=%r db_env=%r task_env_set=%s run_env=%r stack=%s",
        op,
        trigger or "",
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
        _stack_summary(),
    )


def _install_connection_trace(conn: sqlite3.Connection, database: Any) -> None:
    db_label = _connection_label(conn, database)

    def _authorize(action: int, arg1: Optional[str], arg2: Optional[str], _db: Optional[str],
                   trigger: Optional[str]) -> int:
        try:
            op = _identity_write(action, arg1, arg2)
            if op is not None:
                _log_identity_write(conn, db_label, op, trigger)
        except Exception:
            # A raising authorizer makes SQLite deny the statement; diagnostics
            # must never be able to break the write.
            with contextlib.suppress(Exception):
                logger.debug("[kanban-sql-trace] failed to record tasks write for db=%s", db_label, exc_info=True)
        return sqlite3.SQLITE_OK

    conn.set_authorizer(_authorize)


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
    logger.info(
        "[kanban-sql-trace] enabled process-wide SQLite tasks-identity tracing "
        "(default-on; set HERMES_KANBAN_SQL_TRACE=0 to disable; SQL values are never seen)"
    )
    return True
