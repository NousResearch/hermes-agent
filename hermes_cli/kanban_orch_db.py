"""ORCH V4 canonical DB connector.

Hard rules:
- PRAGMA foreign_keys=ON with read-back
- runtime-private capability UDF installed fail-closed by default
- BEGIN IMMEDIATE helpers for mutation transactions
- never open live fleet paths unless caller opts in explicitly
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from hermes_cli.kanban_orch_capability import (
    CapabilityContext,
    CapabilityGrant,
    bind_context,
    get_context,
    install_fail_closed_udf,
    install_test_open_udf,
    unbind_context,
)

LIVE_FORBIDDEN_PATHS = frozenset(
    {
        "/home/claw/.hermes/kanban.db",
        str(Path.home() / ".hermes" / "kanban.db"),
    }
)


class OrchDBError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _resolve(path: str | os.PathLike[str]) -> str:
    return str(Path(path).expanduser().resolve())


def assert_not_live_path(path: str | os.PathLike[str], *, allow_live: bool = False) -> str:
    resolved = _resolve(path)
    if not allow_live and resolved in {_resolve(p) for p in LIVE_FORBIDDEN_PATHS}:
        raise OrchDBError("live_path_forbidden")
    return resolved


def open_orch_db(
    path: str | os.PathLike[str],
    *,
    allow_live: bool = False,
    create: bool = False,
    test_open_capability: bool = False,
    readonly: bool = False,
) -> sqlite3.Connection:
    """Open an ORCH DB with FK ON + capability UDF.

    test_open_capability is for unit tests only.
    """
    resolved = assert_not_live_path(path, allow_live=allow_live)
    if readonly:
        if not os.path.exists(resolved):
            raise OrchDBError("db_not_found")
        uri = f"file:{resolved}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        if not create and not os.path.exists(resolved):
            raise OrchDBError("db_not_found")
        conn = sqlite3.connect(resolved)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    if int(fk) != 1:
        conn.close()
        raise OrchDBError("foreign_keys_not_on")
    if test_open_capability:
        install_test_open_udf(conn)
    else:
        install_fail_closed_udf(conn)
    from hermes_cli.kanban_orch_digest_udf import install_digest_udfs

    install_digest_udfs(conn)
    return conn


def grant(conn: sqlite3.Connection, grant_spec: CapabilityGrant) -> None:
    ctx = get_context(conn)
    if ctx is None:
        ctx = bind_context(conn)
    ctx.grant(grant_spec)


def begin_immediate(conn: sqlite3.Connection) -> None:
    conn.execute("BEGIN IMMEDIATE")


def close_orch_db(conn: sqlite3.Connection) -> None:
    try:
        unbind_context(conn)
    finally:
        conn.close()


class OrchDB:
    """Context manager wrapper around open_orch_db."""

    def __init__(self, path: str | os.PathLike[str], **kwargs: Any):
        self.path = path
        self.kwargs = kwargs
        self.conn: sqlite3.Connection | None = None

    def __enter__(self) -> sqlite3.Connection:
        self.conn = open_orch_db(self.path, **self.kwargs)
        return self.conn

    def __exit__(self, *args: Any) -> None:
        if self.conn is not None:
            close_orch_db(self.conn)
            self.conn = None


__all__ = [
    "LIVE_FORBIDDEN_PATHS",
    "OrchDBError",
    "OrchDB",
    "assert_not_live_path",
    "open_orch_db",
    "grant",
    "begin_immediate",
    "close_orch_db",
    "CapabilityGrant",
    "CapabilityContext",
]
