"""ORCH V4 runtime-private capability gate.

Contract: orch_capability_ok(kind, board, tenant, object_id, revision, epoch, target_key)
must be fail-closed unless the current connection holds an explicit grant.
Never log or serialize the raw token.
"""

from __future__ import annotations

import secrets
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Any


class CapabilityError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class CapabilityGrant:
    kind: str
    board: str = "*"
    tenant: str = "*"
    object_id: str = "*"
    revision: int | str = "*"
    epoch: int | str = "*"
    target_key: str = "*"


@dataclass
class CapabilityContext:
    """In-process grant set bound to one SQLite connection."""

    token: str = field(default_factory=lambda: secrets.token_hex(16))
    grants: set[CapabilityGrant] = field(default_factory=set)

    def grant(self, grant: CapabilityGrant) -> None:
        if type(grant.kind) is not str or not grant.kind:
            raise CapabilityError("invalid_grant_kind")
        self.grants.add(grant)

    def grant_open_for_tests(self) -> None:
        """Explicit test-only open gate. Production code must not call this."""
        self.grants.add(CapabilityGrant(kind="*"))

    def clear(self) -> None:
        self.grants.clear()

    def check(
        self,
        kind: Any,
        board: Any,
        tenant: Any,
        object_id: Any,
        revision: Any,
        epoch: Any,
        target_key: Any,
    ) -> int:
        if type(kind) is not str or not kind:
            return 0
        board_s = "" if board is None else str(board)
        tenant_s = "" if tenant is None else str(tenant)
        object_s = "" if object_id is None else str(object_id)
        target_s = "" if target_key is None else str(target_key)
        for g in self.grants:
            if g.kind not in {"*", kind}:
                continue
            if g.board not in {"*", board_s}:
                continue
            if g.tenant not in {"*", tenant_s}:
                continue
            if g.object_id not in {"*", object_s}:
                continue
            if g.revision != "*" and str(g.revision) != str(revision):
                continue
            if g.epoch != "*" and str(g.epoch) != str(epoch):
                continue
            if g.target_key not in {"*", target_s}:
                continue
            return 1
        return 0


_lock = threading.RLock()
_conn_contexts: dict[int, CapabilityContext] = {}


def bind_context(conn: sqlite3.Connection, ctx: CapabilityContext | None = None) -> CapabilityContext:
    """Bind a capability context to conn and install the SQLite UDF."""
    if not isinstance(conn, sqlite3.Connection):
        raise CapabilityError("invalid_connection")
    context = ctx or CapabilityContext()
    conn_id = id(conn)

    def _udf(kind, board, tenant, object_id, revision, epoch, target_key):
        with _lock:
            active = _conn_contexts.get(conn_id)
        if active is None:
            return 0
        return active.check(kind, board, tenant, object_id, revision, epoch, target_key)

    with _lock:
        _conn_contexts[conn_id] = context
    # Re-register every time so callers cannot keep a stale always-1 UDF.
    conn.create_function("orch_capability_ok", 7, _udf)
    return context


def get_context(conn: sqlite3.Connection) -> CapabilityContext | None:
    with _lock:
        return _conn_contexts.get(id(conn))


def unbind_context(conn: sqlite3.Connection) -> None:
    with _lock:
        _conn_contexts.pop(id(conn), None)


def install_fail_closed_udf(conn: sqlite3.Connection) -> CapabilityContext:
    """Default production posture: UDF present, no grants => deny."""
    return bind_context(conn, CapabilityContext())


def install_test_open_udf(conn: sqlite3.Connection) -> CapabilityContext:
    """Test helper: explicit open grant. Never use on live connectors."""
    ctx = CapabilityContext()
    ctx.grant_open_for_tests()
    return bind_context(conn, ctx)


__all__ = [
    "CapabilityError",
    "CapabilityGrant",
    "CapabilityContext",
    "bind_context",
    "get_context",
    "unbind_context",
    "install_fail_closed_udf",
    "install_test_open_udf",
]
