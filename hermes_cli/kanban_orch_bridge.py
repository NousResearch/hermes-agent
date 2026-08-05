"""ORCH V4 Bridge — dual connection coordinator for sidecar architecture.

Source: 拆卡機制四大缺陷-詳細實作計畫.md §15.0 Sidecar Architecture Decision.

The bridge is the ONLY cross-DB write path. It enforces:
- Native kanban.db: read-only (mode=ro URI); any write attempt raises BridgeError
- Sidecar orch_v4.db: readwrite; all orch_* mutations go here
- Soft FK: parent_task_id written to sidecar only after native RO confirms existence
- Board/tenant scope lock after first bind
- Real capability grants on sidecar (fail-closed UDF)
- Durable orch_requests binding for parent tasks
- Atomic create-only sidecar init (O_CREAT|O_EXCL|O_NOFOLLOW)

Hard rules:
- native_writable=False is the ONLY allowed mode in this slice
- No ATTACH DATABASE for writes
- No native schema mutation (ALTER, CREATE TRIGGER on native, etc.)
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass
from typing import Any

from hermes_cli.kanban_orch_api import BoundParent, bootstrap_board_only_request, ensure_board_identity
from hermes_cli.kanban_orch_capability import install_fail_closed_udf, unbind_context


class BridgeError(ValueError):
    """Bridge protocol violation."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class NativeTaskRef:
    """Soft FK reference to a native task (read-only)."""

    task_id: str
    title: str
    status: str
    board: str
    tenant: str = ""


def _create_exclusive_file(path: str, mode: int = 0o600) -> None:
    """Atomically create a new empty file; fail if path exists or is a symlink."""
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, mode)
    except FileExistsError as exc:
        raise BridgeError("sidecar_exists") from None
    except OSError as exc:
        # symlink / race / permission
        raise BridgeError("sidecar_create_failed") from None
    try:
        os.close(fd)
    except OSError:
        pass


class OrchBridge:
    """Dual-connection bridge between native kanban.db and sidecar orch_v4.db."""

    def __init__(
        self,
        native_path: str,
        sidecar_path: str,
        *,
        native_writable: bool = False,
        board_instance_id: str | None = None,
        tenant_scope: str = "",
    ):
        if native_writable:
            raise BridgeError("native_write_forbidden_by_sidecar_decision")
        if not os.path.exists(native_path):
            raise BridgeError("native_db_not_found")
        if not os.path.exists(sidecar_path):
            raise BridgeError("sidecar_db_not_found")

        self._native = sqlite3.connect(f"file:{native_path}?mode=ro", uri=True)
        self._native.row_factory = sqlite3.Row

        self._sidecar = sqlite3.connect(sidecar_path)
        self._sidecar.row_factory = sqlite3.Row
        self._sidecar.execute("PRAGMA foreign_keys=ON")
        fk = self._sidecar.execute("PRAGMA foreign_keys").fetchone()[0]
        if int(fk) != 1:
            self._native.close()
            self._sidecar.close()
            raise BridgeError("sidecar_foreign_keys_off")
        install_fail_closed_udf(self._sidecar)

        self._native_path = native_path
        self._sidecar_path = sidecar_path
        self._locked_board = board_instance_id
        self._locked_tenant = "" if tenant_scope is None else str(tenant_scope)
        self._task_columns = {
            row[1] for row in self._native.execute("PRAGMA table_info(tasks)").fetchall()
        }

    def read_native_task(self, task_id: str) -> NativeTaskRef | None:
        cols = ["id", "title", "status"]
        if "tenant" in self._task_columns:
            cols.append("tenant")
        if "board" in self._task_columns:
            cols.append("board")
        sql = f"SELECT {', '.join(cols)} FROM tasks WHERE id = ?"
        row = self._native.execute(sql, (task_id,)).fetchone()
        if row is None:
            return None
        tenant = row["tenant"] if "tenant" in row.keys() and row["tenant"] is not None else ""
        board = row["board"] if "board" in row.keys() and row["board"] is not None else "default"
        return NativeTaskRef(
            task_id=row["id"],
            title=row["title"] or "",
            status=row["status"] or "",
            board=str(board),
            tenant=str(tenant),
        )

    def assert_task_exists(self, task_id: str, *, tenant_scope: str = "") -> NativeTaskRef:
        ref = self.read_native_task(task_id)
        if ref is None:
            raise BridgeError("soft_fk_violation:task_not_found")
        wanted_tenant = "" if tenant_scope is None else str(tenant_scope)
        if "tenant" in self._task_columns and ref.tenant != wanted_tenant:
            raise BridgeError("soft_fk_violation:tenant_mismatch")
        return ref

    def read_native_board_identity(self, board_key: str) -> dict[str, Any] | None:
        return None

    def ensure_board_mirror(
        self,
        board_instance_id: str,
        canonical_board_key: str,
    ) -> None:
        self._assert_scope(board_instance_id, self._locked_tenant if self._locked_board else "")
        ensure_board_identity(
            self._sidecar,
            board_instance_id=board_instance_id,
            canonical_board_key=canonical_board_key,
        )
        self._sidecar.commit()

    def _assert_scope(self, board_instance_id: str, tenant_scope: str) -> None:
        tenant = "" if tenant_scope is None else str(tenant_scope)
        if self._locked_board is None:
            self._locked_board = board_instance_id
            self._locked_tenant = tenant
            return
        if board_instance_id != self._locked_board or tenant != self._locked_tenant:
            raise BridgeError("board_tenant_scope_mismatch")

    def bind_parent_task(
        self,
        board_instance_id: str,
        tenant_scope: str,
        orch_id: str,
        parent_task_id: str,
    ) -> BoundParent:
        self._assert_scope(board_instance_id, tenant_scope)
        self.assert_task_exists(parent_task_id, tenant_scope=tenant_scope)
        try:
            bound = bootstrap_board_only_request(
                self._sidecar,
                board_instance_id=board_instance_id,
                tenant_scope=tenant_scope,
                parent_task_id=parent_task_id,
                orch_id=orch_id,
                title=f"bound:{parent_task_id}",
            )
        except Exception as exc:
            code = getattr(exc, "code", None)
            if code:
                raise BridgeError(f"bind_failed:{code}") from None
            raise BridgeError("bind_failed") from None
        return bound

    def native_write_forbidden(self) -> None:
        return None

    def native_sha256(self) -> str:
        h = hashlib.sha256()
        with open(self._native_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def close(self) -> None:
        try:
            unbind_context(self._sidecar)
        finally:
            self._native.close()
            self._sidecar.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def init_sidecar_db(sidecar_path: str) -> None:
    """Create a fresh sidecar DB with schema applied (atomic O_EXCL create)."""
    if os.path.lexists(sidecar_path):
        raise BridgeError("sidecar_exists")
    _create_exclusive_file(sidecar_path)
    conn = sqlite3.connect(sidecar_path)
    try:
        from hermes_cli.kanban_orch_schema_sidecar import apply_sidecar_schema

        apply_sidecar_schema(conn, test_open_capability=False)
    finally:
        conn.close()


__all__ = [
    "BridgeError",
    "NativeTaskRef",
    "OrchBridge",
    "init_sidecar_db",
    "_create_exclusive_file",
]
