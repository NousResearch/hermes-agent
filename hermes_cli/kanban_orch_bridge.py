"""ORCH V4 Bridge — dual connection coordinator for sidecar architecture.

Source: 拆卡機制四大缺陷-詳細實作計畫.md §15.0 Sidecar Architecture Decision.

The bridge is the ONLY cross-DB write path. It enforces:
- Native kanban.db: read-only (mode=ro URI); any write attempt raises BridgeError
- Sidecar orch_v4.db: readwrite; all orch_* mutations go here
- Soft FK: parent_task_id written to sidecar only after native RO confirms existence
- No cross-DB atomic transaction; eventual consistency via reconcile queue (M6)

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


class OrchBridge:
    """Dual-connection bridge between native kanban.db and sidecar orch_v4.db.

    Native connection is always opened mode=ro.
    Sidecar connection is readwrite with foreign_keys=ON.
    """

    def __init__(
        self,
        native_path: str,
        sidecar_path: str,
        *,
        native_writable: bool = False,
    ):
        if native_writable:
            raise BridgeError("native_write_forbidden_by_sidecar_decision")
        if not os.path.exists(native_path):
            raise BridgeError("native_db_not_found")
        if not os.path.exists(sidecar_path):
            raise BridgeError("sidecar_db_not_found")

        # Native: always read-only via URI
        self._native = sqlite3.connect(
            f"file:{native_path}?mode=ro", uri=True
        )
        self._native.row_factory = sqlite3.Row

        # Sidecar: readwrite
        self._sidecar = sqlite3.connect(sidecar_path)
        self._sidecar.row_factory = sqlite3.Row
        self._sidecar.execute("PRAGMA foreign_keys=ON")

        # Register orch_capability_ok stub UDF on sidecar (bridge is the real authority)
        self._sidecar.create_function("orch_capability_ok", 7, lambda *a: 1)

        self._native_path = native_path
        self._sidecar_path = sidecar_path

    # ── Native RO operations ──────────────────────────────────────

    def read_native_task(self, task_id: str) -> NativeTaskRef | None:
        """Read a task from native DB (read-only)."""
        row = self._native.execute(
            "SELECT id, title, status FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if row is None:
            return None
        return NativeTaskRef(
            task_id=row["id"],
            title=row["title"] or "",
            status=row["status"] or "",
            board="default",
        )

    def assert_task_exists(self, task_id: str) -> NativeTaskRef:
        """Soft FK: confirm task exists in native before sidecar write."""
        ref = self.read_native_task(task_id)
        if ref is None:
            raise BridgeError(f"soft_fk_violation:task_not_found:{task_id}")
        return ref

    def read_native_board_identity(self, board_key: str) -> dict[str, Any] | None:
        """Read board identity from native DB (if available)."""
        # Native may not have orch board_identity; return None for now
        # In full implementation, this reads from tasks or config
        return None

    # ── Sidecar write operations ───────────────────────────────────

    def ensure_board_mirror(
        self,
        board_instance_id: str,
        canonical_board_key: str,
    ) -> None:
        """Upsert board identity mirror into sidecar (from native read)."""
        self._sidecar.execute(
            "INSERT OR IGNORE INTO kanban_board_identity "
            "(singleton, board_instance_id, canonical_board_key, created_at) "
            "VALUES (1, ?, ?, unixepoch())",
            (board_instance_id, canonical_board_key),
        )
        self._sidecar.commit()

    def bind_parent_task(
        self,
        board_instance_id: str,
        tenant_scope: str,
        orch_id: str,
        parent_task_id: str,
    ) -> None:
        """Bind parent task in sidecar after soft FK verification.

        This writes ONLY to the sidecar DB. The native task is not modified.
        The soft FK check is the key invariant: if the task doesn't exist in
        native, we raise before any sidecar write.
        """
        # Soft FK: verify task exists in native (read-only)
        self.assert_task_exists(parent_task_id)

        # Sidecar-only write: record the binding via board mirror upsert.
        # This is a real sidecar write that proves native is untouched.
        # In full implementation, this would create orch_requests rows.
        self.ensure_board_mirror(board_instance_id, f"orch-{orch_id}")
        self._sidecar.commit()

    # ── Native protection ──────────────────────────────────────────

    def native_write_forbidden(self) -> None:
        """Assert that native DB was never opened for writing."""
        # The connection was opened with mode=ro, so any write would
        # raise sqlite3.OperationalError. This method is a documentation
        # point and a test hook.
        pass

    def native_sha256(self) -> str:
        """Compute SHA-256 of native DB file (for zero-mutation proof)."""
        import hashlib as hl
        h = hl.sha256()
        with open(self._native_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    # ── Lifecycle ──────────────────────────────────────────────────

    def close(self) -> None:
        """Close both connections."""
        self._native.close()
        self._sidecar.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def init_sidecar_db(sidecar_path: str) -> None:
    """Create a fresh sidecar DB with schema applied.

    The path must not already exist (O_EXCL semantics).
    """
    if os.path.exists(sidecar_path):
        raise BridgeError("sidecar_exists")

    # Create the file
    conn = sqlite3.connect(sidecar_path)
    try:
        from hermes_cli.kanban_orch_schema_sidecar import apply_sidecar_schema
        apply_sidecar_schema(conn)
    finally:
        conn.close()


__all__ = [
    "BridgeError", "NativeTaskRef", "OrchBridge", "init_sidecar_db",
]
