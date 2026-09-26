"""Durable SQLite contract for future cross-machine session handoff.

This is intentionally an additive, no-route foundation.  Existing runtime turns
continue to use ``session_turn_leases``.  A future backend-neutral consumer may
select this protocol only after it can carry the opaque owner receipt through
all turn and external-effect boundaries.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Literal


TurnState = Literal["running", "indeterminate", "settled"]


@dataclass(frozen=True)
class RuntimeOwner:
    """Stable cross-host owner identity; never infer it from a PID alone."""

    installation_id: str
    host: str
    process_generation: str

    def validate(self) -> bool:
        return bool(self.installation_id and self.host and self.process_generation)


@dataclass(frozen=True)
class RuntimeOwnershipReceipt:
    """Opaque durable claim receipt. Every mutation is fenced by ``fence``."""

    namespace: str
    session_id: str
    owner: RuntimeOwner
    fence: int
    expires_at: float


class SessionRuntimeOwnershipMixin:
    """SQLite implementation of the bounded owner/turn state machine.

    SQLite's write transaction and local clock are compatibility semantics only.
    A PostgreSQL adapter must use one server-time CAS transaction for each
    transition; it must not copy this clock source or process-local capability
    behavior into a shared runtime.
    """

    _RUNTIME_LOCAL_CAPABILITIES = frozenset({"browser", "computer_use", "approval_wait"})

    def _execute_write(self, *args: Any, **kwargs: Any) -> Any: ...

    @staticmethod
    def _runtime_namespace(namespace: str | None) -> str:
        return (namespace or "").strip()

    @staticmethod
    def _runtime_owner_columns(owner: RuntimeOwner) -> tuple[str, str, str]:
        if not owner.validate():
            raise ValueError("runtime owner needs installation_id, host, and process_generation")
        return owner.installation_id, owner.host, owner.process_generation

    def acquire_session_runtime_ownership(
        self, session_id: str, owner: RuntimeOwner, *, ttl_seconds: float = 300.0,
        namespace: str | None = None,
    ) -> RuntimeOwnershipReceipt | None:
        """Acquire or renew an owner lease; takeover increments the monotonic fence."""
        if not session_id:
            return None
        installation_id, host, process_generation = self._runtime_owner_columns(owner)
        namespace = self._runtime_namespace(namespace)
        now = time.time()
        expires_at = now + max(0.1, float(ttl_seconds))

        def _do(conn):
            row = conn.execute(
                "SELECT installation_id, host, process_generation, fence, expires_at "
                "FROM session_runtime_owners WHERE namespace = ? AND session_id = ?",
                (namespace, session_id),
            ).fetchone()
            if row is None:
                fence = 1
                conn.execute(
                    "INSERT INTO session_runtime_owners "
                    "(namespace, session_id, installation_id, host, process_generation, fence, expires_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (namespace, session_id, installation_id, host, process_generation, fence, expires_at, now),
                )
                return fence
            same_owner = (row["installation_id"], row["host"], row["process_generation"]) == (
                installation_id, host, process_generation)
            if same_owner:
                conn.execute(
                    "UPDATE session_runtime_owners SET expires_at = ?, updated_at = ? "
                    "WHERE namespace = ? AND session_id = ? AND fence = ?",
                    (expires_at, now, namespace, session_id, row["fence"]),
                )
                return int(row["fence"])
            if float(row["expires_at"]) > now:
                return None
            fence = int(row["fence"]) + 1
            conn.execute(
                "UPDATE session_runtime_owners SET installation_id = ?, host = ?, process_generation = ?, "
                "fence = ?, expires_at = ?, updated_at = ? WHERE namespace = ? AND session_id = ? AND fence = ?",
                (installation_id, host, process_generation, fence, expires_at, now, namespace, session_id, row["fence"]),
            )
            conn.execute(
                "UPDATE session_runtime_turns SET state = 'indeterminate', updated_at = ? "
                "WHERE namespace = ? AND session_id = ? AND state = 'running' AND owner_fence < ?",
                (now, namespace, session_id, fence),
            )
            return fence

        fence = self._execute_write(_do)
        return None if fence is None else RuntimeOwnershipReceipt(namespace, session_id, owner, int(fence), expires_at)

    def renew_session_runtime_ownership(self, receipt: RuntimeOwnershipReceipt, *, ttl_seconds: float = 300.0) -> RuntimeOwnershipReceipt | None:
        """Renew only the exact current owner/fence receipt."""
        installation_id, host, process_generation = self._runtime_owner_columns(receipt.owner)
        now = time.time()
        expires_at = now + max(0.1, float(ttl_seconds))
        def _do(conn):
            updated = conn.execute(
                "UPDATE session_runtime_owners SET expires_at = ?, updated_at = ? "
                "WHERE namespace = ? AND session_id = ? AND installation_id = ? AND host = ? "
                "AND process_generation = ? AND fence = ? AND expires_at > ?",
                (expires_at, now, receipt.namespace, receipt.session_id, installation_id, host,
                 process_generation, receipt.fence, now),
            ).rowcount
            return bool(updated)
        return RuntimeOwnershipReceipt(receipt.namespace, receipt.session_id, receipt.owner, receipt.fence, expires_at) if self._execute_write(_do) else None

    def release_session_runtime_ownership(self, receipt: RuntimeOwnershipReceipt) -> bool:
        installation_id, host, process_generation = self._runtime_owner_columns(receipt.owner)
        def _do(conn):
            return conn.execute(
                "DELETE FROM session_runtime_owners WHERE namespace = ? AND session_id = ? "
                "AND installation_id = ? AND host = ? AND process_generation = ? AND fence = ?",
                (receipt.namespace, receipt.session_id, installation_id, host, process_generation, receipt.fence),
            ).rowcount > 0
        return bool(self._execute_write(_do))

    def begin_session_runtime_turn(self, receipt: RuntimeOwnershipReceipt, turn_id: str) -> bool:
        """Record a turn before effects. Duplicate same-fence begin is idempotent."""
        if not turn_id:
            return False
        now = time.time()
        def _do(conn):
            owner = conn.execute(
                "SELECT fence, expires_at FROM session_runtime_owners WHERE namespace = ? AND session_id = ?",
                (receipt.namespace, receipt.session_id),
            ).fetchone()
            if owner is None or int(owner["fence"]) != receipt.fence or float(owner["expires_at"]) <= now:
                return False
            row = conn.execute(
                "SELECT state, owner_fence FROM session_runtime_turns WHERE namespace = ? AND session_id = ? AND turn_id = ?",
                (receipt.namespace, receipt.session_id, turn_id),
            ).fetchone()
            if row is not None:
                return row["state"] == "running" and int(row["owner_fence"]) == receipt.fence
            conn.execute(
                "INSERT INTO session_runtime_turns (namespace, session_id, turn_id, state, owner_fence, receipt_json, created_at, updated_at) "
                "VALUES (?, ?, ?, 'running', ?, NULL, ?, ?)",
                (receipt.namespace, receipt.session_id, turn_id, receipt.fence, now, now),
            )
            return True
        return bool(self._execute_write(_do))

    def resolve_session_runtime_turn(self, receipt: RuntimeOwnershipReceipt, turn_id: str, *, state: TurnState, receipt_data: dict | None = None) -> bool:
        """Explicitly settle or mark indeterminate; never claims external completion without a receipt."""
        if state not in {"settled", "indeterminate"} or not turn_id:
            return False
        if state == "settled" and receipt_data is None:
            raise ValueError("settled turn requires a verified receipt")
        now = time.time()
        serialized = None if receipt_data is None else json.dumps(receipt_data, sort_keys=True)
        def _do(conn):
            owner = conn.execute(
                "SELECT fence, expires_at FROM session_runtime_owners WHERE namespace = ? AND session_id = ?",
                (receipt.namespace, receipt.session_id),
            ).fetchone()
            if owner is None or int(owner["fence"]) != receipt.fence or float(owner["expires_at"]) <= now:
                return False
            return conn.execute(
                "UPDATE session_runtime_turns SET state = ?, receipt_json = ?, updated_at = ? "
                "WHERE namespace = ? AND session_id = ? AND turn_id = ? "
                "AND state IN ('running', 'indeterminate')",
                (state, serialized, now, receipt.namespace, receipt.session_id, turn_id),
            ).rowcount > 0
        return bool(self._execute_write(_do))

    @classmethod
    def supports_session_runtime_handoff_capability(cls, capability: str) -> bool:
        """Process-local capability state deliberately never crosses this handoff boundary."""
        return capability not in cls._RUNTIME_LOCAL_CAPABILITIES
