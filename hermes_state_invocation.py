"""Strict invocation-binding evidence on the native session row (``model_config.invocation_binding``).

The record is bound once per session (identical re-binds are idempotent), counted per request, and never
rewritten to look better: an unsettled ``prepared`` request stays visible as in-flight, and a violation is
appended, not erased. ``agent.invocation_binding`` owns what is written; this mixin owns the transactions.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from agent.invocation_binding import RECORD_KEY, InvocationBindingError
from hermes_state_common import _sql_json_extract
from hermes_state_sessions import _parse_model_config

_MAX_VIOLATIONS = 20
_LINEAGE_CAP = 64
_SETTLED_STATES = {"responded": "responded_count", "failed": "failed_count"}


class SessionInvocationBindingMixin:
    """Write-once binding, request ordinals and violations for one invocation's native session."""

    def _invocation_txn(self, session_id: str, fn: Callable[[Any, dict[str, Any]], Any]) -> Any:
        """Read ``model_config``, let *fn* mutate it, write it back, all in one write transaction."""
        def _do(conn):
            row = conn.execute("SELECT model_config FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if row is None:
                raise InvocationBindingError("native_session_missing")
            config = _parse_model_config(row[0])
            result = fn(conn, config)
            conn.execute("UPDATE sessions SET model_config = ? WHERE id = ?", (json.dumps(config), session_id))
            return result
        return self._execute_write(_do)

    @staticmethod
    def _bound_invocation_record(config: dict[str, Any], invocation_id: str) -> dict[str, Any]:
        record = config.get(RECORD_KEY)
        if not isinstance(record, dict):
            raise InvocationBindingError("native_binding_missing")
        if record.get("invocation_id") != invocation_id:
            raise InvocationBindingError("binding_conflict")
        return record

    @staticmethod
    def _invocation_reused_elsewhere(conn, session_id: str, invocation_id: str) -> bool:
        """Another session already carries this invocation id and is not one of this session's compression
        ancestors (a continuation legitimately inherits the binding of the session it continues)."""
        rows = conn.execute(
            f"SELECT id FROM sessions WHERE id != ? AND {_sql_json_extract('model_config', f'$.{RECORD_KEY}.invocation_id')} = ?",
            (session_id, invocation_id),
        ).fetchall()
        if not rows:
            return False
        ancestors: set[str] = set()
        current = session_id
        for _ in range(_LINEAGE_CAP):
            parent = conn.execute("SELECT parent_session_id FROM sessions WHERE id = ?", (current,)).fetchone()
            if not parent or not parent[0] or parent[0] in ancestors:
                break
            ancestors.add(parent[0])
            current = parent[0]
        return any(row[0] not in ancestors for row in rows)

    def bind_invocation(self, session_id: str, record: dict[str, Any]) -> str:
        """Write *record* once. ``"bound"`` on first write, ``"existing"`` for an identical re-bind (restart,
        resume); a different binding on the session or a reused invocation id raises."""
        def _bind(conn, config):
            existing = config.get(RECORD_KEY)
            if isinstance(existing, dict):
                if existing.get("invocation_id") == record["invocation_id"] and existing.get("digest") == record["digest"]:
                    return "existing"
                raise InvocationBindingError("binding_conflict")
            if self._invocation_reused_elsewhere(conn, session_id, record["invocation_id"]):
                raise InvocationBindingError("invocation_reused")
            config[RECORD_KEY] = record
            return "bound"
        return self._invocation_txn(session_id, _bind)

    def record_invocation_request(self, session_id: str, invocation_id: str, snapshot: dict[str, Any]) -> int:
        """Append the next request as ``prepared`` (before any byte is sent); returns its ordinal."""
        def _request(conn, config):
            record = self._bound_invocation_record(config, invocation_id)
            ordinal = int(record.get("request_count") or 0) + 1
            record["request_count"] = ordinal
            record["last_request"] = {**snapshot, "ordinal": ordinal, "state": "prepared"}
            return ordinal
        return self._invocation_txn(session_id, _request)

    def settle_invocation_request(
        self, session_id: str, invocation_id: str, ordinal: int, state: str, error_type: Optional[str] = None,
    ) -> None:
        """Count a request as ``responded`` or ``failed``; ``last_request`` only changes if it is still this one."""
        counter = _SETTLED_STATES.get(state)
        if counter is None:
            raise InvocationBindingError("native_record_failed")

        def _settle(conn, config):
            record = self._bound_invocation_record(config, invocation_id)
            record[counter] = int(record.get(counter) or 0) + 1
            last = record.get("last_request")
            if isinstance(last, dict) and last.get("ordinal") == ordinal and last.get("state") == "prepared":
                last["state"] = state
                if error_type:
                    last["error_type"] = error_type
        self._invocation_txn(session_id, _settle)

    def record_invocation_violation(self, session_id: str, invocation_id: str, code: str) -> None:
        def _violation(conn, config):
            record = self._bound_invocation_record(config, invocation_id)
            record["violations"] = (list(record.get("violations") or []) + [code])[-_MAX_VIOLATIONS:]
        self._invocation_txn(session_id, _violation)

    def get_invocation_binding(self, session_id: str) -> Optional[dict[str, Any]]:
        record = self.get_session_model_config_value(session_id, RECORD_KEY)
        return record if isinstance(record, dict) else None
