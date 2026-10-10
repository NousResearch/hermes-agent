"""Generic transactional pre-delete identity capture for plugin-owned handoff.

No plugin callback or external cleanup policy runs in the deletion transaction.
"""

import json
import uuid
from pathlib import Path

from hermes_state_common import _id_chunks, _placeholders

_USER_DELETE_SURFACES = {"user_rest": "dashboard_rest", "user_rpc": "session_rpc"}
_IDENTITY_COLUMNS = "id, source, session_key, chat_id, chat_type, thread_id, parent_session_id"


class SessionDeletionMixin:
    def _prepare_session_deletion(self, conn, session_ids, deletion_origin):
        """Capture after guards/fences and before mutation, inside the delete transaction."""
        surface = _USER_DELETE_SURFACES.get(deletion_origin)
        if surface is None:
            return
        from hermes_state_sessions import _collect_delegate_child_ids

        session_ids = [*session_ids, *_collect_delegate_child_ids(conn, session_ids)]
        identities = [dict(row) for chunk in _id_chunks(sorted(set(session_ids))) for row in conn.execute(
            f"SELECT {_IDENTITY_COLUMNS} FROM sessions WHERE id IN ({_placeholders(chunk)}) ORDER BY id", chunk,
        )]
        if not identities:
            return
        store = Path(self.db_path).resolve()
        deletion = {
            "operation_id": uuid.uuid4().hex,
            "reason": "explicit_user",
            "surface": surface,
            "store_id": str(store),
            "profile_home": str(store.parent),
            "identities": sorted(identities, key=lambda row: row["id"]),
        }
        # Persist immutable core evidence, not plugin results; a failed commit rolls this back too.
        conn.execute(
            "INSERT INTO session_deletion_receipts (operation_id, deletion_json) VALUES (?, ?)",
            (deletion["operation_id"], json.dumps(deletion)),
        )

    def forget_session_deletion_receipts(self, *, through_sequence: int) -> int:
        """Administrative reclamation after ALL consumers acknowledge; never a per-plugin ack."""
        if through_sequence < 0:
            raise ValueError("through_sequence must be nonnegative")
        return self._execute_write(lambda conn: conn.execute(
            "DELETE FROM session_deletion_receipts WHERE sequence <= ?", (through_sequence,),
        ).rowcount)

    def get_session_deletion_receipt(self, operation_id: str):
        """Committed minimal context, or None; usable from another connection/process."""
        row = self._read_one(
            "SELECT deletion_json FROM session_deletion_receipts WHERE operation_id = ?", (operation_id,),
        )
        return json.loads(row[0]) if row else None

    def list_session_deletion_receipts(self, *, after_sequence: int = 0, limit: int = 100):
        """Replay committed explicit-user receipts in this store; plugin owns its durable cursor."""
        if after_sequence < 0 or not 1 <= limit <= 1000:
            raise ValueError("after_sequence must be nonnegative and limit must be between 1 and 1000")
        with self._read_ctx() as conn:
            rows = conn.execute(
                "SELECT sequence, deletion_json FROM session_deletion_receipts "
                "WHERE sequence > ? ORDER BY sequence LIMIT ?", (after_sequence, limit),
            ).fetchall()
        return [{"sequence": row["sequence"], "deletion": json.loads(row["deletion_json"])} for row in rows]
