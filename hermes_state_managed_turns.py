"""Read-only admission receipts for opt-in TUI managed turns.

The transcript insert and receipt live in one SessionDB transaction in
hermes_state_messages.append_message; this reader never reserves or runs a turn.
"""
from __future__ import annotations


class SessionManagedTurnsMixin:
    def get_managed_turn(self, session_id: str, managed_turn_key: str) -> dict[str, int] | None:
        """Read only receipts in the verified compression lineage of this session."""
        lineage = self._resume_lineage_ids(session_id)  # type: ignore[attr-defined]  # SessionDB mixin
        placeholders = ",".join("?" for _ in lineage)
        with self._read_ctx() as conn:
            row = conn.execute(
                f"SELECT user_row_id FROM managed_turn_submissions "
                f"WHERE idempotency_key = ? AND session_id IN ({placeholders})",
                (managed_turn_key, *lineage)).fetchone()
        return {"user_row_id": int(row[0])} if row is not None else None
