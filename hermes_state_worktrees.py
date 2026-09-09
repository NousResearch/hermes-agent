"""Durable conversation worktree identities and lifecycle transitions.

Restored from the carried worktree implementation in e828efe6ec; SessionDB
supplies its existing guarded read and transactional write paths.
"""
from __future__ import annotations

from dataclasses import dataclass
import sqlite3
import time
from typing import Optional, Tuple


class ConversationWorktreeConflict(RuntimeError):
    """A root session attempted to change its claimed Git identity."""


@dataclass(frozen=True)
class ConversationWorktreeRecord:
    """Durable immutable Git identity plus the lifecycle state of one root."""

    root_session_id: str
    worktree_path: str
    branch: str
    base_commit: str
    repo_common_dir: str
    state: str
    failure_phase: Optional[str]
    failure_message: Optional[str]
    created_at: float
    updated_at: float


class SessionWorktreesMixin:
    @staticmethod
    def _conversation_worktree_record(row: sqlite3.Row) -> ConversationWorktreeRecord:
        return ConversationWorktreeRecord(
            root_session_id=row["root_session_id"],
            worktree_path=row["worktree_path"],
            branch=row["branch"],
            base_commit=row["base_commit"],
            repo_common_dir=row["repo_common_dir"],
            state=row["state"],
            failure_phase=row["failure_phase"],
            failure_message=row["failure_message"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @classmethod
    def _conversation_worktree_record_on_conn(
        cls, conn: sqlite3.Connection, root_session_id: str
    ) -> Optional[ConversationWorktreeRecord]:
        row = conn.execute(
            "SELECT root_session_id, worktree_path, branch, base_commit, "
            "repo_common_dir, state, failure_phase, failure_message, "
            "created_at, updated_at "
            "FROM conversation_worktree_bindings WHERE root_session_id = ?",
            (root_session_id,),
        ).fetchone()
        return cls._conversation_worktree_record(row) if row is not None else None

    def get_conversation_worktree(
        self, root_session_id: str
    ) -> Optional[ConversationWorktreeRecord]:
        """Return the durable binding for a root session, if it was claimed."""
        with self._read_ctx() as conn:
            return self._conversation_worktree_record_on_conn(conn, root_session_id)

    def claim_conversation_worktree(
        self,
        *,
        root_session_id: str,
        worktree_path: str,
        branch: str,
        base_commit: str,
        repo_common_dir: str,
    ) -> ConversationWorktreeRecord:
        """Claim an immutable Git identity, or return the identical claim.

        Claims are made before session persistence, so the binding table has no
        foreign key to ``sessions``. A retry may observe the same identity,
        but it may never replace any identity field of an existing root.
        """
        identity = (worktree_path, branch, base_commit, repo_common_dir)

        def _do(conn: sqlite3.Connection) -> ConversationWorktreeRecord:
            existing = self._conversation_worktree_record_on_conn(
                conn, root_session_id
            )
            if existing is not None:
                existing_identity = (
                    existing.worktree_path,
                    existing.branch,
                    existing.base_commit,
                    existing.repo_common_dir,
                )
                if existing_identity != identity:
                    raise ConversationWorktreeConflict(
                        "conversation worktree identity already claimed for "
                        f"root session {root_session_id!r}"
                    )
                return existing

            now = time.time()
            conn.execute(
                "INSERT INTO conversation_worktree_bindings ("
                "root_session_id, worktree_path, branch, base_commit, "
                "repo_common_dir, state, created_at, updated_at"
                ") VALUES (?, ?, ?, ?, ?, 'creating', ?, ?)",
                (root_session_id, *identity, now, now),
            )
            record = self._conversation_worktree_record_on_conn(conn, root_session_id)
            if record is None:  # pragma: no cover - INSERT is in this transaction.
                raise RuntimeError("conversation worktree claim was not persisted")
            return record

        return self._execute_write(_do)

    def _set_conversation_worktree_state(
        self,
        root_session_id: str,
        *,
        state: str,
        allowed_current_states: Tuple[str, ...],
        failure_phase: Optional[str] = None,
        failure_message: Optional[str] = None,
    ) -> ConversationWorktreeRecord:
        def _do(conn: sqlite3.Connection) -> ConversationWorktreeRecord:
            existing = self._conversation_worktree_record_on_conn(
                conn, root_session_id
            )
            if existing is None:
                raise RuntimeError(
                    "cannot transition an unclaimed conversation worktree "
                    f"for root session {root_session_id!r}"
                )
            if existing.state == state:
                return existing
            if existing.state not in allowed_current_states:
                raise ConversationWorktreeConflict(
                    "cannot transition conversation worktree for root session "
                    f"{root_session_id!r} from {existing.state!r} to {state!r}"
                )

            conn.execute(
                "UPDATE conversation_worktree_bindings "
                "SET state = ?, failure_phase = ?, failure_message = ?, updated_at = ? "
                "WHERE root_session_id = ?",
                (state, failure_phase, failure_message, time.time(), root_session_id),
            )
            record = self._conversation_worktree_record_on_conn(conn, root_session_id)
            if record is None:  # pragma: no cover - guarded by prior SELECT.
                raise RuntimeError("conversation worktree transition was not persisted")
            return record

        return self._execute_write(_do)

    def mark_conversation_worktree_ready(
        self, root_session_id: str
    ) -> ConversationWorktreeRecord:
        """Record that a claimed worktree was verified and bootstrapped."""
        return self._set_conversation_worktree_state(
            root_session_id,
            state="ready",
            allowed_current_states=("creating", "creation_failed"),
        )

    def mark_conversation_worktree_failed(
        self,
        root_session_id: str,
        *,
        failure_phase: str,
        failure_message: str,
    ) -> ConversationWorktreeRecord:
        """Retain a failed creation claim with its safe diagnostic summary."""
        return self._set_conversation_worktree_state(
            root_session_id,
            state="creation_failed",
            allowed_current_states=("creating", "creation_failed"),
            failure_phase=failure_phase,
            failure_message=failure_message,
        )

    def mark_conversation_worktree_removed(
        self, root_session_id: str
    ) -> ConversationWorktreeRecord:
        """Record explicit cleanup without discarding the immutable identity."""
        return self._set_conversation_worktree_state(
            root_session_id,
            state="removed",
            allowed_current_states=("creating", "ready", "creation_failed", "retained"),
        )
