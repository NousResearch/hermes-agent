"""Guarded fresh session births, distinct from idempotent ensure/upsert."""

from hermes_state_lineage import session_lineage_guard


class FreshSessionMixin:
    def _assert_fresh_session_conn(self, conn, session_id):
        from hermes_state_lineage import FreshSessionCollision
        from hermes_state_mutation_retirement import RETIRED_PREFIX
        if (conn.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone()
                or conn.execute("SELECT 1 FROM state_meta WHERE key=?", (RETIRED_PREFIX + session_id,)).fetchone()):
            raise FreshSessionCollision("Generated session ID already exists")

    def _assert_fresh_session_id(self, session_id):
        from hermes_state_lineage import FreshSessionCollision, SessionIdOwnedElsewhereError, SessionLineageAmbiguousError
        try:
            with session_lineage_guard(self.db_path, session_id, timeout_s=self._TRANSCRIPT_WRITE_PATIENCE_S) as snapshot:
                if snapshot is not None and snapshot.rows_by_id.get(session_id):
                    raise FreshSessionCollision("Generated session ID already exists")
                with self._read_ctx() as conn:
                    self._assert_fresh_session_conn(conn, session_id)
        except (SessionIdOwnedElsewhereError, SessionLineageAmbiguousError) as exc:
            raise FreshSessionCollision("Generated session ID already exists") from exc

    def _create_fresh_session(self, session_id, source, **kwargs):
        """Create only a genuinely new ID; hold ownership through insertion commit."""
        from hermes_state_lineage import FreshSessionCollision, SessionIdOwnedElsewhereError, SessionLineageAmbiguousError
        from hermes_state_worker_lifecycle import insert_session_row_in_transaction
        try:
            with session_lineage_guard(self.db_path, session_id, timeout_s=self._TRANSCRIPT_WRITE_PATIENCE_S) as snapshot:
                if snapshot is not None and snapshot.rows_by_id.get(session_id):
                    raise FreshSessionCollision("Generated session ID already exists")
                def insert(conn):
                    self._assert_fresh_session_conn(conn, session_id)
                    insert_session_row_in_transaction(self, conn, session_id, source, **kwargs)
                self._execute_write(insert, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S)
        except (SessionIdOwnedElsewhereError, SessionLineageAmbiguousError) as exc:
            raise FreshSessionCollision("Generated session ID already exists") from exc
        return session_id
