"""Main's ``gateway/session_managed_worker.py`` bodies (as of the pinned main), the legacy side of the backend parity checks."""



def _execution_tip(db, execution_id):
    """The session this worker execution last wrote to, as the owner's ledger records it."""
    if not execution_id:
        return None
    with db._read_ctx() as conn:
        row = conn.execute('SELECT session_id FROM worker_executions WHERE execution_id=?', (execution_id,)).fetchone()
    return row[0] if row else None
