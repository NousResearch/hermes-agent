"""Lineage worker exclusion is one indexed probe, also on a state.db created before the index."""
import sqlite3

from hermes_state import SessionDB


def test_lineage_worker_probe_is_indexed_on_new_and_upgraded_stores(tmp_path):
    from hermes_state_runtime_workers import worker_states
    path = tmp_path / 'state.db'
    with SessionDB(path) as db:
        db.create_session('root', source='cli')
    raw = sqlite3.connect(path)
    raw.execute('DROP INDEX IF EXISTS idx_worker_executions_session_status')  # an older store
    raw.commit()
    raw.close()
    with SessionDB(path) as db:  # reopening migrates it; doing it twice must be a no-op
        pass
    with SessionDB(path) as db:
        for parent, child in (('root', 'c1'), ('c1', 'c2')):
            db.create_session(child, 'cli', parent_session_id=parent)
            db.end_session(parent, 'compression')
        statements = []
        with db._read_ctx() as conn:
            conn.set_trace_callback(statements.append)
            assert worker_states(conn, 'root') == set()
            conn.set_trace_callback(None)
            probes = [s for s in statements if 'FROM worker_executions' in s]
            assert len(probes) == 1, f'one probe per lineage row: {probes}'
            plan = ' '.join(str(r[-1]) for r in conn.execute('EXPLAIN QUERY PLAN ' + probes[0]))
        assert 'idx_worker_executions_session_status' in plan, plan
