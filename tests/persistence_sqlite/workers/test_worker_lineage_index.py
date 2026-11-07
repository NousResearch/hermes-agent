"""Lineage worker exclusion is one indexed probe, also on a state.db created before the index."""
import sqlite3
from contextlib import closing

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.sessions import create_session, end_session


def test_lineage_worker_probe_is_indexed_on_new_and_upgraded_stores(tmp_path):
    from persistence.sqlite.workers._runtime_workers import worker_states
    path = tmp_path / 'state.db'
    with closing(StoreHolder(path)) as holder:
        create_session(holder, 'root', 'cli')
    raw = sqlite3.connect(path)
    raw.execute('DROP INDEX IF EXISTS idx_worker_executions_session_status')  # an older store
    raw.commit()
    raw.close()
    with closing(StoreHolder(path)):  # reopening migrates it; doing it twice must be a no-op
        pass
    with closing(StoreHolder(path)) as holder:
        for parent, child in (('root', 'c1'), ('c1', 'c2')):
            create_session(holder, child, 'cli', parent_session_id=parent)
            end_session(holder, parent, 'compression')
        statements = []
        with holder._read_ctx() as conn:
            conn.set_trace_callback(statements.append)
            assert worker_states(conn, 'root') == set()
            conn.set_trace_callback(None)
            probes = [s for s in statements if 'FROM worker_executions' in s]
            assert len(probes) == 1, f'one probe per lineage row: {probes}'
            plan = ' '.join(str(r[-1]) for r in conn.execute('EXPLAIN QUERY PLAN ' + probes[0]))
        assert 'idx_worker_executions_session_status' in plan, plan
