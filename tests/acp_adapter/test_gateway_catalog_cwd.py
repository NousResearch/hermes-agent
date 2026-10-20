"""Read-only ACP picker understands canonical column-backed cwd."""
from acp_adapter.catalog import catalog_sessions
from hermes_state import SessionDB


def test_catalog_prefers_canonical_cwd_without_rewriting_storage(tmp_path):
    db_path = tmp_path / 'state.db'
    with SessionDB(db_path) as db:
        db.create_session(session_id='canonical-acp', source='acp', cwd=str(tmp_path),
                          model_config={'cwd': '/stale-legacy-workspace'})
        db.append_message('canonical-acp', 'user', 'saved editor turn')
    before = db_path.read_bytes()
    rows = catalog_sessions(db_path, str(tmp_path))
    assert [row['session_id'] for row in rows] == ['canonical-acp']
    assert rows[0]['cwd'] == str(tmp_path)
    assert not catalog_sessions(db_path, '/stale-legacy-workspace')
    assert db_path.read_bytes() == before


def test_catalog_includes_local_rows_created_by_acp(tmp_path):
    db_path = tmp_path / 'state.db'
    with SessionDB(db_path) as db:
        db.create_session(session_id='refreshed-acp', source='acp', cwd=str(tmp_path),
                          model_config={'cwd': str(tmp_path)})
        db._execute_write(lambda conn: conn.execute(
            'UPDATE sessions SET source = ? WHERE id = ?', ('local', 'refreshed-acp')))
        db.append_message('refreshed-acp', 'user', 'saved editor turn')
        # A newer unrelated local row would consume a post-filtered LIMIT.
        db.create_session(session_id='plain-local', source='local', cwd=str(tmp_path))
        db.append_message('plain-local', 'user', 'not an editor session')
        limited = db.list_sessions_rich(source='acp', created_source='acp', limit=1)
        assert [row['id'] for row in limited] == ['refreshed-acp']
    rows = catalog_sessions(db_path, str(tmp_path))
    assert [row['session_id'] for row in rows] == ['refreshed-acp']


def test_created_source_filter_accepts_any_current_source_and_preserves_positional_calls(tmp_path):
    db_path = tmp_path / 'state.db'
    with SessionDB(db_path) as db:
        db.create_session(session_id='rerouted-acp', source='acp', cwd=str(tmp_path))
        db._execute_write(lambda conn: conn.execute(
            'UPDATE sessions SET source = ? WHERE id = ?', ('gateway', 'rerouted-acp')))
        db.append_message('rerouted-acp', 'user', 'saved editor turn')
        assert [row['id'] for row in db.list_sessions_rich(created_source='acp')] == ['rerouted-acp']

        # The new optional filter is trailing, preserving the pre-existing positional API.
        db.create_session(session_id='direct-acp', source='acp', cwd=str(tmp_path))
        db.append_message('direct-acp', 'user', 'direct editor turn')
        positional = db.list_sessions_rich('acp', None, None, None, 20, 0)
        assert [row['id'] for row in positional] == ['direct-acp']
