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
