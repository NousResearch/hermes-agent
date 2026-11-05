"""Read-only ACP picker understands canonical column-backed cwd."""
from acp_adapter.catalog import catalog_sessions
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_messages, seed_session


def test_catalog_prefers_canonical_cwd_without_rewriting_storage(tmp_path):
    db_path = tmp_path / 'state.db'
    with open_persistence(profile_scope(home=tmp_path)) as db:
        seed_session(db, 'canonical-acp', 'acp', cwd=str(tmp_path),
                     model_config={'cwd': '/stale-legacy-workspace'})
        seed_messages(db, 'canonical-acp', {'role': 'user', 'content': 'saved editor turn'})
    before = db_path.read_bytes()
    rows = catalog_sessions(db_path, str(tmp_path))
    assert [row['session_id'] for row in rows] == ['canonical-acp']
    assert rows[0]['cwd'] == str(tmp_path)
    assert not catalog_sessions(db_path, '/stale-legacy-workspace')
    assert db_path.read_bytes() == before
