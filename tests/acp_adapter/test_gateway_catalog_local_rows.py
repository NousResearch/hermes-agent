"""session/list finds editor threads the gateway owner stores as ``local`` (#143547 ACP-24)."""
from acp_adapter.catalog import catalog_sessions
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_messages, seed_session


def test_catalog_lists_gateway_owned_editor_threads_only(tmp_path):
    with open_persistence(profile_scope(home=tmp_path)) as db:
        for sid, created in (('editor-thread', 'acp'), ('cli-chat', 'cli')):
            seed_session(db, sid, 'local', cwd=str(tmp_path))
            raw_holder(db)._execute_write(lambda conn, sid=sid, created=created: conn.execute(
                'UPDATE sessions SET created_source = ? WHERE id = ?', (created, sid)))
            seed_messages(db, sid, {'role': 'user', 'content': 'saved turn'})
    assert [row['session_id'] for row in catalog_sessions(tmp_path, str(tmp_path))] == ['editor-thread']
