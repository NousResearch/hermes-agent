"""session/list finds editor threads the gateway owner stores as ``local`` (#143547 ACP-24)."""
from acp_adapter.catalog import catalog_sessions
from hermes_state import SessionDB


def test_catalog_lists_gateway_owned_editor_threads_only(tmp_path):
    db_path = tmp_path / 'state.db'
    with SessionDB(db_path) as db:
        for sid, created in (('editor-thread', 'acp'), ('cli-chat', 'cli')):
            db.create_session(session_id=sid, source='local', cwd=str(tmp_path))
            db._execute_write(lambda conn, sid=sid, created=created: conn.execute(
                'UPDATE sessions SET created_source = ? WHERE id = ?', (created, sid)))
            db.append_message(sid, 'user', 'saved turn')
    assert [row['session_id'] for row in catalog_sessions(db_path, str(tmp_path))] == ['editor-thread']
