"""``/new`` promotes the predecessor row to an explicit reset boundary. When that write fails the old
row stays recoverable and may win restart recovery, so the gateway must say so in its log rather than
drop the failure."""
import logging
import sqlite3

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from hermes_state import SessionDB


def test_failed_reset_boundary_is_logged(tmp_path, caplog):
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    store._db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        entry = store.get_or_create_session(SessionSource(platform=Platform.TELEGRAM, chat_id='123', user_id='u1'))
        old_id = entry.session_id
        store._db.create_session(session_id=old_id, source='telegram', user_id='u1')
        # A storage fault on exactly the boundary write; every other write of the reset succeeds.
        with sqlite3.connect(tmp_path / 'state.db') as conn:
            conn.execute("CREATE TRIGGER reset_fault BEFORE UPDATE OF end_reason ON sessions "
                         "WHEN NEW.end_reason = 'session_reset' BEGIN SELECT RAISE(ABORT, 'storage fault'); END")
        with caplog.at_level(logging.WARNING, logger='gateway.session'):
            assert store.reset_session(entry.session_key) is not None
        row = store._db.get_session(old_id)
        assert row is not None and row['end_reason'] is None
        assert f'Failed to end predecessor session row {old_id}' in caplog.text
        assert 'storage fault' in caplog.text
    finally:
        store._db.close()
