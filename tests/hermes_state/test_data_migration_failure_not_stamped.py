"""A version-gated data migration that fails must not be recorded as done: the store keeps its old
schema_version so the next open runs the step again."""
import json
import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_common import SCHEMA_VERSION


def _rewind_version(path, version, *statements):
    with sqlite3.connect(path) as conn:
        conn.execute('UPDATE schema_version SET version = ?', (version,))
        for sql in statements:
            conn.execute(sql)


def _version(path):
    with sqlite3.connect(path) as conn:
        return conn.execute('SELECT version FROM schema_version').fetchone()[0]


def _delegate_from(path, session_id):
    with sqlite3.connect(path) as conn:
        raw = conn.execute('SELECT model_config FROM sessions WHERE id = ?', (session_id,)).fetchone()[0]
    return json.loads(raw or '{}').get('_delegate_from')


def _seed_untagged_delegate(path):
    db = SessionDB(db_path=path)
    try:
        db.create_session('parent', source='cli')
        db.create_session('child', source='subagent', parent_session_id='parent')
    finally:
        db.close()
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE sessions SET model_config = NULL WHERE id = 'child'")


def test_failed_v33_retag_keeps_version_and_reruns_on_next_open(tmp_path):
    path = tmp_path / 'state.db'
    _seed_untagged_delegate(path)
    # A storage fault on exactly the v33 re-tag UPDATE (an unregistered SQL function fails at prepare).
    _rewind_version(path, 32, "CREATE TRIGGER fault BEFORE UPDATE OF model_config ON sessions "
                              "BEGIN SELECT storage_fault(); END")

    with pytest.raises(sqlite3.OperationalError, match='storage_fault'):
        SessionDB(db_path=path).close()
    assert _version(path) == 32
    assert _delegate_from(path, 'child') is None

    with sqlite3.connect(path) as conn:
        conn.execute('DROP TRIGGER fault')
    SessionDB(db_path=path).close()
    assert _version(path) == SCHEMA_VERSION
    assert _delegate_from(path, 'child') == 'parent'


def test_v16_tags_delegates_despite_one_malformed_model_config(tmp_path):
    """A pre-v16 store: the v16 step tags every subagent row; one unreadable row is skipped, not the step."""
    path = tmp_path / 'state.db'
    db = SessionDB(db_path=path)
    try:
        db.create_session('parent', source='cli')
        for sid in ('child', 'damaged-child'):
            db.create_session(sid, source='cli', parent_session_id='parent')
    finally:
        db.close()
    _rewind_version(path, 15, "UPDATE sessions SET model_config = NULL WHERE id = 'child'",
                    "UPDATE sessions SET model_config = 'not json' WHERE id = 'damaged-child'")

    SessionDB(db_path=path).close()
    assert _version(path) == SCHEMA_VERSION
    assert _delegate_from(path, 'child') == 'parent'
