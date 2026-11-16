"""The gateway's Bots roster follows a reset Bot Chat into its lineage children.

A canonical ``/new`` (or a compression-exhaustion auto-reset) keeps the Bot Chat's id and moves the
transcript to children named in the local receipt's ``lineage``. The Desktop Bots pane reads the
row from ``profiles.list`` (``gateway/session_group_controls.py`` ``_read_bot_sessions``), so its
``canonical_session`` must preview the conversation's newest message and date the row from the
freshest segment, not from the segment before the reset.
"""
import json

import pytest

from gateway.session_group_controls import _read_bot_sessions
from hermes_state import SessionDB
from hermes_state_local import POLICY_PREFIX

OWNER = 'uid:desk'


@pytest.fixture
def db(tmp_path):
    store = SessionDB(db_path=tmp_path / 'state.db')
    yield store
    store.close()


def _segment(db, sid, *, at, text=None, title=None, parent=None, end_reason=None):
    """One stored segment of the local chat ``local-root`` with fixed stamps."""
    db.create_session(sid, source='gui', user_id=OWNER, chat_id='local-root', parent_session_id=parent)
    if text:
        db.append_message(sid, 'assistant', text, timestamp=at)
    with db._lock:
        db._conn.execute(
            'UPDATE sessions SET title=?, started_at=?, last_activity_at=?, hidden=1, ended_at=?, end_reason=? '
            'WHERE id=?', (title, at, at, at + 1 if end_reason else None, end_reason, sid))


def _reset_bot_chat(db, *, receipt):
    _segment(db, 'local-root', at=1000, text='before the reset', title='Bot Chat', end_reason='session_reset')
    _segment(db, 'seg1', at=2000, text='after the reset', parent='local-root', end_reason='session_reset')
    _segment(db, 'seg2', at=3000, parent='seg1')  # just reset again, nothing said yet
    if receipt:
        with db._lock:
            db._conn.execute(
                'INSERT INTO state_meta (key, value) VALUES (?, ?)',
                (POLICY_PREFIX + 'local-root',
                 json.dumps({'session_id': 'local-root', 'lineage': ['local-root', 'seg1', 'seg2']})))


def test_canonical_row_previews_the_newest_reset_segment(db):
    _reset_bot_chat(db, receipt=True)

    row = _read_bot_sessions(db, OWNER, False)['canonical_session']

    assert (row['id'], row['resolved_id']) == ('local-root', 'local-root')
    assert row['preview'] == 'after the reset'  # the empty newest segment falls back to the one before it
    assert row['last_active'] == 3000


def test_a_chat_without_a_receipt_lineage_keeps_the_tip_row(db):
    _reset_bot_chat(db, receipt=False)

    row = _read_bot_sessions(db, OWNER, False)['canonical_session']

    assert row['preview'] == 'before the reset'
    assert row['last_active'] == 1000
