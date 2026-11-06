"""Desktop's Archive is ``PATCH /api/sessions/{id}`` -> ``session.mutate`` operation ``sidebar``, a
store write that never calls ``set_session_archived``. It must retire the canonical Bot Chat's name
the same way, live or already reaped, or the next bot click resolves the archived row and hangs on
"Waking up" (#139779)."""
import pytest

import hermes_state_runtime as rt
from hermes_state import SessionDB


@pytest.mark.parametrize('reaped', [False, True])
def test_sidebar_archive_mutation_retires_canonical_bot_chat(tmp_path, reaped):
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        db.create_session('canon', source='desktop')
        assert db.set_session_title('canon', SessionDB.CANONICAL_BOT_CHAT_TITLE)
        assert db.set_session_hidden('canon', True)
        if reaped:  # the ws-orphan reaper closed it while the machine slept
            db.end_session('canon', 'ws_orphan_reap')
        rt.mutate_runtime_session(db, epoch=epoch, principal_id='human', session_id='canon', request_id='arch',
                                  expected_revision=0, operation='sidebar',
                                  payload={'archived': True, 'pinned': False})
        row = db.get_session('canon')
        assert row['archived'] and row['title'] is None and row['end_reason'] is None
        assert db.get_session_by_title(SessionDB.CANONICAL_BOT_CHAT_TITLE) is None
        # The next click mints a fresh "Bot Chat" (same title, new row).
        db.create_session('fresh', source='desktop')
        assert db.set_session_title('fresh', SessionDB.CANONICAL_BOT_CHAT_TITLE)
        assert db.get_session_by_title(SessionDB.CANONICAL_BOT_CHAT_TITLE)['id'] == 'fresh'
    finally:
        db.close()
