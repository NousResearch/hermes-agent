"""Desktop's Archive is ``PATCH /api/sessions/{id}`` -> ``session.mutate`` operation ``sidebar``, a
store write that never calls ``set_session_archived``. It must retire the canonical Bot Chat's name
the same way, live or already reaped, or the next bot click resolves the archived row and hangs on
"Waking up" (#139779)."""
import pytest

import hermes_agent.hermes_state_runtime as rt
from hermes_agent.hermes_state import SessionDB


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


@pytest.mark.asyncio
@pytest.mark.parametrize('hidden', [False, True])
async def test_delivery_after_archive_of_delivery_minted_bot_chat_mints_fresh(hidden, monkeypatch):
    """The delivery door mints the canonical chat VISIBLE (cron, teammate, ``chat -c "Bot Chat"
    --create-if-missing``) and generation 0 reuses an untitled row as an interrupted mint. After a
    sidebar Archive the next delivery must land in a fresh "Bot Chat", never wedge on
    ``admission_conflict`` (visible) nor re-title the archived row (hidden) (#139779)."""
    from hermes_agent.gateway import run
    from hermes_agent.gateway.session_bot import _target
    from hermes_constants import get_hermes_home
    from tests.gateway.test_follow_profile_followups import _profile
    from tests.gateway.test_session_follow_profile_config import _owner
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run, '_hermes_home', home)
    _profile(home, 'm', 1, 'k')
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    db = authority.db
    try:
        await _target(authority, owner.actor)
        old = db.get_session_by_title('Bot Chat')['id']
        db.append_message(old, 'user', 'old history')
        if hidden:
            db.set_session_hidden(old, True)
        rt.mutate_runtime_session(db, epoch=authority.epoch, principal_id='human', session_id=old,
                                  request_id='arch', expected_revision=db.get_session(old)['runtime_revision'] or 0,
                                  operation='sidebar', payload={'archived': True})
        _ref, _live, entry = await _target(authority, owner.actor)
        assert entry.session_id != old
        assert db.get_session_by_title('Bot Chat')['id'] == entry.session_id
        assert {k: db.get_session(old)[k] for k in ('title', 'archived', 'hidden')} == {
            'title': None, 'archived': 1, 'hidden': 0}
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.parametrize('writer', ['sidebar', 'set_session_archived'])
def test_archive_of_visible_reaped_bot_chat_retires_the_name(tmp_path, writer):
    """A VISIBLE canonical (delivery/cron/``-c --create-if-missing`` mint) the reaper already closed is
    still the user's chat: their Archive retires it, or the roster lookup un-archives it on the next
    poll and clicks keep landing in it."""
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        db.create_session('canon', source='gateway_botmode')
        assert db.set_session_title('canon', SessionDB.CANONICAL_BOT_CHAT_TITLE)
        db.end_session('canon', 'ws_orphan_reap')
        if writer == 'sidebar':
            rt.mutate_runtime_session(db, epoch=epoch, principal_id='human', session_id='canon', request_id='arch',
                                      expected_revision=0, operation='sidebar', payload={'archived': True})
        else:
            assert db.set_session_archived('canon', True)
        row = db.get_session('canon')
        assert (row['archived'], row['title'], row['end_reason']) == (1, None, None)
        assert db.get_session_by_title(SessionDB.CANONICAL_BOT_CHAT_TITLE) is None
        assert not db.unarchive_recoverable_session('canon')
    finally:
        db.close()


def test_bulk_archive_spares_compressed_bot_chat_with_visible_tip(tmp_path):
    """Compression carries the title onto a continuation that is not hidden; bulk archive is
    housekeeping and must still spare the canonical chat by its title."""
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        db.create_session('root', source='desktop')
        assert db.set_session_title('root', SessionDB.CANONICAL_BOT_CHAT_TITLE)
        assert db.set_session_hidden('root', True)
        db.end_session('root', 'compression')
        db.create_session('tip', source='desktop', parent_session_id='root')
        assert db.set_session_title('tip', SessionDB.CANONICAL_BOT_CHAT_TITLE)
        assert not db.get_session('tip')['hidden']
        assert db.archive_sessions(include_open=True) == 0
        assert db.get_session_by_title(SessionDB.CANONICAL_BOT_CHAT_TITLE)['id'] == 'tip'
    finally:
        db.close()
