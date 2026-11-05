"""Desktop's Archive is ``PATCH /api/sessions/{id}`` -> ``session.mutate`` operation ``sidebar``, a
runtime mutation that never goes through ``SessionRepository.set_archived``. It must retire the canonical
Bot Chat's name the same way, live or already reaped, or the next bot click resolves the archived row and
hangs on "Waking up" (#139779)."""
import pytest

from persistence.authority.runtime import Sidebar
from persistence.conversation.sessions import CANONICAL_BOT_CHAT_TITLE
from persistence.housekeeping.maintenance import PruneCriteria
from persistence_assembly import open_persistence, profile_scope
from session_retention import archive_sessions
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_session


def _sidebar(db, epoch, session_id, *, expected_revision=0, **fields):
    db.runtime.mutate(epoch, principal_id='human', session_id=session_id, request_id='arch',
                      expected_revision=expected_revision, command=Sidebar(**fields), owner=None)


@pytest.mark.parametrize('reaped', [False, True])
def test_sidebar_archive_mutation_retires_canonical_bot_chat(tmp_path, reaped):
    db = open_persistence(profile_scope(home=tmp_path))
    try:
        epoch = db.runtime.begin_epoch(instance_id='owner')
        seed_session(db, 'canon', 'desktop')
        assert db.sessions.set_title('canon', CANONICAL_BOT_CHAT_TITLE)
        assert db.sessions.set_hidden('canon', True)
        if reaped:  # the ws-orphan reaper closed it while the machine slept
            db.sessions.end('canon', 'ws_orphan_reap')
        _sidebar(db, epoch, 'canon', archived=True, pinned=False)
        row = session_row(db, 'canon')
        assert row['archived'] and row['title'] is None and row['end_reason'] is None
        assert db.sessions.get_by_title(CANONICAL_BOT_CHAT_TITLE) is None
        # The next click mints a fresh "Bot Chat" (same title, new row).
        seed_session(db, 'fresh', 'desktop')
        assert db.sessions.set_title('fresh', CANONICAL_BOT_CHAT_TITLE)
        assert db.sessions.get_by_title(CANONICAL_BOT_CHAT_TITLE).session_id == 'fresh'
    finally:
        db.close()


@pytest.mark.parametrize('writer', ['sidebar', 'set_session_archived'])
def test_archive_of_visible_reaped_bot_chat_retires_the_name(tmp_path, writer):
    """A VISIBLE canonical (delivery/cron/``-c --create-if-missing`` mint) the reaper already closed is
    still the user's chat: their Archive retires it, or the roster lookup un-archives it on the next
    poll and clicks keep landing in it."""
    db = open_persistence(profile_scope(home=tmp_path))
    try:
        epoch = db.runtime.begin_epoch(instance_id='owner')
        seed_session(db, 'canon', 'gateway_botmode')
        assert db.sessions.set_title('canon', CANONICAL_BOT_CHAT_TITLE)
        db.sessions.end('canon', 'ws_orphan_reap')
        if writer == 'sidebar':
            _sidebar(db, epoch, 'canon', archived=True)
        else:
            assert db.sessions.set_archived('canon', True).changed
        row = session_row(db, 'canon')
        assert (row['archived'], row['title'], row['end_reason']) == (1, None, None)
        assert db.sessions.get_by_title(CANONICAL_BOT_CHAT_TITLE) is None
        assert not db.sessions.unarchive_recoverable('canon')
    finally:
        db.close()


def test_bulk_archive_spares_compressed_bot_chat_with_visible_tip(tmp_path):
    """Compression carries the title onto a continuation that is not hidden; bulk archive is
    housekeeping and must still spare the canonical chat by its title."""
    db = open_persistence(profile_scope(home=tmp_path))
    try:
        seed_session(db, 'root', 'desktop')
        assert db.sessions.set_title('root', CANONICAL_BOT_CHAT_TITLE)
        assert db.sessions.set_hidden('root', True)
        db.sessions.end('root', 'compression')
        seed_session(db, 'tip', 'desktop', parent_session_id='root')
        assert db.sessions.set_title('tip', CANONICAL_BOT_CHAT_TITLE)
        assert not session_row(db, 'tip')['hidden']
        assert archive_sessions(db, db.scope.profile_dir, criteria=PruneCriteria(include_open=True)) == 0
        assert db.sessions.get_by_title(CANONICAL_BOT_CHAT_TITLE).session_id == 'tip'
    finally:
        db.close()
