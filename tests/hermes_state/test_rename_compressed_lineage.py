"""A user rename addressed to an ended compression segment lands on the conversation's live tip (#152624)."""
import pytest

from hermes_state import SessionDB
import hermes_state_runtime as rt


def _compressed(db, root_title=None):
    db.create_session('chat', source='desktop')
    if root_title:
        db.set_session_title('chat', root_title)
    db.append_message('chat', role='user', content='hi')
    db.publish_compression_child(parent_session_id='chat', child_session_id='chat-2', source='desktop',
                                 messages=[{'role': 'user', 'content': 'hi'}], require_compression_lease=False)
    db.set_auto_title('chat-2', 'Plugin para sessoes', source=db.TITLE_SOURCE_LLM)
    return rt.begin_runtime_epoch(db, instance_id='owner')


def _rename(db, epoch, sid, title, request_id):
    revision = db.get_session(sid)['runtime_revision']
    return rt.mutate_runtime_session(db, epoch=epoch, principal_id='human', session_id=sid, request_id=request_id,
                                     expected_revision=revision, operation='sidebar', payload={'title': title})


def test_rename_through_ended_segment_is_what_the_listing_shows(tmp_path):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        epoch = _compressed(db)
        receipt = _rename(db, epoch, 'chat', 'SESSION TALK', 'r1')
        assert receipt['title'] == 'SESSION TALK'
        assert db.get_session('chat-2')['title'] == 'SESSION TALK'
        assert db.get_session('chat')['title'] is None
        [row] = [s for s in db.list_sessions_rich() if 'chat' in s.get('_lineage_ids', [s['id']])]
        assert row['title'] == 'SESSION TALK'
        # The receipt's revision is the addressed row's, so the client's next fence on that id holds.
        assert receipt['revision'] == db.get_session('chat')['runtime_revision'] == 1
        assert _rename(db, epoch, 'chat', 'Second name', 'r2')['title'] == 'Second name'
        assert db.get_session('chat-2')['title'] == 'Second name'


def test_name_held_by_ended_ancestor_moves_to_the_tip(tmp_path):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        epoch = _compressed(db, root_title='Old name')
        _rename(db, epoch, 'chat', 'Old name', 'r1')
        assert db.get_session('chat-2')['title'] == 'Old name'
        assert db.get_session('chat')['title'] is None
        # A rename sent to the tip itself (the classic CLI path) is unchanged.
        assert _rename(db, epoch, 'chat-2', 'Tip name', 'r2')['title'] == 'Tip name'
        assert db.get_session('chat-2')['title'] == 'Tip name'


def test_a_tip_rename_makes_an_older_copy_of_any_segment_stale(tmp_path):
    """Client A read the ended segment, client B then renamed the live tip: A's rename (now landing
    on that same tip) must refuse instead of silently overwriting B's name."""
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        epoch = _compressed(db)
        stale = db.get_session('chat')['runtime_revision']
        _rename(db, epoch, 'chat-2', 'B name', 'b1')
        with pytest.raises(rt.RuntimeStoreError, match='revision_conflict'):
            rt.mutate_runtime_session(db, epoch=epoch, principal_id='human', session_id='chat', request_id='a1',
                                      expected_revision=stale, operation='sidebar', payload={'title': 'A name'})
        assert db.get_session('chat-2')['title'] == 'B name'


def test_clearing_through_an_ended_segment_clears_the_listed_name(tmp_path):
    """A rotation cut off before the title copy leaves the name on the root, and the untitled tip
    lists under it: a clear must remove that fallback too, or the sidebar keeps the old name."""
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        epoch = _compressed(db, root_title='Old name')
        db.set_session_title('chat-2', '')  # the copy never happened: the tip is untitled
        assert _rename(db, epoch, 'chat', '', 'c1')['title'] is None
        [row] = [s for s in db.list_sessions_rich() if 'chat' in s.get('_lineage_ids', [s['id']])]
        assert row['title'] is None


def test_a_name_freed_from_another_chat_advances_that_chats_revision(tmp_path):
    """Taking a name an abandoned empty chat held clears it there too: that row's revision must move,
    or a client's older copy of it still passes the fence."""
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('ghost', source='desktop')
        db.set_session_title('ghost', 'Shared name')
        db.end_session('ghost', 'user_exit')
        epoch = _compressed(db)
        before = db.get_session('ghost')['runtime_revision']
        _rename(db, epoch, 'chat', 'Shared name', 'g1')
        assert (db.get_session('ghost')['title'], db.get_session('chat-2')['title']) == (None, 'Shared name')
        assert db.get_session('ghost')['runtime_revision'] > before


def test_clearing_through_a_canonical_bot_chat_ancestor_keeps_its_name(tmp_path):
    """The hidden canonical Bot Chat is found by its exact name: a clear must not reach past the
    untitled tip and empty it on the ancestor, or the bot loses its conversation."""
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        epoch = _compressed(db, root_title=db.CANONICAL_BOT_CHAT_TITLE)
        db.set_session_hidden('chat', True)
        db.set_session_title('chat-2', '')  # the rotation's title copy never happened
        with pytest.raises(ValueError, match='canonical Bot Chat'):
            _rename(db, epoch, 'chat', '', 'b1')
        assert db.resolve_session_by_title(db.CANONICAL_BOT_CHAT_TITLE) is not None
