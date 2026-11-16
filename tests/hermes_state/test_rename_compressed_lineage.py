"""A user rename addressed to an ended compression segment lands on the conversation's live tip (#152624)."""
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
