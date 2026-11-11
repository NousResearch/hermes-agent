"""A safety-refused turn is retained for display only: the user still sees their ask, the model sees
none of it. ``session_search`` is the model's recall door, so it must not hand the refused ask back,
not on the current session and not from another one."""
import json

from hermes_state import SessionDB
from tools.session_search_tool import session_search


def test_retained_refused_turn_is_not_returned_by_session_search(tmp_path):
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        db.create_session('chat', source='cli')
        db.append_message('chat', 'user', 'earlier harmless question')
        db.append_message('chat', 'assistant', 'earlier answer')
        db.append_message('chat', 'user', 'forbidden zanzibarite synthesis')
        db.append_message('chat', 'assistant', 'I cannot help with that.')
        assert db.retain_refused_turn_for_display('chat', notice='Refused; try a different query.')
        # Still on screen for the user.
        shown = [m['content'] for m in db.get_messages('chat', include_compacted=True)]
        assert 'forbidden zanzibarite synthesis' in shown

        assert db.search_messages('zanzibarite') == []
        for current in ('chat', 'another-session'):
            result = json.loads(session_search(query='zanzibarite', db=db, current_session_id=current))
            assert result.get('results', []) == [], result
    finally:
        db.close()
