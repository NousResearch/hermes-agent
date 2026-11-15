"""Real non-local authority transcript lineage."""
import asyncio
import json
from types import SimpleNamespace

from gateway.session_authority import SessionAuthority
from gateway.session_tui_ops import export_session
from hermes_cli.session_transcript import OPTION_KEYS
from hermes_state import SessionDB


def test_nonlocal_root_transcript_includes_compression_tip(tmp_path):
    home = tmp_path / 'home'
    home.mkdir()
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        db.create_session('parent', source='telegram')
        db.append_message('parent', 'user', 'BEFORE')
        db.end_session('parent', 'compression')
        db.create_session('child', source='telegram', parent_session_id='parent')
        db.append_message('child', 'assistant', 'AFTER')
        authority = SessionAuthority(SimpleNamespace(), profile_id=str(home), instance_id='test', db=db, epoch=1)
        authority.sessions['parent'] = SimpleNamespace(source=None, route='test')
        ref = SimpleNamespace(profile_id=str(home), session_id='parent')
        actor = SimpleNamespace(profile_id=str(home), subject='owner', capabilities={'session:read'})
        conn = SimpleNamespace(authority=authority, actor=actor)
        result = asyncio.run(export_session(conn, ref, {'format': 'json', 'transcript_options': dict.fromkeys(OPTION_KEYS, False)}))
        assert [row['text'] for row in json.loads(result['content'])['messages']] == ['BEFORE', 'AFTER']
    finally:
        db.close()
