"""Real non-local authority transcript lineage."""
import asyncio
import json
from types import SimpleNamespace

from gateway.session_authority import SessionAuthority
from gateway.session_tui_ops import export_session
from hermes_cli.session_transcript import OPTION_KEYS
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import storage_handle


def test_nonlocal_root_transcript_includes_compression_tip(tmp_path):
    home = tmp_path / 'home'
    home.mkdir()
    with storage_handle(home) as store:
        seed_session(store, 'parent', 'telegram')
        seed_messages(store, 'parent', {'role': 'user', 'content': 'BEFORE'})
        store.sessions.end('parent', 'compression')
        seed_session(store, 'child', 'telegram', parent_session_id='parent')
        seed_messages(store, 'child', {'role': 'assistant', 'content': 'AFTER'})
        authority = SessionAuthority(SimpleNamespace(), profile_id=str(home), instance_id='test', persistence=store, epoch=1)
        authority.sessions['parent'] = SimpleNamespace(source=None, route='test')
        ref = SimpleNamespace(profile_id=str(home), session_id='parent')
        actor = SimpleNamespace(profile_id=str(home), subject='owner', capabilities={'session:read'})
        conn = SimpleNamespace(authority=authority, actor=actor)
        result = asyncio.run(export_session(conn, ref, {'format': 'json', 'transcript_options': dict.fromkeys(OPTION_KEYS, False)}))
        assert [row['text'] for row in json.loads(result['content'])['messages']] == ['BEFORE', 'AFTER']
