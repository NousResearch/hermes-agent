import asyncio
import json
from types import SimpleNamespace

from gateway import session_tui_ops, session_local_recovery


def test_real_owner_authorization_and_export_on_current_host(tmp_path):
    import pytest
    from gateway.session_authority import SessionAuthority
    from hermes_state import SessionDB
    from hermes_state_runtime import RuntimeStoreError

    home = tmp_path / 'home'
    home.mkdir()
    db = SessionDB(db_path=tmp_path / 'owner.db')
    try:
        db.create_session('s', source='webui')
        db.append_message('s', 'user', 'owned message')
        authority = SessionAuthority(SimpleNamespace(), profile_id=str(home), instance_id='test', db=db, epoch=1)
        authority.sessions['s'] = SimpleNamespace(source=None, route='test')
        ref = SimpleNamespace(profile_id=str(home), session_id='s')
        actor = SimpleNamespace(profile_id=str(home), subject='owner', capabilities={'session:read'})
        conn = SimpleNamespace(authority=authority, actor=actor)
        opts = dict(include_tools=False, include_reasoning=False, include_images=True, include_videos=True,
                    include_files=True, include_links=True, include_asides=True, collapse_details=True)
        result = asyncio.run(session_tui_ops.export_session(conn, ref, {'format': 'json', 'transcript_options': opts}))
        assert [m['text'] for m in json.loads(result['content'])['messages']] == ['owned message']
        actor.capabilities = set()
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            asyncio.run(session_tui_ops.export_session(conn, ref, {'format': 'json', 'transcript_options': opts}))
        actor.capabilities = {'session:read'}
        actor.profile_id = str(tmp_path / 'foreign')
        with pytest.raises(RuntimeStoreError, match='profile_mismatch'):
            asyncio.run(session_tui_ops.export_session(conn, ref, {'format': 'json', 'transcript_options': opts}))
    finally:
        db.close()


def test_real_sqlite_compression_lineage_is_visible(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    from gateway import session_ancillary
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        db.create_session('parent', source='webui')
        kept = db.append_message('parent', 'user', 'archived user')
        removed = db.append_message('parent', 'assistant', 'undone answer')
        edited = db.append_message('parent', 'user', 'old edited text')
        db.append_message('parent', 'user', 'replacement text')
        db._execute_write(lambda c: c.execute('UPDATE messages SET active=0, compacted=1 WHERE id=?', (kept,)))
        db._execute_write(lambda c: c.execute('UPDATE messages SET active=0, compacted=0 WHERE id=?', (removed,)))
        db._execute_write(lambda c: c.execute('UPDATE messages SET active=0, compacted=0 WHERE id=?', (edited,)))
        db.end_session('parent', 'compression')
        db.create_session('child', source='webui', parent_session_id='parent')
        db.append_message('child', 'assistant', 'child answer')
        monkeypatch.setattr(session_tui_ops, '_live_ref', lambda *args: None)
        monkeypatch.setattr(session_ancillary, 'physical_target', lambda *args: 'child')
        monkeypatch.setattr(session_local_recovery, 'transcript_target', lambda *args: 'child')
        conn = SimpleNamespace(authority=SimpleNamespace(db=db))
        opts = dict(include_tools=False, include_reasoning=False, include_images=True, include_videos=True,
                    include_files=True, include_links=True, include_asides=True, collapse_details=True)
        result = asyncio.run(session_tui_ops.export_session(conn, SimpleNamespace(session_id='child'),
                             {'format': 'json', 'transcript_options': opts}))
        assert [m['text'] for m in json.loads(result['content'])['messages']] == ['archived user', 'replacement text', 'child answer']
    finally:
        db.close()


def test_aggregate_image_and_serialized_limits_reject_without_truncation(tmp_path, monkeypatch):
    import base64
    import pytest
    from hermes_cli import session_transcript as transcript
    from hermes_state_runtime import RuntimeStoreError
    from gateway import session_ancillary
    db = __import__('hermes_state').SessionDB(db_path=tmp_path / 'limits.db')
    try:
        db.create_session('s', source='webui')
        image = 'data:image/png;base64,' + base64.b64encode(bytes.fromhex('89504e470d0a1a0a') + b'x' * 100).decode()
        db.append_message('s', 'user', [{'type': 'image_url', 'image_url': {'url': image}},
                                         {'type': 'image_url', 'image_url': {'url': image}}])
        monkeypatch.setattr(session_tui_ops, '_live_ref', lambda *args: None)
        monkeypatch.setattr(session_ancillary, 'physical_target', lambda *args: 's')
        monkeypatch.setattr(session_local_recovery, 'transcript_target', lambda *args: 's')
        conn = SimpleNamespace(authority=SimpleNamespace(db=db))
        opts = dict(include_tools=False, include_reasoning=False, include_images=True, include_videos=True,
                    include_files=True, include_links=True, include_asides=True, collapse_details=True)
        params = {'format': 'json', 'transcript_options': opts}
        monkeypatch.setattr(transcript, 'MAX_IMAGE_BYTES', 150, raising=False)
        with pytest.raises(RuntimeStoreError, match='export_too_large'):
            asyncio.run(session_tui_ops.export_session(conn, SimpleNamespace(session_id='s'), params))
        monkeypatch.setattr(transcript, 'MAX_IMAGE_BYTES', 1000, raising=False)
        monkeypatch.setattr(transcript, 'MAX_SERIALIZED_BYTES', 100, raising=False)
        with pytest.raises(RuntimeStoreError, match='export_too_large'):
            asyncio.run(session_tui_ops.export_session(conn, SimpleNamespace(session_id='s'), params))
        legacy = asyncio.run(session_tui_ops.export_session(conn, SimpleNamespace(session_id='s'), {'format': 'json'}))
        assert 'transcript_version' not in legacy and image in legacy['content']
    finally:
        db.close()


def test_owner_passes_configured_workspace_root(tmp_path, monkeypatch):
    from gateway import session_ancillary, session_policy
    from tui_gateway import server
    from hermes_state import SessionDB
    home = tmp_path / 'home'
    home.mkdir()
    workspace = tmp_path / 'workspace'
    root = workspace / '.hermes' / 'attachments'
    root.mkdir(parents=True)
    image = root / 'image.png'
    image.write_bytes(bytes.fromhex('89504e470d0a1a0a') + b'workspace')
    db = SessionDB(db_path=tmp_path / 'workspace.db')
    try:
        db.create_session('s', source='webui')
        db.append_message('s', 'user', [{'type': 'image_url', 'image_url': {'url': str(image)}}])
        monkeypatch.setattr(session_tui_ops, '_live_ref', lambda *args: None)
        monkeypatch.setattr(session_ancillary, 'physical_target', lambda *args: 's')
        monkeypatch.setattr(session_local_recovery, 'transcript_target', lambda *args: 's')
        monkeypatch.setattr(session_policy, 'policy_for_source', lambda *args: SimpleNamespace(cwd=str(workspace)))
        monkeypatch.setattr(server, '_profile_attachments_storage', lambda home: 'workspace')
        monkeypatch.setattr(server, '_cwd_is_remote', lambda home: False)
        authority = SimpleNamespace(db=db, profile_id=str(home), runner=object(), sessions={'s': SimpleNamespace(source='webui')})
        opts = dict(include_tools=False, include_reasoning=False, include_images=True, include_videos=True,
                    include_files=True, include_links=True, include_asides=True, collapse_details=True)
        result = asyncio.run(session_tui_ops.export_session(SimpleNamespace(authority=authority), SimpleNamespace(session_id='s'),
                             {'format': 'json', 'transcript_options': opts}))
        assert 'data_url' in json.loads(result['content'])['messages'][0]['attachments'][0]
    finally:
        db.close()


def test_owner_export_projects_filtered_transcript(monkeypatch):
    from gateway import session_ancillary
    monkeypatch.setattr(session_tui_ops, '_live_ref', lambda *args: None)
    monkeypatch.setattr(session_ancillary, 'physical_target', lambda *args: 'physical')
    monkeypatch.setattr(session_local_recovery, 'transcript_target', lambda *args: 'physical')
    class DB:
        def assert_export_safe(self, sid):
            assert sid == 'physical'
        def get_messages(self, sid, **kwargs):
            return self.export_session(sid)['messages']
        def export_session(self, sid, **kwargs):
            return {'id': sid, 'messages': [
                {'role': 'system', 'content': 'private', 'active': 1, 'compacted': 0},
                {'role': 'user', 'content': 'hello', 'active': 1, 'compacted': 0},
                {'role': 'tool', 'content': 'secret tool', 'active': 1, 'compacted': 0}]}
    conn = SimpleNamespace(authority=SimpleNamespace(db=DB()))
    opts = dict(include_tools=False, include_reasoning=False, include_images=True, include_videos=True,
                include_files=True, include_links=True, include_asides=True, collapse_details=True)
    result = asyncio.run(session_tui_ops.export_session(conn, SimpleNamespace(session_id='logical'),
                         {'format': 'json', 'transcript_options': opts}))
    assert result['transcript_version'] == 1
    doc = json.loads(result['content'])
    assert doc['session_id'] == 'physical'
    assert [m['text'] for m in doc['messages']] == ['hello']


def test_redaction_covers_title_as_well_as_turns(monkeypatch):
    from gateway import session_ancillary
    monkeypatch.setattr(session_tui_ops, '_live_ref', lambda *args: None)
    monkeypatch.setattr(session_ancillary, 'physical_target', lambda *args: 'physical')
    monkeypatch.setattr(session_local_recovery, 'transcript_target', lambda *args: 'physical')
    secret = 'sk-' + 'a' * 48
    class DB:
        def assert_export_safe(self, sid): pass
        def get_messages(self, sid, **kwargs):
            return self.export_session(sid)['messages']
        def export_session(self, sid, **kwargs):
            return {'id': sid, 'title': secret, 'messages': [{'role': 'user', 'content': secret, 'active': 1, 'compacted': 0}]}
    conn = SimpleNamespace(authority=SimpleNamespace(db=DB()))
    opts = dict(include_tools=False, include_reasoning=False, include_images=True, include_videos=True,
                include_files=True, include_links=True, include_asides=True, collapse_details=True)
    result = asyncio.run(session_tui_ops.export_session(conn, SimpleNamespace(session_id='logical'),
                         {'format': 'json', 'redact': True, 'transcript_options': opts}))
    assert secret not in result['content']
