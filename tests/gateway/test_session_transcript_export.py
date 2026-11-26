import asyncio
import json
from types import SimpleNamespace

from gateway import session_tui_ops, session_transcript_export
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import storage_handle


def test_real_owner_authorization_and_export_on_current_host(tmp_path):
    import pytest
    from gateway.session_authority import SessionAuthority
    from persistence.core.errors import AuthorityRefusal

    home = tmp_path / 'home'
    home.mkdir()
    db = storage_handle(home)
    try:
        seed_session(db, 's', 'webui')
        seed_messages(db, 's', {'role': 'user', 'content': 'owned message'})
        authority = SessionAuthority(SimpleNamespace(), profile_id=str(home), instance_id='test', persistence=db, epoch=1)
        authority.sessions['s'] = SimpleNamespace(source=None, route='test')
        ref = SimpleNamespace(profile_id=str(home), session_id='s')
        actor = SimpleNamespace(profile_id=str(home), subject='owner', capabilities={'session:read'})
        conn = SimpleNamespace(authority=authority, actor=actor)
        opts = dict(include_tools=False, include_reasoning=False, include_images=True, include_videos=True,
                    include_files=True, include_links=True, include_asides=True, collapse_details=True)
        result = asyncio.run(session_tui_ops.export_session(conn, ref, {'format': 'json', 'transcript_options': opts}))
        assert [m['text'] for m in json.loads(result['content'])['messages']] == ['owned message']
        actor.capabilities = set()
        with pytest.raises(AuthorityRefusal, match='permission_denied'):
            asyncio.run(session_tui_ops.export_session(conn, ref, {'format': 'json', 'transcript_options': opts}))
        actor.capabilities = {'session:read'}
        actor.profile_id = str(tmp_path / 'foreign')
        with pytest.raises(AuthorityRefusal, match='profile_mismatch'):
            asyncio.run(session_tui_ops.export_session(conn, ref, {'format': 'json', 'transcript_options': opts}))
    finally:
        db.close()


def test_real_sqlite_compression_lineage_is_visible(tmp_path, monkeypatch):
    from gateway import session_ancillary
    db = storage_handle(tmp_path)
    try:
        seed_session(db, 'parent', 'webui')
        kept = seed_messages(db, 'parent', {'role': 'user', 'content': 'archived user'})[0]
        removed = seed_messages(db, 'parent', {'role': 'assistant', 'content': 'undone answer'})[0]
        edited = seed_messages(db, 'parent', {'role': 'user', 'content': 'old edited text'})[0]
        seed_messages(db, 'parent', {'role': 'user', 'content': 'replacement text'})
        raw_holder(db)._execute_write(lambda c: c.execute('UPDATE messages SET active=0, compacted=1 WHERE id=?', (kept,)))
        raw_holder(db)._execute_write(lambda c: c.execute('UPDATE messages SET active=0, compacted=0 WHERE id=?', (removed,)))
        raw_holder(db)._execute_write(lambda c: c.execute('UPDATE messages SET active=0, compacted=0 WHERE id=?', (edited,)))
        db.sessions.end('parent', 'compression')
        seed_session(db, 'child', 'webui', parent_session_id='parent')
        seed_messages(db, 'child', {'role': 'assistant', 'content': 'child answer'})
        monkeypatch.setattr(session_tui_ops, '_live_ref', lambda *args: None)
        monkeypatch.setattr(session_ancillary, 'physical_target', lambda *args: 'child')
        monkeypatch.setattr(session_transcript_export, 'transcript_target', lambda *args: 'child')
        conn = SimpleNamespace(authority=SimpleNamespace(persistence=db, profile_id=str(tmp_path), sessions={}))
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
    from persistence.core.errors import AuthorityRefusal
    from gateway import session_ancillary
    db = storage_handle(tmp_path)
    try:
        seed_session(db, 's', 'webui')
        image = 'data:image/png;base64,' + base64.b64encode(bytes.fromhex('89504e470d0a1a0a') + b'x' * 100).decode()
        seed_messages(db, 's', {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': image}},
                                         {'type': 'image_url', 'image_url': {'url': image}}]})
        monkeypatch.setattr(session_tui_ops, '_live_ref', lambda *args: None)
        monkeypatch.setattr(session_ancillary, 'physical_target', lambda *args: 's')
        monkeypatch.setattr(session_transcript_export, 'transcript_target', lambda *args: 's')
        conn = SimpleNamespace(authority=SimpleNamespace(persistence=db, profile_id=str(tmp_path), sessions={}))
        opts = dict(include_tools=False, include_reasoning=False, include_images=True, include_videos=True,
                    include_files=True, include_links=True, include_asides=True, collapse_details=True)
        params = {'format': 'json', 'transcript_options': opts}
        monkeypatch.setattr(transcript, 'MAX_IMAGE_BYTES', 150, raising=False)
        with pytest.raises(AuthorityRefusal, match='export_too_large'):
            asyncio.run(session_tui_ops.export_session(conn, SimpleNamespace(session_id='s'), params))
        monkeypatch.setattr(transcript, 'MAX_IMAGE_BYTES', 1000, raising=False)
        monkeypatch.setattr(transcript, 'MAX_SERIALIZED_BYTES', 100, raising=False)
        with pytest.raises(AuthorityRefusal, match='export_too_large'):
            asyncio.run(session_tui_ops.export_session(conn, SimpleNamespace(session_id='s'), params))
        legacy = asyncio.run(session_tui_ops.export_session(conn, SimpleNamespace(session_id='s'), {'format': 'json'}))
        assert 'transcript_version' not in legacy and image in legacy['content']
    finally:
        db.close()


def test_owner_passes_configured_workspace_root(tmp_path, monkeypatch):
    from gateway import session_ancillary, session_policy
    from tui_gateway import server
    home = tmp_path / 'home'
    home.mkdir()
    workspace = tmp_path / 'workspace'
    root = workspace / '.hermes' / 'attachments'
    root.mkdir(parents=True)
    image = root / 'image.png'
    image.write_bytes(bytes.fromhex('89504e470d0a1a0a') + b'workspace')
    db = storage_handle(home)
    try:
        seed_session(db, 's', 'webui')
        seed_messages(db, 's', {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': str(image)}}]})
        monkeypatch.setattr(session_tui_ops, '_live_ref', lambda *args: None)
        monkeypatch.setattr(session_ancillary, 'physical_target', lambda *args: 's')
        monkeypatch.setattr(session_transcript_export, 'transcript_target', lambda *args: 's')
        monkeypatch.setattr(session_policy, 'policy_for_source', lambda *args: SimpleNamespace(cwd=str(workspace)))
        monkeypatch.setattr(server, '_profile_attachments_storage', lambda home: 'workspace')
        monkeypatch.setattr(server, '_cwd_is_remote', lambda home: False)
        authority = SimpleNamespace(persistence=db, profile_id=str(home), runner=object(), sessions={'s': SimpleNamespace(source='webui')})
        opts = dict(include_tools=False, include_reasoning=False, include_images=True, include_videos=True,
                    include_files=True, include_links=True, include_asides=True, collapse_details=True)
        result = asyncio.run(session_tui_ops.export_session(SimpleNamespace(authority=authority), SimpleNamespace(session_id='s'),
                             {'format': 'json', 'transcript_options': opts}))
        assert 'data_url' in json.loads(result['content'])['messages'][0]['attachments'][0]
    finally:
        db.close()


def _direct_export(tmp_path, monkeypatch, messages, *, title=None, redact=False):
    with storage_handle(tmp_path) as db:
        seed_session(db, 'physical', 'webui', title=title)
        seed_messages(db, 'physical', *messages)
        monkeypatch.setattr(session_tui_ops, '_live_ref', lambda *args: None)
        monkeypatch.setattr(session_transcript_export, 'transcript_target', lambda *args: 'physical')
        authority = SimpleNamespace(persistence=db, profile_id=str(tmp_path), sessions={})
        opts = dict(include_tools=False, include_reasoning=False, include_images=True, include_videos=True,
                    include_files=True, include_links=True, include_asides=True, collapse_details=True)
        return asyncio.run(session_tui_ops.export_session(
            SimpleNamespace(authority=authority), SimpleNamespace(session_id='logical'),
            {'format': 'json', 'redact': redact, 'transcript_options': opts}))


def test_owner_export_projects_filtered_transcript(tmp_path, monkeypatch):
    result = _direct_export(tmp_path, monkeypatch, [
        {'role': 'system', 'content': 'private'}, {'role': 'user', 'content': 'hello'},
        {'role': 'tool', 'content': 'secret tool'}])
    assert result['transcript_version'] == 1
    doc = json.loads(result['content'])
    assert doc['session_id'] == 'physical'
    assert [m['text'] for m in doc['messages']] == ['hello']


def test_redaction_covers_title_as_well_as_turns(tmp_path, monkeypatch):
    secret = 'sk-' + 'a' * 48
    result = _direct_export(tmp_path, monkeypatch, [{'role': 'user', 'content': secret}], title=secret, redact=True)
    assert secret not in result['content']
