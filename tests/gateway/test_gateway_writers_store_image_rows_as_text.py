"""The gateway's transcript writers store an image row as the agent's flush stores it: text, never the base64.

Older builds' compaction stored a vision result as its part list (``\\x00json:`` JSON, base64 included), and
nothing migrates those rows. A writer that reads the transcript back and stores it again as a new generation
(``/retry`` under a session authority or through ``rewrite_transcript``, ``/branch``, the canonical
``session.mutate compress`` commit, in place or rotated) gets the part list from the loader and wrote it again,
base64 included, into the rows every rebuilt request sends (#148146). Real SQLite and the real handlers.
"""
import asyncio
import weakref

import pytest

from session_rules.turns import _durable_content
from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run_runtime import initialize_gateway_runtime
from gateway.session import SessionSource
from gateway.session_ingress_context import native_callback, register_transport_home
from tests.gateway.test_retry_authority_refusal import _FakeAdapter, _queued_texts, _retry, served
from tests.gateway.test_resume_latest_local_cli import _call, _owner
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_messages
from tests.waiting import async_wait_until

_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
_PARTS = [{"type": "text", "text": "Image loaded into your context."},
          {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _B64}}]
# The rows an older build's compaction left: an active vision result stored as its part list.
_LEGACY = [
    {"role": "user", "content": "what is on this label?"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "vision_analyze", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "tool_name": "vision_analyze", "content": _PARTS},
    {"role": "assistant", "content": "The label says 42."},
]


async def _gateway_session(runner, root):
    await initialize_gateway_runtime(runner)
    register_transport_home(runner, None, root)
    adapter = _FakeAdapter()
    runner.adapters = {Platform.DISCORD: adapter}
    runner._is_user_authorized_for_source = lambda *a, **k: True
    source = SessionSource(platform=Platform.DISCORD, chat_id='c1', chat_type='dm', user_id='u1', user_name='probe')
    source._transport_adapter_ref = weakref.ref(adapter)
    sid = (await runner.async_session_store.get_or_create_session(source)).session_id
    db = runner.session_store._persistence_for_session_id(sid)
    seed_messages(db, sid, *(dict(m) for m in _LEGACY))
    for row in ({'role': 'user', 'content': 'retry me'}, {'role': 'assistant', 'content': 'old answer'}):
        runner.session_store.append_to_transcript(sid, row)
    return source, sid, db


async def _retry_writer(request, monkeypatch, *, under_authority):
    runner, root = request.getfixturevalue('served')
    source, sid, db = await _gateway_session(runner, root)
    from gateway.session_authorities import active_authority
    authority = active_authority(runner)
    authority._schedule = lambda ref: None  # the re-sent turn stays queued; the rewrite is what is tested
    if under_authority:
        assert runner._retry_owned_store(sid) is not None  # replace_messages_in_transaction inside the admission
    else:
        monkeypatch.setattr(runner, '_retry_owned_store', lambda _sid: None)  # SessionStore.rewrite_transcript
    task = asyncio.ensure_future(_retry(runner, root, source))
    await async_wait_until(lambda: bool(_queued_texts(authority, sid)), timeout=4)
    task.cancel()
    assert _queued_texts(authority, sid) == ['retry me']
    return db, sid


async def _branch_writer(request, monkeypatch):
    runner, root = request.getfixturevalue('served')
    source, sid, db = await _gateway_session(runner, root)
    event = MessageEvent(text='/branch --here', message_type=MessageType.TEXT, source=source, message_id='m-branch')
    with native_callback(runner, event, root, None):
        await runner._handle_branch_command(event)
    child = raw_holder(db)._read_one("SELECT id FROM sessions WHERE parent_session_id = ?", (sid,))[0]
    return db, child


async def _compress_writer(request, monkeypatch, *, in_place):
    """The commit ``session.mutate compress`` runs, handed the transcript ``prepare_compress`` loads."""
    from persistence.sqlite.authority._mutation_publish import compress_in_transaction
    from persistence.sqlite.authority._runtime_picks import local_snapshot

    tmp_path = request.getfixturevalue('tmp_path')
    monkeypatch.chdir(tmp_path)  # the owner's session.create seeds a profile tree relative to the cwd
    store, authority, owner = _owner(tmp_path / 'home', monkeypatch)
    request.addfinalizer(store.close_all_db_handles)
    sid = (await _call(owner, 'session.create', request_id='create', source='cli', cwd=str(tmp_path),
                       toolsets=[], model='m'))['session_id']
    await owner.close()
    db = authority.persistence
    seed_messages(db, sid, *(dict(m) for m in _LEGACY))

    def write(conn):
        snapshot = local_snapshot(raw_holder(db), conn, sid)
        from tests.fixtures.storage_read import conversation
        loaded = conversation(db, snapshot['target'])
        prepared = {'snapshot': snapshot, 'in_place': in_place, 'messages': [
            {'role': 'user', 'content': '[CONTEXT COMPACTION] a label was read', '_compressed_summary': True},
            *(m for m in loaded if m.get('role') in {'user', 'assistant', 'tool'})]}
        return compress_in_transaction(raw_holder(db), conn, sid, {}, prepared)
    return db, raw_holder(db)._execute_write(write)[1]['target_session_id']


@pytest.mark.asyncio
async def test_gateway_append_projects_parts_without_mutating_live_content(served):  # health: allow F811 -- pytest injects the imported served fixture by parameter name
    runner, root = served
    _source, sid, db = await _gateway_session(runner, root)
    live = {'role': 'user', 'content': _PARTS}
    runner.session_store.append_to_transcript(sid, live)
    row = raw_holder(db)._read_one(
        "SELECT content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT 1", (sid,))
    assert row['content'] == _durable_content(_PARTS)
    assert live['content'] == _PARTS


_WRITERS = {
    'retry_in_admission': lambda request, mp: _retry_writer(request, mp, under_authority=True),
    'retry_rewrite_transcript': lambda request, mp: _retry_writer(request, mp, under_authority=False),
    'branch': _branch_writer,
    'compress_in_place': lambda request, mp: _compress_writer(request, mp, in_place=True),
    'compress_rotated': lambda request, mp: _compress_writer(request, mp, in_place=False),
}


@pytest.mark.asyncio
@pytest.mark.parametrize('writer', list(_WRITERS))
async def test_a_rewritten_legacy_image_row_is_stored_as_text(request, monkeypatch, writer):
    db, session_id = await _WRITERS[writer](request, monkeypatch)
    rows = raw_holder(db)._read_all(
        "SELECT content FROM messages WHERE session_id = ? AND active = 1 AND role = 'tool'", (session_id,))
    assert [row[0] for row in rows] == [_durable_content(_PARTS)]
    assert not raw_holder(db)._read_all(
        "SELECT COUNT(*) FROM messages WHERE session_id = ? AND active = 1 "
        "AND instr(CAST(content AS BLOB), CAST(';base64,' AS BLOB)) > 0", (session_id,))[0][0]
