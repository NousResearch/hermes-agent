"""``session.history`` on the canonical owner: the durable display transcript with each row's ``row_id``.

Main's TUI sidecar served it (``tui_gateway/methods_session.py``); Desktop's edit-after-interrupt resolves the edited turn's row from it, and without an answer the edit only appends (TUI-15, #134889).
"""
from types import SimpleNamespace

import pytest

from tests.gateway.test_session_mutation_model_busy import _owner


@pytest.mark.asyncio
async def test_session_history_returns_the_display_transcript_with_row_ids(tmp_path, monkeypatch):
    async def handle(event):
        return 'ok'
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    try:
        from gateway.session_local_recovery import transcript_target
        target = transcript_target(authority, ref)
        first = db.append_message(target, 'user', 'first question')
        answer = db.append_message(target, 'assistant', 'first answer')
        interrupted = db.append_message(target, 'user', 'interrupted question')
        reply = await owner.dispatch({'id': 1, 'method': 'session.history',
                                      'params': {'session_id': ref.session_id, 'profile': None}})
        assert 'result' in reply, reply
        messages = reply['result']['messages']
        assert [(m['role'], m['text'], m['row_id']) for m in messages] == [
            ('user', 'first question', first), ('assistant', 'first answer', answer),
            ('user', 'interrupted question', interrupted)]
        assert reply['result']['count'] == len(messages)
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_session_history_of_a_compressed_messaging_session_reads_its_continuation(tmp_path, monkeypatch):
    from gateway.config import Platform
    from gateway.session_authority import LiveSession

    async def handle(event):
        return 'ok'
    store, db, authority, owner, _ref = _owner(tmp_path, monkeypatch, handle)
    try:
        db.create_session('root', source='telegram', session_key='agent:main:telegram:dm:1')
        before = db.append_message('root', 'user', 'before compression')
        assert db.try_acquire_compression_lock('root', 'holder')
        db.publish_compression_child(parent_session_id='root', child_session_id='child', source='telegram',
                                     messages=[{'role': 'user', 'content': 'summary'}],
                                     compression_lock_holder='holder')
        after = db.append_message('child', 'user', 'after compression')
        authority.sessions['root'] = LiveSession(
            SimpleNamespace(platform=Platform.TELEGRAM, user_id='human'), 'agent:main:telegram:dm:1')
        reply = await owner.dispatch({'id': 1, 'method': 'session.history', 'params': {'session_id': 'root'}})
        assert 'result' in reply, reply
        rows = {m['text']: m['row_id'] for m in reply['result']['messages'] if m['role'] == 'user'}
        assert rows.get('before compression') == before, rows
        assert rows.get('after compression') == after, rows
    finally:
        await owner.close()
        store.close_all_db_handles()
