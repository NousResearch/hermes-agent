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


@pytest.mark.asyncio
async def test_session_history_keeps_visible_replies_and_audio_after_in_place_compaction(tmp_path, monkeypatch):
    async def handle(event):
        return 'ok'
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    try:
        from hermes_state import SessionDB
        assert isinstance(db, SessionDB)
        from gateway.session_local_recovery import transcript_target
        target = transcript_target(authority, ref)
        db.append_message(target, 'user', 'report the result')
        prose = db.append_message(target, 'assistant', 'The first visible reply.', finish_reason='stop')
        audio_text = f'[[audio_as_voice]]\nMEDIA:{tmp_path / "reply.ogg"}'
        audio = db.append_message(target, 'assistant', audio_text, finish_reason='verification_required')
        db.archive_and_compact(target, [{'role': 'user', 'content': 'compressed model context'}])
        assert audio_text not in str(db.get_messages_as_conversation(target))
        reply = await owner.dispatch({'id': 1, 'method': 'session.history',
                                      'params': {'session_id': ref.session_id}})
        assert 'result' in reply, reply
        rows = {m['row_id']: m['text'] for m in reply['result']['messages'] if 'row_id' in m}
        assert rows.get(prose) == 'The first visible reply.'
        assert rows.get(audio) == audio_text
        assert next(m for m in reply['result']['messages'] if m.get('row_id') == audio)['archived'] is True
        assert reply['result']['count'] == len(reply['result']['messages'])
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_session_history_does_not_resurrect_user_rewound_replies(tmp_path, monkeypatch):
    async def handle(event):
        return 'ok'
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    try:
        from hermes_state import SessionDB
        assert isinstance(db, SessionDB)
        from gateway.session_local_recovery import transcript_target
        target = transcript_target(authority, ref)
        question = {'role': 'user', 'content': 'kept question'}
        db.append_message(target, 'user', question['content'])
        db.append_message(target, 'assistant', 'reply withdrawn by the user')
        db.replace_messages(target, [question], archive_dropped=True)
        reply = await owner.dispatch({'id': 1, 'method': 'session.history',
                                      'params': {'session_id': ref.session_id}})
        assert 'result' in reply, reply
        assert [m['text'] for m in reply['result']['messages']] == ['kept question']
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('visible', [True, False])
async def test_session_history_count_matches_projection_after_internal_rows_are_hidden(tmp_path, monkeypatch, visible):
    async def handle(event):
        return 'ok'
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    try:
        from hermes_state import SessionDB
        assert isinstance(db, SessionDB)
        from gateway.session_local_recovery import transcript_target
        target = transcript_target(authority, ref)
        if visible:
            db.append_message(target, 'user', 'visible question')
        db.append_message(target, 'user', '[System: verification required]')
        reply = await owner.dispatch({'id': 1, 'method': 'session.history',
                                      'params': {'session_id': ref.session_id}})
        assert 'result' in reply, reply
        result = reply['result']
        assert [m['text'] for m in result['messages']] == (['visible question'] if visible else [])
        assert result['count'] == (1 if visible else 0)
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_session_history_marks_archived_rows_and_bounds_only_those_main_never_returned(tmp_path, monkeypatch):
    from gateway.config import Platform
    from gateway.session_authority import LiveSession
    import gateway.session_local_recovery as recovery

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
        db.append_message('child', 'assistant', '', tool_calls=[
            {'id': 'c1', 'type': 'function', 'function': {'name': 'terminal', 'arguments': '{"command": "ls"}'}}])
        db.append_message('child', 'tool', 'listing', tool_call_id='c1', tool_name='terminal')
        reply_row = db.append_message('child', 'assistant', 'reply a compaction archived')
        db.archive_and_compact('child', [{'role': 'user', 'content': 'compressed model context'}])
        after = db.append_message('child', 'user', 'after compaction')
        authority.sessions['root'] = LiveSession(
            SimpleNamespace(platform=Platform.TELEGRAM, user_id='human'), 'agent:main:telegram:dm:1')
        reply = await owner.dispatch({'id': 1, 'method': 'session.history', 'params': {'session_id': 'root'}})
        assert 'result' in reply, reply
        messages = reply['result']['messages']
        flags = {m['row_id']: m.get('archived') for m in messages if 'row_id' in m}
        assert flags.get(before) is True and flags.get(reply_row) is True, flags
        assert after in flags and flags[after] is None, flags
        assert next(m for m in messages if m['role'] == 'tool').get('archived') is True
        assert reply['result']['count'] == len(messages) and 'archived_truncated' not in messages[0]
        # Past the resume byte budget only the rows a compaction archived go; the ancestor main returned stays.
        monkeypatch.setattr(recovery, 'RESUME_HISTORY_BYTES', 1)
        reply = await owner.dispatch({'id': 2, 'method': 'session.history', 'params': {'session_id': 'root'}})
        messages = reply['result']['messages']
        flags = {m['row_id']: m.get('archived') for m in messages if 'row_id' in m}
        assert flags.get(before) is True and flags[after] is None and reply_row not in flags, flags
        assert all(m['role'] != 'tool' for m in messages) and messages[0]['archived_truncated'] is True
        assert reply['result']['count'] == len(messages)
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('rewound', [False, True])
async def test_session_history_follows_the_display_lineage_of_a_rotated_compression(tmp_path, monkeypatch, rewound):
    """The child carries the tail: the reply shows the carried reply once, and a turn rewound in the child stays
    gone, as ``session.attach`` shows them (main also returned the ancestor's copies, and kept them after the
    rewind)."""
    from gateway.config import Platform
    from gateway.session_authority import LiveSession

    async def handle(event):
        return 'ok'
    store, db, authority, owner, _ref = _owner(tmp_path, monkeypatch, handle)
    try:
        db.create_session('root', source='telegram', session_key='agent:main:telegram:dm:1')
        for role, text in zip(('user', 'assistant') * 2, ('OLD_Q', 'OLD_A', 'TAIL_Q', 'TAIL_A')):
            db.append_message('root', role, text)
        assert db.try_acquire_compression_lock('root', 'holder')
        db.publish_compression_child(parent_session_id='root', child_session_id='child', source='telegram',
                                     messages=[{'role': 'user', 'content': 'SUMMARY'},
                                               {'role': 'user', 'content': 'TAIL_Q'},
                                               {'role': 'assistant', 'content': 'TAIL_A'}],
                                     compression_lock_holder='holder')
        db.append_message('child', 'user', 'NEW_Q')
        if rewound:
            db.rewind_to_message('child', next(m['id'] for m in db.get_messages('child') if m['content'] == 'TAIL_Q'))
        authority.sessions['root'] = LiveSession(
            SimpleNamespace(platform=Platform.TELEGRAM, user_id='human'), 'agent:main:telegram:dm:1')
        reply = await owner.dispatch({'id': 1, 'method': 'session.history', 'params': {'session_id': 'root'}})
        assert 'result' in reply, reply
        assert [(m['text'], bool(m.get('archived'))) for m in reply['result']['messages']] == [
            ('OLD_Q', True), ('OLD_A', True), ('SUMMARY', False),
            *([] if rewound else [('TAIL_Q', False), ('TAIL_A', False), ('NEW_Q', False)])]
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_session_history_stays_under_the_frame_cap_after_many_compactions(tmp_path, monkeypatch):
    """Each in-place compaction archives a generation the reply shows. Past the resume byte budget only the
    newest archived rows stay, so the reply stays under the gateway clients' 8 MiB frame cap whenever main's
    does; every live row stays."""
    import json

    async def handle(event):
        return 'ok'
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    try:
        from gateway.session_local_recovery import transcript_target
        target = transcript_target(authority, ref)
        for generation in range(3):  # ~9 MiB archived in all: past the frame cap if every row were sent
            db.append_messages_batch(target, [
                {'role': 'user' if n % 2 == 0 else 'assistant', 'content': f'G{generation} {n} ' + 'z' * 100_000}
                for n in range(30)])
            db.archive_and_compact(target, [{'role': 'user', 'content': f'summary {generation}'}])
        for n in range(10):
            db.append_message(target, 'user' if n % 2 == 0 else 'assistant', f'LIVE{n} ' + 'l' * 100_000)
        reply = await owner.dispatch({'id': 1, 'method': 'session.history',
                                      'params': {'session_id': ref.session_id}})
        assert 'result' in reply, reply
        assert len(json.dumps(reply)) < 8 * 1024 * 1024
        messages = reply['result']['messages']
        assert messages[0]['archived_truncated'] is True and reply['result']['count'] == len(messages)
        assert [m['row_id'] for m in messages if not m.get('archived')] == db.get_active_message_ids(target)
        archived = [m['text'][:3] for m in messages if m.get('archived')]
        assert archived and set(archived) == {'G2 '}  # the newest generation, cut to what fits
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('write', ['rewind', 'compact', 'born-then-rewound', 'born-then-rewound-with-no-live-row'])
async def test_session_history_reads_again_when_a_live_row_retires_between_its_reads(tmp_path, monkeypatch, write):
    """A rewind or compaction landing after the lineage read and before the live one would bring a withdrawn
    reply back (archived), put the archived rows after the new summary, or show a turn born and rewound
    meanwhile; the reply is the one a read after the write gives."""
    async def handle(event):
        return 'ok'
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    try:
        from gateway.session_local_recovery import transcript_target
        target = transcript_target(authority, ref)
        first = db.append_message(target, 'user', 'question')
        db.append_message(target, 'assistant', 'reply')
        if write.endswith('no-live-row'):
            db.rewind_to_message(target, first)
        ids, read, born, fired = db.get_active_message_ids, db.get_messages_as_conversation, [], []

        def active_ids(*args, **kwargs):
            out = ids(*args, **kwargs)
            if write.startswith('born') and not born:  # a turn starts right after the live ids are read
                born.append(db.append_message(target, 'user', 'born then rewound'))
            return out

        def racing(*args, **kwargs):
            rows = read(*args, **kwargs)
            if kwargs.get('include_compacted') and not fired:
                fired.append(write)
                if write == 'rewind':
                    db.replace_messages(target, [{'role': 'user', 'content': 'question'}], archive_dropped=True)
                elif write == 'compact':
                    db.archive_and_compact(target, [{'role': 'user', 'content': 'compressed model context'}])
                else:
                    db.rewind_to_message(target, born[0])
            return rows
        monkeypatch.setattr(db, 'get_active_message_ids', active_ids)
        monkeypatch.setattr(db, 'get_messages_as_conversation', racing)
        raced = await owner.dispatch({'id': 1, 'method': 'session.history',
                                      'params': {'session_id': ref.session_id}})
        fresh = await owner.dispatch({'id': 2, 'method': 'session.history',
                                      'params': {'session_id': ref.session_id}})
        assert fired and 'result' in raced, raced
        assert raced['result'] == fresh['result']
    finally:
        await owner.close()
        store.close_all_db_handles()
