"""``session.transcript``: a stored conversation read without attaching (#150091), its compaction handoffs as rows."""
import base64
import json
import os
from types import SimpleNamespace

import pytest

from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import NewMessage
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_read import conversation
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.test_session_mutation_model_busy import _owner


async def _call(connection, method, **params):
    return await connection.dispatch({'id': 1, 'method': method, 'params': params})


async def _pages(connection, session_id, limit, between=None, kinds=('message',), resets=None, **extra):
    """Every page of *session_id*'s transcript as a list of row-text lists (rows of *kinds*), as a reader
    keeps them: a ``reset`` reply drops what it held. *between* runs after the first page. Oldest order
    ends at the first empty page, latest order at a null cursor; *resets* collects each reply's ``reset``."""
    pages, cursor, latest = [], None, extra.get('order') == 'latest'
    while True:
        reply = await _call(connection, 'session.transcript', session_id=session_id, limit=limit,
                            **extra, **({'cursor': cursor} if cursor else {}))
        assert 'result' in reply, reply
        result = reply['result']
        if resets is not None:
            resets.append(result['reset'])
        assert result['reset'] is (cursor is None) or not latest, result
        if result['reset']:
            pages = []
        if not latest and not result['rows']:
            assert result['next_cursor'], result
            return pages
        pages.append([row.get('text') for row in result['rows'] if row['row_kind'] in kinds])
        cursor = result['next_cursor']
        if between is not None:
            between()
            between = None
        if cursor is None:
            return pages


def _flat(pages):
    return [text for page in pages for text in page]


def _summary(body, compacted):
    from agent.context_compressor import SUMMARY_PREFIX, _build_recovery_footer, _build_verbatim_user_section
    return (f"{SUMMARY_PREFIX}\n{body}"
            + _build_verbatim_user_section([{'role': 'user', 'content': 'question 1'}])
            + _build_recovery_footer('sid', compacted))


def _append(db, session_id, role, content, **extra):
    """One stored message row, as an append writes it."""
    seed_messages(db, session_id, {'role': role, 'content': content, **extra})


def _compact(db, target, messages, tail_count):
    """An in-place compaction of *target* to *messages* (live dicts), the last *tail_count* of them carried."""
    db.messages.compact(target, [NewMessage.from_live(message) for message in messages],
                        watermark=db.messages.active_watermark(target), carried_tail_count=tail_count)


@pytest.mark.asyncio
async def test_transcript_pages_both_ways_across_a_compaction_and_never_attach(tmp_path, monkeypatch):
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local_recovery import transcript_target

    async def handle(event):
        return 'ok'
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    try:
        target = transcript_target(authority, ref)
        for i in (1, 2, 3):
            _append(db, target, 'user', f'question {i}')
            _append(db, target, 'assistant', f'answer {i}')

        def compact():
            tail = [{'role': 'user', 'content': 'question 3'}, {'role': 'assistant', 'content': 'answer 3'}]
            _compact(db, target, [{'role': 'user', 'content': _summary('## Goal\nship it', 4)}, *tail], len(tail))

        everything = ['question 1', 'answer 1', 'question 2', 'answer 2', 'question 3', 'answer 3']
        # The first page is read before the compaction: the next read starts over, each message once, in order.
        resets = []
        assert _flat(await _pages(owner, ref.session_id, 2, between=compact, resets=resets)) == everything
        assert resets[:2] == [True, True] and not any(resets[2:]), resets
        newest = await _pages(owner, ref.session_id, 2, order='latest')
        assert newest[0] == ['question 3', 'answer 3']
        assert _flat(newest[::-1]) == everything
        # The handoff is a row at its position, without the parts written for the model.
        rows = (await _call(owner, 'session.transcript', session_id=ref.session_id))['result']['rows']
        assert [(row['row_kind'], row.get('text')) for row in rows] == [
            ('message', 'question 1'), ('message', 'answer 1'), ('message', 'question 2'), ('message', 'answer 2'),
            ('compaction', '## Goal\nship it'), ('message', 'question 3'), ('message', 'answer 3')], rows
        [handoff] = [row for row in rows if row['row_kind'] == 'compaction']
        assert handoff['role'] == 'system' and handoff['compacted_messages'] == 4 and handoff['timestamp'] > 0
        assert authority.sessions[ref.session_id].subscribers == {}
        assert ref.session_id not in owner.subscriptions

        first = await _call(owner, 'session.transcript', session_id=ref.session_id, limit=2)
        cursor = first['result']['next_cursor']
        for params in ({'session_id': 'another-session', 'cursor': cursor},
                       {'session_id': ref.session_id, 'cursor': cursor, 'order': 'latest'},
                       {'session_id': ref.session_id, 'cursor': base64.urlsafe_b64encode(b'[' * 100_000).decode()}):
            refused = await _call(owner, 'session.transcript', **params)
            assert refused['error']['data']['reason'] in {'invalid_params', 'not_found'}, refused
            if params['session_id'] == ref.session_id:
                assert refused['error']['data']['reason'] == 'invalid_params', refused
        other = AuthorityConnection(authority, object(), {'user_id': 'someone-else'})
        refused = await _call(other, 'session.transcript', session_id=ref.session_id)
        assert refused['error']['data']['reason'] == 'permission_denied', refused
        await other.close()
    finally:
        await owner.close()
        store.close_all_db_handles()


def _compact_carrying_tail(db, target, summary, tail_rows=2):
    """An in-place compaction as the compressor commits it: the held tail dicts (stored timestamp and
    ``message_uid``) carried after the summary, their originals superseded."""
    tail = conversation(db, target)[-tail_rows:]
    _compact(db, target, [{'role': 'user', 'content': summary}, *tail], len(tail))


@pytest.mark.asyncio
async def test_a_page_boundary_inside_the_carried_tail_never_repeats_a_row(tmp_path, monkeypatch):
    from agent.context_compressor import SUMMARY_PREFIX
    from gateway.session_local_recovery import transcript_target

    async def handle(event):
        return 'ok'
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    try:
        target = transcript_target(authority, ref)
        for i in (1, 2, 3):
            _append(db, target, 'user', f'question {i}')
            _append(db, target, 'assistant', f'answer {i}')
        resets = []
        pages = await _pages(owner, ref.session_id, 5, resets=resets, between=lambda: _compact_carrying_tail(
            db, target, f'{SUMMARY_PREFIX}\n## Goal\none'))
        # The first page ended inside the tail the compaction then carried: the next read starts over.
        assert resets[:2] == [True, True] and not any(resets[2:]), resets
        assert _flat(pages) == ['question 1', 'answer 1', 'question 2', 'answer 2', 'question 3', 'answer 3']
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_every_compaction_is_a_row_when_a_later_one_reuses_a_summary_carrier(tmp_path, monkeypatch):
    from agent.context_compressor import (
        SUMMARY_PREFIX, _MERGED_PRIOR_CONTEXT_HEADER, _MERGED_SUMMARY_DELIMITER, _SUMMARY_END_MARKER,
    )
    from gateway.session_local_recovery import transcript_target

    async def handle(event):
        return 'ok'
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    try:
        target = transcript_target(authority, ref)
        _append(db, target, 'user', 'question 1')
        _append(db, target, 'assistant', 'answer 1')
        _append(db, target, 'user', 'question 2', message_uid='q2')
        _append(db, target, 'assistant', 'answer 2')
        held = conversation(db, target)
        # The first compaction merges its summary into the carried prompt (the carrier keeps its uid) ...
        carrier = {**held[2], '_compressed_summary': True, 'content': (
            f"{_MERGED_PRIOR_CONTEXT_HEADER}\nquestion 2\n{_MERGED_SUMMARY_DELIMITER}\n{SUMMARY_PREFIX}\n"
            f"## Goal\nfirst\n{_SUMMARY_END_MARKER}")}
        _compact(db, target, [carrier, held[3]], 2)
        _append(db, target, 'user', 'question 3')
        _append(db, target, 'assistant', 'answer 3')
        # ... and the second unwraps it again: the prompt's own identity, with a summary of its own before it.
        held = conversation(db, target)
        prompt = {**held[0], 'content': 'question 2', '_compressed_summary': False}
        _compact(db, target, [{'role': 'user', 'content': f'{SUMMARY_PREFIX}\n## Goal\nsecond'}, prompt, *held[1:]],
                 len(held))
        pages = await _pages(owner, ref.session_id, 3, kinds=('compaction',))
        assert _flat(pages) == ['## Goal\nsecond', '## Goal\nfirst'], pages  # each at its position
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_page_stays_under_its_byte_cap_and_shows_images_as_placeholders(tmp_path, monkeypatch):
    from gateway.session_local_recovery import transcript_target
    from gateway.session_stored_reads import PAGE_BYTES

    async def handle(event):
        return 'ok'
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    try:
        target = transcript_target(authority, ref)
        image = 'data:image/png;base64,' + base64.b64encode(os.urandom(1_500_000)).decode()
        for i in range(5):
            _append(db, target, 'user', [{'type': 'text', 'text': f'look {i}'},
                                         {'type': 'image_url', 'image_url': {'url': image}}])
            _append(db, target, 'assistant', f'reply {i} ' + 'w' * 700_000)
        cursor, texts = None, []
        while True:
            reply = await _call(owner, 'session.transcript', session_id=ref.session_id, limit=10,
                                **({'cursor': cursor} if cursor else {}))
            assert len(json.dumps(reply)) < PAGE_BYTES + 64 * 1024, len(json.dumps(reply))
            texts += [row['text'].split()[0:2] for row in reply['result']['rows']]
            cursor = reply['result']['next_cursor']
            if not reply['result']['rows']:
                break
        assert [t for t in texts if t[0] == 'look'] == [['look', str(i)] for i in range(5)]
        assert len([t for t in texts if t[0] == 'reply']) == 5
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_preview_of_an_ended_conversation_leaves_the_live_list_alone(tmp_path, monkeypatch):
    from gateway.session_local_recovery import transcript_target

    async def handle(event):
        return 'ok'
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    try:
        target = transcript_target(authority, ref)
        _append(db, target, 'user', 'old question')
        db.sessions.end(target, 'cli_close')
        del authority.sessions[ref.session_id]  # as a restart leaves an ended conversation
        before = (await _call(owner, 'session.list'))['result']['sessions']
        assert _flat(await _pages(owner, ref.session_id, 10)) == ['old question']
        assert (await _call(owner, 'session.list'))['result']['sessions'] == before == []
        assert authority.sessions[ref.session_id].subscribers == {}
    finally:
        await owner.close()
        store.close_all_db_handles()


def _legacy_owner(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store.persistence
    seed_session(db, 'old-tui', 'tui', model='m', cwd=str(tmp_path))
    _append(db, 'old-tui', 'user', 'before compression')
    _append(db, 'old-tui', 'assistant', 'old answer')
    db.lineage.publish_compression_child(
        'old-tui', CompressionChild(session_id='tui-tip', source='tui', model='m'),
        [NewMessage(role='user', content='after compression'), NewMessage(role='assistant', content='new answer')],
        require_compression_lease=False)
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    return store, db, runner


def _native(authority, tmp_path, user):
    from gateway.session_controls import AuthorityConnection
    return AuthorityConnection(authority, object(), {'provider': 'local', 'user_id': user,
        'profile_id': str(tmp_path), 'instance_id': 'owner'})


@pytest.mark.asyncio
async def test_an_old_compressed_transcript_reads_across_its_lineage_without_adoption(tmp_path, monkeypatch):
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    import storage_limits

    store, db, runner = _legacy_owner(tmp_path, monkeypatch)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='owner')
    native = _native(authority, tmp_path, 'native')
    remote = AuthorityConnection(authority, object(), {'provider': 'remote', 'user_id': 'remote',
        'profile_id': str(tmp_path), 'instance_id': 'owner'}, operator=True)
    try:
        listed = await _call(native, 'session.list', scope='stored')
        [row] = listed['result']['sessions']
        before = raw_holder(db)._conn.total_changes
        assert _flat(await _pages(native, row['session_id'], 3)) == [
            'before compression', 'old answer', 'after compression', 'new answer']
        assert raw_holder(db)._conn.total_changes == before  # nothing adopted, nothing written
        assert authority.sessions == {}
        # Past the resume's guard a rotated lineage pages its current segment only.
        monkeypatch.setattr(storage_limits, 'resolved_max_resume_messages', lambda: 2)
        assert _flat(await _pages(native, row['session_id'], 3)) == ['after compression', 'new answer']
        refused = await _call(remote, 'session.transcript', session_id=row['session_id'])
        assert refused['error']['data']['reason'] == 'not_found', refused
    finally:
        await native.close()
        await remote.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_an_old_row_is_judged_at_its_tip_whichever_lineage_id_is_asked(tmp_path, monkeypatch):
    """Another principal's stamp on the tip refuses the list, resume and every read of the lineage,
    the root's id included (the stored list and resume judge the tip)."""
    from gateway.session_authority import initialize_session_authority

    store, db, runner = _legacy_owner(tmp_path, monkeypatch)
    raw_holder(db)._execute_write(lambda conn: conn.execute("UPDATE sessions SET user_id='alice' WHERE id='tui-tip'"))
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='owner')
    bob = _native(authority, tmp_path, 'bob')
    try:
        assert (await _call(bob, 'session.list', scope='stored'))['result']['sessions'] == []
        for sid in ('tui-tip', 'old-tui'):
            refused = await _call(bob, 'session.transcript', session_id=sid)
            assert refused['error']['data']['reason'] == 'not_found', (sid, refused)
    finally:
        await bob.close()
        store.close_all_db_handles()
