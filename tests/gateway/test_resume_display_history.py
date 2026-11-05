"""A resume after a compression shows the whole display history: the turns the compression
summarized away come back marked ``archived``, and only the live rows stay undo/retry targets. An undone
turn stays gone, a merged prompt shows once, and the archived part of the reply is bounded.

Real store, authority, local session, compaction verbs and ``AuthorityConnection.resume``; the runner
is a stand-in with no cached agent, and the resume ``info`` block is stubbed out.
"""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from agent.context_compressor import SUMMARY_PREFIX
from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import NewMessage
from tests.fixtures.storage_read import session_row, transcript_rows
from tests.fixtures.storage_seed import seed_messages

SUMMARY = SUMMARY_PREFIX + '\nearlier turns summarized'


async def _local_session(tmp_path, monkeypatch):
    from gateway import run, session_local
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import SessionActor
    monkeypatch.setattr(run, '_load_gateway_config', dict)
    monkeypatch.setattr(session_local, 'local_session_info', lambda *a: {})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store.persistence, adapters={}, _draining=False,
                             _cached_agent_for=lambda *a, **k: None, _evict_cached_agent=lambda *a, **k: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='first')
    actor = SessionActor('owner', 'fixture', frozenset({'session:create', 'session:read', 'session:control'}), 'sock')
    ref = session_local.create_local_session(authority, actor, dict(
        request_id='resume', source='gui', cwd=str(tmp_path), model='frozen', toolsets=[]))
    return authority, actor, ref, authority.persistence.bindings.receipt(ref.session_id).entry['session_id']


async def _resume(authority, actor, ref, **params):
    from gateway.session_controls import AuthorityConnection
    conn = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    conn.actor = actor
    authority.events[actor.transport_id] = object()
    return await conn.resume(ref, {'session_id': ref.session_id, **params})


def _append(db, sid, *texts):
    for role, text in zip(('user', 'assistant') * len(texts), texts):
        seed_messages(db, sid, {'role': role, 'content': text})


def _compact(db, physical, mode):
    compacted = [{'role': 'user', 'content': SUMMARY}, {'role': 'user', 'content': 'TAIL_Q'},
                 {'role': 'assistant', 'content': 'TAIL_A'}]
    if mode == 'in_place':
        db.messages.compact(physical, [NewMessage.from_live(m) for m in compacted], carried_tail_count=2)
        return physical
    assert db.coordination.try_acquire_compression_lock(session_id=physical, holder='fixture')
    try:
        db.lineage.publish_compression_child(physical, CompressionChild(session_id='child', source='gui'),
                                             [NewMessage.from_live(m) for m in compacted],
                                             compression_lock_holder='fixture')
    finally:
        db.coordination.release_compression_lock(session_id=physical, holder='fixture')
    return 'child'


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['in_place', 'rotated'])
async def test_resume_shows_the_turns_a_compression_archived(tmp_path, monkeypatch, mode):
    authority, actor, ref, physical = await _local_session(tmp_path, monkeypatch)
    db = authority.persistence
    _append(db, physical, 'OLD_Q', 'OLD_A', 'REWOUND_Q', 'REWOUND_A')
    target = next(m['id'] for m in transcript_rows(db, physical) if m['content'] == 'REWOUND_Q')
    for m in transcript_rows(db, physical):
        if m['id'] >= target:
            db.messages.deactivate_message(physical, m['id'])
    _append(db, physical, 'TAIL_Q', 'TAIL_A')
    tip = _compact(db, physical, mode)
    rows = (await _resume(authority, actor, ref))['messages']
    # The earlier turns are shown and marked archived; the carried tail appears once; a rewound turn stays out.
    # The summary sits before the tail it precedes in the store, in both modes.
    assert [(r['content'], bool(r.get('archived'))) for r in rows] == [
        ('OLD_Q', True), ('OLD_A', True), (SUMMARY, False), ('TAIL_Q', False), ('TAIL_A', False)]
    # The rows not archived are exactly the live transcript, each with its durable row id.
    assert [r['row_id'] for r in rows if not r.get('archived')] == [m['id'] for m in transcript_rows(db, tip)]


@pytest.mark.asyncio
async def test_cli_undo_prefix_keeps_the_summary_after_a_rotated_compression(tmp_path, monkeypatch):
    """``GatewaySessionStore.rewind_user_turn`` and ``/compress`` take the live rows as the CLI's history:
    the prefix before the newest user turn holds the summary the store keeps active (review of #140971)."""
    from agent.context_compressor import history_before_user_originated_turn
    from session_rules.turns import user_indices
    authority, actor, ref, physical = await _local_session(tmp_path, monkeypatch)
    db = authority.persistence
    _append(db, physical, 'OLD_Q', 'OLD_A', 'TAIL_Q', 'TAIL_A')
    _compact(db, physical, 'rotated')
    _append(db, 'child', 'NEW_Q', 'NEW_A')
    live = [r for r in (await _resume(authority, actor, ref))['messages'] if not r.get('archived')]
    prefix, turn = history_before_user_originated_turn(live, user_indices(live)[-1])
    assert turn['content'] == 'NEW_Q'
    assert [m['content'] for m in prefix] == [SUMMARY, 'TAIL_Q', 'TAIL_A']


@pytest.mark.asyncio
async def test_undo_and_retry_choose_only_among_live_turns(tmp_path, monkeypatch):
    from hermes_cli.gateway_chat_commands import _rewind_to
    authority, actor, ref, physical = await _local_session(tmp_path, monkeypatch)
    db = authority.persistence
    _append(db, physical, 'OLD_Q', 'OLD_A', 'TAIL_Q', 'TAIL_A')
    _compact(db, physical, 'in_place')
    _append(db, physical, 'NEW_Q', 'NEW_A')
    snapshot = await _resume(authority, actor, ref)
    live = {r['content']: r['row_id'] for r in snapshot['messages'] if not r.get('archived')}
    assert _rewind_to(3)(snapshot) is None  # two live turns: the archived OLD_Q is never a target
    assert _rewind_to(2)(snapshot) == {'target_message_id': live['TAIL_Q']}
    assert _rewind_to(1, retry=True)(snapshot) == {'target_message_id': live['NEW_Q']}


@pytest.mark.asyncio
async def test_an_undone_turn_stays_gone_after_a_rotated_compression(tmp_path, monkeypatch):
    """The rewind removes the child's copies; the parent's copies of the same turn must not come back."""
    from gateway.session_mutations import mutate_session
    authority, actor, ref, physical = await _local_session(tmp_path, monkeypatch)
    db = authority.persistence
    _append(db, physical, 'OLD_Q', 'OLD_A', 'TAIL_Q', 'TAIL_A')
    tip = _compact(db, physical, 'rotated')
    handle = authority._handle(ref)
    await mutate_session(authority, actor, ref, {
        'session_id': ref.session_id, 'request_id': 'undo', 'operation': 'rewind',
        'expected_revision': handle.revision, 'expected_generation': handle.execution_generation,
        'payload': {'target_message_id': next(m['id'] for m in transcript_rows(db, tip) if m['content'] == 'TAIL_Q')}})
    rows = (await _resume(authority, actor, ref))['messages']
    assert [(r['content'], bool(r.get('archived'))) for r in rows] == [
        ('OLD_Q', True), ('OLD_A', True), (SUMMARY, False)]


@pytest.mark.asyncio
async def test_a_micro_compaction_merge_shows_its_prompts_once(tmp_path, monkeypatch):
    """A merged prompt row (``model_only``) stays a live undo target but is not painted next to its originals."""
    from hermes_cli.gateway_chat_commands import _rewind_to
    authority, actor, ref, physical = await _local_session(tmp_path, monkeypatch)
    db = authority.persistence
    _append(db, physical, 'Q1', 'A1', 'Q2', 'A2', 'Q3', 'A3')
    db.messages.compact(physical, [NewMessage.from_live(m) for m in [
        {'role': 'user', 'content': 'Q1\n\nQ2', 'display_metadata': {'model_only': True}},
        {'role': 'assistant', 'content': SUMMARY}, {'role': 'user', 'content': 'Q3'},
        {'role': 'assistant', 'content': 'A3'}]], carried_tail_count=2)
    snapshot = await _resume(authority, actor, ref, display_projection=True)
    assert [m['text'] for m in snapshot['display_messages'] if m['role'] == 'user'] == ['Q1', 'Q2', 'Q3']
    merged = next(r for r in snapshot['messages'] if r['content'] == 'Q1\n\nQ2')
    assert not merged.get('archived') and _rewind_to(2)(snapshot) == {'target_message_id': merged['row_id']}


@pytest.mark.asyncio
async def test_a_long_archived_history_keeps_the_resume_reply_bounded(tmp_path, monkeypatch):
    """Archived history is cut to the newest rows that fit what the live rows leave of a byte budget, so the
    reply (with Ink's ``display_messages``) stays under the gateway clients' 8 MiB frame cap whenever main's
    live-only reply does, large live rows included; live rows all stay."""
    import json
    authority, actor, ref, physical = await _local_session(tmp_path, monkeypatch)
    db = authority.persistence
    for n in range(40):
        _append(db, physical, f'Q{n} ' + 'x' * 120_000, f'A{n}')
    _append(db, physical, 'TAIL_Q', 'TAIL_A')
    tip = _compact(db, physical, 'in_place')
    for n in range(24):  # 2.3 MiB of live rows: main's reply (twice that for Ink) fits the frame
        _append(db, tip, f'LIVE{n} ' + 'l' * 100_000, f'R{n}')
    reply = await _resume(authority, actor, ref, display_projection=True)
    assert len(json.dumps(reply)) < 8 * 1024 * 1024
    rows = reply['messages']
    assert rows[0]['archived_truncated'] and rows[-1]['content'] == 'R23'
    assert [r['row_id'] for r in rows if not r.get('archived')] == [m['id'] for m in transcript_rows(db, tip)]
    archived = [r['content'][:4] for r in rows if r.get('archived') and r['role'] == 'user']
    assert archived and archived[-1] == 'Q39 ' and 'Q0 x' not in archived  # the newest are kept


@pytest.mark.asyncio
async def test_a_row_appended_during_prepare_counts_against_the_archive_budget(tmp_path, monkeypatch):
    """A live row landing between the off-loop prepare and the locked read shrinks what the archive may
    use, as a fresh read would: the raced reply is no larger than the fresh one (review of #140971)."""
    import json
    from gateway import session_local_recovery as recovery
    authority, actor, ref, physical = await _local_session(tmp_path, monkeypatch)
    db = authority.persistence
    for n in range(28):  # ~2.8 MiB of history, then summarized away in place (all archived)
        seed_messages(db, physical, {'role': 'user' if n % 2 == 0 else 'assistant', 'content': f'{n}:' + 'h' * 100 * 1024})
    db.messages.compact(physical, [NewMessage.from_live({'role': 'user', 'content': SUMMARY})])
    real, raced = recovery.prepare_history, []

    async def prepare_then_append(authority_, ref_):
        prepared = await real(authority_, ref_)
        if not raced:
            raced.extend(seed_messages(db, physical, {'role': 'assistant', 'content': 'n' * 2 * 1024 * 1024}))
        return prepared
    monkeypatch.setattr(recovery, 'prepare_history', prepare_then_append)
    reply = await _resume(authority, actor, ref, display_projection=True)
    fresh = await _resume(authority, actor, ref, display_projection=True)
    assert raced and reply['messages'][-1]['content'].startswith('nnn')
    assert reply['messages'][0].get('archived_truncated')
    assert len(json.dumps(reply['messages'])) <= len(json.dumps(fresh['messages']))
    assert len(json.dumps(reply)) < 8 * 1024 * 1024


@pytest.mark.asyncio
async def test_a_delete_during_the_resume_prepare_answers_not_found(tmp_path, monkeypatch):
    """``attach`` awaits the lineage read between its authorize and the live-entry lookup: a delete committed
    by another connection in that window answers ``not_found``, never an internal error (review of #140971)."""
    from gateway import session_local_recovery as recovery
    from gateway.session_controls import AuthorityConnection
    authority, _actor, ref, physical = await _local_session(tmp_path, monkeypatch)
    authority.runner.session_authority = authority  # one process owns both writes of a delete
    seed_messages(authority.persistence, physical, {'role': 'user', 'content': 'hello'})

    def connection():
        return AuthorityConnection(authority, object(), {
            'provider': 'local', 'user_id': 'owner', 'profile_id': 'fixture', 'instance_id': authority.instance_id,
            'capabilities': ['session:create', 'session:read', 'session:control']})
    real, deleted = recovery.prepare_history, []

    async def prepare_then_delete(authority_, ref_):
        prepared = await real(authority_, ref_)
        if not deleted:
            row = session_row(authority.persistence, ref.session_id)
            deleted.append(await connection().dispatch({'id': 9, 'method': 'session.mutate', 'params': {
                'session_id': ref.session_id, 'request_id': 'delete-1', 'expected_revision': row['runtime_revision'],
                'expected_generation': row['runtime_generation'], 'operation': 'delete', 'payload': {}}}))
        return prepared
    monkeypatch.setattr(recovery, 'prepare_history', prepare_then_delete)
    reply = await connection().dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': ref.session_id}})
    assert 'result' in deleted[0], deleted  # the delete really committed inside the window
    assert reply.get('error', {}).get('message') == 'not_found', reply


@pytest.mark.asyncio
async def test_a_committed_but_unretired_delete_during_the_prepare_answers_not_found(tmp_path, monkeypatch):
    """The delete's commit lands off the loop before its retirement runs: the live entry is still there,
    so the re-authorize passes, but the row is gone. The resume answers ``not_found``, not an internal error."""
    from gateway import session_local_recovery, session_mutations
    from gateway.session_controls import AuthorityConnection
    authority, _actor, ref, physical = await _local_session(tmp_path, monkeypatch)
    authority.runner.session_authority = authority  # one process owns both writes of a delete
    seed_messages(authority.persistence, physical, {'role': 'user', 'content': 'hello'})

    def connection():
        return AuthorityConnection(authority, object(), {
            'provider': 'local', 'user_id': 'owner', 'profile_id': 'fixture', 'instance_id': authority.instance_id,
            'capabilities': ['session:create', 'session:read', 'session:control']})
    from gateway.session_persistence import authority_persistence
    from persistence.authority.runtime import Delete
    runtime = authority_persistence(authority).runtime
    prepare, write, deletes = session_local_recovery.prepare_history, runtime.mutate, []
    committed, release = threading.Event(), threading.Event()

    def write_then_hold(*args, **kwargs):
        result = write(*args, **kwargs)
        if isinstance(kwargs.get('command'), Delete):
            committed.set()
            assert release.wait(10), 'the delete was never released'
        return result

    async def prepare_then_delete(authority_, ref_):
        prepared = await prepare(authority_, ref_)
        row = session_row(authority.persistence, ref.session_id)
        deletes.append(asyncio.ensure_future(connection().dispatch({'id': 9, 'method': 'session.mutate', 'params': {
            'session_id': ref.session_id, 'request_id': 'delete-1', 'operation': 'delete', 'payload': {},
            'expected_revision': row['runtime_revision'], 'expected_generation': row['runtime_generation']}})))
        assert await asyncio.to_thread(committed.wait, 10)
        assert session_row(authority.persistence, ref.session_id) is None and ref.session_id in authority.sessions
        release.set()  # the commit has landed; the retirement runs once the loop gets to it
        return prepared
    monkeypatch.setattr(runtime, 'mutate', write_then_hold)
    monkeypatch.setattr(session_local_recovery, 'prepare_history', prepare_then_delete)
    try:
        reply = await connection().dispatch({'id': 1, 'method': 'session.resume',
                                             'params': {'session_id': ref.session_id}})
    finally:
        release.set()
    assert 'result' in await deletes[0], deletes
    assert reply.get('error', {}).get('message') == 'not_found', reply
    assert ref.session_id not in authority.sessions


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['in_place', 'rotated'])
async def test_rollback_turn_rewind_never_targets_an_archived_turn(tmp_path, monkeypatch, mode):
    """/rollback's history rewind picks the newest live user turn; with none left (the last one undone after
    a compression) it rewinds nothing instead of aiming the owner's rewind at an archived row it refuses."""
    from gateway.session_mutations import mutate_session
    from gateway.session_tui_ops import _rewind_last_turn
    authority, actor, ref, physical = await _local_session(tmp_path, monkeypatch)
    db = authority.persistence
    _append(db, physical, 'OLD_Q', 'OLD_A', 'TAIL_Q', 'TAIL_A')
    tip = _compact(db, physical, mode)
    handle = authority._handle(ref)
    await mutate_session(authority, actor, ref, {
        'session_id': ref.session_id, 'request_id': 'undo', 'operation': 'rewind',
        'expected_revision': handle.revision, 'expected_generation': handle.execution_generation,
        'payload': {'target_message_id': next(m['id'] for m in transcript_rows(db, tip) if m['content'] == 'TAIL_Q')}})
    live = [m['content'] for m in transcript_rows(db, tip)]
    assert await _rewind_last_turn(SimpleNamespace(authority=authority, actor=actor), ref) == 0
    assert [m['content'] for m in transcript_rows(db, tip)] == live == [SUMMARY]
