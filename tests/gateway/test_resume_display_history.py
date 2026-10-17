"""A resume after a compression shows the whole display history: the turns the compression
summarized away come back marked ``archived``, and only the live rows stay undo/retry targets. An undone
turn stays gone, a merged prompt shows once, and the archived part of the reply is bounded.

Real store, authority, local session, compaction verbs and ``AuthorityConnection.resume``; the runner
is a stand-in with no cached agent, and the resume ``info`` block is stubbed out.
"""
from types import SimpleNamespace

import pytest

from agent.context_compressor import SUMMARY_PREFIX

SUMMARY = SUMMARY_PREFIX + '\nearlier turns summarized'


async def _local_session(tmp_path, monkeypatch):
    from gateway import run, session_local
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from hermes_state_local import local_receipt
    monkeypatch.setattr(run, '_load_gateway_config', dict)
    monkeypatch.setattr(session_local, 'local_session_info', lambda *a: {})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda *a, **k: None, _evict_cached_agent=lambda *a, **k: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='first')
    actor = Principal('owner', 'fixture', frozenset({'session:create', 'session:read', 'session:control'}), 'sock')
    ref = session_local.create_local_session(authority, actor, dict(
        request_id='resume', source='gui', cwd=str(tmp_path), model='frozen', toolsets=[]))
    return authority, actor, ref, local_receipt(authority.db, ref.session_id)['entry']['session_id']


async def _resume(authority, actor, ref, **params):
    from gateway.session_controls import AuthorityConnection
    conn = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    conn.actor = actor
    authority.events[actor.transport_id] = object()
    return await conn.resume(ref, {'session_id': ref.session_id, **params})


def _append(db, sid, *texts):
    for role, text in zip(('user', 'assistant') * len(texts), texts):
        db.append_message(sid, role, text)


def _compact(db, physical, mode):
    compacted = [{'role': 'user', 'content': SUMMARY}, {'role': 'user', 'content': 'TAIL_Q'},
                 {'role': 'assistant', 'content': 'TAIL_A'}]
    if mode == 'in_place':
        db.archive_and_compact(physical, compacted, tail_count=2)
        return physical
    assert db.try_acquire_compression_lock(physical, 'fixture')
    try:
        db.publish_compression_child(parent_session_id=physical, child_session_id='child', source='gui',
                                     messages=compacted, compression_lock_holder='fixture')
    finally:
        db.release_compression_lock(physical, 'fixture')
    return 'child'


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['in_place', 'rotated'])
async def test_resume_shows_the_turns_a_compression_archived(tmp_path, monkeypatch, mode):
    authority, actor, ref, physical = await _local_session(tmp_path, monkeypatch)
    db = authority.db
    _append(db, physical, 'OLD_Q', 'OLD_A', 'REWOUND_Q', 'REWOUND_A')
    db.rewind_to_message(physical, next(m['id'] for m in db.get_messages(physical) if m['content'] == 'REWOUND_Q'))
    _append(db, physical, 'TAIL_Q', 'TAIL_A')
    tip = _compact(db, physical, mode)
    rows = (await _resume(authority, actor, ref))['messages']
    # The earlier turns are shown and marked archived; the carried tail appears once; a rewound turn stays out.
    assert [(r['content'], bool(r.get('archived'))) for r in rows if r['content'] != SUMMARY] == [
        ('OLD_Q', True), ('OLD_A', True), ('TAIL_Q', False), ('TAIL_A', False)]
    # The rows not archived are exactly the live transcript, each with its durable row id.
    assert sorted(r['row_id'] for r in rows if not r.get('archived')) == [m['id'] for m in db.get_messages(tip)]


@pytest.mark.asyncio
async def test_undo_and_retry_choose_only_among_live_turns(tmp_path, monkeypatch):
    from hermes_cli.gateway_chat_commands import _rewind_to
    authority, actor, ref, physical = await _local_session(tmp_path, monkeypatch)
    db = authority.db
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
    db = authority.db
    _append(db, physical, 'OLD_Q', 'OLD_A', 'TAIL_Q', 'TAIL_A')
    tip = _compact(db, physical, 'rotated')
    handle = authority._handle(ref)
    await mutate_session(authority, actor, ref, {
        'session_id': ref.session_id, 'request_id': 'undo', 'operation': 'rewind',
        'expected_revision': handle.revision, 'expected_generation': handle.execution_generation,
        'payload': {'target_message_id': next(m['id'] for m in db.get_messages(tip) if m['content'] == 'TAIL_Q')}})
    rows = (await _resume(authority, actor, ref))['messages']
    assert [(r['content'], bool(r.get('archived'))) for r in rows] == [
        ('OLD_Q', True), ('OLD_A', True), (SUMMARY, False)]


@pytest.mark.asyncio
async def test_a_micro_compaction_merge_shows_its_prompts_once(tmp_path, monkeypatch):
    """A merged prompt row (``model_only``) stays a live undo target but is not painted next to its originals."""
    from hermes_cli.gateway_chat_commands import _rewind_to
    authority, actor, ref, physical = await _local_session(tmp_path, monkeypatch)
    db = authority.db
    _append(db, physical, 'Q1', 'A1', 'Q2', 'A2', 'Q3', 'A3')
    db.archive_and_compact(physical, [
        {'role': 'user', 'content': 'Q1\n\nQ2', 'display_metadata': {'model_only': True}},
        {'role': 'assistant', 'content': SUMMARY}, {'role': 'user', 'content': 'Q3'},
        {'role': 'assistant', 'content': 'A3'}], tail_count=2)
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
    db = authority.db
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
    assert sorted(r['row_id'] for r in rows if not r.get('archived')) == [m['id'] for m in db.get_messages(tip)]
    archived = [r['content'][:4] for r in rows if r.get('archived') and r['role'] == 'user']
    assert archived and archived[-1] == 'Q39 ' and 'Q0 x' not in archived  # the newest are kept
