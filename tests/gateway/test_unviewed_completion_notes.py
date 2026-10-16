"""An unviewed local completion becomes a note through the gateway FIFO, never a lost or early one.

Regression for the #134889 LC-7 review. The completion is admitted durably like any producer input (a
coalesced batch is one admission, found by any of its identities); when that admission comes up for
execution with no viewer attached, the executor writes the note (no model call) and settles the
admission. Busy sessions, queue order, owner restarts and lost acknowledgements are the admission
contract's, not a retry budget's.
"""
import pytest

from tests.gateway.test_completion_admission import pending
from tests.gateway.test_session_end_delegations import _PROCESS, _admissions, _deliver, _local, _notes
from tools import async_delegation as delegation


@pytest.fixture(autouse=True)
def _no_delegations():
    delegation._reset_for_tests()
    yield
    delegation._reset_for_tests()


def _human(authority, ref, request_id, text):
    from hermes_state_runtime import admit_session_input
    return admit_session_input(authority.db, epoch=authority.epoch, principal_id='test-owner',
                               session_id=ref.session_id, request_id=request_id, payload={'text': text})


def _turns(runner, authority, target, marker='RESULT_'):
    """Record every turn the drain runs, with the notes the transcript held when it started."""
    seen = []

    async def handle(event):
        seen.append((event.text, len(_texts(authority, target, marker))))
        return 'ok'
    runner._handle_message = handle
    return seen


def _texts(authority, target, marker):
    return [m for m in authority.db.get_messages(target) if marker in str(m.get('content'))]


def _target(runner, authority, ref):
    return runner.session_store.lookup_by_session_key(authority.sessions[ref.session_id].route).session_id


def _process_event(authority, ref, target, name='proc_busy'):
    """What the process watcher hands ``_deliver_completion_notification`` for a pinned completion."""
    return {'type': 'completion', 'session_id': name, 'started_at': 1700000000.0, 'command': 'make',
            'exit_code': 0, 'output': f'RESULT_{name}', 'session_key': authority.sessions[ref.session_id].route,
            'parent_session_id': target}


@pytest.mark.asyncio
async def test_busy_session_keeps_the_completion_durably_until_its_turn(tmp_path, monkeypatch):
    """No ``session_busy`` refusal (so no delivery-failure budget): the completion is committed behind
    the running input and becomes a note when it reaches the head, with notifications off too."""
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        target = _target(runner, authority, ref)
        _human(authority, ref, 'human', 'FIRST')
        await _deliver(runner, authority, ref, _PROCESS)
        evt = _process_event(authority, ref, target)
        assert await runner._deliver_completion_notification(f'[finished] RESULT_{evt["session_id"]}', evt) is True
        assert [r['status'] for r in _admissions(authority, ref)] == ['queued', 'queued']
        assert _notes(authority, target) == []

        seen = _turns(runner, authority, target)
        await authority._drain(ref)
        assert seen == [('FIRST', 0)], 'the human input runs first and sees no note yet'
        assert len(_notes(authority, target)) == 2
        assert [r['outcome'] for r in _admissions(authority, ref)] == ['completed', 'completed']
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_input_admitted_after_the_completion_sees_its_note(tmp_path, monkeypatch):
    """The idle decision is the FIFO's: a human input committed after the completion runs after its note."""
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        target = _target(runner, authority, ref)
        await _deliver(runner, authority, ref, _PROCESS)
        _human(authority, ref, 'human', 'AFTER')
        seen = _turns(runner, authority, target)
        await authority._drain(ref)
        assert seen == [('AFTER', 1)]
        assert len(_notes(authority, target)) == 1
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_viewer_back_before_execution_gets_the_turn(tmp_path, monkeypatch):
    """Whether it is a note is decided when the admission runs, not when it was committed."""
    runner, authority, ref, actor = await _local(tmp_path, monkeypatch)
    try:
        target = _target(runner, authority, ref)
        await _deliver(runner, authority, ref, _PROCESS)
        authority.sessions[ref.session_id].subscribers['back'] = actor
        seen = _turns(runner, authority, target)
        await authority._drain(ref)
        assert [text for text, _ in seen] == [f'[finished] RESULT_{_PROCESS[1]}']
        assert [r['outcome'] for r in _admissions(authority, ref)] == ['completed']
        assert _notes(authority, target) == []
        assert not [m for m in authority.db.get_messages(target)
                    if (m.get('display_metadata') or {}).get('completion_admission')]
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_note_keeps_the_cached_agent(tmp_path, monkeypatch):
    """A plain disconnect keeps the cached agent; the owner's own note re-baselines its message count,
    so the next turn reuses it instead of reading the note as an outside write and rebuilding."""
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        target, route = _target(runner, authority, ref), authority.sessions[ref.session_id].route
        agent, before = object(), authority.db.get_session(target)['message_count']
        runner._agent_cache[route] = (agent, 'sig', before, target)
        await _deliver(runner, authority, ref, _PROCESS)
        await authority._drain(ref)
        assert len(_notes(authority, target)) == 1
        after = authority.db.get_session(target)['message_count']
        assert after == before + 1
        assert runner._agent_cache[route] == (agent, 'sig', after, target), 'the next turn would rebuild the agent'
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_watch_notice_without_a_viewer_is_a_note(tmp_path, monkeypatch):
    from gateway.platforms.event import MessageEvent
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        target = _target(runner, authority, ref)
        live = authority.sessions[ref.session_id]
        for kind in ('watch_match', 'watch_disabled'):
            identity = f'["{kind}","proc_watch","evt-1"]'
            event = MessageEvent(text=f'[watch] RESULT_{kind}', source=live.source, internal=True, metadata={
                'gateway_session_key': live.route, 'gateway_session_id': target})
            await authority.admit_automation(runner._adapter_for_source(live.source), event, identity)
        seen = _turns(runner, authority, target)
        await authority._drain(ref)
        assert seen == []
        assert [m['content'] for m in _notes(authority, target)] == ['[watch] RESULT_watch_match',
                                                                    '[watch] RESULT_watch_disabled']
    finally:
        authority.db.close()


async def _restart(tmp_path, monkeypatch, old):
    """A real owner restart over the same state.db: new runner, new epoch, local sessions recovered."""
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from hermes_state import AsyncSessionDB, SessionDB
    path = old.db.db_path
    old.db.close()
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    db = SessionDB(path)
    runner.session_store._db = db
    runner._session_db = AsyncSessionDB(db)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='lc7-restarted', db=db)
    assert authority.epoch > old.epoch
    return runner, authority


def _delegation_note(authority, target, name):
    return [m for m in authority.db.get_messages(target) if name in str(m.get('content'))]


@pytest.mark.asyncio
async def test_lost_ack_restart_and_regrouped_batch_note_each_result_once(tmp_path, monkeypatch):
    """A is noted but its producer claim is never acknowledged; after a real owner restart A is offered
    again grouped with a fresh B. A resolves to its committed admission, B is noted, each once."""
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    route = authority.sessions[ref.session_id].route
    target = _target(runner, authority, ref)
    first = dict(pending(route, 'deleg_alpha'), parent_session_id=target)
    lost = []
    monkeypatch.setattr(type(runner), '_settle_durable_claim', staticmethod(lambda *args: lost.append(args)))
    assert await runner._deliver_async_delegation_group([first]) is True
    await authority._drain(ref)
    assert len(_delegation_note(authority, target, 'deleg_alpha')) == 1
    assert lost and lost[0][0] == 'complete', 'the acknowledgement was the step that got lost'
    monkeypatch.undo()

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(delegation, '_CLAIM_LEASE_S', -1.0)  # the dead owner's claim is reclaimable
    runner, authority = await _restart(tmp_path, monkeypatch, authority)
    from gateway.session_authority import SessionAuthority
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    try:
        second = dict(pending(route, 'deleg_bravo'), parent_session_id=target)
        assert await runner._deliver_async_delegation_group([dict(first), second]) is True
        await authority._drain(ref)
        assert len(_delegation_note(authority, target, 'deleg_alpha')) == 1
        assert len(_delegation_note(authority, target, 'deleg_bravo')) == 1, 'a fresh sibling was dropped'
        assert [r['outcome'] for r in _admissions(authority, ref)] == ['completed', 'completed']
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('producer', ['delegation', 'process'])
async def test_executed_turn_stays_a_turn_after_lost_ack_detach_and_restart(tmp_path, monkeypatch, producer):
    """A watched completion ran as its turn, but the producer never saw the acknowledgement. The viewer
    leaves, the owner restarts, and the producer offers it again through its real path: it resolves to
    the executed admission, so there is no second turn and no note."""
    runner, authority, ref, actor = await _local(tmp_path, monkeypatch)
    route, target = authority.sessions[ref.session_id].route, _target(runner, authority, ref)
    name = f'{producer}_ran'
    evt = (dict(pending(route, name), parent_session_id=target) if producer == 'delegation'
           else _process_event(authority, ref, target, name))

    async def offer(runner):
        if producer == 'delegation':
            return await runner._deliver_async_delegation_group([dict(evt)])
        return await runner._deliver_completion_notification(f'[finished] RESULT_{name}', dict(evt))
    authority.sessions[ref.session_id].subscribers['watching'] = actor
    with monkeypatch.context() as lost_ack:
        settled = []
        lost_ack.setattr(type(runner), '_settle_durable_claim', staticmethod(lambda *args: settled.append(args)))
        assert await offer(runner) is True
    seen = _turns(runner, authority, target, name)
    await authority._drain(ref)
    assert [text for text, _ in seen] == [_admissions(authority, ref)[0]['payload']['text']]
    assert name in seen[0][0] and seen[0][1] == 0, 'the watched completion ran as a turn'
    assert [r['outcome'] for r in _admissions(authority, ref)] == ['completed']
    if producer == 'delegation':
        assert settled and settled[0][0] == 'complete', 'the acknowledgement was the step that got lost'
        monkeypatch.setattr(delegation, '_CLAIM_LEASE_S', -1.0)  # the dead owner's claim is reclaimable

    authority.sessions[ref.session_id].subscribers.clear()  # the viewer leaves
    runner, authority = await _restart(tmp_path, monkeypatch, authority)
    try:
        assert not authority.sessions[ref.session_id].subscribers
        seen = _turns(runner, authority, target, name)
        assert await offer(runner) is True, 'the retry resolves to the executed admission'
        await authority._drain(ref)
        assert seen == [], 'never a second turn'
        assert not [m for m in authority.db.get_messages(target)
                    if (m.get('display_metadata') or {}).get('completion_admission')], 'never also a note'
        assert _texts(authority, target, name) == []
        assert [r['outcome'] for r in _admissions(authority, ref)] == ['completed']
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_queued_completion_survives_owner_restart_as_one_note(tmp_path, monkeypatch):
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    target = _target(runner, authority, ref)
    _human(authority, ref, 'human', 'FIRST')
    await _deliver(runner, authority, ref, _PROCESS)
    runner, authority = await _restart(tmp_path, monkeypatch, authority)
    try:
        seen = _turns(runner, authority, target)
        await authority._drain(ref)
        await authority._drain(ref)
        assert seen == [('FIRST', 0)]
        assert len(_notes(authority, target)) == 1
    finally:
        authority.db.close()


# Every last-viewer close a local client sends (``session.detach {end_reason}``): the classic CLI's exit,
# /new, /resume and /branch; Ink's /new or resume switch and its exit.
_CLOSES = ['cli_close', 'new_session', 'resumed_other', 'branched', 'tui_close', 'tui_shutdown']


@pytest.mark.asyncio
@pytest.mark.parametrize('reason', _CLOSES)
@pytest.mark.parametrize('producer', ['delegation', 'process'])
async def test_completion_after_new_is_noted_on_the_closed_session(tmp_path, monkeypatch, producer, reason):
    """The viewer closed the session (exit, /new): work finishing afterwards is a note on that original
    session, never dropped and never delivered into the session opened next."""
    from gateway.session_local import create_local_session
    from hermes_state_local import end_idle_local_session
    runner, authority, ref, actor = await _local(tmp_path, monkeypatch)
    try:
        target = _target(runner, authority, ref)
        assert end_idle_local_session(authority.db, epoch=authority.epoch, session_id=ref.session_id,
                                      target_id=target, reason=reason)
        fresh = create_local_session(authority, actor, dict(request_id='lc7-next', source='cli', model='fixture',
                                                            toolsets=[]))
        route = authority.sessions[ref.session_id].route
        if producer == 'delegation':
            evt = dict(pending(route, 'deleg_after_new'), parent_session_id=target)
            assert await runner._deliver_async_delegation_group([evt]) is True
            name = 'deleg_after_new'
        else:
            evt = _process_event(authority, ref, target, 'proc_after_new')
            assert await runner._deliver_completion_notification('[finished] RESULT_proc_after_new', evt) is True
            name = 'proc_after_new'
        await authority._drain(ref)
        assert len(_delegation_note(authority, target, name)) == 1
        row = authority.db.get_session(target)
        assert row['ended_at'] and row['end_reason'] == reason, 'the note must not reopen the closed session'
        fresh_target = _target(runner, authority, fresh)
        assert not _delegation_note(authority, fresh_target, name)
        assert _admissions(authority, fresh) == []
    finally:
        authority.db.close()


def _compress(runner, authority, ref, parent, child):
    """A real compression rotation of the local conversation's physical transcript, route included."""
    db = authority.db
    assert db.try_acquire_compression_lock(parent, 'lc7')
    try:
        db.publish_compression_child(parent_session_id=parent, child_session_id=child, source='cli',
                                     messages=[{'role': 'user', 'content': 'SUMMARY'}], compression_lock_holder='lc7')
    finally:
        db.release_compression_lock(parent, 'lc7')
    assert runner.session_store.advance_compression_session(authority.sessions[ref.session_id].route, parent, child)


@pytest.mark.asyncio
@pytest.mark.parametrize('reason', _CLOSES)
@pytest.mark.parametrize('producer', ['delegation', 'process'])
async def test_compressed_then_closed_completion_is_noted_on_the_closed_tip(tmp_path, monkeypatch, producer, reason):
    """Work started before compression stays pinned to the ancestor; the user then closed the compressed
    continuation (exit, /new). The result is a note on that closed tip, not the ancestor, not the replacement."""
    from gateway.session_local import create_local_session
    from hermes_state_local import end_idle_local_session
    runner, authority, ref, actor = await _local(tmp_path, monkeypatch)
    try:
        ancestor = _target(runner, authority, ref)
        _compress(runner, authority, ref, ancestor, 'lc7-tip')
        assert _target(runner, authority, ref) == 'lc7-tip'
        assert end_idle_local_session(authority.db, epoch=authority.epoch, session_id=ref.session_id,
                                      target_id='lc7-tip', reason=reason)
        fresh = create_local_session(authority, actor, dict(request_id='lc7-next', source='cli', model='fixture',
                                                            toolsets=[]))
        route, name = authority.sessions[ref.session_id].route, f'{producer}_compressed'
        if producer == 'delegation':
            evt = dict(pending(route, name), parent_session_id=ancestor)
            assert await runner._deliver_async_delegation_group([evt]) is True
        else:
            evt = _process_event(authority, ref, ancestor, name)
            assert await runner._deliver_completion_notification(f'[finished] RESULT_{name}', evt) is True
        seen = _turns(runner, authority, 'lc7-tip')
        await authority._drain(ref)
        assert seen == []
        assert len(_delegation_note(authority, 'lc7-tip', name)) == 1
        assert not _delegation_note(authority, ancestor, name)
        row = authority.db.get_session('lc7-tip')
        assert row['ended_at'] and row['end_reason'] == reason, 'the note must not reopen the closed tip'
        assert [r['outcome'] for r in _admissions(authority, ref)] == ['completed']
        assert not _delegation_note(authority, _target(runner, authority, fresh), name)
        assert _admissions(authority, fresh) == []
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_closed_tip_note_needs_the_route_to_name_it(tmp_path, monkeypatch):
    """Only the conversation's own continuation qualifies: a local route that names another session is
    never retargeted onto the closed lineage."""
    from hermes_state_local import end_idle_local_session
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        ancestor = _target(runner, authority, ref)
        _compress(runner, authority, ref, ancestor, 'lc7-tip')
        assert end_idle_local_session(authority.db, epoch=authority.epoch, session_id=ref.session_id,
                                      target_id='lc7-tip', reason='new_session')
        route = authority.sessions[ref.session_id].route
        assert await runner._classify_completion_owner(ancestor, route) == ('deliver', ancestor, 'lc7-tip')
        authority.db.create_session('lc7-other', source='cli')
        runner.session_store._entries[route].session_id = 'lc7-other'
        assert (await runner._classify_completion_owner(ancestor, route))[0] != 'deliver'
    finally:
        authority.db.close()
