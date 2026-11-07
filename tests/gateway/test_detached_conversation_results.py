"""What an old conversation started keeps landing in it after its chat moved on (re-review of #144840).

Teknium's ruling: nothing a conversation the chat moved past (A) produces reaches the chat, and A stays
fully usable from the apps, its reminders and background results included. Real GatewayRunner,
DiscordAdapter (``send`` recorded), SessionAuthority, drain and SessionStore over a real SessionDB; the
agent run is the stand-in of ``test_authority_lane_moves_on``.
"""
import asyncio
import time
from pathlib import Path

import pytest

from tests.gateway.test_authority_lane_moves_on import _app_turn, _chat
from tests.gateway.test_detached_conversation_lifecycle import _moved_on
from tests.waiting import async_wait_until


def _noted(authority, session_id, text):
    return any(text in str(message.get('content')) for message in authority.db.get_messages(session_id))


@pytest.mark.asyncio
async def test_a_reminder_the_app_set_on_the_old_conversation_fires_after_a_resume_and_after_a_restart(monkeypatch):
    """N1: the reminder names A (``local:<A>``). Back on the chat (its ``/resume``) A's fire is A's own
    turn there; with A not mounted (any restart) it mounts off the chat and is written into A. It is
    never refused for a local receipt A never had, so a recurring reminder keeps firing."""
    from gateway.session_cron import deliver_to_local_chat
    from gateway.session_mutations import retire_live_sessions
    _runner, adapter, authority, source, sent, ran, _gates, lane, ref_a = await _moved_on(monkeypatch)
    a, home = ref_a.session_id, Path(authority.db.db_path).parent

    await _chat(adapter, source, sent, '/resume First chat')
    assert await asyncio.to_thread(deliver_to_local_chat, home, a, 'fire after resume') is None
    await async_wait_until(lambda: f"reply to 'fire after resume' from {a}" in sent, timeout=10, what="A's turn in the chat")
    assert [turn[1:] for turn in ran if turn[0] == 'fire after resume'] == [(a, lane)]

    await _chat(adapter, source, sent, '/new')
    retire_live_sessions(authority, [a])  # what a restart leaves: A not mounted
    chat_before = len(sent)
    assert await asyncio.to_thread(deliver_to_local_chat, home, a, 'fire after restart') is None
    await async_wait_until(lambda: _noted(authority, a, 'fire after restart'), timeout=10, what='the note in A')
    assert authority.sessions[a].detached and sent[chat_before:] == []


@pytest.mark.asyncio
async def test_a_note_into_the_old_conversation_leaves_the_row_the_chats_new_ended_closed(monkeypatch):
    """N2a: a reminder written into A as a note (no viewer, no turn) keeps the ``/new``'s end stamp, as
    a note into a closed local session does; only a turn reopens the row."""
    from gateway.session_cron import deliver_to_local_chat
    _runner, _adapter, authority, _source, _sent, ran, _gates, _lane, ref_a = await _moved_on(monkeypatch)
    a = ref_a.session_id
    ran_before = len(ran)
    assert await asyncio.to_thread(deliver_to_local_chat, Path(authority.db.db_path).parent, a, 'a reminder') is None
    await async_wait_until(lambda: _noted(authority, a, 'a reminder'), timeout=10, what='the note in A')
    await asyncio.wait_for(authority.sessions[a].task, 10)
    assert ran[ran_before:] == []
    assert authority.db.get_session(a)['end_reason'] == 'session_reset'


def _chat_started(kind, lane, a):
    """A result of work the chat's own turn on A started before its ``/new``: pinned to A, on the chat's route."""
    if kind == 'delegation':
        return {'type': 'async_delegation', 'delegation_id': 'deleg-1', 'session_key': lane,
                'parent_session_id': a, 'dispatched_at': 1.0, 'summary': 'delegated result', 'status': 'completed'}
    return {'type': 'completion', 'event_id': 'c-1', 'session_id': 'proc_chat', 'started_at': 1000.0,
            'session_key': lane, 'parent_session_id': a, 'command': 'make', 'exit_code': 0, 'output': 'built',
            'platform': 'discord', 'chat_id': '42', 'chat_type': 'dm', 'user_id': '42', 'emitted_at': time.time()}


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['delegation', 'process'])
async def test_a_result_the_chat_started_on_the_old_conversation_lands_in_it_once_the_app_used_it(monkeypatch, kind):
    """N2b: the app's turn reopened A after the chat's ``/new``. A result of work the chat's turn on A
    started lands in A once (a note, no viewer), never in the chat, and is not refused on every retry."""
    import tools.async_delegation as ad
    runner, _adapter, authority, _source, sent, ran, _gates, lane, ref_a = await _moved_on(monkeypatch)
    a = ref_a.session_id
    assert await _app_turn(authority, ref_a, 'app on A') == ('terminal', 'completed')
    evt = _chat_started(kind, lane, a)
    chat_before, ran_before = len(sent), len(ran)
    if kind == 'delegation':
        ad._persist_dispatch(evt)
        ad._persist_completion(evt, {'status': 'completed', 'summary': evt['summary']})
        results = [await runner._deliver_completion_notification('delegated result', dict(evt)) for _ in range(2)]
        assert results == [True, None]
        assert ad.get_durable_delegation('deleg-1')['delivery_state'] == 'delivered'
        text = 'delegated result'
    else:
        assert await runner._inject_watch_notification('[process proc_chat finished]', evt) is True
        assert '_refusal_permanent' not in evt
        text = '[process proc_chat finished]'
    await async_wait_until(lambda: _noted(authority, a, text), timeout=10, what='the result in A')
    assert ran[ran_before:] == []
    assert sent[chat_before:] == [], f'the chat got the old conversation\'s result: {sent[chat_before:]}'


@pytest.mark.asyncio
async def test_a_result_of_the_apps_turn_that_finishes_after_the_chat_resumed_the_conversation_runs_there(monkeypatch):
    """A process the app's turn on A started (on A's own key) finishes after the chat ``/resume``d A:
    it runs as A's turn on the chat's route, never refused for a key that no longer names A."""
    from tests.gateway.test_detached_conversation_lifecycle import _process_event
    runner, adapter, authority, source, sent, ran, _gates, lane, ref_a = await _moved_on(monkeypatch)
    a = ref_a.session_id
    assert await _app_turn(authority, ref_a, 'start a job') == ('terminal', 'completed')
    evt = _process_event(runner, authority.sessions[a], a, exit_code=0, output='done')
    await _chat(adapter, source, sent, '/resume First chat')
    ran_before = len(ran)

    assert await runner._inject_watch_notification('[process proc_1 finished]', evt) is True
    await async_wait_until(lambda: len(ran) > ran_before, timeout=10, what="A's result turn")
    assert [turn[1:] for turn in ran[ran_before:]] == [(a, lane)]


@pytest.mark.asyncio
async def test_a_secondary_profiles_old_conversation_gets_its_result_after_the_source_cache_forgot_it(
        tmp_path, monkeypatch):
    """N3, two profiles in one process: a process the app's turn on a ``fitness`` conversation A
    started finishes after the runner's bounded source cache dropped A. Its key names the profile, so
    the event resolves A's live source and binds ``fitness``'s scope, unscoped as the drain starts,
    and the result is written into A in ``fitness``'s store."""
    import hermes_state
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from gateway.session_authorities import SessionAuthorities
    from gateway.session_authority import LiveSession, initialize_session_authority
    from gateway.session_detached import detach
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB
    root = tmp_path / 'hermes'
    profile = root / 'profiles' / 'fitness'
    profile.mkdir(parents=True)
    (profile / 'config.yaml').write_text('{}\n', encoding='utf-8')
    monkeypatch.setenv('HERMES_HOME', str(root))
    monkeypatch.setattr(hermes_state, 'DEFAULT_DB_PATH', hermes_state._IMPORT_DEFAULT_DB_PATH)
    runner = GatewayRunner(GatewayConfig(multiplex_profiles=True))
    launch = await initialize_session_authority(runner, profile_id=str(root), instance_id='owner',
                                                db=SessionDB(root / 'state.db'))
    fitness = await initialize_session_authority(runner, profile_id=str(profile), instance_id='owner',
                                                 db=SessionDB(profile / 'state.db'), register=False)
    runner.session_authorities = SessionAuthorities(root, multiplexed=True)
    runner.session_authorities.add(root, launch)
    runner.session_authorities.add(profile, fitness, name='fitness')
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='555', chat_type='dm', user_id='u1', profile='fitness',
                           message_id='chat-message-1')  # as the chat's last inbound left it
    route = runner.session_store._generate_session_key(source)
    assert route.startswith('agent:fitness:')
    a = 'conversation-a'
    fitness.db.create_session(a, source='telegram', session_key=route)
    live = fitness.sessions[a] = LiveSession(source, route)
    detach(fitness, live, a)  # the chat moved on; the app keeps A
    evt = {'type': 'completion', 'event_id': 'c-1', 'session_id': 'proc_1', 'started_at': 1000.0,
           'session_key': live.route, 'parent_session_id': a, 'command': 'make', 'exit_code': 0,
           'output': 'built', 'platform': 'local', 'chat_id': a, 'chat_type': 'dm', 'user_id': 'u1',
           'emitted_at': time.time()}
    runner._session_sources.clear()

    assert runner._build_process_event_source(evt) is live.source
    async with runner._completion_event_scope(evt):
        assert Path(get_hermes_home()) == profile
        assert await runner._inject_watch_notification('[process proc_1 finished]', evt) is True
    await async_wait_until(lambda: _noted(fitness, a, '[process proc_1 finished]'), timeout=10,
                           what="the result in fitness's A")
