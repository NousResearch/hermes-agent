"""The lifecycle of a conversation its chat moved past (review of #144840 at 03d04918dde).

Teknium's ruling: once a messaging chat names another conversation, nothing the old one produces
reaches the chat, and the old one stays fully usable from the apps. Real GatewayRunner, DiscordAdapter
(``send`` recorded), SessionAuthority, drain and SessionStore over a real SessionDB; the agent run is
the stand-in of ``test_authority_lane_moves_on``.
"""
import asyncio
from types import SimpleNamespace

import pytest

from tests.gateway.test_authority_lane_moves_on import _app, _app_turn, _chat, _gateway
from tests.waiting import async_wait_until


async def _moved_on(monkeypatch):
    """Chat 42 talked in A (open in an app), then ``/new`` and a message in B."""
    runner, adapter, authority, source, sent, ran, gates = await _gateway(monkeypatch)
    lane = runner.session_store._generate_session_key(source)
    await _chat(adapter, source, sent, 'hello')
    ref_a = authority.register(source)
    authority.db.set_session_title(ref_a.session_id, 'First chat')
    await _chat(adapter, source, sent, '/new')
    await _chat(adapter, source, sent, 'hello again')
    return runner, adapter, authority, source, sent, ran, gates, lane, ref_a


@pytest.mark.asyncio
async def test_the_chats_context_exhaustion_reset_moves_the_old_conversation_off_the_chat(monkeypatch):
    """F1: the auto-reset after context exhaustion rotates the chat to B without a /new. The app's
    next prompt on A runs in A, and the chat gets nothing of it."""
    from gateway.session_results import execution_result
    runner, adapter, authority, source, sent, ran, _gates = await _gateway(monkeypatch)
    lane = runner.session_store._generate_session_key(source)
    await _chat(adapter, source, sent, 'hello')
    ref_a = authority.register(source)
    prepare = runner._hmwa_prepare_turn

    async def exhausting(event, source, session_entry, session_key, *args):
        if event.text != 'too big':
            return await prepare(event, source, session_entry, session_key, *args)
        execution_result.get()['result'] = {'final_response': 'Context too large', 'messages': []}
        entry = await runner._hmwa_rotate_exhausted_session(session_entry, session_key, source)
        return 'Context too large', entry
    monkeypatch.setattr(runner, '_hmwa_prepare_turn', exhausting)
    await _chat(adapter, source, sent, 'too big')
    await async_wait_until(lambda: runner.session_store.peek_session_id(lane) != ref_a.session_id,
                           what='the route to rotate')
    chat_replies = len(sent)

    assert await _app_turn(authority, ref_a, 'app prompt on A') == ('terminal', 'completed')
    assert ran[-1][:2] == ('app prompt on A', ref_a.session_id) and ran[-1][2] != lane
    assert sent[chat_replies:] == [], f'the chat got the old conversation\'s reply: {sent[chat_replies:]}'


@pytest.mark.asyncio
async def test_what_the_old_conversation_schedules_for_here_names_the_app_not_the_chat(monkeypatch):
    """F3: a detached turn binds the session variables a Desktop session's turn binds, so a reminder,
    kanban subscription or send_message origin it creates never names the chat."""
    from gateway.session import build_session_context
    from gateway.session_context import get_session_env
    from tools.cronjob_job_args import _origin_from_env
    runner, _adapter, authority, _source, _sent, _ran, _gates, _lane, ref_a = await _moved_on(monkeypatch)
    live = authority.sessions[ref_a.session_id]
    entry = runner.session_store.get_or_create_session(live.source)
    with runner._session_env_scope(build_session_context(live.source, runner.config, entry)):
        env = {name: get_session_env(f'HERMES_SESSION_{name}') for name in ('PLATFORM', 'CHAT_ID', 'KEY')}
        origin = _origin_from_env()
    assert env == {'PLATFORM': 'local', 'CHAT_ID': ref_a.session_id, 'KEY': live.route}
    assert origin is None or origin['platform'] == 'local', origin


@pytest.mark.asyncio
async def test_the_old_conversation_is_live_again_once_the_app_uses_it(monkeypatch):
    """F4: the chat's /new ended A's row; the app's turn reopens it, as an API or local turn reopens
    theirs, so its background results and a rotating compression are not refused as ended."""
    _runner, _adapter, authority, _source, _sent, _ran, _gates, _lane, ref_a = await _moved_on(monkeypatch)
    assert authority.db.get_session(ref_a.session_id)['end_reason'] == 'session_reset'
    assert await _app_turn(authority, ref_a, 'from the app') == ('terminal', 'completed')
    row = authority.db.get_session(ref_a.session_id)
    assert (row['ended_at'], row['end_reason']) == (None, None)


@pytest.mark.asyncio
async def test_a_stop_from_either_side_reaches_the_apps_turn_after_the_chat_resumes_it(monkeypatch):
    """F5: the app's turn on A is running when the chat ``/resume``s A. That turn keeps its own key;
    the app's Stop and the chat's /stop both still reach it."""
    from gateway.platforms.event import MessageEvent
    from gateway.session_contract import Submission
    runner, adapter, authority, source, sent, ran, gates, lane, ref_a = await _moved_on(monkeypatch)
    stops = []
    agent = SimpleNamespace(interrupt=lambda *a, **k: stops.append('interrupt'), session_id=ref_a.session_id)
    prepare = runner._hmwa_prepare_turn

    async def adopting(event, source, session_entry, session_key, *args):
        if event.text == 'slow app turn':
            runner._running_agents[session_key] = agent  # the turn's agent, as a real build installs it
        return await prepare(event, source, session_entry, session_key, *args)
    monkeypatch.setattr(runner, '_hmwa_prepare_turn', adopting)
    gates['slow app turn'] = asyncio.Event()
    await authority.submit(_app(), Submission('slow', ref_a, {'text': 'slow app turn'}, 'queue'))
    await async_wait_until(lambda: ran and ran[-1][0] == 'slow app turn', what='the app turn to start')
    await _chat(adapter, source, sent, '/resume First chat')
    assert authority.sessions[ref_a.session_id].route == lane

    assert authority._runs_own_agent(ref_a) and authority.agent(ref_a) is agent  # the app's Stop target
    before = len(sent)
    await adapter.handle_message(MessageEvent(text='/stop', source=source, message_id='stop'))
    await async_wait_until(lambda: len(sent) > before, what='the /stop reply')
    from agent.i18n import t
    assert sent[before:] != [t('gateway.stop.no_active')] and stops
    gates['slow app turn'].set()


@pytest.mark.asyncio
async def test_the_chats_resume_drops_the_agent_the_app_cached_for_the_old_conversation(monkeypatch):
    """F6: the agent cached under A's detached key goes when the chat takes A back, instead of being
    reused or committing memory at a later idle eviction while A runs in the chat."""
    runner, adapter, authority, source, sent, _ran, _gates, lane, ref_a = await _moved_on(monkeypatch)
    key = authority.sessions[ref_a.session_id].route
    assert key != lane
    with runner._agent_cache_lock:
        runner._agent_cache[key] = (SimpleNamespace(_last_activity_ts=0, session_id=ref_a.session_id), ('sig',))
    await _chat(adapter, source, sent, '/resume First chat')
    assert authority.sessions[ref_a.session_id].route == lane
    assert key not in runner._agent_cache


def _process_event(runner, live, session_id, kind='completion', **fields):
    """A process event as the registry records it for a process the turn on *live* started."""
    import time
    from gateway.session import build_session_context
    from gateway.session_context import get_session_env
    entry = runner.session_store.get_or_create_session(live.source)
    with runner._session_env_scope(build_session_context(live.source, runner.config, entry)):
        env = {name: get_session_env(f'HERMES_SESSION_{name}') for name in ('KEY', 'PLATFORM', 'CHAT_ID', 'CHAT_TYPE')}
    return {'type': kind, 'event_id': f'{kind}-1', 'session_id': 'proc_1', 'started_at': 1000.0,
            'session_key': env['KEY'], 'parent_session_id': session_id, 'command': 'sleep 1',
            'platform': env['PLATFORM'], 'chat_id': env['CHAT_ID'], 'chat_type': env['CHAT_TYPE'],
            'user_id': '42', 'emitted_at': time.time(), **fields}


def _automation_rows(authority, session_id, route):
    from hermes_state_runtime import list_session_admissions
    return [row for row in list_session_admissions(authority.db, session_id=session_id, pending_only=False)
            if row['principal_id'] == 'automation:' + route]


@pytest.mark.asyncio
@pytest.mark.parametrize('viewer', [False, True], ids=['no-viewer', 'app-viewing'])
async def test_a_background_result_of_the_old_conversation_reaches_the_app_not_the_chat(monkeypatch, viewer):
    """A process the app's turn on A started finishes: its result lands in A (a note in its history
    with no viewer, a turn of A with the app watching), never in the chat, and is not retried. The
    runner's bounded source cache no longer holds A's source by then."""
    runner, _adapter, authority, _source, sent, ran, _gates, _lane, ref_a = await _moved_on(monkeypatch)
    assert await _app_turn(authority, ref_a, 'start a job') == ('terminal', 'completed')
    live = authority.sessions[ref_a.session_id]
    if viewer:
        await authority.attach(_app(), ref_a)
    evt = _process_event(runner, live, ref_a.session_id, exit_code=0, output='done')
    runner._session_sources.clear()
    chat_before, ran_before = len(sent), len(ran)

    assert await runner._inject_watch_notification('[process proc_1 finished]', evt) is True

    def settled():
        return [row for row in _automation_rows(authority, ref_a.session_id, live.route) if row['status'] == 'terminal']
    rows = await async_wait_until(settled, timeout=10, what="the result's admission on A")
    assert rows[0]['outcome'] == 'completed'
    if viewer:
        await async_wait_until(lambda: len(ran) > ran_before, what='the result turn')
        assert [turn[1:] for turn in ran[ran_before:]] == [(ref_a.session_id, live.route)]
    else:
        assert ran[ran_before:] == []
        assert any('[process proc_1 finished]' in str(message.get('content'))
                   for message in authority.db.get_messages(ref_a.session_id))
    assert sent[chat_before:] == [], f'the chat got the old conversation\'s result: {sent[chat_before:]}'


@pytest.mark.asyncio
async def test_a_result_queued_on_the_old_conversation_still_runs_once_the_chat_resumes_it(monkeypatch):
    """The result is queued behind the app's running turn when the chat ``/resume``s A: it runs as A's
    turn on the chat's route, not refused as admitted under another route."""
    from gateway.session_contract import Submission
    runner, adapter, authority, source, sent, ran, gates, lane, ref_a = await _moved_on(monkeypatch)
    await authority.attach(_app(), ref_a)
    live = authority.sessions[ref_a.session_id]
    detached_route = live.route
    gates['slow app turn'] = asyncio.Event()
    await authority.submit(_app(), Submission('slow', ref_a, {'text': 'slow app turn'}, 'queue'))
    await async_wait_until(lambda: ran and ran[-1][0] == 'slow app turn', what='the app turn to start')
    evt = _process_event(runner, live, ref_a.session_id, exit_code=0, output='done')
    assert await runner._inject_watch_notification('[process proc_1 finished]', evt) is True
    await _chat(adapter, source, sent, '/resume First chat')
    assert live.route == lane
    gates['slow app turn'].set()

    def settled():
        return [row for row in _automation_rows(authority, ref_a.session_id, detached_route)
                if row['status'] == 'terminal']
    rows = await async_wait_until(settled, timeout=10, what="the result's admission on A")
    assert rows[0]['outcome'] == 'completed'
    assert [turn[1:] for turn in ran if turn[0].startswith('[process proc_1 finished]')] == [(ref_a.session_id, lane)]


@pytest.mark.asyncio
async def test_the_old_conversations_finished_process_withdraws_its_waiting_heartbeat(monkeypatch):
    """A liveness tick of A's process waits behind the app's running turn; the process's completion
    cancels it, as on any route, instead of leaving a stale "still running" turn queued."""
    from gateway.session_automation import deliver_heartbeat
    from gateway.session_contract import Submission
    runner, _adapter, authority, _source, _sent, ran, gates, _lane, ref_a = await _moved_on(monkeypatch)
    live = authority.sessions[ref_a.session_id]
    gates['slow app turn'] = asyncio.Event()
    await authority.submit(_app(), Submission('slow', ref_a, {'text': 'slow app turn'}, 'queue'))
    await async_wait_until(lambda: ran and ran[-1][0] == 'slow app turn', what='the app turn to start')
    tick = _process_event(runner, live, ref_a.session_id, 'heartbeat', seq=1, elapsed=60, interval=60,
                          output='still building', omitted=0)
    assert await deliver_heartbeat(runner, tick) is True
    completion = _process_event(runner, live, ref_a.session_id, exit_code=0, output='done')
    assert await runner._inject_watch_notification('[process proc_1 finished]', completion) is True
    ticks = [row for row in _automation_rows(authority, ref_a.session_id, live.route)
             if row['request_id'].startswith('["heartbeat"')]
    assert [(row['status'], row['outcome']) for row in ticks] == [('terminal', 'cancelled')]
    gates['slow app turn'].set()


@pytest.mark.asyncio
async def test_a_reminder_the_app_set_on_the_old_conversation_fires_into_it(monkeypatch):
    """A reminder created from the app on A names A (``platform: local``): its fire is written into A,
    never refused for a local session A never was, and never sent to the chat."""
    from pathlib import Path
    from gateway.session_cron import deliver_to_local_chat
    _runner, _adapter, authority, _source, sent, _ran, _gates, _lane, ref_a = await _moved_on(monkeypatch)
    chat_before = len(sent)
    home = Path(authority.db.db_path).parent

    assert await asyncio.to_thread(deliver_to_local_chat, home, ref_a.session_id, 'reminder from A') is None
    await async_wait_until(lambda: any('reminder from A' in str(message.get('content'))
                                       for message in authority.db.get_messages(ref_a.session_id)),
                           timeout=10, what='the reminder note in A')
    assert sent[chat_before:] == []


@pytest.mark.asyncio
async def test_a_late_route_announcement_never_hands_the_chat_back_its_old_conversation(monkeypatch):
    """Review on #144840: a worker's ``/resume`` of A announces its route after a newer ``/new``
    moved the chat on to C. The stale announcement must not reattach A to the chat: the app's next
    prompt on A still runs in A, off the chat, and the chat keeps C."""
    import threading
    runner, adapter, authority, source, sent, ran, _gates, lane, ref_a = await _moved_on(monkeypatch)
    store = runner.session_store
    announce, paused, release = store._announce_route, threading.Event(), threading.Event()

    def held(session_key, entry):  # the worker pauses after its store write, before announcing
        if entry.session_id == ref_a.session_id and not paused.is_set():
            paused.set()
            release.wait(10)
        announce(session_key, entry)
    monkeypatch.setattr(store, '_announce_route', held)
    worker = asyncio.create_task(asyncio.to_thread(store.switch_session, lane, ref_a.session_id))
    await async_wait_until(paused.is_set, what='the worker to pause before announcing')
    await _chat(adapter, source, sent, '/new')
    await _chat(adapter, source, sent, 'newest')
    c = store.peek_session_id(lane)
    assert c != ref_a.session_id
    release.set()
    await worker  # its announcement was handed to the loop before the worker's result, so it has run

    assert store.peek_session_id(lane) == c
    chat_replies = len(sent)
    assert await _app_turn(authority, ref_a, 'app prompt on A') == ('terminal', 'completed')
    assert ran[-1][:2] == ('app prompt on A', ref_a.session_id) and ran[-1][2] != lane
    assert sent[chat_replies:] == [], f'the chat got the old conversation\'s reply: {sent[chat_replies:]}'
    await _chat(adapter, source, sent, 'still C')
    assert ran[-1][1:] == (c, lane) and sent[-1] == f"reply to 'still C' from {c}"
