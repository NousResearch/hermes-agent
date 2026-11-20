"""Producers that target a local (Desktop/TUI/CLI) chat reach its owner's FIFO (#143547 KB-22, RT-90, CRON-11).

``LocalSessionAdapter`` has no message handler and its ``send()`` only acknowledges (a turn's output
rides the event stream), so a Kanban wake, a due heartbeat or a cron result that took the messaging
adapter path was refused or silently dropped.
"""
import asyncio
import threading
import time

import pytest

from tests.fixtures.storage_read import transcript_rows
from tests.gateway.test_session_end_delegations import _local


def _admissions(authority, ref, needle):
    rows = authority.persistence.admissions.list(session_id=ref.session_id, pending_only=False)
    return [row for row in rows if needle in str(row.payload)]


@pytest.mark.asyncio
async def test_kanban_wake_is_admitted_into_a_local_chat(tmp_path, monkeypatch):
    from gateway.wake import deliver_wake
    runner, authority, ref, _ = await _local(tmp_path, monkeypatch)
    live = authority.sessions[ref.session_id]
    adapter = runner._delivery_adapter_for(live.source)
    await deliver_wake(adapter, text='card t1 done', session_id=ref.session_id, source=live.source)
    assert len(_admissions(authority, ref, 'card t1 done')) == 1


@pytest.mark.asyncio
async def test_kanban_wake_names_the_receipt_an_idle_classic_cli_opens_its_turn_with(tmp_path, monkeypatch):
    """An idle classic CLI only opens a gateway-admitted turn whose ``message.start`` carries a card:
    a wake without one ran and was saved, and the open chat never showed the reply."""
    from gateway.platforms.event import MessageEvent
    from gateway.response_filters import INTERNAL_NOTIFICATION_DISPLAY_KIND, display_kind_for_event
    from gateway.session_automation import admission_display
    from gateway.wake import deliver_wake
    from hermes_cli.gateway_tools_notify import _card
    runner, authority, ref, _ = await _local(tmp_path, monkeypatch)
    live = authority.sessions[ref.session_id]
    adapter = runner._delivery_adapter_for(live.source)
    wake = '[kanban] Task t1 completed.\nTitle: card t1'
    await deliver_wake(adapter, text=wake, session_id=ref.session_id, source=live.source,
                       display_text='[kanban] Task t1 completed.')
    [row] = _admissions(authority, ref, 'Task t1 completed')
    card = _card({'payload': admission_display(row.payload)})
    assert card == {'display_kind': INTERNAL_NOTIFICATION_DISPLAY_KIND,
                    'display_text': '[kanban] Task t1 completed.', 'text': wake}
    # The receipt is for the terminal only: the stored row stays the machinery kind every viewer hides.
    event = MessageEvent(text=wake, source=live.source, internal=True,
                         metadata={key: card[key] for key in ('display_kind', 'display_text')})
    assert display_kind_for_event(event) == INTERNAL_NOTIFICATION_DISPLAY_KIND


def test_kanban_notifier_hands_the_wake_its_first_line_as_the_receipt(tmp_path, monkeypatch):
    from gateway import wake as wake_module
    from gateway.config import Platform
    from hermes_cli import kanban_db as kb
    from tests.gateway.test_kanban_wake_scope import (
        UnscopedAdapter, _completed_subscription, _one_notifier_tick, _runner)
    monkeypatch.setenv('HERMES_KANBAN_DB', str(tmp_path / 'wake-receipt.db'))
    kb.init_db()
    task_id = _completed_subscription(platform='telegram', chat_id='chat-dm', chat_type='dm')
    delivered, deliver = [], wake_module.deliver_wake

    async def recording(adapter, **kwargs):
        delivered.append(kwargs)
        await deliver(adapter, **kwargs)

    monkeypatch.setattr(wake_module, 'deliver_wake', recording)
    asyncio.run(_one_notifier_tick(monkeypatch, _runner(UnscopedAdapter(), Platform.TELEGRAM)))
    [wake] = delivered
    assert wake['display_text'] == wake['text'].splitlines()[0]
    assert task_id in wake['display_text']


def _overdue_heartbeat(session_id):
    from hermes_cli.heartbeat import HeartbeatManager, save_heartbeat
    state = HeartbeatManager(session_id).set('status tick', 60)
    state.created_at = time.time() - 120  # one interval overdue
    save_heartbeat(session_id, state)


def _heartbeat_admissions(authority, ref):
    rows = authority.persistence.admissions.list(session_id=ref.session_id, pending_only=False)
    return [row for row in rows if row.payload.get('local_automation_v1', {}).get('heartbeat')]


@pytest.mark.asyncio
async def test_due_heartbeat_is_admitted_into_a_local_chat(tmp_path, monkeypatch):
    from gateway.run_heartbeat_restore import restore_heartbeat_watches
    runner, authority, ref, _ = await _local(tmp_path, monkeypatch)
    _overdue_heartbeat(ref.session_id)
    await restore_heartbeat_watches(runner)
    await runner._heartbeat_poll_once(runner._heartbeat_watch)
    assert _heartbeat_admissions(authority, ref)


@pytest.mark.asyncio
async def test_overdue_heartbeat_fires_once_with_a_classic_cli_also_polling(tmp_path, monkeypatch):
    from gateway.run_heartbeat_restore import restore_heartbeat_watches
    from gateway.session_viewer_automation import invoke
    from hermes_cli.heartbeat import HeartbeatManager
    runner, authority, ref, _ = await _local(tmp_path, monkeypatch)
    _overdue_heartbeat(ref.session_id)
    await restore_heartbeat_watches(runner)
    loop, claim, polled = asyncio.get_running_loop(), HeartbeatManager.due_prompt, []

    def due_prompt(self, now=None):
        # The CLI's due_prompt RPC (worker thread) has loaded the overdue row when the gateway poller claims it.
        if threading.current_thread() is not threading.main_thread() and not polled:
            polled.append(True)
            asyncio.run_coroutine_threadsafe(runner._heartbeat_poll_once(runner._heartbeat_watch), loop).result()
        return claim(self, now)

    monkeypatch.setattr(HeartbeatManager, 'due_prompt', due_prompt)
    cli = await asyncio.to_thread(invoke, runner, 'heartbeat', ref.session_id, 'due_prompt', [], {})
    assert polled
    assert len(_heartbeat_admissions(authority, ref)) + bool(cli['value']) == 1


async def _attached_cli(tmp_path, monkeypatch, request):
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    from hermes_cli import goals
    from persistence_assembly import open_persistence
    runner, authority, _, _ = await _local(tmp_path, monkeypatch)
    # The RPC runs under the owner's profile scope, ``Path('default')`` here: keep it in tmp_path, on its db.
    monkeypatch.chdir(tmp_path)
    store = open_persistence(authority.persistence.scope)
    request.addfinalizer(store.close)
    monkeypatch.setattr(goals, '_get_session_db', lambda: store)
    cli = AuthorityConnection(authority, object(), {'user_id': 'human', 'profile_id': 'default'})
    ref = create_local_session(authority, cli.actor, dict(request_id='cli', source='cli', model='fixture', toolsets=[]))
    cli.subscriptions[ref.session_id] = 'sub-1'

    async def automation(kind, call, *args):
        reply = await cli.dispatch({'id': 1, 'method': 'session.automation', 'params': {
            'session_id': ref.session_id, 'kind': kind, 'call': call, 'args': list(args), 'kwargs': {}}})
        return reply['result']
    return runner, authority, ref, automation


@pytest.mark.asyncio
async def test_a_classic_cli_driving_its_heartbeat_keeps_the_tick(tmp_path, monkeypatch, request):
    """The classic CLI paints only the ticks its own watchdog fires, so the poller leaves them to it."""
    from gateway.run_heartbeat_restore import restore_heartbeat_watches
    runner, authority, ref, automation = await _attached_cli(tmp_path, monkeypatch, request)
    await automation('heartbeat', 'set', 'status tick', 60)  # /heartbeat set starts the CLI's watchdog
    _overdue_heartbeat(ref.session_id)
    await restore_heartbeat_watches(runner)
    await runner._heartbeat_poll_once(runner._heartbeat_watch)
    assert not _heartbeat_admissions(authority, ref)
    assert 'status tick' in (await automation('heartbeat', 'due_prompt'))['value']


@pytest.mark.asyncio
async def test_a_classic_cli_without_a_heartbeat_watchdog_leaves_the_tick_to_the_poller(tmp_path, monkeypatch, request):
    """A CLI reopening a chat runs goal/loop reads but starts its watchdog only on /heartbeat set or resume."""
    from gateway.run_heartbeat_restore import restore_heartbeat_watches
    runner, authority, ref, automation = await _attached_cli(tmp_path, monkeypatch, request)
    _overdue_heartbeat(ref.session_id)
    await restore_heartbeat_watches(runner)
    await automation('goal', 'state')
    await automation('loop', 'state')
    await runner._heartbeat_poll_once(runner._heartbeat_watch)
    assert _heartbeat_admissions(authority, ref)


@pytest.mark.asyncio
async def test_a_classic_cli_resume_with_no_heartbeat_leaves_the_tick_to_the_poller(tmp_path, monkeypatch, request):
    """/heartbeat resume with nothing to resume starts no watchdog, so a heartbeat set later elsewhere still fires."""
    from gateway.run_heartbeat_restore import restore_heartbeat_watches
    runner, authority, ref, automation = await _attached_cli(tmp_path, monkeypatch, request)
    assert (await automation('heartbeat', 'resume'))['value'] is None
    _overdue_heartbeat(ref.session_id)  # Desktop sets it
    await restore_heartbeat_watches(runner)
    await runner._heartbeat_poll_once(runner._heartbeat_watch)
    assert _heartbeat_admissions(authority, ref)


@pytest.mark.asyncio
@pytest.mark.parametrize('live_adapters', [False, True], ids=['no-local-adapter-yet', 'in-gateway-ticker'])
@pytest.mark.parametrize('viewer', [False, True], ids=['unviewed', 'viewed'])
async def test_cron_result_for_a_local_origin_is_a_note_not_a_turn(tmp_path, monkeypatch, live_adapters, viewer):
    from cron import scheduler_delivery as sd
    from tests.gateway.test_session_end_delegations import _record_turns
    from tests.gateway.test_unviewed_completion_notes import _target
    runner, authority, ref, actor = await _local(tmp_path, monkeypatch)
    if viewer:
        authority.sessions[ref.session_id].subscribers['viewer'] = actor
    job = {'id': 'j1', 'name': 'digest', 'deliver': 'origin',
           'origin': {'platform': 'local', 'chat_id': ref.session_id}}
    # The ticker delivers from its own thread; the owner admits on its loop.
    error = await asyncio.to_thread(sd._deliver_result, job, 'CRON_RESULT_j1',
                                    adapters=runner.adapters if live_adapters else None,
                                    loop=asyncio.get_running_loop())
    assert error is None
    turns = _record_turns(runner)
    await authority._drain(ref)
    assert turns == [], 'the finished result must not cost a model turn'
    [row] = _admissions(authority, ref, 'CRON_RESULT_j1')
    assert row.outcome == 'completed'
    [note] = [m for m in transcript_rows(authority.persistence, _target(runner, authority, ref))
              if 'CRON_RESULT_j1' in str(m.get('content'))]
    assert note['role'] == 'user' and note['display_metadata']['completion_admission'] == row.admission_id
    assert 'summarize for the chat' not in note['content'], 'no model framing on a note'


@pytest.mark.asyncio
async def test_cron_result_for_a_local_origin_without_its_owner_is_an_error(tmp_path, monkeypatch):
    from cron import scheduler_delivery as sd
    from gateway.session_cron import unbind_owner
    runner, authority, ref, _ = await _local(tmp_path, monkeypatch)
    unbind_owner(authority)  # a standalone `hermes cron run`: no gateway in this process
    job = {'id': 'j1', 'name': 'digest', 'deliver': 'origin',
           'origin': {'platform': 'local', 'chat_id': ref.session_id}}
    error = await asyncio.to_thread(sd._deliver_result, job, 'CRON_RESULT_j1', adapters=runner.adapters,
                                    loop=asyncio.get_running_loop())
    assert error and 'local' in error
    assert not _admissions(authority, ref, 'CRON_RESULT_j1')
