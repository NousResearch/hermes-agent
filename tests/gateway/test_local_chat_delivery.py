"""Producers that target a local (Desktop/TUI/CLI) chat reach its owner's FIFO (#143547 KB-22, RT-90, CRON-11).

``LocalSessionAdapter`` has no message handler and its ``send()`` only acknowledges (a turn's output
rides the event stream), so a Kanban wake, a due heartbeat or a cron result that took the messaging
adapter path was refused or silently dropped.
"""
import asyncio
import threading
import time

import pytest

from tests.gateway.test_session_end_delegations import _local


def _admissions(authority, ref, needle):
    from hermes_state_runtime import list_session_admissions
    rows = list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False)
    return [row for row in rows if needle in str(row['payload'])]


@pytest.mark.asyncio
async def test_kanban_wake_is_admitted_into_a_local_chat(tmp_path, monkeypatch):
    from gateway.wake import deliver_wake
    runner, authority, ref, _ = await _local(tmp_path, monkeypatch)
    live = authority.sessions[ref.session_id]
    adapter = runner._delivery_adapter_for(live.source)
    await deliver_wake(adapter, text='card t1 done', session_id=ref.session_id, source=live.source)
    assert len(_admissions(authority, ref, 'card t1 done')) == 1


def _overdue_heartbeat(session_id):
    from hermes_cli.heartbeat import HeartbeatManager, save_heartbeat
    state = HeartbeatManager(session_id).set('status tick', 60)
    state.created_at = time.time() - 120  # one interval overdue
    save_heartbeat(session_id, state)


def _heartbeat_admissions(authority, ref):
    from hermes_state_runtime import list_session_admissions
    rows = list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False)
    return [row for row in rows if row['payload'].get('local_automation_v1', {}).get('heartbeat')]


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


@pytest.mark.asyncio
@pytest.mark.parametrize('live_adapters', [False, True], ids=['no-local-adapter-yet', 'in-gateway-ticker'])
async def test_cron_result_for_a_local_origin_is_admitted_not_acknowledged(tmp_path, monkeypatch, live_adapters):
    from cron import scheduler_delivery as sd
    runner, authority, ref, _ = await _local(tmp_path, monkeypatch)
    job = {'id': 'j1', 'name': 'digest', 'deliver': 'origin',
           'origin': {'platform': 'local', 'chat_id': ref.session_id}}
    # The ticker delivers from its own thread; the owner admits on its loop.
    error = await asyncio.to_thread(sd._deliver_result, job, 'CRON_RESULT_j1',
                                    adapters=runner.adapters if live_adapters else None,
                                    loop=asyncio.get_running_loop())
    assert error is None
    assert len(_admissions(authority, ref, 'CRON_RESULT_j1')) == 1


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
