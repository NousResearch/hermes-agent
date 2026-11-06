"""Characterize the installed pause verb with the actual unpatched admission/after-turn path.

Only the socket bind and final service stop are replaced. No network or inference.
"""
import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from gateway.control_socket import GatewayControlServer
from gateway.run_bootstrap import _start_gateway_start_control_socket
from gateway.runtime_bootstrap import TicketStore
from gateway.session_authority import SessionAuthority
from hermes_state_runtime import RuntimeStoreError, get_session_admission
from tests.gateway.test_external_drain_authority_fence import _admit, native_lane as native_lane


async def _control(runner, monkeypatch):
    runner.session_runtime_descriptor = {'instance_id': 'fixture', 'state': 'ready', 'capabilities': [],
                                         'served_profiles': ['default']}
    runner.session_ticket_store = TicketStore('fixture', ['default'])
    monkeypatch.setattr(GatewayControlServer, 'start', AsyncMock(return_value=True))
    return await _start_gateway_start_control_socket(runner)


async def _pause(server):
    # Run the real socket dispatch and bootstrap-created executor handler. The
    # handler marshals real request_restart onto the owner loop before ACK.
    return json.loads((await asyncio.to_thread(server.handle_request_line,
        b'{"protocol":1,"id":1,"verb":"pause-for-update"}')).decode())


@pytest.mark.asyncio
async def test_existing_pause_fences_native_before_ack_when_already_idle(native_lane, monkeypatch):
    runner, authority, _adapter, event = native_lane
    receipt = await _admit(runner, authority, event)
    server = await _control(runner, monkeypatch)
    stopped = []

    async def stop(**kwargs):
        stopped.append((runner._active_work_count(), kwargs))

    monkeypatch.setattr(runner, 'stop', stop)
    try:
        ack = await _pause(server)
        assert ack['ok'] and ack['result']['pausing'] is True
        assert runner._draining is True
        with pytest.raises(RuntimeStoreError, match='runtime_draining'):
            await _admit(runner, authority, event)
        with pytest.raises(RuntimeStoreError, match='runtime_draining'):
            await authority._claim_next(receipt.ref, authority.sessions[receipt.ref.session_id])
        await asyncio.wait_for(runner._restart_task, 5)
        assert stopped == [(0, {'restart': True, 'detached_restart': False, 'service_restart': True})]
    finally:
        server.cleanup_files()


@pytest.mark.asyncio
async def test_existing_pause_cap_proceeds_to_stop_with_actual_execution_active(native_lane, monkeypatch):
    runner, authority, _adapter, event = native_lane
    receipt = await _admit(runner, authority, event)
    entered, release = asyncio.Event(), asyncio.Event()
    stop_counts = []

    async def execute(owner, ref, row):
        entered.set()
        await asyncio.wait_for(release.wait(), 5)
        return 'fixture'

    async def stop(**kwargs):
        stop_counts.append(runner._active_work_count())

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    monkeypatch.setattr(runner, 'stop', stop)
    runner._restart_after_turn_timeout = 0.01  # finite cap scaled only for this characterization
    SessionAuthority._schedule(authority, receipt.ref)
    await asyncio.wait_for(entered.wait(), 5)
    assert runner._active_work_count() > 0
    server = await _control(runner, monkeypatch)
    try:
        ack = await _pause(server)
        assert ack['ok'] and ack['result']['pausing'] is True
        await asyncio.wait_for(runner._restart_task, 5)
        assert not release.is_set()
        assert stop_counts and stop_counts[0] > 0
        assert get_session_admission(authority.db, admission_id=receipt.admission_id)['status'] == 'started'
    finally:
        release.set()
        await asyncio.wait_for(authority.sessions[receipt.ref.session_id].task, 5)
        server.cleanup_files()


@pytest.mark.asyncio
async def test_global_pause_is_not_an_authority_admission_or_claim_fence(native_lane, monkeypatch):
    from agent.estop import engage, is_engaged
    from dataclasses import replace
    runner, authority, _adapter, event = native_lane
    first = await _admit(runner, authority, event)
    engage(reason='fixture-maintenance')
    assert is_engaged()
    assert runner._hm_estop_gate(event, event.source, False) is not None
    assert runner._hm_estop_gate(event, event.source, True) is None
    # Already running sessions and recognized commands deliberately bypass ESTOP.
    with monkeypatch.context() as active:
        active.setattr(runner, '_is_session_running', lambda key: True)
        assert runner._hm_estop_gate(event, event.source, False) is None
    new_receipt = await _admit(runner, authority, replace(event, message_id='direct-native-under-pause'))
    assert new_receipt.status == 'queued'
    claim, _ = await authority._claim_next(first.ref, authority.sessions[first.ref.session_id])
    assert claim['status'] == 'started'
