"""Canonical busy corrections stay visible: the authority reports where the text
landed, and ``session.pending`` / ``session.steer_clear`` read and drop that slot.

Shared-gateway TUI mode talks to this authority, not ``tui_gateway``. A tool-time
``session.redirect`` is stored as a pending steer (the agent will not kill the
tool); a model-time redirect waits in the redirect slot. The pending snapshot is
what the queue strip and a reconnect render.
"""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from agent.interrupt_control import InterruptControlMixin
from gateway.session_busy_controls import clear_steer, correct, pending_view
from gateway.session_contract import Principal
from tests.gateway.test_session_authority_cancel_settlement import REF, _authority, _submit


OWNER = Principal(
    'human', 'owned', frozenset({'session:submit', 'session:read', 'session:control'}), 'tui')


class CorrectionAgent(InterruptControlMixin):
    _supports_active_turn_redirect = True

    def __init__(self, *, executing_tools=False, model_active=False):
        self._pending_steer = None
        self._pending_redirect = None
        self._interrupt_requested = False
        self._executing_tools = executing_tools
        self._model_request_active = threading.Event()
        if model_active:
            self._model_request_active.set()


def _connection(authority):
    return SimpleNamespace(authority=authority, actor=OWNER)


def _install(authority, agent):
    authority.runner._cached_agent_for = {'route': agent}.get
    authority.runner._running_agents = {'route': agent}


async def _running(monkeypatch, authority):
    from gateway import session_finite

    ready, release = asyncio.Event(), asyncio.Event()

    async def execute(authority, ref, row):
        ready.set()
        await asyncio.wait_for(release.wait(), 5)
        return 'done'

    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    await _submit(authority, 'running')
    await asyncio.wait_for(ready.wait(), 5)
    return release, authority._handle(REF).execution_generation


@pytest.mark.asyncio
async def test_tool_time_redirect_is_a_pending_steer_the_snapshot_and_clear_share(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    agent = CorrectionAgent(executing_tools=True)
    _install(authority, agent)
    connection = _connection(authority)
    with db:
        release, generation = await _running(monkeypatch, authority)
        try:
            reply = await correct(connection, REF, {
                'session_id': 's', 'text': 'add the marker word', 'execution_generation': generation,
            }, verb='redirect')
            assert reply['status'] == 'redirected'
            assert reply['pending'] == 'steer'
            snap = await pending_view(connection, REF, {'session_id': 's'})
            assert snap['steer'] == 'add the marker word'
            assert 'redirect' not in snap
            assert snap['queue'] == []
            cleared = await clear_steer(connection, REF, {'session_id': 's'})
            assert cleared == {'status': 'cleared', 'cleared': True}
            assert agent._pending_steer is None
            assert (await pending_view(connection, REF, {'session_id': 's'}))['steer'] is None
        finally:
            release.set()
            await asyncio.wait_for(authority.sessions['s'].task, 5)


@pytest.mark.asyncio
async def test_model_redirect_reports_the_redirect_slot(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    agent = CorrectionAgent(model_active=True)
    _install(authority, agent)
    connection = _connection(authority)
    with db:
        release, generation = await _running(monkeypatch, authority)
        try:
            reply = await correct(connection, REF, {
                'session_id': 's', 'text': 'answer with Monday only', 'execution_generation': generation,
            }, verb='redirect')
            assert reply['status'] == 'redirected'
            assert reply['pending'] == 'redirect'
            snap = await pending_view(connection, REF, {'session_id': 's'})
            assert snap['steer'] is None
            assert snap['redirect'] == 'answer with Monday only'
        finally:
            release.set()
            await asyncio.wait_for(authority.sessions['s'].task, 5)


@pytest.mark.asyncio
async def test_steer_before_adoption_is_in_the_pending_snapshot_until_cleared(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    connection = _connection(authority)
    with db:
        release, generation = await _running(monkeypatch, authority)
        try:
            reply = await correct(connection, REF, {
                'session_id': 's', 'text': 'use SQLite', 'execution_generation': generation,
            }, verb='steer')
            assert reply['status'] == 'queued'
            assert reply['pending'] == 'steer'
            assert (await pending_view(connection, REF, {'session_id': 's'}))['steer'] == 'use SQLite'
            assert (await clear_steer(connection, REF, {'session_id': 's'}))['cleared'] is True
            assert ref_pending(authority) == []
            assert (await clear_steer(connection, REF, {'session_id': 's'})) == {
                'status': 'noop', 'cleared': False}
        finally:
            release.set()
            await asyncio.wait_for(authority.sessions['s'].task, 5)


def ref_pending(authority):
    _generation, texts = authority.pending_steers.get('s', (None, []))
    return list(texts)
