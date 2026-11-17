"""Authorized messaging Stop remains a control during a graceful restart wait.

Draft for tests/gateway/: real runner, Discord adapter, authority and SQLite.
Only the model turn and Discord send are substituted. No candidate-only imports.
"""
import asyncio
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from agent.interrupt_control import InterruptControlMixin
from tests.gateway.test_messaging_stop_owner_turn import _owner
from tests.waiting import async_wait_until


class ModelTurn(InterruptControlMixin):
    """No provider/model, but the production hard-interrupt implementation is unchanged."""

    def __init__(self):
        self._interrupt_requested = False
        self._hard_interrupt_requested = threading.Event()
        self._execution_thread_id = None
        self._active_children = []
        self._active_children_lock = threading.Lock()
        self.quiet_mode = True


@asynccontextmanager
async def _running_turn(monkeypatch):
    from gateway.platforms.event import MessageEvent
    from gateway.run_turn_runner import TurnRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import SessionRef
    from gateway.session_results import execution_result
    from gateway.turn_context import TurnContext
    from hermes_state_runtime import list_session_admissions

    runner, adapter, sent = _owner(monkeypatch)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='q1-owner')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')
    key = runner.session_store._generate_session_key(source)
    entered, release = asyncio.Event(), asyncio.Event()
    model = ModelTurn()
    state = SimpleNamespace(runner=runner, adapter=adapter, authority=authority,
                            source=source, key=key, sent=sent, model=model)

    async def turn(event, source, quick_key, run_generation):
        ctx = TurnContext(session_id=runner.session_store.peek_session_id(key), session_key=key, source=source)
        owner, sid, generation = TurnRunner(runner, ctx)._approval_owner
        assert owner is authority
        assert generation == run_generation
        owner.adopt_agent(sid, generation, model)
        state.ref = SessionRef(authority.profile_id, sid)
        state.generation = generation
        entered.set()
        await release.wait()
        execution_result.get()['result'] = {
            'final_response': '', 'messages': [], 'interrupted': model._interrupt_requested,
        }

    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    try:
        await adapter.handle_message(MessageEvent(text='work', source=source, message_id='q1-work'))
        await asyncio.wait_for(entered.wait(), 10)
        rows = list_session_admissions(authority.db, session_id=state.ref.session_id, pending_only=False)
        assert [(r['request_id'], r['status'], r['generation']) for r in rows] == [
            ('q1-work', 'started', state.generation)]
        assert authority._handle(state.ref).execution_state == 'running'
        assert authority.adopted[state.ref.session_id] == (state.generation, model)
        state.target = runner.session_store.lookup_by_session_key(key).session_id
        yield state
    finally:
        # Never cancel the canonical drain as test cleanup: let the model substitute return.
        release.set()
        await async_wait_until(
            lambda: all(live.task is None or live.task.done() for live in authority.sessions.values())
            and not adapter._background_tasks,
            timeout=10, what='owned turn and adapter tasks to settle')
        authority.db.close()


def _restart_wait(state):
    from gateway.drain_control import open_restart_wait, restart_wait_open
    state.runner._running = state.runner._draining = True
    state.runner.session_runtime_descriptor = {}
    open_restart_wait(state.runner)
    assert restart_wait_open(state.runner)
    assert state.authority._handle(state.ref).execution_generation == state.generation
    assert state.authority._handle(state.ref).execution_state == 'running'


async def _stop(state):
    from gateway.platforms.event import MessageEvent
    try:
        await asyncio.wait_for(state.adapter.handle_message(
            MessageEvent(text='/stop', source=state.source, message_id='q1-stop')), 10)
    except Exception as exc:
        # Assert the actual interruption first, even if this ingress propagates a storage error.
        return exc
    return None


def _assert_exact_stop(state, error):
    assert state.model._hard_interrupt_requested.is_set(), (
        f'authorized Stop never reached adopted generation {state.generation}; ingress error={error!r}')
    assert state.model._interrupt_requested
    assert state.authority.delivered_stops.get(state.ref.session_id) == state.generation, (
        'Stop did not traverse the authority exact-generation interrupt boundary')


def _goal(state, monkeypatch, *, parked):
    from hermes_cli import goals
    monkeypatch.setattr(goals, '_get_session_db', lambda: state.authority.db)
    manager = goals.GoalManager(state.target)
    manager.set('finish fixture work')
    if parked:
        manager.wait_for_seconds(600)
        assert manager.state.waiting_until
    return manager


@pytest.mark.asyncio
@pytest.mark.parametrize('restart_wait', [False, True], ids=['ordinary', 'restart-wait'])
async def test_authorized_stop_interrupts_adopted_generation_without_goal(monkeypatch, restart_wait):
    """Comparison oracle: this contract predates the goal-wait contribution."""
    async with _running_turn(monkeypatch) as state:
        assert state.authority.db.get_meta('goal:' + state.target) is None
        if restart_wait:
            _restart_wait(state)
        error = await _stop(state)
        _assert_exact_stop(state, error)
        assert error is None
        assert state.authority.db.get_meta('goal:' + state.target) is None


@pytest.mark.asyncio
@pytest.mark.parametrize('parked', [False, True], ids=['active', 'parked'])
async def test_restart_wait_stop_persists_pause_without_goal_resumption(monkeypatch, parked):
    """New-feature acceptance, deliberately separate from the base-pass no-goal oracle."""
    from hermes_cli import goals
    async with _running_turn(monkeypatch) as state:
        _goal(state, monkeypatch, parked=parked)
        _restart_wait(state)
        error = await _stop(state)
        _assert_exact_stop(state, error)
        assert error is None
        manager = goals.GoalManager(state.target)  # read persisted state, not the old manager's cache
        assert manager.state.status == 'paused'
        assert manager.state.paused_reason == 'stop_command'
        assert not manager.state.waiting_until
        assert manager.state.waiting_on_pid is None
        assert manager.state.waiting_on_session is None
        before = state.authority.db.get_meta('goal:' + state.target)
        # Remove the drain as an alternative explanation for no continuation.
        state.runner._draining = False
        assert not manager.is_active()
        decision = manager.evaluate_after_turn('interrupted fixture turn')
        assert decision['should_continue'] is False
        assert decision['continuation_prompt'] is None
        assert state.authority.db.get_meta('goal:' + state.target) == before


@pytest.mark.asyncio
async def test_denied_sender_cannot_stop_or_pause_during_restart_wait(monkeypatch):
    async with _running_turn(monkeypatch) as state:
        _goal(state, monkeypatch, parked=True)
        before = state.authority.db.get_meta('goal:' + state.target)
        _restart_wait(state)
        # Same real sender and route, now excluded by the actual authorization allowlist.
        monkeypatch.setenv('DISCORD_ALLOWED_USERS', '99')
        assert not state.runner._is_user_authorized_for_source(state.source)
        assert await _stop(state) is None
        assert not state.model._hard_interrupt_requested.is_set()
        assert not state.model._interrupt_requested
        assert state.ref.session_id not in state.authority.delivered_stops
        assert state.authority._handle(state.ref).execution_generation == state.generation
        assert state.authority._handle(state.ref).execution_state == 'running'
        assert state.authority.db.get_meta('goal:' + state.target) == before


@pytest.mark.asyncio
async def test_goal_pause_storage_failure_cannot_swallow_stop(monkeypatch, caplog):
    """Real SQLite failure limited to this goal write; ledger and interruption remain available."""
    async with _running_turn(monkeypatch) as state:
        _goal(state, monkeypatch, parked=True)
        before = state.authority.db.get_meta('goal:' + state.target)

        def install_failure(conn):
            # SQL literals are quoted from a fixture-owned identifier, not user input.
            key = ('goal:' + state.target).replace("'", "''")
            for operation in ('INSERT', 'UPDATE'):
                conn.execute(f'''CREATE TRIGGER q1_goal_fail_{operation.lower()}
                    BEFORE {operation} ON state_meta WHEN NEW.key = '{key}'
                    BEGIN SELECT RAISE(ABORT, 'q1_goal_pause_write_failed'); END''')
        state.authority.db._execute_write(install_failure)
        caplog.clear()
        error = await _stop(state)
        _assert_exact_stop(state, error)
        assert state.authority.db.get_meta('goal:' + state.target) == before, 'failed pause partially committed'
        surfaced = '\n'.join([str(error or ''), caplog.text, *map(str, state.sent)])
        assert 'q1_goal_pause_write_failed' in surfaced, 'goal persistence failure silently disappeared'
