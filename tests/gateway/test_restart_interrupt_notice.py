"""#134889 F1: ``agent.restart_drain_timeout`` stays 0, and a turn a restart interrupts says so.

With the 0 s default a restart or stop interrupts in-flight turns at once. The admission settles
``interrupted`` (never replayed: ``tests/gateway/test_journey_new_session_queued_work.py`` restart
case, ``tests/gateway/test_prompt_queued_transfer.py`` owner-restart case), and its
``message.complete`` carries a named notice in ``warning``, the slot Ink and Desktop already render.
"""
import asyncio
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from gateway.session_settlement_recovery import (
    RESTART_INTERRUPT_NOTICE,
    SHUTDOWN_INTERRUPT_NOTICE,
    _publish_completion,
)
from tui_gateway.contracts.registry import EVENTS


def test_restart_drain_timeout_default_stays_zero():
    from gateway.restart import DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG['agent']['restart_drain_timeout'] == 0
    assert DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT == 0.0


class _Stream:
    def __init__(self):
        self.frames = []

    def publish(self, session_id, payload):
        self.frames.append(payload)


def _publish(outcome, captured=None):
    stream = _Stream()
    live = SimpleNamespace(controls=SimpleNamespace(snapshot=lambda *a: None), event_stream=stream)
    row = {'admission_id': 'adm-1', 'request_id': 'sub-1'}
    _publish_completion(SimpleNamespace(runner=None), live, SimpleNamespace(session_id='s'), row,
                        {'outcome': outcome}, 'partial reply', captured)
    (frame,) = stream.frames
    return frame


def _cause(message, **extra):
    return {'result': {'interrupted': True, 'interrupt_message': message, **extra}}


@pytest.mark.parametrize(('cause', 'notice'), [('Gateway restarting', RESTART_INTERRUPT_NOTICE),
                                               ('Gateway shutting down', SHUTDOWN_INTERRUPT_NOTICE)])
def test_turn_interrupted_by_the_owner_stop_carries_a_named_notice(cause, notice):
    frame = _publish('interrupted', _cause(cause))
    assert frame['status'] == 'interrupted' and frame['warning'] == notice
    assert 'will not be run again' in notice
    EVENTS['message.complete'].payload.model_validate({k: frame[k] for k in ('text', 'status', 'warning')})


def test_other_endings_get_no_restart_notice():
    # A completed turn, a user Stop (no recorded cause, or its own) and a turn with no result.
    assert 'warning' not in _publish('completed', _cause('Gateway restarting'))
    assert 'warning' not in _publish('interrupted', {'result': {'interrupted': True}})
    assert 'warning' not in _publish('interrupted', _cause('User requested stop'))
    assert 'warning' not in _publish('interrupted')


def test_a_turn_warning_is_not_overwritten():
    captured = _cause('Gateway restarting', local_viewer=True, warning='own warning')
    assert _publish('interrupted', captured)['warning'] == 'own warning'


def test_classic_cli_view_prints_the_restart_notice_for_an_interrupted_turn(capsys):
    """The attached classic CLI (``hermes chat``) prints the notice on stderr; a completed turn's
    warning stays unprinted there, as before."""
    from hermes_cli.gateway_chat_view import GatewayChatView

    view = GatewayChatView(object(), {'stored_session_id': 'sid'})
    view._complete('a1', {'text': '', 'status': 'interrupted', 'warning': RESTART_INTERRUPT_NOTICE})
    view._complete('a2', {'text': 'done', 'status': 'complete', 'warning': 'other warning'})
    err = capsys.readouterr().err
    assert RESTART_INTERRUPT_NOTICE in err and 'other warning' not in err


@pytest.mark.asyncio
async def test_user_stop_is_not_relabelled_when_shutdown_starts_during_settlement(tmp_path):
    """A user Stop whose settlement a competing SQLite writer delays until after shutdown began is
    still the user's Stop: the notice follows the turn's recorded cause, not the runner's flags."""
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import SessionRef
    from gateway.session_results import admission_result
    from gateway.session_settlement_recovery import commit_and_publish
    from hermes_state import SessionDB
    from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input

    runner = SimpleNamespace(_draining=False, _running=True, _restart_requested=False)
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('s', source='test')
        epoch = begin_runtime_epoch(db, instance_id='owner')
        authority = SessionAuthority(runner, profile_id='owned', instance_id='owner', db=db, epoch=epoch)
        live = authority.sessions['s'] = LiveSession(None, 'route')
        ref = SessionRef('owned', 's')
        admit_session_input(db, epoch=epoch, principal_id='person', session_id='s', request_id='r',
                            payload={'text': 'please work'})
        row = claim_session_input(db, epoch=epoch, session_id='s')
        authority._stamp_claim(ref, live, row)
        frames = []
        live.event_stream.observers.add(frames.append)
        captured = {'result': {'final_response': '', 'messages': [], 'interrupted': True,
                               'completed': False, 'interrupt_message': 'User requested stop'}}
        reached = threading.Event()
        settlement = {}
        lock = sqlite3.connect(tmp_path / 'state.db')
        lock.execute('BEGIN IMMEDIATE')

        def trace(frame, event, arg):
            if event == 'call' and frame.f_code.co_name == 'settle_session_input':
                reached.set()
            return trace

        def commit():
            sys.settrace(trace)
            try:
                commit_and_publish(authority, live, ref, row, '', 'interrupted', captured, settlement)
            finally:
                sys.settrace(None)

        with ThreadPoolExecutor(max_workers=1) as pool:
            task = pool.submit(commit)
            try:
                assert await asyncio.to_thread(reached.wait, 10), 'settlement not reached'
                assert not task.done(), 'competing SQLite writer should block settlement'
                runner._draining = True
                runner._running = False
                lock.rollback()
                await asyncio.wrap_future(task)
            finally:
                lock.close()
        complete = [f['params']['payload'] for f in frames if f['params']['type'] == 'message.complete']
        persisted = admission_result(db, row['admission_id'])
        assert len(complete) == 1
        assert persisted['result']['interrupt_message'] == 'User requested stop'
        assert 'warning' not in complete[0], complete[0]


STOPPABLE_CHILD = """import json, os, sys, psutil
me = psutil.Process()
out = os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0)
def send(frame):
    out.write((json.dumps(frame) + '\\n').encode())
send({'type': 'hello', 'pid': me.pid, 'birth': me.create_time(), 'ancestors': [p.pid for p in me.parents()[:3]]})
for line in sys.stdin.buffer:
    frame = json.loads(line)
    if frame == {'type': 'stop'}:
        send({'type': 'result', 'result': {'final_response': '', 'interrupted': True, 'completed': False}})
    elif frame == {'type': 'finish'}:
        send({'type': 'finished'})
        break
"""


@pytest.mark.platforms("linux")
@pytest.mark.parametrize('user_stopped_first', [False, True])
def test_managed_worker_turn_settles_with_the_cause_of_the_stop_that_reached_it(tmp_path, monkeypatch,
                                                                                  user_stopped_first):
    """A managed-worker turn (real child, real SQLite authority) that the gateway restart stops
    settles with the restart notice; one the user had already stopped before shutdown began gets
    none. The worker's wire result carries no cause: the owner records which stop it sent."""
    import subprocess

    from gateway import session_managed_worker as managed
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import Principal, SessionRef, Submission
    from gateway.session_policy import build_policy
    from gateway.session_settlement_recovery import commit_and_publish
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch, claim_session_input
    from tests.gateway.restart_test_helpers import make_restart_runner
    from tests.waiting import async_wait_until

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    runner, _adapter = make_restart_runner()
    runner._restart_requested = True
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('s', source='test')
    authority = SessionAuthority(runner, profile_id='owned', instance_id='i', db=db,
                                 epoch=begin_runtime_epoch(db, instance_id='i'))
    runner.session_authority = authority
    live = authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='human', chat_id='c'), 'route')
    actor = Principal('human', 'owned', frozenset({'session:submit', 'session:control'}), 'cli')
    ref = SessionRef('owned', 's')
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)

    class Child(subprocess.Popen):
        def __init__(self, args, **kw):
            super().__init__([sys.executable, '-c', STOPPABLE_CHILD], **kw)
    monkeypatch.setattr(managed.subprocess, 'Popen', Child)
    policy = build_policy({'cwd': str(tmp_path), 'model': 'm'}, {}, profile_terminal=False)

    async def scenario():
        await authority.submit(actor, Submission('turn', ref, {'text': 'go'}, 'queue'))
        row = claim_session_input(db, epoch=authority.epoch, session_id='s')
        authority._stamp_claim(ref, live, row)
        task = asyncio.create_task(managed.execute_managed(authority, ref, row, policy))
        await async_wait_until(lambda: (w := getattr(authority, '_managed_workers', {}).get('s')) is not None and w.writer.is_alive(),
                               timeout=30, what='the managed worker to receive its bootstrap')
        if user_stopped_first:
            assert managed.interrupt_managed(authority, actor, ref, row['generation'])
        runner._interrupt_running_agents(runner._shutdown_interrupt_reason())  # stop()'s timed-out path
        async with asyncio.timeout(30):
            return row, await task

    with db:
        row, response = asyncio.run(scenario())
        frames = []
        live.event_stream.observers.add(frames.append)
        commit_and_publish(authority, live, ref, row, response, 'interrupted',
                           authority.pending_results[row['admission_id']], {})
    (complete,) = [f['params']['payload'] for f in frames if f['params']['type'] == 'message.complete']
    assert complete['status'] == 'interrupted'
    if user_stopped_first:
        assert 'warning' not in complete, complete
    else:
        assert complete['warning'] == RESTART_INTERRUPT_NOTICE


@pytest.mark.parametrize('user_stopped_first', [False, True])
def test_in_process_turn_keeps_the_user_stop_when_shutdown_interrupts_it_again(user_stopped_first):
    """An in-process turn the user stopped is still unwinding (e.g. inside a tool) when the restart
    interrupts its agent: the shutdown reason must not overwrite the pending Stop, so the turn
    records no restart cause and gets no notice. A turn only the restart stopped still gets it."""
    from agent.interrupt_compat import request_hard_interrupt
    from gateway.session_settlement_recovery import owner_stop_notice
    from run_agent import AIAgent
    from tests.gateway.restart_test_helpers import make_restart_runner
    from tools.interrupt import set_interrupt

    agent = AIAgent.__new__(AIAgent)
    agent._interrupt_requested = False
    agent._interrupt_message = agent._tool_interrupt_reason = None
    agent._hard_interrupt_requested = threading.Event()
    agent._execution_thread_id = None
    agent._interrupt_thread_signal_pending = False
    agent._active_children, agent._active_children_lock = [], threading.Lock()
    agent.quiet_mode = True
    runner, _adapter = make_restart_runner()
    runner._restart_requested = True
    runner._running_agents = {'route': agent}
    try:
        if user_stopped_first:
            request_hard_interrupt(agent)  # SessionAuthority.interrupt: the user's Stop carries no message
        runner._interrupt_running_agents(runner._shutdown_interrupt_reason())
        assert agent._interrupt_requested and agent._hard_interrupt_requested.is_set()
        result = {'interrupted': True, **({'interrupt_message': agent._interrupt_message} if agent._interrupt_message else {})}
        notice = owner_stop_notice({'result': result}, 'interrupted')
        if user_stopped_first:
            assert agent._interrupt_message is None and notice is None
        else:
            assert agent._interrupt_message == 'Gateway restarting' and notice == RESTART_INTERRUPT_NOTICE
    finally:
        set_interrupt(False)


def _redirect_agent():
    from run_agent import AIAgent
    agent = AIAgent.__new__(AIAgent)
    agent._interrupt_requested = False
    agent._interrupt_message = agent._tool_interrupt_reason = None
    agent._hard_interrupt_requested = threading.Event()
    agent._execution_thread_id = None
    agent._interrupt_thread_signal_pending = False
    agent._active_children, agent._active_children_lock = [], threading.Lock()
    agent.quiet_mode = True
    agent._pending_redirect = None
    agent._pending_redirect_lock = threading.Lock()
    agent._model_request_active = threading.Event(); agent._model_request_active.set()
    agent._executing_tools = False
    agent._lock = threading.Lock()
    agent.turn_active = True
    return agent


def test_restart_after_a_pending_redirect_still_gets_the_restart_notice():
    """A mid-request correction is not a Stop: the restart drops it and is the turn's cause."""
    from gateway.session_settlement_recovery import owner_stop_notice
    from tests.gateway.restart_test_helpers import make_restart_runner
    from tools.interrupt import set_interrupt
    agent = _redirect_agent()
    runner, _ = make_restart_runner()
    runner._restart_requested = True
    runner._running_agents = {'route': agent}
    try:
        assert agent.redirect("actually use python 3.12") is True
        runner._interrupt_running_agents(runner._shutdown_interrupt_reason())
        result = {'interrupted': True, **({'interrupt_message': agent._interrupt_message} if agent._interrupt_message else {})}
        notice = owner_stop_notice({'result': result}, 'interrupted')
        assert agent._pending_redirect is None  # the restart dropped the correction: the turn ends interrupted
        assert notice == RESTART_INTERRUPT_NOTICE, (agent._interrupt_message, notice)
    finally:
        set_interrupt(False)
