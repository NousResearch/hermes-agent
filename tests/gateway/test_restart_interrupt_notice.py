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
        assert _notices(db) == []
    else:
        assert complete['warning'] == RESTART_INTERRUPT_NOTICE
        assert _notices(db) == [RESTART_INTERRUPT_NOTICE]  # GW-21: a reconnecting viewer reads it too


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


@pytest.mark.parametrize('cause', ['Gateway restarting', 'Gateway shutting down'])
@pytest.mark.parametrize('row', [
    {'principal_id': 'cron-owner', 'payload': {'text': 'job'}},
    {'principal_id': 'automation:agent:main:telegram:dm:x', 'payload': {'text': 'done'}},
    {'principal_id': 'person', 'payload': {'text': 'hook', 'native_text_v1': {'webhook_delivery': {}}}},
])
def test_automation_turn_the_owner_stop_interrupted_is_never_asked_to_resend(cause, row):
    """#134889 GW-20: nobody sent an unattended turn, so the owner-stop notice is the neutral
    automation copy (GW-10's rule), never "Send your message again"."""
    from agent.turn_failure_copy import AUTOMATION_INTERRUPTED_NOTICE
    stream = _Stream()
    live = SimpleNamespace(controls=SimpleNamespace(snapshot=lambda *a: None), event_stream=stream)
    _publish_completion(SimpleNamespace(runner=None), live, SimpleNamespace(session_id='s'),
                        {'admission_id': 'adm-1', 'request_id': 'sub-1', **row},
                        {'outcome': 'interrupted'}, '', _cause(cause))
    (frame,) = stream.frames
    assert frame['warning'] == AUTOMATION_INTERRUPTED_NOTICE
    assert 'again' not in frame['warning'].replace('not run again', '')


def _settle_on_real_store(tmp_path, cause, principal='person', payload=None):
    """One claimed admission on a real SessionDB settled through commit_and_publish (the owner's
    worker-thread settlement) with *cause* recorded. Returns (db, row, frames, settlement)."""
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import SessionRef
    from gateway.session_settlement_recovery import commit_and_publish
    from hermes_state import SessionDB
    from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input

    runner = SimpleNamespace(_draining=False, _running=True, _restart_requested=True)
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('s', source='test')
    epoch = begin_runtime_epoch(db, instance_id='owner')
    authority = SessionAuthority(runner, profile_id='owned', instance_id='owner', db=db, epoch=epoch)
    live = authority.sessions['s'] = LiveSession(None, 'route')
    ref = SessionRef('owned', 's')
    admit_session_input(db, epoch=epoch, principal_id=principal, session_id='s', request_id='r',
                        payload=payload or {'text': 'please work'})
    row = claim_session_input(db, epoch=epoch, session_id='s')
    authority._stamp_claim(ref, live, row)
    db.append_message('s', 'user', 'please work')  # the turn's own rows, as the agent wrote them
    db.append_message('s', 'assistant', 'partial reply')
    frames = []
    live.event_stream.observers.add(frames.append)
    result = {'final_response': 'partial reply', 'messages': [], 'interrupted': True, 'completed': False}
    if cause:
        result['interrupt_message'] = cause
    settlement = {}
    commit_and_publish(authority, live, ref, row, 'partial reply', 'interrupted', {'result': result}, settlement)
    return db, authority, row, frames, settlement


def _notices(db):
    from agent.turn_failure_copy import FAILED_TURN_DISPLAY_KIND
    return [m['content'] for m in db.get_messages('s') if m.get('display_kind') == FAILED_TURN_DISPLAY_KIND]


@pytest.mark.parametrize(('cause', 'notice'), [('Gateway restarting', RESTART_INTERRUPT_NOTICE),
                                               ('Gateway shutting down', SHUTDOWN_INTERRUPT_NOTICE)])
def test_owner_stop_reason_is_in_the_transcript_for_a_viewer_that_reconnects(tmp_path, cause, notice):
    """#134889 GW-21: a viewer that reconnects after the restart reads the transcript, not the live
    frame. The reason commits as a Hermes notice in the settlement's own write, once; the live
    ``message.complete.warning`` stays."""
    from gateway.session_results import finish_result
    db, authority, row, frames, settlement = _settle_on_real_store(tmp_path, cause)
    with db:
        assert settlement['settled']['outcome'] == 'interrupted'
        assert _notices(db) == [notice]
        (complete,) = [f['params']['payload'] for f in frames if f['params']['type'] == 'message.complete']
        assert complete['warning'] == notice
        # A settlement retry that reads the committed row back never writes the notice again.
        finish_result(db, epoch=authority.epoch, row=row, response='partial reply', outcome='interrupted',
                      result={'result': {'final_response': 'partial reply', 'messages': [], 'interrupted': True,
                                         'interrupt_message': cause}})
        assert _notices(db) == [notice]


@pytest.mark.parametrize('cause', [None, 'User requested stop'])
def test_a_user_stop_writes_no_transcript_notice(tmp_path, cause):
    db, *_ = _settle_on_real_store(tmp_path, cause)
    with db:
        assert _notices(db) == []


def test_automation_turn_transcript_notice_is_the_neutral_copy(tmp_path):
    from agent.turn_failure_copy import AUTOMATION_INTERRUPTED_NOTICE
    db, *_ = _settle_on_real_store(tmp_path, 'Gateway restarting', principal='cron-owner')
    with db:
        assert _notices(db) == [AUTOMATION_INTERRUPTED_NOTICE]



def _core_closed_turn(tmp_path, monkeypatch, *, principal='person', earlier_failed=False):
    """A claimed admission whose agent was interrupted by the restart before any assistant text:
    the real AIAgent persists the user row and the real core closer
    (``agent/conversation_loop.py::_close_durable_failed_turn``) writes its failed-turn boundary.
    ``earlier_failed``: a previous turn already ended on a failed-turn boundary before this claim."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    from agent.conversation_loop import _close_durable_failed_turn
    from agent.turn_failure_copy import FAILED_TURN_DISPLAY_KIND, FAILED_TURN_NOTICE
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import SessionRef
    from hermes_state import SessionDB
    from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input
    from run_agent import AIAgent

    runner = SimpleNamespace(_draining=False, _running=True, _restart_requested=True)
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('s', source='test')
    if earlier_failed:
        db.append_message('s', 'user', 'earlier request')
        db.append_message('s', 'assistant', FAILED_TURN_NOTICE, display_kind=FAILED_TURN_DISPLAY_KIND)
    epoch = begin_runtime_epoch(db, instance_id='owner')
    authority = SessionAuthority(runner, profile_id='owned', instance_id='owner', db=db, epoch=epoch)
    live = authority.sessions['s'] = LiveSession(None, 'route')
    ref = SessionRef('owned', 's')
    payload = {'text': 'please work'}
    admit_session_input(db, epoch=epoch, principal_id=principal, session_id='s', request_id='r', payload=payload)
    row = claim_session_input(db, epoch=epoch, session_id='s')
    authority._stamp_claim(ref, live, row)
    result = {'final_response': '', 'interrupted': True, 'completed': False, 'interrupt_message': 'Gateway restarting'}
    if not earlier_failed:
        agent = AIAgent(session_db=db, session_id='s', model='test-model', provider='openai-compat', api_key='test',
                        base_url='http://127.0.0.1:1/v1', quiet_mode=True, skip_context_files=True, skip_memory=True)
        messages = [{'role': 'user', 'content': 'please work'}]
        agent._flush_messages_to_session_db(messages)  # the turn-start flush of the accepted user row
        result['messages'] = messages
        _close_durable_failed_turn(agent, result)
    return db, authority, live, ref, row, result


@pytest.mark.parametrize('principal', ['person', 'cron-owner'])
def test_restart_notice_replaces_the_core_failed_turn_boundary_of_the_same_turn(tmp_path, monkeypatch, principal):
    """#134889 GW-21 review (GW-A): the agent's own closer already ended this turn with
    "Send it again"; settlement replaces that boundary with the restart notice in its own write,
    so the transcript holds one Hermes boundary, and an automation row never keeps resend advice."""
    from agent.turn_failure_copy import AUTOMATION_INTERRUPTED_NOTICE, FAILED_TURN_NOTICE
    from gateway.session_settlement_recovery import commit_and_publish
    db, authority, live, ref, row, result = _core_closed_turn(tmp_path, monkeypatch, principal=principal)
    with db:
        assert _notices(db) == [FAILED_TURN_NOTICE]  # the real core closer wrote it
        commit_and_publish(authority, live, ref, row, '', 'interrupted', {'result': result}, {})
        expected = RESTART_INTERRUPT_NOTICE if principal == 'person' else AUTOMATION_INTERRUPTED_NOTICE
        assert _notices(db) == [expected]
        assert [m['role'] for m in db.get_messages('s')] == ['user', 'assistant']


def test_an_earlier_turns_failed_boundary_is_kept_and_the_restart_notice_appended(tmp_path, monkeypatch):
    """A failed-turn boundary written before this claim belongs to an earlier turn: it stays."""
    from agent.turn_failure_copy import FAILED_TURN_NOTICE
    from gateway.session_settlement_recovery import commit_and_publish
    db, authority, live, ref, row, result = _core_closed_turn(tmp_path, monkeypatch, earlier_failed=True)
    with db:
        commit_and_publish(authority, live, ref, row, '', 'interrupted', {'result': result}, {})
        assert _notices(db) == [FAILED_TURN_NOTICE, RESTART_INTERRUPT_NOTICE]


@pytest.mark.platforms("linux")
@pytest.mark.parametrize('ending', ['spawned', 'cancelled'])
def test_shutdown_stop_before_the_managed_worker_spawned_carries_its_cause(tmp_path, monkeypatch, ending):
    """#134889 GW-22: the gateway's stop lands while the managed worker is still being spawned
    (no worker to signal yet). The authority latches the cause for that generation, so the
    pre-bootstrap interrupted result carries it: the spawned child is stopped before it is ever
    bootstrapped, or the cancelled spawn settles with the cause. Real child, real SQLite."""
    import subprocess

    from gateway import session_managed_worker as managed
    from gateway.run_runtime import stop_authority_turns
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import Principal, SessionRef, Submission
    from gateway.session_policy import build_policy
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch, claim_session_input
    from tests.gateway.restart_test_helpers import make_restart_runner
    from tests.waiting import async_wait_until

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    runner, _adapter = make_restart_runner()
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('s', source='test')
    authority = SessionAuthority(runner, profile_id='owned', instance_id='i', db=db,
                                 epoch=begin_runtime_epoch(db, instance_id='i'))
    live = authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='human', chat_id='c'), 'route')
    actor = Principal('human', 'owned', frozenset({'session:submit', 'session:control'}), 'cli')
    ref = SessionRef('owned', 's')
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    spawning, release, bootstrapped = threading.Event(), threading.Event(), []

    class Child(subprocess.Popen):
        def __init__(self, args, **kw):
            spawning.set()
            release.wait(30)
            super().__init__([sys.executable, '-c', STOPPABLE_CHILD], **kw)
    monkeypatch.setattr(managed.subprocess, 'Popen', Child)
    monkeypatch.setattr(managed, '_bootstrap', lambda *a: bootstrapped.append(a) or pytest.fail('bootstrapped'))
    policy = build_policy({'cwd': str(tmp_path), 'model': 'm'}, {}, profile_terminal=False)

    async def scenario():
        await authority.submit(actor, Submission('turn', ref, {'text': 'go'}, 'queue'))
        row = claim_session_input(db, epoch=authority.epoch, session_id='s')
        authority._stamp_claim(ref, live, row)
        task = asyncio.create_task(managed.execute_managed(authority, ref, row, policy))
        await asyncio.to_thread(spawning.wait, 30)
        assert stop_authority_turns(authority, reason='Gateway restarting') == 1
        if ending == 'cancelled':
            task.cancel()
        release.set()
        async with asyncio.timeout(20):
            await async_wait_until(task.done, timeout=20, what='the stopped turn to end')
        return row, await task

    with db:
        row, response = asyncio.run(scenario())
    result = authority.pending_results[row['admission_id']]['result']
    assert response == '' and not bootstrapped
    assert result['interrupted'] and result['interrupt_message'] == 'Gateway restarting'



@pytest.mark.platforms("linux")
@pytest.mark.parametrize('stopper', ['shutdown', 'user'])
def test_a_stop_during_context_reference_expansion_never_spawns_the_worker(tmp_path, monkeypatch, stopper):
    """#134889 GW-22 review (GW-B): ``managed_turn_text`` (``@`` expansion; it may fetch the model's
    context length) runs before the worker exists. A gateway shutdown Stop there is latched for the
    generation, so no worker is spawned and the turn settles interrupted with the restart cause. A
    user's Stop there is honoured too and leaves no cause. Real authority and SQLite; only the
    expansion is slowed and Popen is observed."""
    import subprocess

    from gateway import session_context_refs
    from gateway import session_managed_worker as managed
    from gateway.run_runtime import stop_authority_turns
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import Principal, SessionRef, Submission
    from gateway.session_policy import build_policy
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch, claim_session_input
    from tests.gateway.restart_test_helpers import make_restart_runner

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    runner, _adapter = make_restart_runner()
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('s', source='test')
    authority = SessionAuthority(runner, profile_id='owned', instance_id='i', db=db,
                                 epoch=begin_runtime_epoch(db, instance_id='i'))
    live = authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='human', chat_id='c'), 'route')
    actor = Principal('human', 'owned', frozenset({'session:submit', 'session:control'}), 'cli')
    ref = SessionRef('owned', 's')
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    expanding, release, spawned = threading.Event(), threading.Event(), []

    async def slow_expansion(authority, row, policy):  # stands in for a slow context-length lookup
        expanding.set()
        await asyncio.to_thread(release.wait, 30)
        return None, None
    monkeypatch.setattr(session_context_refs, 'managed_turn_text', slow_expansion)

    class Child(subprocess.Popen):
        def __init__(self, args, **kw):
            spawned.append(True)
            super().__init__([sys.executable, '-c', STOPPABLE_CHILD], **kw)
    monkeypatch.setattr(managed.subprocess, 'Popen', Child)
    policy = build_policy({'cwd': str(tmp_path), 'model': 'm'}, {}, profile_terminal=False)

    async def scenario():
        await authority.submit(actor, Submission('turn', ref, {'text': 'look at @file:x'}, 'queue'))
        row = claim_session_input(db, epoch=authority.epoch, session_id='s')
        authority._stamp_claim(ref, live, row)
        task = asyncio.create_task(managed.execute_managed(authority, ref, row, policy))
        await asyncio.to_thread(expanding.wait, 30)
        if stopper == 'shutdown':
            assert stop_authority_turns(authority, reason='Gateway shutting down') == 1
        else:
            await authority.interrupt(actor, ref, row['generation'])
        release.set()
        async with asyncio.timeout(30):
            return row, await task

    with db:
        row, response = asyncio.run(scenario())
    result = authority.pending_results[row['admission_id']]['result']
    assert response == '' and spawned == [] and result['interrupted']
    if stopper == 'shutdown':
        assert result['interrupt_message'] == 'Gateway shutting down'
    else:
        assert 'interrupt_message' not in result
    assert 's' not in authority._managed_spawning and 's' not in authority.pending_stops


def test_a_user_stop_first_keeps_no_shutdown_cause_before_spawn():
    """A user's Stop latched for the pending generation came first: shutdown records no cause."""
    from gateway.run_runtime import stop_authority_turns
    live = SimpleNamespace(event_stream=SimpleNamespace(execution={'execution_generation': 4}))
    authority = SimpleNamespace(_managed_workers={}, _managed_spawning={'s': 4}, sessions={'s': live},
                                pending_stops={'s': 4}, pending_stop_causes={})
    assert stop_authority_turns(authority, reason='Gateway restarting') == 1
    assert authority.pending_stop_causes == {} and authority.pending_stops == {'s': 4}
    authority.pending_stops = {}
    stop_authority_turns(authority)  # no reason (profile retire): latches the Stop, never a cause
    assert authority.pending_stop_causes == {} and authority.pending_stops == {'s': 4}
    authority.pending_stops = {}
    assert stop_authority_turns(authority, reason='Gateway shutting down') == 1
    assert authority.pending_stop_causes == {'s': (4, 'Gateway shutting down')}
