"""The live state a ``message_agent`` sender hears about its teammate's Bot Chat.

Real owner pieces (``SessionAuthority`` over a real ``SessionDB``, the real claim and the real
approval/clarify registration paths); only the gateway runner is a stand-in for its drain flag.
"""
from types import SimpleNamespace

import pytest

from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_bot_state import bot_chat_state
from hermes_state import SessionDB
from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input
from tools import approval, clarify_gateway
from tools.approval_gateway_wait import _ApprovalEntry


@pytest.fixture()
def owner(tmp_path):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('s', source='test')
        epoch = begin_runtime_epoch(db, instance_id='current')
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id='bot',
                                     instance_id='current', db=db, epoch=epoch)
        yield SimpleNamespace(authority=authority, db=db, epoch=epoch)


def _start_turn(owner):
    owner.authority.sessions['s'] = LiveSession(None, 'route')
    admit_session_input(owner.db, epoch=owner.epoch, principal_id='bot:alpha', session_id='s',
                        request_id='dm', payload={'text': 'Message from alpha: status?'})
    return claim_session_input(owner.db, epoch=owner.epoch, session_id='s')['generation']


def test_idle_until_a_turn_runs_then_working(owner):
    assert bot_chat_state(owner.authority, 's') == 'idle'  # no live session on this owner yet
    owner.authority.sessions['s'] = LiveSession(None, 'route')
    assert bot_chat_state(owner.authority, 's') == 'idle'  # open, nothing claimed
    admit_session_input(owner.db, epoch=owner.epoch, principal_id='bot:alpha', session_id='s',
                        request_id='dm', payload={'text': 'queued behind nothing'})
    assert bot_chat_state(owner.authority, 's') == 'idle'  # queued is not running
    claim_session_input(owner.db, epoch=owner.epoch, session_id='s')
    assert bot_chat_state(owner.authority, 's') == 'working'


def test_waiting_on_clarify_then_back_to_working_once_answered(owner):
    generation = _start_turn(owner)
    entry = clarify_gateway.register('state-question', 'route', 'Red or blue?', ['red', 'blue'])
    try:
        owner.authority.register_clarify('s', generation, entry)
        assert bot_chat_state(owner.authority, 's') == 'waiting_clarify'
        clarify_gateway.resolve_gateway_clarify(entry.clarify_id, 'blue')
        assert bot_chat_state(owner.authority, 's') == 'working'
    finally:
        clarify_gateway.clear_session(entry.session_key)


def test_waiting_on_approval_wins_over_clarify_and_clears_on_resolve(owner):
    generation = _start_turn(owner)
    entry = _ApprovalEntry({'command': 'rm -rf build', 'request_id': 'state-approval'})
    approval._gateway_queues.setdefault('route', []).append(entry)
    question = clarify_gateway.register('state-question-2', 'route', 'Sure?', ['yes', 'no'])
    try:
        owner.authority.register_approval('s', generation, 'route', entry.data)
        owner.authority.register_clarify('s', generation, question)
        assert bot_chat_state(owner.authority, 's') == 'waiting_approval'
        approval.resolve_gateway_approval('route', 'deny', request_id='state-approval')
        assert bot_chat_state(owner.authority, 's') == 'waiting_clarify'
    finally:
        approval._gateway_queues.pop('route', None)
        clarify_gateway.clear_session(question.session_key)


@pytest.mark.parametrize('stop', ['draining', 'retiring'])
def test_paused_while_the_owner_admits_nothing_new(owner, stop):
    _start_turn(owner)
    if stop == 'draining':
        owner.authority.runner._draining = True
    else:
        owner.authority.retiring = True
    assert bot_chat_state(owner.authority, 's') == 'paused'
