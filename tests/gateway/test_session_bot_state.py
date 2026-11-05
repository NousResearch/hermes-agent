"""The live state a ``message_agent`` sender hears about its teammate's Bot Chat.

Real owner pieces (``SessionAuthority`` over a real store, the real claim and the real
approval/clarify registration paths); only the gateway runner is a stand-in for its drain flag.
"""
from types import SimpleNamespace

import pytest

from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_bot_state import bot_chat_state
from persistence_assembly import open_persistence, profile_scope
from tools import approval, clarify_gateway
from tools.approval_gateway_wait import _ApprovalEntry
from tests.fixtures.storage_seed import seed_session


@pytest.fixture()
def owner(tmp_path):
    with open_persistence(profile_scope(home=tmp_path)) as db:
        seed_session(db, 's', 'test')
        epoch = db.runtime.begin_epoch(instance_id='current')
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id='bot',
                                     instance_id='current', persistence=db, epoch=epoch)
        yield SimpleNamespace(authority=authority, db=db, epoch=epoch)


def _start_turn(owner):
    owner.authority.sessions['s'] = LiveSession(None, 'route')
    owner.db.admissions.admit(epoch=owner.epoch, principal_id='bot:alpha', session_id='s',
                              request_id='dm', payload={'text': 'Message from alpha: status?'})
    return owner.db.admissions.claim(epoch=owner.epoch, session_id='s').generation


def test_idle_until_a_turn_runs_then_working(owner):
    assert bot_chat_state(owner.authority, 's') == 'idle'  # no live session on this owner yet
    owner.authority.sessions['s'] = LiveSession(None, 'route')
    assert bot_chat_state(owner.authority, 's') == 'idle'  # open, nothing claimed
    owner.db.admissions.admit(epoch=owner.epoch, principal_id='bot:alpha', session_id='s',
                              request_id='dm', payload={'text': 'queued behind nothing'})
    assert bot_chat_state(owner.authority, 's') == 'idle'  # queued is not running
    owner.db.admissions.claim(epoch=owner.epoch, session_id='s')
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
