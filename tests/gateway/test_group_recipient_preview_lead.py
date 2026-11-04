"""The advisory preview and the real Discussion planner agree on a lead-routed room's first round.

Covers the blocking finding from HG-U2: a lead room's unaddressed new thread has no known first
round until the lead's routing turn answers, so the preview must say "the lead decides" instead of
naming every member. Every preview call goes through the real canonical dispatcher and is compared
with the real planner over the same room log; the room's SQLite image must not change.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import hosted_room_context as context
from gateway import hosted_room_discussion as discussion
from gateway import hosted_rooms
from gateway.hosted_rooms import create_room, local_authority_gateway_id
from gateway.session_controls import AuthorityConnection
from gateway.session_hosted_service import CanonicalHostedRoomService
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch

ROOM_ID = 'room-lead'
PROFILES = ('ada', 'linus', 'grace')
MEMBERS = [{'member_id': f'member-{profile}', 'profile': profile, 'handle': profile} for profile in PROFILES]


@pytest.fixture
def room(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text('model:\n  default: fixture\n')
    with SessionDB(tmp_path / 'state.db') as db:
        authority = SimpleNamespace(db=db, profile_id=str(tmp_path), instance_id='owner', events={},
                                    epoch=begin_runtime_epoch(db, instance_id='test'))
        service = authority.hosted_room_service = CanonicalHostedRoomService(authority, None)
        monkeypatch.setattr(service.runtime, 'status', lambda: {'running': True, 'stopping': False})
        owner = AuthorityConnection(authority, object(), {'user_id': 'owner'})
        service.authorize_room(owner.actor.subject, ROOM_ID, create=True)
        create_room(db.db_path, room_id=ROOM_ID, name='Kernel', authority_gateway_id=local_authority_gateway_id(),
                    members=MEMBERS)
        yield SimpleNamespace(db=db, path=db.db_path, owner=owner)


def _events(room) -> list[dict]:
    return hosted_rooms.read_events(room.path, room_id=ROOM_ID, limit=hosted_rooms.MAX_LOG_LIMIT)['events']


def _latest(room) -> int:
    events = _events(room)
    return int(events[-1]['seq']) if events else 0


def _plan(room, source_seq: int) -> discussion.DiscussionDecision:
    state = hosted_rooms.room_state(room.path, room_id=ROOM_ID)
    state['members'] = MEMBERS
    return discussion.plan_next_task(
        state, _events(room), local_profiles=PROFILES,
        room_context=context.room_context(room.path, room_id=ROOM_ID, before_seq=source_seq))


def _say(room, text: str, *, event_id: str = 'u1', thread_id: str = 't1') -> int:
    return hosted_rooms.append_event(
        room.path, room_id=ROOM_ID, event_id=event_id, kind='message.user',
        actor={'kind': 'user', 'id': 'desktop'}, payload={'text': text, 'thread_id': thread_id},
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(), authority_epoch=1)['seq']


def _finish(room, decision: discussion.DiscussionDecision, text: str, status: str = 'settled') -> list[dict]:
    result = {'text': text} if status == 'settled' else {'error': text}
    plan = discussion.plan_publication(
        {**hosted_rooms.room_state(room.path, room_id=ROOM_ID), 'members': MEMBERS}, _events(room),
        decision.task, status=status, result=result, local_profiles=PROFILES)
    return [hosted_rooms.append_event(room.path, **event.append_kwargs(ROOM_ID)) for event in plan.events]


def _lead_room(room, event_id: str = 'ctx') -> None:
    context.set_room_context(
        room.path, room_id=ROOM_ID, event_id=event_id, purpose='Kernel work',
        notes='linus: scheduler; grace: compiler', routing='lead', lead_member_id='member-ada')


def _preview(room, text: str, *, thread_id: str | None = None, latest_seq: int | None = None) -> dict:
    params = {'room_id': ROOM_ID, 'text': text, 'latest_seq': _latest(room) if latest_seq is None else latest_seq}
    if thread_id is not None:
        params['thread_id'] = thread_id
    reply = asyncio.run(room.owner.dispatch({'id': 1, 'method': 'groups.preview', 'params': params}))
    assert 'result' in reply, reply
    return reply['result']


def test_unaddressed_new_thread_shows_lead_decides_and_the_real_planner_confirms_only_the_pick(room):
    _lead_room(room)
    with room.db._read_ctx() as conn:
        before = tuple(conn.iterdump())
    text = 'The build is slow, can someone look?'
    projected = _preview(room, text)
    assert projected['mode'] == 'lead'
    assert projected['members'] == []
    assert projected['lead'] == {'member_id': 'member-ada', 'handle': 'ada'}
    assert projected['latest_seq'] == _latest(room) == 1
    with room.db._read_ctx() as conn:
        assert tuple(conn.iterdump()) == before  # advisory only: no assignment, message or context write

    seq = _say(room, text)
    routing = _plan(room, seq)
    assert (routing.status, routing.reason, routing.task.member.member_id) == ('task', 'lead_routing', 'member-ada')
    _finish(room, routing, 'Sure: {"assign": ["@Linus"]}')
    turn = _plan(room, seq)
    # The actual first round is Linus alone; naming all three members in the preview was the defect.
    assert (turn.reason, turn.task.member.member_id) == ('member_turn', 'member-linus')


@pytest.mark.parametrize(('text', 'expected'), [
    ('@linus take a look', ['member-linus']),
    ('@everyone please look', ['member-ada', 'member-linus', 'member-grace']),
    ('@ada @GRACE please look', ['member-ada', 'member-grace']),
])
def test_explicit_mention_and_broadcast_bypass_the_lead_in_preview_and_planner(room, text, expected):
    _lead_room(room)
    projected = _preview(room, text)
    assert projected['mode'] == 'members' and projected['lead'] is None
    assert [member['member_id'] for member in projected['members']] == expected
    seq = _say(room, text, event_id='mention')
    assert _plan(room, seq).task.member.member_id == expected[0]


def test_an_unknown_handle_is_not_an_address_and_the_lead_still_decides(room):
    _lead_room(room)
    projected = _preview(room, '@nobody please look')
    assert projected['mode'] == 'lead'
    seq = _say(room, '@nobody please look', event_id='unknown')
    assert _plan(room, seq).reason == 'lead_routing'


def test_an_unaddressed_message_in_an_existing_thread_keeps_everyone(room):
    _lead_room(room)
    _say(room, 'thread opener', event_id='open', thread_id='thread-1')
    projected = _preview(room, 'a follow-up that names nobody', thread_id='thread-1')
    assert projected['mode'] == 'members'
    assert [member['member_id'] for member in projected['members']] == [
        'member-ada', 'member-linus', 'member-grace']
    seq = _say(room, 'a follow-up that names nobody', event_id='follow', thread_id='thread-1')
    assert _plan(room, seq).task.member.member_id == 'member-ada'


def test_an_everyone_room_projects_all_members_and_the_planner_starts_with_the_rotated_first(room):
    context.set_room_context(room.path, room_id=ROOM_ID, event_id='ctx', purpose='', notes='')
    projected = _preview(room, 'The build is slow')
    assert projected['mode'] == 'members'
    assert [member['member_id'] for member in projected['members']] == [
        'member-ada', 'member-linus', 'member-grace']
    seq = _say(room, 'The build is slow')
    decision = _plan(room, seq)
    assert (decision.reason, decision.task.member.member_id) == ('member_turn', 'member-ada')


def test_a_routing_change_is_a_new_fenced_projection_even_though_revision_is_unchanged(room):
    _lead_room(room)
    revision = hosted_rooms.room_state(room.path, room_id=ROOM_ID)['revision']
    lead_view = _preview(room, 'Who takes this?')
    context.set_room_context(room.path, room_id=ROOM_ID, event_id='ctx-2', purpose='', notes='',
                             routing='everyone')
    everyone_view = _preview(room, 'Who takes this?')
    assert everyone_view['revision'] == revision == lead_view['revision']
    assert everyone_view['latest_seq'] > lead_view['latest_seq']
    assert (lead_view['mode'], everyone_view['mode']) == ('lead', 'members')
    assert [member['member_id'] for member in everyone_view['members']] == [
        'member-ada', 'member-linus', 'member-grace']


def test_a_preview_older_than_the_room_log_fails_closed_without_writing(room):
    _lead_room(room)
    observed = _latest(room)
    _say(room, 'another message', event_id='late')
    with room.db._read_ctx() as conn:
        before = tuple(conn.iterdump())
    reply = asyncio.run(room.owner.dispatch({'id': 1, 'method': 'groups.preview',
        'params': {'room_id': ROOM_ID, 'text': 'draft', 'latest_seq': observed}}))
    assert reply['error']['message'] == 'stale_preview'
    with room.db._read_ctx() as conn:
        assert tuple(conn.iterdump()) == before
