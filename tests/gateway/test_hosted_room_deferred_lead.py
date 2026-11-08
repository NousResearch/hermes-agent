"""A lead's lost routing turn gives its message to the lead once, and never routes it again."""

from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as state
from gateway import hosted_rooms
from tui_gateway.hosted_room_service import HostedRoomService
from tests.gateway.test_hosted_room_lead import MEMBERS, PROFILES, ROOM_ID, _lead_room
from tests.tui_gateway.test_hosted_room_driver_runtime import BINDING, PROFILE, _identity, db as db_fixture
from tests.tui_gateway.test_hosted_room_service import _FakeRPC, _server
from tests.tui_gateway.test_hosted_room_slow_reads import _restarted_mid_turn

db = db_fixture
ROUTE_TO_LINUS = '{"assign": ["linus"]}'


class _LeadRPC(_FakeRPC):
    """In-process bots: the lead's first routing turn stays out until the test answers it, and the
    lead routes to linus after that. Every other turn answers at once unless held."""

    def __init__(self):
        super().__init__()
        self.held, self.routing_reply = {}, None

    def submit(self, *, profile, prompt, on_terminal, **kwargs):
        if '{"assign"' in prompt:
            if self.routing_reply is None:
                self.routing_reply = on_terminal
            else:
                on_terminal({'status': 'settled', 'text': ROUTE_TO_LINUS})
        elif profile in self.held:
            self.held[profile] = on_terminal
        else:
            return super().submit(profile=profile, prompt=prompt, on_terminal=on_terminal, **kwargs)
        return {'accepted': True}


def _events(room):
    return hosted_rooms.read_events(room.db, room_id=ROOM_ID, limit=hosted_rooms.MAX_LOG_LIMIT)['events']


def _answered_by(room, thread_id='t1'):
    return [event['payload']['member_id'] for event in _events(room)
            if event['kind'] == 'message.member' and event['payload']['thread_id'] == thread_id]


def _assignments(room):
    return [event['payload']['member_ids'] for event in _events(room) if event['kind'] == 'room.assignment']


@pytest.fixture
def lost_routing(tmp_path, monkeypatch):
    """A lead routing turn this process lost sight of, at its deadline: its bot is idle with no reply."""
    def build(*, confirmed=True, held=()):
        db, now = tmp_path / 'state.db', [100.0]
        service = HostedRoomService(_server(), db_path=db)
        rpc = service.rpc = service.runtime.rpc = _LeadRPC()
        rpc.held.update(dict.fromkeys(held))
        runtime = service.runtime
        runtime.clock = lambda: now[0]
        runtime.lease_ttl_seconds, runtime.indeterminate_defer_seconds = 30, 5
        if not confirmed:  # the lead's daemon or gateway cannot say whether its run still goes on
            runtime.confirm_deferred_stop = lambda *_: False
        service.local_profiles = lambda: PROFILES
        service.create_room(room_id=ROOM_ID, name='Kernel', members=MEMBERS)
        _lead_room(db)
        service.send(room_id=ROOM_ID, event_id='u1',
                     payload={'text': 'The build is slow, can someone look?', 'thread_id': 't1'})
        routing, = state.list_tasks(db, room_id=ROOM_ID, status='queued')
        assert routing['payload'].get('lead_routing') is True
        binding = service.bindings()[0]
        monkeypatch.setattr(runtime._wake, 'wait', lambda _timeout: runtime._stop.set())

        def process():  # the worker's passes until the room has nothing left to start
            for _ in range(10):
                runtime._stop.clear()
                runtime._process_room(binding)
                if not state.list_tasks(db, room_id=ROOM_ID, status='queued'):
                    return
            raise AssertionError('the group chat kept admitting turns')

        process()
        assert state.get_task(db, routing['identity'])['status'] == 'running'
        lease = runtime._leases[ROOM_ID]
        state.mark_attempt_indeterminate(db, state.TaskAttempt(routing['identity'], lease, 1, 0), lease,
                                         clock=runtime.clock)
        room = SimpleNamespace(db=db, now=now, service=service, binding=binding, routing=routing['identity'],
                               process=process, routing_status=lambda: state.get_task(db, routing['identity'])['status'],
                               late_reply=lambda: rpc.routing_reply({'status': 'settled', 'text': ROUTE_TO_LINUS}))
        for now[0] in (102.0, 108.0):
            process()
        return room
    return build


def test_a_lost_routing_turn_gives_the_message_to_the_lead_and_frees_other_threads(lost_routing):
    room = lost_routing(held=('ada',))
    room.service.send(room_id=ROOM_ID, event_id='u2',
                      payload={'text': 'Release notes, anyone?', 'thread_id': 't2'})
    assert room.routing_status() == 'cancelled'
    # The lead takes the message itself, as after Discard, and nothing is offered for the lost run.
    assert not room.service.status(ROOM_ID)['pending_actions']
    assert [event['payload']['reason'] for event in _events(room)
            if event['kind'] == 'turn.cancelled'] == ['lead_answers_itself']
    room.service.rpc.held.pop('ada')({'status': 'settled', 'text': 'Ada here: the cache was cold.'})
    room.process()
    assert _answered_by(room) == ['member-ada']
    # The next thread is no longer held: its own routing turn runs and linus answers it.
    assert _answered_by(room, 't2') == ['member-linus']


@pytest.mark.parametrize('when', ['before_the_lead_answers', 'after_the_discussion_completed'])
def test_the_lost_routing_runs_late_reply_routes_nothing(lost_routing, when):
    room = lost_routing(held=('ada',))
    if when == 'before_the_lead_answers':
        room.late_reply()
    room.service.rpc.held.pop('ada')({'status': 'settled', 'text': 'Ada here: the cache was cold.'})
    room.process()
    assert any(event['kind'] == 'room.activity' for event in _events(room))
    if when == 'after_the_discussion_completed':
        room.late_reply()
        room.process()
    assert _answered_by(room) == ['member-ada']
    assert _assignments(room) == []  # Desktop never shows "ada gave this to linus"
    assert room.routing_status() == 'cancelled'


def test_retry_of_the_lost_routing_turn_is_not_offered(lost_routing):
    room = lost_routing()
    with pytest.raises(state.InvalidTaskTransitionError):
        room.service.retry_room_task(ROOM_ID, task_id=room.routing.task_id)
    assert _answered_by(room) == ['member-ada']


@pytest.mark.parametrize('outcome', ['retry', 'late_reply'])
def test_a_routing_turn_that_may_still_run_keeps_its_message_until_it_answers(lost_routing, outcome):
    room = lost_routing(confirmed=False)
    assert room.routing_status() == 'deferred'
    assert _answered_by(room) == []  # the message waits for the lead's pick, as before
    assert [action['kind'] for action in room.service.status(ROOM_ID)['pending_actions']] == ['retry']
    if outcome == 'retry':
        assert room.service.retry_room_task(ROOM_ID, task_id=room.routing.task_id)['status'] == 'queued'
    else:
        room.late_reply()
    room.process()
    assert _assignments(room) == [['member-linus']]
    assert _answered_by(room) == ['member-linus']


def _lost_routing_on_the_gateway_route(db, now, monkeypatch):
    identity = _identity()
    state.admit_task(db, identity, clock=lambda: now[0], payload={
        'target_profile': PROFILE, 'source_event_seq': 1, 'prompt': 'who takes this?', 'lead_routing': True})
    member, runtime = _restarted_mid_turn(db, monkeypatch, lambda: now[0], indeterminate_defer_seconds=5)
    runtime.confirm_deferred_stop = runtime.deferred_stop_confirmed  # the service's local-member answer
    return identity, member, runtime


def _passes(runtime, now, count):
    for _ in range(count):
        runtime._stop.clear()
        runtime._run_room_once(BINDING)
        now[0] += 5


def test_a_slow_unconfirmed_stop_still_defers_a_lost_routing_turn(db, monkeypatch):
    """A hung daemon or a black-holed peer waits out its 30 s transport timeout on every call, longer
    than the lease the deferral then commits under."""
    now = [100.0]
    identity, member, runtime = _lost_routing_on_the_gateway_route(db, now, monkeypatch)
    next(iter(member.states.values()))['active'] = False

    def hung_daemon(_room_id, _task):
        now[0] += 30
        return False

    runtime.confirm_deferred_stop = hung_daemon
    _passes(runtime, now, 12)
    assert state.get_task(db, identity)['status'] == 'deferred', runtime._last_error


@pytest.mark.parametrize('unreadable', [('info',), ('history', 'results')])
def test_a_routing_turn_that_could_not_be_read_is_not_cancelled_while_it_runs(db, monkeypatch, unreadable):
    now = [100.0]
    identity, member, runtime = _lost_routing_on_the_gateway_route(db, now, monkeypatch)
    session = next(iter(member.states.values()))
    now[0] += 31  # the dead process's lease has lapsed
    state.recover_room(db, runtime._ensure_lease(BINDING), clock=runtime.clock)
    assert state.get_task(db, identity)['status'] == 'indeterminate'

    def timed_out(**_kwargs):
        raise TimeoutError('transport call timed out after 30 seconds')

    for read in unreadable:
        setattr(member, read, timed_out)
    if unreadable != ('info',):  # what a bot with no reply and no run says: only its receipts are unread
        session['active'] = False
    _passes(runtime, now, 12)
    assert state.get_task(db, identity)['status'] == 'deferred', runtime._last_error
