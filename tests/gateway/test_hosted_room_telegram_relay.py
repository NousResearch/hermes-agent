"""The DM grant is independent of ordinary room ownership."""
import json
from types import SimpleNamespace

import pytest

from gateway import hosted_rooms as rooms, session_context
from gateway.session_hosted_service import CanonicalHostedRoomService
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch, RuntimeStoreError


@pytest.fixture
def relay(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr('pathlib.Path.home', lambda: tmp_path / 'user')
    db = SessionDB(tmp_path / 'state.db')
    authority = SimpleNamespace(db=db, profile_id=str(tmp_path),
                                epoch=begin_runtime_epoch(db, instance_id='test'))
    service = authority.hosted_room_service = CanonicalHostedRoomService(authority, None)
    service.authorize_room('desktop-owner', 'fixture-room', create=True)
    rooms.create_room(db.db_path, room_id='fixture-room', name='Fixture',
                      authority_gateway_id=rooms.local_authority_gateway_id(), members=[
                          {'member_id': 'worker', 'profile': 'server', 'handle': 'server'},
                          {'member_id': 'other', 'profile': 'default', 'handle': 'other'}])
    monkeypatch.setattr(service.runtime, 'status', lambda: {'running': True, 'stopping': False})
    monkeypatch.setattr(service, 'profile_homes', lambda: {'server': tmp_path, 'default': tmp_path})
    monkeypatch.setattr(service, 'prepare_room', lambda binding: None)
    config = {'enabled': True, 'profile': 'default', 'allowed_senders': ['fixture-sender'],
              'room_id': 'fixture-room', 'members': ['server', 'default']}
    monkeypatch.setattr('gateway.hosted_room_telegram_relay.CALL_TIMEOUT_SECONDS', 2.0)
    context = dict(platform='telegram', chat_type='dm', user_id='fixture-sender',
                   chat_id='fixture-sender', profile='default', session_key='agent:main:fixture',
                   session_id='fixture-session', message_id='fixture-message')
    runner = SimpleNamespace(session_authority=authority)

    def call(recipient='server', message='Run fixture task'):
        from gateway.hosted_room_telegram_relay import delegate
        (tmp_path / 'config.yaml').write_text(json.dumps({'telegram_room_relay': config}))
        session_context.set_session_vars(**context)
        return delegate(runner, recipient=recipient, message=message)

    yield SimpleNamespace(call=call, config=config, context=context, service=service,
                          runner=runner, db=db, home=tmp_path)
    from gateway.hosted_room_telegram_relay import _worker_lock
    assert _worker_lock.acquire(timeout=2)
    _worker_lock.release()
    session_context.reset_session_vars()
    db.close()


def events(relay):
    return rooms.read_events(relay.db.db_path, room_id='fixture-room')['events']


def wait_for_worker():
    from gateway.hosted_room_telegram_relay import _worker_lock
    assert _worker_lock.acquire(timeout=2)
    _worker_lock.release()


@pytest.mark.parametrize('field,value', [
    ('platform', 'discord'), ('chat_type', 'group'), ('user_id', 'foreign'),
    ('profile', 'other'), ('profile', ''), ('chat_id', ''), ('chat_id', 'foreign-chat'),
    ('session_key', ''), ('session_id', ''), ('message_id', ''), ('platform', ''),
])
def test_identity_denials_have_no_side_effect(relay, field, value):
    relay.context[field] = value
    before = events(relay)
    assert relay.call()['status'] == 'denied'
    assert events(relay) == before


@pytest.mark.parametrize('change', [
    {'enabled': False}, {'enabled': 'true'}, {'profile': 'other'}, {'room_id': ''},
    {'allowed_senders': []}, {'allowed_senders': ['fixture-sender', 'another-sender']},
    {'allowed_senders': ['']}, {'members': ['other']}, {'members': ['server', 'other']},
])
def test_configuration_denials_have_no_side_effect(relay, change):
    relay.config.update(change)
    before = events(relay)
    assert relay.call()['status'] == 'denied'
    assert events(relay) == before


@pytest.mark.parametrize('problem', ['absent', 'inactive', 'stopping', 'membership', 'unserved', 'draining'])
def test_driver_and_membership_denials(relay, monkeypatch, problem):
    if problem == 'absent':
        relay.runner.session_authority.hosted_room_service = None
    elif problem in {'inactive', 'stopping'}:
        monkeypatch.setattr(relay.service.runtime, 'status', lambda: {
            'running': problem != 'inactive', 'stopping': problem == 'stopping'})
    elif problem == 'membership':
        relay.config['room_id'] = 'missing-room'
    elif problem == 'unserved':
        monkeypatch.setattr(relay.service, 'profile_homes', dict)
    else:
        relay.runner._draining = True
    before = events(relay)
    assert relay.call()['status'] == 'denied'
    assert events(relay) == before


@pytest.mark.parametrize('recipient,message', [('other', 'task'), ('server', ''), ('server', 'x' * 8001)])
def test_invalid_input_does_not_append(relay, recipient, message):
    before = events(relay)
    assert relay.call(recipient, message)['status'] == 'denied'
    assert events(relay) == before


def test_timeout_and_retry_are_one_durable_discussion(relay):
    first = relay.call()
    wait_for_worker()
    second = relay.call()
    wait_for_worker()
    assert first['status'] == second['status'] == 'timeout'
    users = [e for e in events(relay) if e['kind'] == 'message.user']
    assert len(users) == 1
    assert users[0]['event_id'].startswith('user:')
    assert set(users[0]['payload']) == {'thread_id', 'text'}
    before = events(relay)
    assert relay.call(message='Changed retry')['status'] == 'incomplete'
    assert events(relay) == before
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        relay.service.authorize_room('fixture-sender', 'fixture-room')
    assert relay.service.authorize_room('desktop-owner', 'fixture-room')


def test_unbound_context_cannot_borrow_environment(relay, monkeypatch):
    from gateway.hosted_room_telegram_relay import delegate
    relay.call(recipient='invalid')  # write local config without admission
    session_context.reset_session_vars()
    for key, value in relay.context.items():
        monkeypatch.setenv('HERMES_SESSION_' + key.upper(), value)
    before = events(relay)
    assert delegate(relay.runner, recipient='server', message='task')['status'] == 'denied'
    assert events(relay) == before


@pytest.mark.parametrize('outcome', ['settled', 'bounded', 'failed', 'deferred', 'cancelled'])
def test_only_committed_correlated_replies_return(relay, monkeypatch, outcome):
    from gateway import hosted_room_discussion as discussion

    def publish(binding):
        room = relay.service._room('fixture-room')
        log = events(relay)
        user = next(e for e in log if e['kind'] == 'message.user')
        plan = discussion.plan_next_task(room, log, local_profiles=['server', 'default']).task
        if plan is None:
            return
        publication = discussion.plan_publication(room, log, plan,
            status=outcome if outcome not in {'bounded', 'settled'} else 'settled',
            result={'text': 'CURRENT RESULT'},
            execution_generation=1 if outcome == 'deferred' else None,
            local_profiles=['server', 'default'])
        for event in publication.events:
            rooms.append_event(relay.db.db_path, **event.append_kwargs('fixture-room'))
        gateway, epoch = relay.service._owned_authority('fixture-room')
        for suffix, thread, discussion_id, status in [
            ('unrelated', 'foreign-thread', 'foreign-event', 'settled'),
            ('current', user['payload']['thread_id'], user['event_id'],
             'bounded' if outcome == 'bounded' else 'settled')]:
            rooms.append_event(relay.db.db_path, room_id='fixture-room', event_id='activity-' + suffix,
                kind='room.activity', actor={'kind': 'gateway', 'id': gateway},
                payload={'status': status, 'reason_code': 'silent_round', 'thread_id': thread,
                         'discussion_event_id': discussion_id},
                authority_gateway_id=gateway, authority_epoch=epoch)

    monkeypatch.setattr('gateway.hosted_room_telegram_relay.CALL_TIMEOUT_SECONDS', 5)
    monkeypatch.setattr(relay.service, 'prepare_room', publish)
    result = relay.call()
    assert result['status'] == ('completed' if outcome == 'settled' else 'incomplete')
    assert 'foreign' not in json.dumps(result)
    if outcome == 'settled':
        assert result['text'] == 'server: CURRENT RESULT'


def test_default_is_disabled(relay):
    from gateway.hosted_room_telegram_relay import delegate
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    assert DEFAULT_CONFIG['telegram_room_relay']['enabled'] is False
    assert DEFAULT_CONFIG['telegram_room_relay']['members'] == []
    (relay.home / 'config.yaml').write_text('{}')
    session_context.set_session_vars(**relay.context)
    before = events(relay)
    assert delegate(relay.runner, recipient='server', message='task')['status'] == 'denied'
    assert events(relay) == before


def test_relay_preserves_normal_group_followups(relay):
    from gateway import hosted_room_discussion as discussion
    assert relay.call(message='Check this task')['status'] == 'timeout'
    room, log = relay.service._room('fixture-room'), events(relay)
    plan = discussion.plan_next_task(room, log, local_profiles=['server', 'default']).task
    assert plan.member.member_id == 'worker'
    publication = discussion.plan_publication(room, log, plan, status='settled',
        result={'text': '@other please help verify'}, local_profiles=['server', 'default'])
    for event in publication.events:
        rooms.append_event(relay.db.db_path, **event.append_kwargs('fixture-room'))
    decision = discussion.plan_next_task(room, events(relay), local_profiles=['server', 'default'])
    assert decision.status == 'task'
    assert decision.task.member.member_id == 'other'


def test_inbound_retry_survives_session_rotation(relay):
    assert relay.call()['status'] == 'timeout'
    wait_for_worker()
    relay.context['session_id'] = 'replacement-session'
    relay.context['session_key'] = 'agent:main:replacement'
    assert relay.call()['status'] == 'timeout'
    assert len([e for e in events(relay) if e['kind'] == 'message.user']) == 1


@pytest.mark.parametrize('problem', ['wrong-handle', 'wrong-profile', 'peer', 'unowned', 'retiring', 'wrong-home'])
def test_exact_grant_binding_denials(relay, monkeypatch, problem):
    if problem in {'wrong-handle', 'wrong-profile', 'peer'}:
        original = relay.service._room
        def room(room_id):
            value = original(room_id)
            worker = value['members'][0]
            if problem == 'wrong-handle':
                worker['handle'] = 'other'
            elif problem == 'wrong-profile':
                worker['profile'] = 'default'
            else:
                worker['target'] = {'kind': 'peer'}
            return value
        monkeypatch.setattr(relay.service, '_room', room)
    elif problem == 'unowned':
        monkeypatch.setattr(relay.service, 'bindings', lambda: ())
    elif problem == 'retiring':
        relay.runner.session_authority.retiring = True
    else:
        monkeypatch.setattr('gateway.hosted_room_telegram_relay.get_hermes_home', lambda: relay.home / 'other')
    before = events(relay)
    assert relay.call()['status'] == 'denied'
    assert events(relay) == before


def test_result_projection_excludes_unrelated_and_uncommitted_text():
    from gateway.hosted_room_telegram_relay import _correlated_result
    def event(event_id, kind, **payload):
        return {'event_id': event_id, 'kind': kind, 'payload': {
            'thread_id': 'thread', 'discussion_event_id': 'discussion', 'member_id': 'worker', **payload}}
    log = [event('reply', 'message.member', text='CURRENT'),
           event('uncommitted', 'message.member', text='UNCOMMITTED'),
           event('foreign', 'message.member', text='FOREIGN', discussion_event_id='other'),
           event('foreign-thread', 'message.member', text='FOREIGN THREAD', thread_id='other'),
           event('other-member', 'message.member', text='OTHER MEMBER', member_id='other'),
           event('terminal', 'turn.settled', message_event_id='reply'),
           event('other-terminal', 'turn.settled', message_event_id='other-member', member_id='other'),
           event('done', 'room.activity', status='settled')]
    assert _correlated_result(log, event_id='discussion', thread_id='thread', members={'worker': 'server'}) == {
        'status': 'completed', 'text': 'server: CURRENT'}


def test_ordinary_group_apis_still_deny_nonowner_after_relay(relay):
    from gateway.session_group_controls import _group
    from gateway.session_contract import Principal
    relay.call()
    actor = Principal('fixture-sender', str(relay.home), frozenset({'session:read', 'session:submit'}), 'fixture')
    before = events(relay)
    for method, params in [('groups.state', {}), ('groups.log', {}), ('groups.send', {
            'event_id': 'forged', 'payload': {'text': 'forged', 'thread_id': 'forged'}})]:
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            _group(relay.runner.session_authority, actor, method, {'room_id': 'fixture-room', **params})
    assert _group(relay.runner.session_authority, actor, 'groups.list', {})['rooms'] == []
    assert events(relay) == before


def test_tool_dispatch_uses_bound_gateway_context(relay, monkeypatch):
    import tools.bot_room_delegate
    from tools.registry import registry
    assert relay.call(recipient='other')['status'] == 'denied'
    monkeypatch.setattr('gateway.run._gateway_runner_ref', lambda: relay.runner)
    result = json.loads(registry.dispatch('bot_room_delegate', {'recipient': 'server', 'message': 'Run fixture task'},
                                          user_id='forged', room_id='forged'))
    assert result['status'] == 'timeout'
    wait_for_worker()
    assert len([e for e in events(relay) if e['kind'] == 'message.user']) == 1


def test_ambiguous_submission_retry_uses_idempotent_event_key(relay, monkeypatch):
    send = relay.service.send
    calls = []

    def ambiguous_once(**kwargs):
        calls.append(kwargs['event_id'])
        event = send(**kwargs)
        if len(calls) == 1:
            raise RuntimeError('transport failed after commit')
        return event

    monkeypatch.setattr(relay.service, 'send', ambiguous_once)
    assert relay.call()['status'] == 'incomplete'
    assert relay.call()['status'] == 'timeout'
    assert len(calls) == 2 and calls[0] == calls[1]
    assert len([e for e in events(relay) if e['kind'] == 'message.user']) == 1


@pytest.mark.parametrize('length', [0, 16000, 16001])
def test_result_limits_and_silent_pass(length):
    from gateway.hosted_room_telegram_relay import _correlated_result
    payload = {'thread_id': 'thread', 'discussion_event_id': 'discussion', 'member_id': 'worker'}
    log = [{'event_id': 'done', 'kind': 'room.activity', 'payload': {**payload, 'status': 'settled'}}]
    if length:
        log = [{'event_id': 'reply', 'kind': 'message.member', 'payload': {**payload, 'text': 'x' * length}},
               {'event_id': 'turn', 'kind': 'turn.settled', 'payload': {**payload, 'message_event_id': 'reply'}}, *log]
    result = _correlated_result(log, event_id='discussion', thread_id='thread', members={'worker': 'server'})
    rendered = 'server: ' + ('x' * length) if length else ''
    assert result['status'] == ('incomplete' if not length or len(rendered) > 16000 else 'completed')
    assert len(result['text']) == min(len(rendered), 16000)


def test_reconstructed_service_keeps_retry_and_recipient_ceiling(relay, monkeypatch):
    from gateway import hosted_room_discussion as discussion
    assert relay.call()['status'] == 'timeout'
    wait_for_worker()
    service = CanonicalHostedRoomService(relay.runner.session_authority, None)
    monkeypatch.setattr(service.runtime, 'status', lambda: {'running': True})
    monkeypatch.setattr(service, 'profile_homes', lambda: {'server': relay.home, 'default': relay.home})
    monkeypatch.setattr(service, 'prepare_room', lambda binding: None)
    relay.runner.session_authority.hosted_room_service = service
    assert relay.call()['status'] == 'timeout'
    log = events(relay)
    assert len([e for e in log if e['kind'] == 'message.user']) == 1
    room = service._room('fixture-room')
    snapshot = service._policy_snapshot(room)
    decision = discussion.plan_next_task(room, snapshot.events, local_profiles=['server', 'default'])
    assert decision.task.member.member_id == 'worker'


def test_concurrent_retry_is_one_submission(relay):
    from concurrent.futures import ThreadPoolExecutor
    from gateway.hosted_room_telegram_relay import delegate
    relay.call(recipient='invalid')  # prepare config without writing to room
    def invoke():
        session_context.set_session_vars(**relay.context)
        try:
            return delegate(relay.runner, recipient='server', message='task')
        finally:
            session_context.reset_session_vars()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: invoke(), range(2)))
    assert sorted(r['status'] for r in results) == ['busy', 'timeout']
    assert len([e for e in events(relay) if e['kind'] == 'message.user']) == 1


def test_timeout_bounds_blocking_submission(relay, monkeypatch):
    from concurrent.futures import TimeoutError as FutureTimeout
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    monkeypatch.setattr('gateway.hosted_room_telegram_relay.CALL_TIMEOUT_SECONDS', 2.0)
    relay.call(recipient='invalid')  # Persist the test grant without submitting.
    entered, release = Event(), Event()
    original = relay.service.send

    def blocked_send(**kwargs):
        entered.set()
        release.wait(5)
        return original(**kwargs)

    monkeypatch.setattr(relay.service, 'send', blocked_send)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(relay.call)
        assert entered.wait(1.5)
        try:
            result = future.result(timeout=2.5)
            assert result['status'] == 'timeout'
        except FutureTimeout:
            pytest.fail('relay exceeded its fixed end-to-end timeout')
        finally:
            release.set()
    from gateway.hosted_room_telegram_relay import _worker_lock
    assert _worker_lock.acquire(timeout=2)
    _worker_lock.release()
    assert len([e for e in events(relay) if e['kind'] == 'message.user']) == 1


@pytest.mark.parametrize('operation', ['authorize', 'send', 'read'])
def test_blocked_io_is_busy_until_worker_exits_and_late_reply_is_incomplete(relay, monkeypatch, operation):
    import contextvars
    import threading
    from gateway import hosted_room_telegram_relay as bridge

    monkeypatch.setattr(bridge, 'CALL_TIMEOUT_SECONDS', 2.0)
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    marker = contextvars.ContextVar('relay_test_marker', default='missing')
    marker.set('copied')
    seen = []
    target, name = {
        'authorize': (bridge, '_local_grant'),
        'send': (relay.service, 'send'),
        'read': (rooms, 'read_events'),
    }[operation]
    original = getattr(target, name)

    def blocked(*args, **kwargs):
        seen.append((marker.get(), threading.current_thread().daemon))
        entered.set()
        assert release.wait(5)
        try:
            return original(*args, **kwargs)
        finally:
            exited.set()

    monkeypatch.setattr(target, name, blocked)
    try:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as pool:
            context = contextvars.copy_context()
            future = pool.submit(context.run, relay.call)
            assert entered.wait(1.5)
            result = future.result(timeout=2.5)
            assert result['status'] == 'timeout'
            assert ('request_id' in result) == (operation != 'authorize')
        assert relay.call()['status'] == 'busy'
    finally:
        release.set()
        assert exited.wait(2)
        assert bridge._worker_lock.acquire(timeout=2)
        bridge._worker_lock.release()
    assert result['status'] != 'completed'
    if operation == 'authorize':
        assert 'request_id' not in result
        assert len([e for e in events(relay) if e['kind'] == 'message.user']) == 0
    assert all(value == ('copied', True) for value in seen)
    monkeypatch.setattr(target, name, original)
    assert relay.call()['status'] == 'timeout'
    assert len([e for e in events(relay) if e['kind'] == 'message.user']) == 1


def test_unknown_send_can_retry_canonical_idempotent_submission(relay, monkeypatch):
    calls = []

    def unknown(**kwargs):
        calls.append(kwargs)
        raise RuntimeError('delivery unknown')

    monkeypatch.setattr(relay.service, 'send', unknown)
    assert relay.call()['status'] == 'incomplete'
    assert relay.call()['status'] == 'incomplete'
    assert len(calls) == 2


@pytest.mark.parametrize('roster', [['server'], ['server', 'default', 'extra'],
                                    ['server', 'default', 'default'], [], ['default']])
def test_exact_actual_roster_required(relay, roster):
    relay.config['members'] = roster
    before = events(relay)
    assert relay.call()['status'] == 'denied'
    assert events(relay) == before


def test_indexed_event_lookup_hit_and_miss(relay):
    relay.call()
    user = next(e for e in events(relay) if e['kind'] == 'message.user')
    assert rooms.get_event(relay.db.db_path, room_id='fixture-room', event_id=user['event_id']) == user
    assert rooms.get_event(relay.db.db_path, room_id='fixture-room', event_id='missing') is None


def test_late_poll_returns_committed_teammate_result(relay):
    from gateway.hosted_room_telegram_relay import room_result
    from gateway import hosted_room_discussion as discussion
    submitted = relay.call()
    assert submitted['status'] == 'timeout'
    request_id = submitted['request_id']
    wait_for_worker()
    before = events(relay)
    assert room_result(relay.runner, request_id='bad')['status'] == 'denied'
    assert room_result(relay.runner, request_id='0' * 64)['status'] == 'incomplete'
    assert room_result(relay.runner, request_id=request_id)['status'] == 'incomplete'
    assert events(relay) == before
    room = relay.service._room('fixture-room')
    for text in ['@other verify', 'VERIFIED']:
        log = events(relay)
        task = discussion.plan_next_task(room, log, local_profiles=['server', 'default']).task
        publication = discussion.plan_publication(room, log, task, status='settled',
            result={'text': text}, local_profiles=['server', 'default'])
        for event in publication.events:
            rooms.append_event(relay.db.db_path, **event.append_kwargs('fixture-room'))
    user = next(e for e in events(relay) if e['kind'] == 'message.user')
    gateway, epoch = relay.service._owned_authority('fixture-room')
    rooms.append_event(relay.db.db_path, room_id='fixture-room', event_id='poll-terminal',
        kind='room.activity', actor={'kind': 'gateway', 'id': gateway},
        payload={'status': 'settled', 'reason_code': 'silent_round',
                 'thread_id': user['payload']['thread_id'], 'discussion_event_id': user['event_id']},
        authority_gateway_id=gateway, authority_epoch=epoch)
    result = room_result(relay.runner, request_id=request_id)
    assert result['status'] == 'completed'
    assert result['text'] == 'server: @other verify\n\ndefault: VERIFIED'
    relay.context['user_id'] = relay.context['chat_id'] = 'foreign'
    session_context.set_session_vars(**relay.context)
    assert room_result(relay.runner, request_id=request_id)['status'] == 'denied'


def test_completion_at_wait_boundary_is_observed(relay, monkeypatch):
    from gateway import hosted_room_telegram_relay as bridge
    original_event = bridge.threading.Event

    class BoundaryEvent:
        def __init__(self):
            self.inner = original_event()

        def set(self):
            self.inner.set()

        def is_set(self):
            return self.inner.is_set()

        def wait(self, timeout=None):
            assert self.inner.wait(2)
            return False  # wait reports expiry while completion has become visible

    relay.call(recipient='invalid')
    monkeypatch.setattr(bridge.threading, 'Event', BoundaryEvent)
    # Thread itself needs the real Event for its startup handshake.
    original_thread = bridge.threading.Thread
    def thread(**kwargs):
        monkeypatch.setattr(bridge.threading, 'Event', original_event)
        return original_thread(**kwargs)
    monkeypatch.setattr(bridge.threading, 'Thread', thread)
    result = bridge.delegate(relay.runner, recipient='invalid', message='task')
    assert result['status'] == 'denied'


def test_poll_remains_available_during_blocked_submit(relay, monkeypatch):
    import contextvars
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from gateway import hosted_room_telegram_relay as bridge
    entered, release = threading.Event(), threading.Event()
    original = relay.service.send

    def blocked(**kwargs):
        entered.set()
        assert release.wait(8)
        return original(**kwargs)

    monkeypatch.setattr(relay.service, 'send', blocked)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(relay.call)
            assert entered.wait(2)
            result = future.result(timeout=4)
        session_context.set_session_vars(**relay.context)
        marker = contextvars.ContextVar('poll_marker', default='missing')
        marker.set('copied')
        get_event = rooms.get_event
        seen = []
        def lookup(*args, **kwargs):
            seen.append((marker.get(), threading.current_thread().daemon))
            return get_event(*args, **kwargs)
        monkeypatch.setattr(rooms, 'get_event', lookup)
        assert bridge.room_result(relay.runner, request_id=result['request_id'])['status'] == 'incomplete'
        assert seen == [('copied', True)]
    finally:
        release.set()
        wait_for_worker()


@pytest.mark.parametrize('mutation', ['kind', 'thread', 'text'])
def test_poll_rejects_unverified_submission(relay, monkeypatch, mutation):
    from gateway.hosted_room_telegram_relay import room_result
    result = relay.call()
    wait_for_worker()
    user = next(e for e in events(relay) if e['kind'] == 'message.user')
    if mutation == 'kind':
        user['kind'] = 'message.member'
    else:
        user['payload']['thread_id' if mutation == 'thread' else 'text'] = 'foreign'
    monkeypatch.setattr(rooms, 'get_event', lambda *args, **kwargs: user)
    assert room_result(relay.runner, request_id=result['request_id'])['status'] == 'denied'


@pytest.mark.parametrize('problem', ['extra', 'duplicate', 'missing', 'peer', 'unknown'])
def test_actual_room_membership_denied_without_appending(relay, monkeypatch, problem):
    original = relay.service._room
    def changed(room_id):
        room = original(room_id)
        if problem == 'extra':
            room['members'].append({'member_id': 'unexpected', 'profile': 'extra', 'handle': 'extra'})
        elif problem == 'duplicate':
            room['members'].append(dict(room['members'][1]))
        elif problem == 'missing':
            room['members'].pop(1)
        elif problem == 'peer':
            room['members'][1]['target'] = {'kind': 'peer'}
        else:
            room['members'][1]['profile'] = 'unknown'
        return room
    monkeypatch.setattr(relay.service, '_room', changed)
    before = events(relay)
    assert relay.call()['status'] == 'denied'
    assert events(relay) == before


def test_canonical_room_roster_cannot_change_after_ownership(relay):
    with pytest.raises(rooms.RoomConflictError, match='different state'):
        rooms.create_room(relay.db.db_path, room_id='fixture-room', name='Fixture',
            members=[{'member_id': 'worker', 'profile': 'server', 'handle': 'server'},
                     {'member_id': 'other', 'profile': 'default', 'handle': 'other'},
                     {'member_id': 'new', 'profile': 'extra', 'handle': 'extra'}],
            authority_gateway_id=rooms.local_authority_gateway_id())


def test_unexpected_worker_exception_fails_incomplete_and_releases_guard(relay, monkeypatch):
    import gateway.hosted_room_telegram_relay as bridge
    monkeypatch.setattr(bridge, 'CALL_TIMEOUT_SECONDS', 0.2)
    monkeypatch.setattr(bridge, '_authorize', lambda *args: (_ for _ in ()).throw(KeyError('unexpected')))
    result = relay.call()
    assert result == {'status': 'incomplete', 'reason': 'submission_or_result_unknown'}
    assert bridge._worker_lock.acquire(timeout=0.2)
    bridge._worker_lock.release()


def test_empty_committed_member_text_is_not_completion():
    from gateway.hosted_room_telegram_relay import _correlated_result
    def event(event_id, kind, **payload):
        return {'event_id': event_id, 'kind': kind, 'payload': {
            'discussion_event_id': 'discussion', 'thread_id': 'thread', 'member_id': 'worker', **payload}}
    log = [event('reply', 'message.member', text=''),
           event('turn', 'turn.settled', message_event_id='reply'),
           event('done', 'room.activity', status='settled')]
    assert _correlated_result(log, event_id='discussion', thread_id='thread',
                              members={'worker': 'server'}) == {'status': 'incomplete', 'text': ''}


def test_roster_change_while_waiting_suppresses_result(relay, monkeypatch):
    from gateway import hosted_room_discussion as discussion
    changed = {'active': False}
    original_room = relay.service._room

    def room(room_id):
        value = original_room(room_id)
        if changed['active']:
            value['members'].append({'member_id': 'extra', 'profile': 'extra', 'handle': 'extra'})
        return value

    def publish(binding):
        room_value = relay.service._room('fixture-room')
        log = events(relay)
        user = next(e for e in log if e['kind'] == 'message.user')
        task = discussion.plan_next_task(room_value, log, local_profiles=['server', 'default']).task
        assert task is not None
        publication = discussion.plan_publication(room_value, log, task, status='settled',
            result={'text': 'SHOULD NOT LEAK'}, local_profiles=['server', 'default'])
        for item in publication.events:
            rooms.append_event(relay.db.db_path, **item.append_kwargs('fixture-room'))
        gateway, epoch = relay.service._owned_authority('fixture-room')
        rooms.append_event(relay.db.db_path, room_id='fixture-room', event_id='roster-terminal',
            kind='room.activity', actor={'kind': 'gateway', 'id': gateway},
            payload={'status': 'settled', 'reason_code': 'silent_round',
                     'thread_id': user['payload']['thread_id'], 'discussion_event_id': user['event_id']},
            authority_gateway_id=gateway, authority_epoch=epoch)
        changed['active'] = True

    monkeypatch.setattr(relay.service, '_room', room)
    monkeypatch.setattr(relay.service, 'prepare_room', publish)
    result = relay.call()
    assert result['status'] == 'incomplete'
    assert 'SHOULD NOT LEAK' not in json.dumps(result)
