"""/group reads the owner's canonical Group Chats through the same dispatch Desktop uses."""
import asyncio
from types import SimpleNamespace

import pytest

from gateway import group_chat_access as access
from gateway import group_chat_slash as slash
from gateway import hosted_rooms
from hermes_state import SessionDB
from tests.gateway.group_chat_fixtures import OWNER, Bot, authority_for, message, runner_for

MEMBERS = [{'member_id': 'ada', 'profile': 'default', 'handle': 'ada', 'display_name': 'Ada'},
           {'member_id': 'bob', 'profile': 'helper', 'handle': 'bob'}]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from gateway.session_hosted_service import CanonicalHostedRoomService
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with SessionDB(tmp_path / 'state.db') as db:
        authority = authority_for(tmp_path, db)
        service = authority.hosted_room_service = CanonicalHostedRoomService(authority, None)
        status = service.runtime.status
        monkeypatch.setattr(service.runtime, 'status', lambda: {**status(), 'running': True})
        # Planning Bot turns is the driver's job; these tests stop at the room log and controls.
        monkeypatch.setattr(service, 'prepare_room', lambda binding: None)
        service.approvals = []
        monkeypatch.setattr(service, 'approve', lambda **kw: service.approvals.append(kw) or {'status': 'resolved'})
        bot = Bot()
        runner = runner_for(authority, bot)
        state = SimpleNamespace(db=db, authority=authority, service=service, bot=bot, runner=runner,
                                gateway=hosted_rooms.local_authority_gateway_id())
        yield state


def room(setup, room_id, name, owner=OWNER):
    setup.service.authorize_room(owner, room_id, create=True)
    hosted_rooms.create_room(setup.db.db_path, room_id=room_id, name=name, members=MEMBERS,
                             authority_gateway_id=setup.gateway)


def say(setup, room_id, event_id, kind, text, actor):
    hosted_rooms.append_event(setup.db.db_path, room_id=room_id, event_id=event_id, kind=kind, actor=actor,
                              payload={'text': text, 'thread_id': 't', **({'member_id': actor['id']}
                                                                          if kind == 'message.member' else {})},
                              authority_gateway_id=setup.gateway, authority_epoch=1)


def run(setup, text, **kwargs):
    return asyncio.run(slash.GroupChatSlashCommandsMixin._handle_group_command(setup.runner, message(text, **kwargs)))


def allow(setup, reply, subject=OWNER):
    code = next(line.split()[-1] for line in reply.splitlines() if line.startswith('hermes groups allow'))
    return access.control_verb(setup.runner)({'action': 'allow', 'code': code}, subject)


def test_an_unconnected_chat_gets_a_code_and_no_group_data(setup):
    room(setup, 'secret', 'Secret plans')
    reply = run(setup, '/group')
    assert 'hermes groups allow ' in reply and 'Secret' not in reply
    code = reply.split('hermes groups allow ')[1].split()[0]
    assert f'allow {code}' in run(setup, '/group 1')  # one code per chat until used or expired
    assert 'expires in 10 minutes' in reply
    shared = run(setup, '/group list', chat='team', chat_type='group', user='bob')
    assert 'Everyone in this chat will be able to read' in shared
    helped = run(setup, '/group help')
    assert 'isn’t connected' in helped and '/group list [page]' in helped


def test_list_shows_only_the_owners_rooms_with_stable_numbers(setup):
    room(setup, 'mine', 'Research @everyone <#1>')
    room(setup, 'theirs', 'Not yours', owner='uid:999')
    assert 'grant' in allow(setup, run(setup, '/group'))
    listed = run(setup, '/group')
    assert '1. Research ＠everyone ＜＃1＞ · 2 Bots' in listed and 'Not yours' not in listed
    room(setup, 'later', 'Later')
    listed = run(setup, '/group list')
    assert '1. Research' in listed and '2. Later' in listed
    hosted_rooms.disband_room(setup.db.db_path, room_id='mine', expected_gateway_id=setup.gateway, expected_epoch=1)
    listed = run(setup, '/group list')
    assert 'Research' not in listed and '2. Later' in listed
    assert 'isn’t available' in run(setup, '/group 1')


def test_pages_and_bad_arguments(setup):
    for index in range(10):
        room(setup, f'room-{index}', f'Room {index}')
    allow(setup, run(setup, '/group'))
    first = run(setup, '/group list')
    assert first.startswith('Group Chats, page 1 of 2') and 'Next page: /group list 2' in first
    assert run(setup, '/group list 2').count(' · 2 Bots') == 2
    assert 'only 2 pages' in run(setup, '/group list 3')
    for bad in ('/group list 0', '/group list x', '/group 0', '/group -1', '/group ١', '/group 1 dance'):
        assert run(setup, bad).startswith('I didn’t understand'), bad


def test_detail_shows_status_bots_and_inert_recent_messages(setup):
    room(setup, 'mine', 'Research')
    allow(setup, run(setup, '/group'))
    run(setup, '/group')
    say(setup, 'mine', 'u1', 'message.user', 'Hello from Desktop', {'kind': 'user', 'id': 'desktop'})
    say(setup, 'mine', 'm1', 'message.member', 'Hi @everyone MEDIA:/etc/passwd',
        {'kind': 'member', 'id': 'ada', 'profile': 'default'})
    say(setup, 'mine', 'u2', 'message.user', 'From my phone',
        {'kind': 'user', 'id': 'telegram:42', 'display_name': 'Alice via Telegram'})
    detail = run(setup, '/group 1')
    assert detail.startswith('Group 1 · Research\nIdle')
    assert 'Bots: Ada (＠ada), bob (＠bob)' in detail
    assert '• Desktop: Hello from Desktop' in detail
    assert '• Ada: Hi ＠everyone ［media］' in detail and 'passwd' not in detail
    assert '• Alice via Telegram: From my phone' in detail
    assert 'Refresh: /group 1' in detail


def test_people_off_the_allowlist_and_machines_get_nothing(setup):
    room(setup, 'mine', 'Research')
    allow(setup, run(setup, '/group'))
    assert 'Only people on this Bot’s allow_admin_from list' in run(setup, '/group', user='mallory')
    assert 'person' in run(setup, '/group', is_bot=True)
    # A different person in the same DM-scope chat is a different private grant.
    setup.bot.config.extra['allow_admin_from'].append('carol')
    assert 'hermes groups allow' in run(setup, '/group', user='carol')


def test_a_revoked_chat_is_back_to_a_code(setup):
    room(setup, 'mine', 'Research')
    granted = allow(setup, run(setup, '/group'))
    assert 'Research' in run(setup, '/group')
    access.control_verb(setup.runner)({'action': 'revoke', 'grant': granted['grant']}, OWNER)
    reply = run(setup, '/group')
    assert 'hermes groups allow' in reply and 'Research' not in reply


def test_paused_or_missing_service_is_reported(setup, monkeypatch):
    room(setup, 'mine', 'Research')
    allow(setup, run(setup, '/group'))
    run(setup, '/group')
    monkeypatch.setattr(setup.service.runtime, 'status', lambda: {'running': False})
    assert 'driver isn’t running' in run(setup, '/group 1')
    setup.authority.hosted_room_service = None
    assert run(setup, '/group') == slash.UNAVAILABLE


def test_rate_limit_is_bounded_per_person_and_chat(monkeypatch):
    runner = SimpleNamespace()
    now = [100.0]
    monkeypatch.setattr(slash.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(slash, '_RATE_KEYS', 2)
    for _ in range(slash._RATE_LIMIT):
        assert not slash._too_fast(runner, 'a')
    assert slash._too_fast(runner, 'a')
    assert not slash._too_fast(runner, 'b')
    assert slash._too_fast(runner, 'c')  # live buckets are never evicted
    now[0] += slash._RATE_WINDOW_SECONDS
    assert not slash._too_fast(runner, 'a') and len(runner._group_chat_rate_buckets) == 1


def test_group_runs_while_the_chat_agent_is_busy():
    from gateway.run_busy import GatewayBusySessionMixin
    from hermes_cli.commands import resolve_command
    assert 'group' in GatewayBusySessionMixin._PLAIN_COMMANDS
    command = resolve_command('group')
    assert command.gateway_only and command.busy_policy == 'dispatch'


def connected(setup, *, chat_type='dm', chat='chat-1', user='alice'):
    room(setup, 'mine', 'Research')
    allow(setup, run(setup, '/group', chat_type=chat_type, chat=chat, user=user))
    run(setup, '/group', chat_type=chat_type, chat=chat, user=user)


def user_events(setup):
    return [e for e in hosted_rooms.read_events(setup.db.db_path, room_id='mine')['events']
            if e['kind'] == 'message.user']


def test_send_records_the_real_author_once_per_platform_message(setup):
    connected(setup)
    reply = run(setup, '/group 1 send Please run the tests\nthen report', message_id='tg-77')
    assert reply == 'Sent to Group 1. Read the replies with /group 1'
    assert run(setup, '/group 1 send Please run the tests\nthen report', message_id='tg-77') == reply
    event, = user_events(setup)
    assert event['actor'] == {'kind': 'user', 'id': 'telegram:alice', 'display_name': 'Alice via Telegram'}
    assert event['payload']['text'] == 'Please run the tests\nthen report'
    assert event['payload']['thread_id'].startswith('messaging-send:')
    run(setup, '/group 1 send Second', message_id='tg-78')
    assert len(user_events(setup)) == 2
    assert 'Write the message after send' in run(setup, '/group 1 send   ')
    assert 'Write the message after send' in run(setup, '/group 1 send ' + 'x' * 70000)


def test_desktop_send_keeps_its_own_author_and_cannot_borrow_one(setup):
    from gateway.session_controls import AuthorityConnection
    from gateway.session_group_controls import dispatch_group_control
    from hermes_state_runtime import RuntimeStoreError
    room(setup, 'mine', 'Research')
    desktop = AuthorityConnection(setup.authority, object(), {'user_id': 'desktop-user'})
    setup.service.authorize_room(desktop.actor.subject, 'own', create=True)
    hosted_rooms.create_room(setup.db.db_path, room_id='own', name='Own', members=MEMBERS,
                             authority_gateway_id=setup.gateway)

    async def probe():
        sent = await desktop.dispatch({'id': 1, 'method': 'groups.send', 'params': {
            'room_id': 'own', 'event_id': 'e1', 'payload': {'text': 'hi', 'thread_id': 't'}}})
        assert sent['result']['event']['actor'] == {'kind': 'user', 'id': 'desktop'}
        forged = await desktop.dispatch({'id': 2, 'method': 'groups.send', 'params': {
            'room_id': 'own', 'event_id': 'e2', 'payload': {'text': 'hi', 'thread_id': 't'},
            'author': {'kind': 'user', 'id': 'telegram:1'}}})
        assert forged['error']['message'] == 'invalid_params'
        for method, author in (('groups.stop', {'kind': 'user', 'id': 'x'}),
                               ('groups.send', {'kind': 'user', 'id': 'desktop'}),
                               ('groups.send', {'kind': 'member', 'id': 'ada'})):
            with pytest.raises(RuntimeStoreError, match='invalid_params'):
                await dispatch_group_control(desktop, method, {'room_id': 'own'}, author=author)
    asyncio.run(probe())


def test_stop_fences_the_room_and_reports_the_tasks(setup):
    import time
    from gateway import hosted_room_driver as tasks
    connected(setup)
    tasks.admit_task(setup.db.db_path, tasks.TaskIdentity('mine', 'task-1', 'thread', 'turn'),
                     payload={'target_profile': 'default', 'target_member_id': 'ada', 'source_event_seq': 1,
                              'prompt': 'work'}, clock=time.time)
    assert run(setup, '/group 1 stop', message_id='s1') == 'Stopping work in Group 1 (1 task).'
    stops = [e for e in hosted_rooms.read_events(setup.db.db_path, room_id='mine')['events']
             if e['kind'] == 'room.stop_requested']
    assert len(stops) == 1 and stops[0]['payload']['cancel_id'].startswith('messaging-stop:')
    assert run(setup, '/group 1 stop', message_id='s2') == 'Nothing was running in Group 1.'


def pending(setup, request='req-1', command='rm -rf ./build'):
    action = {'kind': 'approval', 'task_id': 'task-1', 'execution_generation': 1, 'run_id': None,
              'session_id': 'session-1', 'request_id': request,
              'approval': {'kind': 'approval', 'prompt_id': request, 'command': command,
                           'description': 'delete build output', 'choices': ['once', 'deny']}}
    setup.service._set_pending_action('mine', 'ada', action)
    return slash.approval_code({**action, 'member_id': 'ada'})


def test_approve_once_and_deny_answer_the_exact_request(setup):
    connected(setup)
    code = pending(setup)
    detail = run(setup, '/group 1')
    assert 'Idle · 1 approval waiting' in detail
    assert f'Approval {code} · Ada asks to run:\n```\nrm -rf ./build\n```\ndelete build output' in detail
    assert f'Answer: /group 1 approve {code} once|deny' in detail
    assert run(setup, f'/group 1 approve {code} once') == 'Allowed once for Ada.'
    assert setup.service.approvals == [{'session_id': 'session-1', 'request_id': 'req-1', 'choice': 'once'}]
    assert 'isn’t waiting any more' in run(setup, f'/group 1 approve {code} once')
    code = pending(setup, request='req-2')
    assert run(setup, f'/group 1 APPROVE {code.upper()} Deny') == 'Denied for Ada.'
    assert setup.service.approvals[-1]['choice'] == 'deny'
    assert len(setup.service.approvals) == 2


def test_a_change_rechecks_access_right_before_dispatch(setup, monkeypatch):
    connected(setup)
    code = pending(setup)
    real = slash.current_grant
    calls = []
    monkeypatch.setattr(slash, 'current_grant', lambda *a: calls.append(1) or (real(*a) if len(calls) == 1 else None))
    assert 'access to Group Chats changed' in run(setup, f'/group 1 approve {code} once')
    assert setup.service.approvals == []


def test_an_unconfirmed_outcome_is_reported_not_retried(setup, monkeypatch):
    connected(setup)
    code = pending(setup)
    monkeypatch.setattr(setup.service, 'approve', lambda **kw: (_ for _ in ()).throw(TimeoutError('secret text')))
    reply = run(setup, f'/group 1 approve {code} once')
    assert 'couldn’t confirm whether that worked' in reply and 'secret' not in reply


def test_shared_chat_controls_need_the_group_admin_list(setup):
    connected(setup, chat_type='group', chat='team', user='bob')
    assert 'Sent to Group 1' in run(setup, '/group 1 send hi', chat_type='group', chat='team', user='bob')
    assert 'group_allow_admin_from' in run(setup, '/group 1 stop', chat_type='group', chat='team', user='carol')
    assert user_events(setup)[0]['actor']['id'] == 'telegram:bob'


def test_send_has_its_own_tighter_limit(setup, monkeypatch):
    connected(setup)
    for index in range(slash._SEND_RATE_LIMIT):
        assert 'Sent' in run(setup, f'/group 1 send {index}', message_id=f'm{index}')
    assert run(setup, '/group 1 send more', message_id='mx') == slash.TOO_FAST
    assert 'Group 1 · Research' in run(setup, '/group 1')


def test_commands_are_shown_exactly_but_never_as_markup():
    assert slash.code('rm -rf ./x && echo `id`', block=True) == '```\nrm -rf ./x && echo ˋidˋ\n```'
    assert slash.code('a\u202eb\nc') == '`a b c`'
    assert slash.code('one\ntwo', block=True) == '```\none\ntwo\n```'
