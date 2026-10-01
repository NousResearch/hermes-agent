"""Always allow in this chat: one exact operation, one Bot, one room, one chat, until it ends."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from gateway import group_chat_access as access
from gateway import group_chat_rules as rules
from gateway import hosted_rooms
from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.group_chat_fixtures import OWNER, Bot, authority_for, message, runner_for

KEY, OTHER_KEY = 'a' * 64, 'b' * 64
MEMBERS = [{'member_id': 'ada', 'profile': 'default', 'handle': 'ada', 'target': {'kind': 'local', 'profile': 'default'}},
           {'member_id': 'bob', 'profile': 'helper', 'handle': 'bob'},
           {'member_id': 'far', 'profile': 'far', 'handle': 'far', 'target': {
               'kind': 'peer', 'peer_id': 'p', 'installation_id': 'i', 'profile': 'far',
               'capability_digest': 'c' * 64}}]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from gateway.session_hosted_service import CanonicalHostedRoomService
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with SessionDB(tmp_path / 'state.db') as db:
        authority = authority_for(tmp_path, db)
        service = authority.hosted_room_service = CanonicalHostedRoomService(authority, None)
        service.approvals = []
        monkeypatch.setattr(service, 'approve', lambda **kw: service.approvals.append(kw) or {
            'status': 'resolved', 'prompt_id': kw['request_id']})
        runner = runner_for(authority, Bot())
        gateway = hosted_rooms.local_authority_gateway_id()
        service.authorize_room(OWNER, 'mine', create=True)
        hosted_rooms.create_room(db.db_path, room_id='mine', name='Research', members=MEMBERS,
                                 authority_gateway_id=gateway)
        event = message('/group')
        chat, adapter = access.resolve_chat(runner, event)
        code, _ = access.request_code(runner, authority, chat, event.source, adapter)
        access.control_verb(runner)({'action': 'allow', 'code': code}, OWNER)
        yield SimpleNamespace(db=db, authority=authority, service=service, runner=runner, chat=chat,
                              gateway=gateway, remember={'grant_id': chat.key, 'by': 'Alice via Telegram'})


def pending(setup, request, *, member='ada', key=KEY, task='task-1', choices=('once', 'deny')):
    approval = {'kind': 'approval', 'prompt_id': request, 'command': 'rm -rf ./build',
                'description': 'delete', 'choices': list(choices)}
    if key:
        approval.update(remember_key=key, remember_context='Local, folder /work')
    action = {'kind': 'approval', 'task_id': task, 'execution_generation': 1, 'run_id': None,
              'session_id': 'session-' + member, 'request_id': request, 'approval': approval}
    setup.service._set_pending_action('mine', member, action)
    return dict(member_id=member, task_id=task, execution_generation=1, request_id=request)


def always(setup, exact):
    return setup.service.approve_room_task('mine', **exact, choice='always', remember=setup.remember)


def test_the_canonical_prompt_names_the_operation_only_when_always_is_offered():
    from gateway.session_events import SessionEvents
    from gateway.session_pending_controls import PendingControls
    controls = PendingControls(SessionEvents())
    base = {'command': 'rm -rf ./build', 'description': 'delete', 'remember_key': KEY,
            'remember_context': 'Local, folder /work'}
    cases = {'r1': {}, 'r2': {'smart_denied': True}, 'r3': {'allow_permanent': False},
             'r4': {'remember_key': 'not-a-key'}, 'r5': {'edit': {'path': '/x'}}}
    for request, change in cases.items():
        controls.register('session', object(), 1, {**base, **change, 'request_id': request})
    prompts = {request: prompt for request, (_, prompt) in controls.pending.items()}
    assert prompts['r1']['remember_key'] == KEY and prompts['r1']['remember_context'] == 'Local, folder /work'
    assert all('remember_key' not in prompts[r] for r in ('r2', 'r3', 'r4', 'r5'))


def test_always_approves_once_then_answers_the_same_operation_by_itself(setup):
    first = pending(setup, 'req-1')
    result = always(setup, first)
    assert result['status'] == 'resolved' and result['remembered'] == rules._rules(
        setup.db._conn)[0]['rule_id'][:6]
    assert setup.service.approvals == [{'session_id': 'session-ada', 'request_id': 'req-1', 'choice': 'once'}]
    pending(setup, 'req-2', task='task-2')
    assert setup.service.approvals[-1] == {'session_id': 'session-ada', 'request_id': 'req-2', 'choice': 'once'}
    pending(setup, 'req-2', task='task-2')  # the driver reports it again: answered only once
    assert len(setup.service.approvals) == 2
    rule, = rules.rules_for(setup.authority, setup.chat.key, hosted_rooms.room_state(setup.db.db_path, room_id='mine'))
    assert (rule['uses'], rule['created_by'], rule['command']) == (1, 'Alice via Telegram', 'rm -rf ./build')


@pytest.mark.parametrize('other', [{'key': OTHER_KEY}, {'member': 'bob'}, {'key': None}])
def test_a_different_operation_or_bot_still_asks(setup, other):
    always(setup, pending(setup, 'req-1'))
    pending(setup, 'req-2', task='task-2', **other)
    assert len(setup.service.approvals) == 1


def test_revoking_the_chat_or_a_new_room_authority_ends_the_rule(setup):
    always(setup, pending(setup, 'req-1'))
    verb = access.control_verb(setup.runner)
    listed, = verb({'action': 'list'}, OWNER)['chats']
    assert listed['remembered'] == [{'group': 'Research', 'bot': 'ada', 'command': 'rm -rf ./build',
                                     'context': 'Local, folder /work', 'uses': 0}]
    verb({'action': 'revoke', 'grant': setup.chat.key[:8]}, OWNER)
    assert rules._rules(setup.db._conn) == []
    pending(setup, 'req-2', task='task-2')
    assert len(setup.service.approvals) == 1


def test_a_new_authority_epoch_never_reuses_an_old_rule(setup):
    always(setup, pending(setup, 'req-1'))
    with setup.db._lock:
        setup.db._conn.execute("UPDATE hosted_rooms SET authority_epoch=2 WHERE room_id='mine'")
        setup.db._conn.commit()
    pending(setup, 'req-2', task='task-2')
    assert len(setup.service.approvals) == 1
    room = hosted_rooms.room_state(setup.db.db_path, room_id='mine')
    assert rules.rules_for(setup.authority, setup.chat.key, room) == []
    listed, = access.control_verb(setup.runner)({'action': 'list'}, OWNER)['chats']
    assert listed['remembered'] == []  # the owner isn't shown a rule that can no longer apply
    rules._save(setup.service, setup.remember, rules.binding(room, 'ada'), {
        'remember_key': OTHER_KEY, 'remember_context': 'Local, folder /work', 'command': 'x'})
    assert [r['operation_key'] for r in rules._rules(setup.db._conn)] == [OTHER_KEY]  # the stale one is pruned


def test_always_needs_an_exact_rememberable_local_request(setup):
    with pytest.raises(RuntimeStoreError, match='unsupported_operation'):
        always(setup, pending(setup, 'req-1', key=None))
    with pytest.raises(RuntimeStoreError, match='unsupported_operation'):
        always(setup, pending(setup, 'req-2', member='far'))
    exact = pending(setup, 'req-3')
    with pytest.raises(RuntimeError, match='no longer pending'):
        always(setup, {**exact, 'request_id': 'req-other'})
    for remember in (None, {'grant_id': setup.chat.key}, 'chat'):
        with pytest.raises(RuntimeStoreError, match='invalid_params'):
            setup.service.approve_room_task('mine', **exact, choice='always', remember=remember)
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        setup.service.approve_room_task('mine', **exact, choice='once', remember=setup.remember)
    assert setup.service.approvals == []


def test_an_unsaved_or_unresolved_always_is_still_only_once(setup, monkeypatch):
    monkeypatch.setattr(rules, 'MAX_RULES_PER_ROOM', 0)
    result = always(setup, pending(setup, 'req-1'))
    assert result['status'] == 'resolved' and result['remembered'] is None
    monkeypatch.setattr(setup.service, 'approve', lambda **kw: {'status': 'already_resolved'})
    monkeypatch.setattr(rules, 'MAX_RULES_PER_ROOM', 32)
    assert always(setup, pending(setup, 'req-2', task='task-2'))['remembered'] is None
    assert rules._rules(setup.db._conn) == []


def test_another_owners_room_or_a_foreign_grant_is_never_remembered(setup):
    setup.service.authorize_room('uid:999', 'theirs', create=True)
    hosted_rooms.create_room(setup.db.db_path, room_id='theirs', name='Theirs', members=MEMBERS,
                             authority_gateway_id=setup.gateway)
    action = {'kind': 'approval', 'task_id': 't', 'execution_generation': 1, 'session_id': 's', 'request_id': 'r',
              'approval': {'remember_key': KEY, 'remember_context': 'Local, folder /work', 'command': 'x'}}
    setup.service._set_pending_action('theirs', 'ada', action)
    result = setup.service.approve_room_task('theirs', member_id='ada', task_id='t', execution_generation=1,
                                             choice='always', request_id='r', remember=setup.remember)
    assert result['remembered'] is None and rules._rules(setup.db._conn) == []


def test_a_transient_failure_is_tried_again_then_left_to_a_person(setup, monkeypatch):
    always(setup, pending(setup, 'req-1'))
    calls = []

    def flaky(**kw):
        calls.append(kw['request_id'])
        if calls.count(kw['request_id']) == 1 or kw['request_id'] == 'req-3':
            raise TimeoutError('the gateway loop was too slow')
        return {'status': 'resolved', 'prompt_id': kw['request_id']}
    monkeypatch.setattr(setup.service, 'approve', flaky)
    for _ in range(3):  # the driver reports the request again while it waits
        pending(setup, 'req-2', task='task-2')
    assert calls == ['req-2', 'req-2']
    for _ in range(5):
        pending(setup, 'req-3', task='task-3')
    assert calls.count('req-3') == rules._TRIES


def test_desktop_cannot_choose_always_over_rpc(setup):
    from gateway.session_controls import AuthorityConnection
    desktop = AuthorityConnection(setup.authority, object(), {'user_id': 'desktop-user'})
    status = setup.service.runtime.status
    setup.service.runtime.status = lambda: {**status(), 'running': True}
    setup.service.authorize_room(desktop.actor.subject, 'own', create=True)
    hosted_rooms.create_room(setup.db.db_path, room_id='own', name='Own', members=MEMBERS,
                             authority_gateway_id=setup.gateway)

    async def probe():
        reply = await desktop.dispatch({'id': 1, 'method': 'groups.approve', 'params': {
            'room_id': 'own', 'member_id': 'ada', 'task_id': 't', 'execution_generation': 1,
            'request_id': 'r', 'choice': 'always'}})
        assert reply['error']['message'] == 'invalid_params'
    asyncio.run(probe())


def test_forget_needs_this_chats_rule_and_an_unambiguous_code(setup):
    always(setup, pending(setup, 'req-1'))
    rule, = rules._rules(setup.db._conn)
    assert rules.forget_rule(setup.authority, 'f' * 64, 'mine', rule['rule_id'][:6]) is None
    assert rules.forget_rule(setup.authority, setup.chat.key, 'mine', rule['rule_id'][:3]) is None
    assert rules.forget_rule(setup.authority, setup.chat.key, 'mine', rule['rule_id'][:6].upper())['rule_id'] == rule['rule_id']
    assert rules._rules(setup.db._conn) == []


def test_damaged_rules_are_ignored(setup):
    always(setup, pending(setup, 'req-1'))
    key, value = setup.db._conn.execute(
        "SELECT key, value FROM state_meta WHERE key LIKE 'gateway.messaging.rule.v1:%'").fetchone()
    for damaged in ('{', json.dumps({**json.loads(value), 'operation_key': 'x'}),
                    json.dumps({**json.loads(value), 'room_id': 'theirs'})):
        setup.authority.db._execute_write(lambda conn: conn.execute(
            'UPDATE state_meta SET value=? WHERE key=?', (damaged, key)))
        assert rules._rules(setup.db._conn) == []
        pending(setup, f'req-{damaged[:3]}', task='task-x' + damaged[:3])
    assert len(setup.service.approvals) == 1
