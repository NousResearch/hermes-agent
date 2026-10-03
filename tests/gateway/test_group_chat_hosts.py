"""/group when a group's host goes offline: every path against mocked groups.succession.* replies.

The room, the chat grants and every other ``groups.*`` call are real (the canonical store and
dispatch Desktop uses). Only the four ``groups.succession.*`` methods are scripted, the way the
gateway answers them, behind the same capability check.
"""
import asyncio
import copy
import dataclasses
from datetime import datetime
import time
from types import SimpleNamespace

import pytest

from gateway import group_chat_access as access
from gateway import group_chat_hosts as hosts
from gateway import group_chat_slash as slash
from gateway import hosted_rooms
from gateway import session_group_controls as controls
from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.group_chat_fixtures import OWNER, Bot, authority_for, message, runner_for

MEMBERS = [{'member_id': 'ada', 'profile': 'default', 'handle': 'ada', 'display_name': 'Ada'},
           {'member_id': 'bob', 'profile': 'helper', 'handle': 'bob'}]
MAC = {'install_id': 'inst-mac', 'name': 'Mac mini'}
VPS = {'install_id': 'inst-vps', 'name': 'Home VPS'}
BOOK = {'install_id': 'inst-book', 'name': 'MacBook'}
OFFLINE_AT = time.time() - 1500
METHODS = {hosts.STATUS: 'session:read', hosts.PREPARE: 'session:control', hosts.PROMOTE: 'session:control',
           hosts.KEEP: 'session:control'}


def hosting(state='host_unreachable', *, this=VPS, host=MAC, backups=None, actions=None, reason=None, **fields):
    """A groups.succession.status reply: Home VPS (this computer) keeps a copy of a group Mac mini hosts."""
    status = {
        'state': state, 'host': {**host, 'reachable': state == 'ok', 'since': OFFLINE_AT},
        'this_install': {**this, 'role': 'host' if this == host else 'backup'}, 'owner': {'name': 'David'},
        'backups': [{**VPS, 'successor': True, 'readiness': 'caught_up', 'behind_by': 0, 'last_seen': None},
                    {**BOOK, 'successor': True, 'readiness': 'behind', 'behind_by': 3, 'last_seen': None}]
        if backups is None else backups,
        'at_risk': {'count': 0}, 'moving': None, 'conflict': None, 'moved': None, 'work': None,
        'actions': [{'action': 'continue', 'targets': ['inst-vps', 'inst-book']}] if actions is None else actions,
        'unavailable_reason': reason}
    return {**status, **fields}


def preview(**fields):
    return {'preview_id': 'pv-1', 'target': {**VPS, 'operator_name': 'David'}, 'owner': {'name': 'David'},
            'behind_by': 0, 'at_risk': {'count': 0},
            'work': {'completed': 0, 'elsewhere': 0, 'unknown': 0, 'waiting_for_host': 0},
            'unavailable_bots': [], 'cautions': [{'code': 'host_may_be_running'}], **fields}


DONE = hosting('ok', host=VPS, actions=[], work={'completed': 2, 'elsewhere': 0, 'unknown': 1, 'waiting_for_host': 2})


def refused(reason, **detail):
    exc = RuntimeStoreError(reason)
    exc.detail = detail
    return exc


class Gateway:
    """groups.succession.* as the gateway answers them, scripted per test; every call is recorded."""

    def __init__(self):
        self.calls = []
        self.status, self.after_promote = hosting(), []
        self.prepare, self.promote, self.keep = preview(), DONE, hosting('ok', host=MAC, actions=[])

    def methods(self, *names):
        return [m for m, _, _ in self.calls if not names or m in names]

    async def __call__(self, connection, method, params):
        self.calls.append((method, copy.deepcopy(params), connection.actor))
        if METHODS[method] not in connection.actor.capabilities:
            raise RuntimeStoreError('permission_denied')
        if method == hosts.STATUS:
            result = self.after_promote.pop(0) if self.after_promote and hosts.PROMOTE in self.methods() \
                else self.status
        else:
            result = {hosts.PREPARE: self.prepare, hosts.PROMOTE: self.promote, hosts.KEEP: self.keep}[method]
        if callable(result):
            result = result(connection.actor)
        if isinstance(result, BaseException):
            raise result
        return copy.deepcopy(result)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from gateway.session_hosted_service import CanonicalHostedRoomService
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(hosts, 'POLL_SECONDS', 0)
    with SessionDB(tmp_path / 'state.db') as db:
        authority = authority_for(tmp_path, db)
        service = authority.hosted_room_service = CanonicalHostedRoomService(authority, None)
        status = service.runtime.status
        monkeypatch.setattr(service.runtime, 'status', lambda: {**status(), 'running': True})
        monkeypatch.setattr(service, 'prepare_room', lambda binding: None)
        bot = Bot()
        runner = runner_for(authority, bot)
        gateway = Gateway()
        real = controls.dispatch_group_control

        async def dispatch(connection, method, params, **kwargs):
            if method in METHODS:
                return await gateway(connection, method, params)
            return await real(connection, method, params, **kwargs)
        monkeypatch.setattr(controls, 'dispatch_group_control', dispatch)
        state = SimpleNamespace(db=db, authority=authority, service=service, bot=bot, runner=runner,
                                gateway=gateway, gateway_id=hosted_rooms.local_authority_gateway_id())
        service.authorize_room(OWNER, 'mine', create=True)
        hosted_rooms.create_room(db.db_path, room_id='mine', name='Research', members=MEMBERS,
                                 authority_gateway_id=state.gateway_id)
        yield state


@pytest.fixture
def advertised(setup, monkeypatch):
    """The gateway lists the succession methods in groups.capabilities, as #105197 registers them."""
    for method, capability in METHODS.items():
        monkeypatch.setitem(controls.GROUP_METHODS, method, capability)
    return setup


def run(setup, text, **kwargs):
    return asyncio.run(slash.GroupChatSlashCommandsMixin._handle_group_command(setup.runner, message(text, **kwargs)))


def connect(setup, subject=OWNER, **kwargs):
    reply = run(setup, '/group', **kwargs)
    code = next(line.split()[-1] for line in reply.splitlines() if line.startswith('hermes groups allow'))
    access.control_verb(setup.runner)({'action': 'allow', 'code': code}, subject)
    return run(setup, '/group list', **kwargs)


SHARED = {'chat_type': 'group', 'chat': 'team', 'user': 'bob'}


def test_without_the_methods_nothing_changes_and_continuing_says_so(setup, monkeypatch):
    for method in METHODS:  # a gateway that doesn't register them
        monkeypatch.delitem(controls.GROUP_METHODS, method, raising=False)
    connect(setup)
    detail = run(setup, '/group 1')
    assert detail.startswith('Group 1 · Research\nIdle\nBots: ') and 'Host' not in detail
    for command in ('/group 1 continue', '/group 1 continue confirm', '/group 1 keep Mac mini'):
        assert run(setup, command) == hosts.UNAVAILABLE
    assert 'continue' not in run(setup, '/group help')
    assert setup.gateway.calls == []


def test_status_gains_the_host_and_who_can_continue(advertised):
    assert 'Research' in connect(advertised)
    gateway = advertised.gateway
    since = hosts.when(OFFLINE_AT)
    detail = run(advertised, '/group 1')
    assert detail.startswith('Group 1 · Research\nIdle\n'
                             f'Host: Mac mini, offline since {since}. Paused.\n'
                             'Can continue on: Home VPS, MacBook.\n'
                             'Reply /group 1 continue to continue it on Home VPS.\nBots: ')
    gateway.status = hosting('ok', actions=[])
    assert 'Idle\nHost: Mac mini.\nCan continue on: Home VPS, MacBook.\nBots: ' in run(advertised, '/group 1')
    # Only the backups the gateway would offer are listed: not one that is offline, on a Hermes too old to
    # keep a copy, or whose permission to keep one expired (its copy is stale). That last one has to be
    # reconnected, and messaging only says so.
    gateway.status = hosting('ok', actions=[{'action': 'continue_anyway'}], backups=[
        {**VPS, 'successor': True, 'readiness': 'caught_up'},
        {**BOOK, 'successor': True, 'readiness': 'offline'},
        {'install_id': 'inst-pi', 'name': 'Old Pi', 'successor': True, 'readiness': 'unsupported'},
        {'install_id': 'inst-studio', 'name': 'Studio', 'successor': True, 'readiness': 'needs_reauthorization'},
        {'install_id': 'inst-attic', 'name': None, 'successor': False, 'readiness': 'needs_reauthorization'}])
    assert ('\nHost: Mac mini.\nCan continue on: Home VPS.\n'
            'Studio needs to be reconnected: its permission to keep a copy of this group expired.\n'
            'Another computer needs to be reconnected: its permission to keep a copy of this group expired.\n'
            'Bots: ') in run(advertised, '/group 1')
    gateway.status = hosting(actions=[], backups=[{**BOOK, 'successor': True, 'readiness': 'needs_reauthorization'}])
    assert ('Paused.\nNo computer can continue this group yet; choose one in Hermes Desktop.\nMacBook needs to be '
            'reconnected: its permission to keep a copy of this group expired.\nBots: ') in run(advertised, '/group 1')
    gateway.status = hosting('host_restarting', actions=[])
    assert '\nHost: Mac mini, restarting.\nCan continue on: Home VPS, MacBook.\nBots: ' in run(advertised, '/group 1')
    gateway.status = hosting(actions=[], reason='not_owner')
    assert ('Paused.\nCan continue on: Home VPS, MacBook.\nOnly David can continue this group on another '
            'computer.\nBots: ') in run(advertised, '/group 1')
    gateway.status = hosting(backups=[{**BOOK, 'successor': False}], actions=[], reason='no_successor')
    assert ('Paused.\nNo computer can continue this group yet; choose one in Hermes Desktop.\nBots: '
            in run(advertised, '/group 1'))
    gateway.status = hosting('moving', actions=[], moving={'to': VPS, 'step': 'fencing', 'started_at': time.time()})
    assert '\nContinuing on Home VPS… Stopping work from Mac mini.\nBots: ' in run(advertised, '/group 1')
    for running, line in ((2, 'after the replies in progress finish (2)'),
                          (1, 'after the replies in progress finish (1)'),
                          (None, 'after the replies in progress finish')):
        gateway.status = hosting('moving', this=MAC, host=MAC, actions=[],
                                 moving={'to': VPS, 'step': 'waiting_for_turns', 'running': running})
        assert f'\nMoving to Home VPS {line}.\nBots: ' in run(advertised, '/group 1')
    gateway.status = hosting('moved_away', this=MAC, actions=[], moved={'to': VPS, 'at': time.time()})
    assert ('\nThis group moved to Home VPS.\nThis computer now keeps a backup copy. Open the group on Home VPS '
            'to keep chatting.\nBots: ') in run(advertised, '/group 1')
    # A state this messaging client doesn't know shows nothing rather than a guess.
    gateway.status = hosting('something_new')
    assert run(advertised, '/group 1').startswith('Group 1 · Research\nIdle\nBots: ')
    gateway.status = refused('internal_error')
    assert run(advertised, '/group 1').startswith('Group 1 · Research\nIdle\nBots: ')
    assert set(gateway.methods()) == {hosts.STATUS}


def test_names_the_gateway_does_not_know_read_as_plain_words(advertised):
    connect(advertised)
    advertised.gateway.status = hosting(host={'install_id': 'inst-mac', 'name': None}, this={'install_id': 'inst-vps'},
                                        backups=[{'install_id': 'inst-vps', 'successor': True}], owner={'name': None})
    detail = run(advertised, '/group 1')
    assert 'Host: the host, offline since ' in detail
    assert 'Can continue on: this computer.\nReply /group 1 continue to continue it on this computer.' in detail
    advertised.gateway.status = hosting('host_restarting', host={'install_id': 'inst-mac', 'name': None})
    assert run(advertised, '/group 1 continue') == 'The host is restarting; the group will continue in a moment.'


def test_continue_shows_the_summary_and_only_confirm_continues(advertised):
    connect(advertised)
    gateway = advertised.gateway
    gateway.prepare = preview(
        behind_by=2, at_risk={'count': 3}, unavailable_bots=[{'member_id': 'ada', 'name': None}],
        work={'completed': 2, 'elsewhere': 1, 'unknown': 1, 'waiting_for_host': 0},
        target={**VPS, 'operator_name': 'Sam'},
        cautions=[{'code': 'host_may_be_running'},
                  {'code': 'participant_not_fenced', 'names': ['MacBook'], 'count': 1},
                  {'code': 'voters_unreachable', 'names': ['Studio', 'Attic'], 'count': 2}])
    assert run(advertised, '/group 1 continue') == '\n'.join([
        'Continue this group on Home VPS?',
        'Home VPS becomes the group’s host. The conversation, members and history stay the same.',
        'Sam will manage this group from Home VPS.',
        '1 Bot runs on Mac mini and stays unavailable until the group moves back to Mac mini: Ada.',
        'Work in progress: 2 finished, 1 still running on other computers, 1 unknown. '
        'Unknown work won’t run again automatically.',
        'Home VPS is missing 3 recent messages. They’ll appear if Mac mini comes back.',
        'Home VPS is catching up 2 messages from another computer.',
        'If Mac mini comes back, it rejoins as a member. Anything it did while offline is shown separately, '
        'not mixed into the conversation.',
        '',
        'Only continue if Mac mini is really offline. If it’s still running somewhere you can’t reach, both '
        'computers may keep working until they reconnect, and you’ll be asked to choose one.',
        'MacBook runs an older Hermes and may still accept work from Mac mini if it is still running.',
        'Studio and Attic can’t be reached, so this computer can’t confirm Mac mini has stopped. Continue only '
        'if Mac mini is really offline.',
        '',
        'Reply /group 1 continue confirm to proceed.'])
    prepared = [params for method, params, _ in gateway.calls if method == hosts.PREPARE]
    assert prepared == [{'room_id': 'mine', 'target_install_id': 'inst-vps'}]
    assert hosts.PROMOTE not in gateway.methods()
    reply = run(advertised, '/group 1 continue confirm', message_id='m-2')
    assert reply == 'Done. “Research” now continues on Home VPS. 1 task unknown, 2 waiting for Mac mini.'
    promoted = [(params, actor) for method, params, actor in gateway.calls if method == hosts.PROMOTE]
    assert [params for params, _ in promoted] == [
        {'room_id': 'mine', 'target_install_id': 'inst-vps', 'preview_id': 'pv-1', 'confirm': True}]
    actor = promoted[0][1]
    assert actor.subject == OWNER and 'session:operator' not in actor.capabilities
    assert actor.transport_id.startswith('messaging:private:')
    # A confirmation answers one summary: confirming again shows a new one and continues nothing.
    assert run(advertised, '/group 1 continue confirm', message_id='m-3').startswith('Continue this group on')
    assert gateway.methods(hosts.PROMOTE) == [hosts.PROMOTE]


def test_a_minimal_summary_and_counted_older_computers(advertised):
    connect(advertised)
    advertised.gateway.prepare = preview(
        behind_by=1, at_risk={'count': 1}, target={**VPS, 'operator_name': 'David'}, cautions=[
            {'code': 'participant_not_fenced', 'names': ['MacBook'], 'count': 3}, {'code': 'later_code'},
            {'code': 'voters_unreachable', 'names': [], 'count': 1}])
    summary = run(advertised, '/group 1 continue')
    assert summary == '\n'.join([
        'Continue this group on Home VPS?',
        'Home VPS becomes the group’s host. The conversation, members and history stay the same.',
        'Home VPS is missing 1 recent message. It’ll appear if Mac mini comes back.',
        'Home VPS is catching up 1 message from another computer.',
        'If Mac mini comes back, it rejoins as a member. Anything it did while offline is shown separately, '
        'not mixed into the conversation.',
        '',
        '3 computers run an older Hermes and may still accept work from Mac mini if it is still running.',
        '1 computer can’t be reached, so this computer can’t confirm Mac mini has stopped. Continue only if '
        'Mac mini is really offline.',
        '',
        'Reply /group 1 continue confirm to proceed.'])


def test_confirm_waits_for_the_move_then_reports_it(advertised, monkeypatch):
    connect(advertised)
    gateway = advertised.gateway
    moving = hosting('moving', actions=[], moving={'to': VPS, 'step': 'catching_up', 'started_at': time.time()})
    gateway.promote = moving
    gateway.after_promote = [hosting('moving', actions=[], moving={'to': VPS, 'step': 'reconciling'}),
                             hosting('ok', host=VPS, actions=[])]
    run(advertised, '/group 1 continue')
    # No work summary in the status: the driver's own task states are counted instead.
    monkeypatch.setattr(advertised.service, 'status', lambda room_id=None: {
        'running': True, 'working': False, 'blocked': False, 'counts': {}, 'pending_actions': [],
        'tasks': [{'task_id': 't1', 'member_id': 'ada', 'state': 'waiting_for_host', 'resource': 'bot'},
                  {'task_id': 't2', 'member_id': 'ada', 'state': 'unknown'}]})
    reply = run(advertised, '/group 1 continue confirm', message_id='m-2')
    assert reply == 'Done. “Research” now continues on Home VPS. 1 task unknown, 1 waiting for Mac mini.'
    assert gateway.methods(hosts.STATUS).count(hosts.STATUS) >= 3
    gateway.promote, gateway.after_promote = moving, []
    monkeypatch.setattr(hosts, 'WAIT_SECONDS', 0)
    run(advertised, '/group 1 continue', message_id='m-3')
    assert (run(advertised, '/group 1 continue confirm', message_id='m-4')
            == 'Continuing on Home VPS… Catching up history. Send /group 1 to check.')


@pytest.mark.parametrize(('error', 'reply'), [
    (refused('room_authority_promised', other={'install_id': 'inst-book', 'name': 'MacBook'}),
     'Couldn’t continue on Home VPS: MacBook is already continuing “Research”.'),
    (refused('room_authority_promised'),
     'Couldn’t continue on Home VPS: another computer is already continuing “Research”.'),
    (refused('preview_stale'),
     'Couldn’t continue on Home VPS: the group changed after that summary. '
     'Reply /group 1 continue to see what changed.'),
    (refused('host_reachable'), 'Couldn’t continue on Home VPS: Mac mini is online again.'),
    (refused('target_not_ready', target={**VPS, 'readiness': 'behind'}),
     'Couldn’t continue on Home VPS: it isn’t ready to continue this group yet.'),
    (refused('not_owner'), hosts.OWNER_ONLY),
    (refused('permission_denied'), hosts.OWNER_ONLY),
    (refused('runtime_coordination_required'), slash.PAUSED),
    (refused('something_else'), 'Couldn’t continue on Home VPS. Reply /group 1 continue to try again.'),
])
def test_a_refused_continue_says_why_in_the_ux_words(advertised, error, reply):
    connect(advertised)
    run(advertised, '/group 1 continue')
    advertised.gateway.promote = error
    assert run(advertised, '/group 1 continue confirm', message_id='m-2') == reply
    assert advertised.gateway.methods(hosts.PROMOTE) == [hosts.PROMOTE]


def test_an_unconfirmed_continue_is_reported_never_retried(advertised):
    connect(advertised)
    run(advertised, '/group 1 continue')
    advertised.gateway.promote = TimeoutError('private transport text')
    reply = run(advertised, '/group 1 continue confirm', message_id='m-2')
    assert 'couldn’t confirm whether that worked' in reply and 'private' not in reply
    advertised.gateway.promote = {'unexpected': True}
    run(advertised, '/group 1 continue', message_id='m-3')
    assert 'couldn’t confirm whether that worked' in run(advertised, '/group 1 continue confirm', message_id='m-4')
    assert advertised.gateway.methods(hosts.PROMOTE) == [hosts.PROMOTE] * 2


def test_a_prepare_refusal_never_leaves_a_summary_to_confirm(advertised):
    connect(advertised)
    advertised.gateway.prepare = refused('room_authority_promised', other=MAC)
    assert run(advertised, '/group 1 continue') == (
        'Couldn’t continue on Home VPS: Mac mini is already continuing “Research”.')
    advertised.gateway.prepare = preview()
    assert run(advertised, '/group 1 continue confirm').startswith('Continue this group on Home VPS?')
    assert hosts.PROMOTE not in advertised.gateway.methods()


@pytest.mark.parametrize(('status', 'reply'), [
    (hosting(reason='not_owner', actions=[]), hosts.OWNER_ONLY),
    (hosting(actions=[{'action': 'continue', 'targets': ['inst-book']}]),
     'This computer can’t continue “Research”. It can continue on: MacBook.'),
    (hosting(backups=[], actions=[], reason='no_successor'),
     'This computer can’t continue “Research”. No computer can continue this group yet; choose one in '
     'Hermes Desktop.'),
    (hosting('host_restarting', actions=[]), 'Mac mini is restarting; the group will continue in a moment.'),
    (hosting('ok', actions=[], reason='host_reachable'), '“Research” doesn’t need to move: Mac mini is online.'),
    (hosting('ok', host=VPS, actions=[]), '“Research” is already hosted on Home VPS.'),
    (hosting('moving', actions=[], moving={'to': BOOK, 'step': 'finishing'}),
     'Continuing on MacBook… Finishing. Send /group 1 to check.'),
    (hosting('moved_away', this=MAC, actions=[], moved={'to': VPS}),
     'This group moved to Home VPS. This computer now keeps a backup copy. Open the group on Home VPS to keep '
     'chatting.'),
])
def test_continue_explains_when_this_computer_cannot(advertised, status, reply):
    connect(advertised)
    advertised.gateway.status = status
    assert run(advertised, '/group 1 continue') == reply
    assert hosts.PREPARE not in advertised.gateway.methods()


def test_continue_needs_the_status_to_be_readable(advertised):
    connect(advertised)
    advertised.gateway.status = refused('not_owner')
    assert run(advertised, '/group 1 continue') == hosts.OWNER_ONLY
    advertised.gateway.status = refused('not_found')
    assert run(advertised, '/group 1 continue') == ('Group 1 isn’t available right now. '
                                                     'Send /group list to check.')
    assert run(advertised, '/group 9 continue').startswith('Group 9 isn’t available.')


def test_a_shared_chat_sees_the_state_but_never_continues_or_keeps(advertised):
    connect(advertised, **SHARED)
    gateway = advertised.gateway
    detail = run(advertised, '/group 1', **SHARED)
    assert ('Paused.\nCan continue on: Home VPS, MacBook.\nOnly David can continue this group on another '
            'computer.\nBots: ') in detail and 'Reply' not in detail
    for command in ('/group 1 continue', '/group 1 continue confirm', '/group 1 keep Mac mini'):
        assert run(advertised, command, **SHARED) == hosts.PRIVATE_ONLY
    gateway.status = hosting('continued_on_two', actions=[{'action': 'keep', 'targets': ['inst-mac', 'inst-vps']}],
                             conflict={'hosts': [{**MAC, 'since': 1}, {**VPS, 'since': 2}]})
    assert ('“Research” was continued on two computers: Mac mini and Home VPS. Waiting for David to choose which '
            'computer keeps the group.') in run(advertised, '/group 1', **SHARED)
    assert not set(gateway.methods()) - {hosts.STATUS}
    # The gateway can tell a shared chat apart too, and refuses it owner-only actions as well.
    assert gateway.calls and all(actor.transport_id.startswith('messaging:shared:') for _, _, actor in gateway.calls)
    # People off the shared chat's admin list get nothing at all.
    assert 'group_allow_admin_from' in run(advertised, '/group 1 continue', **{**SHARED, 'user': 'carol'})


def test_keep_chooses_the_computer_for_a_group_continued_on_two(advertised):
    connect(advertised)
    gateway = advertised.gateway
    gateway.status = hosting('continued_on_two', actions=[{'action': 'keep', 'targets': ['inst-mac', 'inst-vps']}],
                             conflict={'hosts': [{**MAC, 'since': 1}, {**VPS, 'since': 2}]})
    choice = ('“Research” was continued on two computers: Mac mini and Home VPS. '
              'Reply /group 1 keep Mac mini or /group 1 keep Home VPS.')
    assert choice in run(advertised, '/group 1')
    assert run(advertised, '/group 1 continue') == choice
    assert run(advertised, '/group 1 keep') == choice
    assert run(advertised, '/group 1 keep Studio') == choice
    assert run(advertised, '/group 1 keep   mac  MINI') == ('Done. “Research” now continues on Mac mini. '
                                                         'Messages from Home VPS are kept and shown separately.')
    kept = [params for method, params, _ in gateway.calls if method == hosts.KEEP]
    assert kept == [{'room_id': 'mine', 'install_id': 'inst-mac'}]
    gateway.keep = refused('not_owner')
    assert run(advertised, '/group 1 keep Home VPS') == hosts.OWNER_ONLY
    gateway.keep = refused('conflict_resolved')
    assert run(advertised, '/group 1 keep Home VPS') == ('Couldn’t keep Home VPS. '
                                                         'Send /group 1 to see where things stand.')
    gateway.keep = TimeoutError('secret')
    assert 'couldn’t confirm whether that worked' in run(advertised, '/group 1 keep Home VPS')
    gateway.status = hosting('continued_on_two', actions=[{'action': 'keep', 'targets': ['inst-vps', 'inst-mac']}],
                             conflict={'hosts': [{**MAC, 'since': 1}, {**VPS, 'since': 2}], 'running_on': VPS})
    assert ('“Research” was continued on two computers: Home VPS and Mac mini. Home VPS is running the group; '
            'Mac mini stopped. Reply /group 1 keep Home VPS or /group 1 keep Mac mini.') in run(advertised, '/group 1')
    gateway.status = hosting('continued_on_two', actions=[{'action': 'keep', 'targets': ['a', 'b']}],
                             conflict={'hosts': [{'install_id': 'a', 'name': 'Studio'},
                                                 {'install_id': 'b', 'name': 'studio'}]})
    assert run(advertised, '/group 1 keep Studio') == ('Two computers have that name. Choose which one keeps the '
                                                       'group in Hermes Desktop.')
    gateway.status = hosting()
    assert run(advertised, '/group 1 keep Mac mini') == (
        '“Research” wasn’t continued on two computers, so there’s nothing to choose.')
    assert gateway.methods(hosts.KEEP) == [hosts.KEEP] * 4


def test_continue_and_keep_recheck_access_right_before_the_change(advertised, monkeypatch):
    connect(advertised)
    run(advertised, '/group 1 continue')
    real = slash.current_grant
    calls = []
    monkeypatch.setattr(slash, 'current_grant', lambda *a: calls.append(1) or (real(*a) if len(calls) == 1 else None))
    assert 'access to Group Chats changed' in run(advertised, '/group 1 continue confirm', message_id='m-2')
    assert hosts.PROMOTE not in advertised.gateway.methods()


def test_buttons_where_the_chat_has_them_confirm_the_summary_they_came_with(advertised):
    connect(advertised)
    sent = []

    async def picker(event, session_key, *, title, choices, on_choice_selected):
        sent.append(SimpleNamespace(title=title, choices=choices, choose=on_choice_selected))
        return True
    advertised.runner._try_send_choice_picker = picker
    advertised.runner._session_key_for_source = lambda source: 'session'
    assert run(advertised, '/group 1 continue') is None
    assert sent[0].title.startswith('Continue this group on Home VPS?\n')
    assert sent[0].title.endswith('Reply /group 1 continue confirm to proceed.')
    assert [c['label'] for c in sent[0].choices] == ['Continue on Home VPS', 'Cancel']
    assert asyncio.run(sent[0].choose('chat-1', 'confirm')) == (
        'Done. “Research” now continues on Home VPS. 1 task unknown, 2 waiting for Mac mini.')
    assert asyncio.run(sent[0].choose('chat-1', 'confirm')) == (
        'That summary has expired. Reply /group 1 continue to see a new one.')
    assert run(advertised, '/group 1 continue', message_id='m-2') is None
    assert asyncio.run(sent[1].choose('chat-1', 'cancel')) == hosts.CANCELLED
    # Cancelled means nothing is left to confirm: the typed reply shows a new summary.
    assert run(advertised, '/group 1 continue confirm', message_id='m-3') is None
    assert len(sent) == 3 and advertised.gateway.methods(hosts.PROMOTE) == [hosts.PROMOTE]
    # A button only confirms its own summary, never a newer one shown since.
    advertised.gateway.prepare = preview(preview_id='pv-2')
    assert run(advertised, '/group 1 continue', message_id='m-4') is None
    assert asyncio.run(sent[2].choose('chat-1', 'confirm')).startswith('That summary has expired.')
    advertised.gateway.promote = refused('preview_stale')
    assert asyncio.run(sent[3].choose('chat-1', 'confirm')).startswith('Couldn’t continue on Home VPS: the group')
    promoted = [params['preview_id'] for method, params, _ in advertised.gateway.calls if method == hosts.PROMOTE]
    assert promoted == ['pv-1', 'pv-2']


def test_a_summary_expires(advertised):
    connect(advertised)
    run(advertised, '/group 1 continue')
    shown = advertised.runner._group_chat_summaries
    assert len(shown) == 1
    for key, summary in shown.items():
        shown[key] = dataclasses.replace(summary, expires=time.monotonic() - 1)
    assert run(advertised, '/group 1 continue confirm', message_id='m-2').startswith('Continue this group on')
    assert hosts.PROMOTE not in advertised.gateway.methods()


def test_help_lists_continue_and_keep_only_where_offered(advertised):
    connect(advertised)
    listed = run(advertised, '/group help')
    assert ('/group N continue — continue a paused group on this computer\n'
            '/group N keep <computer> — choose the computer that keeps a group continued on two\n'
            '/group help') in listed


def test_waiting_work_is_summed_up_by_host(advertised, monkeypatch):
    connect(advertised)
    monkeypatch.setattr(advertised.service, 'status', lambda room_id=None: {
        'running': True, 'working': True, 'blocked': False, 'counts': {}, 'pending_actions': [],
        'tasks': [{'task_id': 't1', 'member_id': 'ada', 'state': 'waiting_for_host', 'resource': 'bot',
                   'host_name': 'Mac mini'},
                  {'task_id': 't2', 'member_id': 'ada', 'state': 'waiting_for_host', 'resource': 'file',
                   'host_name': 'Mac mini'},
                  {'task_id': 't3', 'member_id': 'bob', 'state': 'waiting_for_host', 'host_name': None},
                  {'task_id': 't4', 'member_id': 'bob', 'state': 'running'}]})
    assert ('Group 1 · Research\nWorking · 2 waiting for Mac mini · 1 waiting for another computer\n'
            in run(advertised, '/group 1'))


@pytest.mark.parametrize(('offset', 'pattern'), [(0, '%H:%M %Z'), (-3 * 86400, '{day} {month} %H:%M %Z')])
def test_offline_since_reads_as_local_time(offset, pattern):
    moment = datetime.now().astimezone()
    stamp = moment.timestamp() + offset
    local = datetime.fromtimestamp(stamp).astimezone()
    expected = local.strftime(pattern.format(day=local.day, month=hosts._MONTHS[local.month - 1]))
    if local.year != moment.year:
        expected = expected.replace(hosts._MONTHS[local.month - 1], f'{hosts._MONTHS[local.month - 1]} {local.year}')
    assert hosts.when(stamp) == expected
    assert hosts.when(local.isoformat()) == expected
    assert hosts.when(None) is None and hosts.when('soon') is None and hosts.when(True) is None
