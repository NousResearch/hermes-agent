"""The owner hears, unasked, when a group moved or paused by itself, or ran on two computers.

The watcher reads like any client, so these tests script only the room log and the
``groups.succession.*`` replies the way the gateway gives them; grants, chats, routing,
buttons and typed commands are the real ones.
"""
import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from gateway import group_chat_access as access
from gateway import group_chat_actions as actions
from gateway import group_chat_hosts as hosts
from gateway import group_chat_notices as notices
from gateway import group_chat_slash as slash
from gateway import session_group_controls as controls
from gateway.config import HomeChannel, Platform
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.group_chat_fixtures import OWNER, Buttons
from tests.gateway.test_group_chat_hosts import (  # noqa: F401 - fixtures
    BOOK, MAC, MEMBERS, SHARED, VPS, advertised, connect, hosting, refused, run, setup)


@pytest.fixture
def watched(advertised, monkeypatch):  # noqa: F811 - the fixture imported above
    """The owner's private chat and a shared chat, a room log the test writes, and one notice pass."""
    runner, bot = advertised.runner, advertised.bot
    adapters = {'default': {Platform.TELEGRAM: bot}}
    runner._adapters_for_profile = lambda profile: adapters.get(profile, {})
    homes, home_sent = [], []
    runner._served_home_channel_transports = lambda: iter(homes)

    async def send_home(platform, home, transport, message, failure_fmt):
        home_sent.append((home.chat_id, message))
        return True
    runner._send_home_channel_message = send_home
    connect(advertised)
    connect(advertised, **SHARED)
    log, real = [], controls.dispatch_group_control

    async def logged(connection, method, params, **kwargs):
        if method == 'groups.log':
            since = params['since_seq']
            if since > len(log):
                raise RuntimeStoreError('invalid_params')
            events = [e for e in log if e['seq'] > since][:params['limit']]
            return {'events': events, 'cursor': events[-1]['seq'] if events else since, 'latest_seq': len(log),
                    'has_more': bool(events) and events[-1]['seq'] < len(log)}
        return await real(connection, method, params, **kwargs)
    monkeypatch.setattr(controls, 'dispatch_group_control', logged)
    advertised.gateway.status = hosting('ok', host=VPS, actions=[])

    def append(kind='authority.transition', *, age=0.0, **payload):
        log.append({'seq': len(log) + 1, 'kind': kind, 'payload': payload, 'created_at': time.time() - age})

    def notify():
        before = len(bot.sent)
        asyncio.run(notices.notify_all(runner))
        return [(chat, text) for chat, text, _ in bot.sent[before:]]

    def home(chat_id, user_id=None):
        homes.append((None, Platform.TELEGRAM, None, HomeChannel(Platform.TELEGRAM, chat_id, 'Home', user_id=user_id),
                      SimpleNamespace(adapter=bot, is_relay=False)))
    return SimpleNamespace(state=advertised, runner=runner, bot=bot, adapters=adapters, append=append, notify=notify,
                           home=home, home_sent=home_sent, here=advertised.gateway_id)


def moved_here(watched, *, age=0.0, **payload):
    watched.append(age=age, **{'reason': 'automatic', 'from_name': 'Mac mini', 'to_name': 'Home VPS',
                               'successor_gateway_id': watched.here, 'proof_kind': 'certified', **payload})


CAREFUL = ('“Research” moved to Home VPS\n'
           'Mac mini went silent for 3 minutes, so Home VPS took over. If Mac mini is actually still running, '
           'the group may now be running in both places.')


def test_the_owner_hears_once_that_the_group_moved_here_by_itself(watched):
    watched.append('message.user', text='before the watcher ever looked')
    moved_here(watched, age=3600)
    assert watched.notify() == []  # older history from before the first look stays history
    moved_here(watched, offline_since=time.time() - 600, at_risk=0)
    assert watched.notify() == [('chat-1', '“Research” moved to Home VPS because Mac mini went offline. '
                                           'It’s running.')]
    assert watched.notify() == []
    moved_here(watched, reason='handover', from_name=None, to_name=None)
    moved_here(watched, reason='manual')
    moved_here(watched, successor_gateway_id='install:' + '0' * 32)  # the computer it moved to tells the owner
    assert watched.notify() == [('chat-1', '“Research” moved to this computer because another computer was '
                                           'shutting down. It’s running.')]


def test_a_move_here_just_before_the_first_look_is_still_told(watched):
    """A room may first show up here because it just moved here: that move is still told, with the
    Bots that stay behind on the old host."""
    moved_here(watched, age=120)
    watched.state.gateway.status = hosting('ok', host=VPS, actions=[],
                                           unavailable_bots=[{'member_id': 'ada', 'name': 'Ada'}])
    assert watched.notify() == [('chat-1', '“Research” moved to Home VPS because Mac mini went offline. It’s '
                                           'running. 1 Bot is unavailable until the group moves back to Mac mini.')]
    assert watched.notify() == []
    moved_here(watched, reason='handover')
    watched.state.gateway.status = hosting('ok', host=VPS, actions=[], unavailable_bots=[
        {'member_id': 'ada', 'name': 'Ada'}, {'member_id': 'bob', 'name': 'bob'}])
    assert watched.notify() == [('chat-1', '“Research” moved to Home VPS because Mac mini was shutting down. '
                                           'It’s running. 2 Bots are unavailable until the group moves back to '
                                           'Mac mini.')]


def test_the_owner_hears_once_per_pause_to_stay_safe(watched):
    gateway = watched.state.gateway
    assert watched.notify() == []
    gateway.status = hosting('paused', this=VPS, host=VPS, paused={'reason': 'lost_majority', 'waiting_for': [MAC]})
    assert watched.notify() == [('chat-1', '“Research” is paused to stay safe: Home VPS can’t reach Mac mini.')]
    assert watched.notify() == []
    gateway.status = refused('internal_error')  # can't tell this time: nothing changes
    assert watched.notify() == []
    gateway.status = hosting('ok', host=VPS, actions=[])
    assert watched.notify() == []
    gateway.status = hosting('paused', this=VPS, host=VPS, paused={'reason': 'lost_majority', 'waiting_for': []})
    assert watched.notify() == [('chat-1', '“Research” is paused to stay safe: Home VPS can’t reach the other '
                                           'computers.')]
    for paused, told in (({'reason': 'no_lease_layer'}, '“Research” is paused to stay safe: Home VPS can’t take '
                           'part in automatic moves right now. Its connection to the other computers isn’t ready.'),
                         ({'reason': 'something_new'}, '“Research” is paused to stay safe.')):
        gateway.status = hosting('ok', host=VPS, actions=[])
        assert watched.notify() == []
        gateway.status = hosting('paused', this=VPS, host=VPS, paused=paused)
        assert watched.notify() == [('chat-1', told)]


def tap(watched, data, *, user='alice', chat='chat-1'):
    result = asyncio.run(slash.GroupChatSlashCommandsMixin._group_chat_action(
        watched.runner, 'telegram', chat, user, data))
    return None if result is None else (result['text'], [name for name, _ in result['buttons']], result['buttons'])


def careful_status(**fields):
    return hosting('ok', host=VPS, actions=[{'action': 'keep', 'targets': ['inst-mac']}],
                   moved_in={'from': MAC, 'at': time.time() - 200, 'proof_kind': 'evidence'}, **fields)


def test_without_buttons_a_careful_move_offers_typed_commands(watched):
    gateway = watched.state.gateway
    watched.notify()
    moved_here(watched, proof_kind='evidence')
    assert watched.notify() == [('chat-1', '\n'.join([
        CAREFUL, '',
        'Keep going: no reply needed.',
        'Go back to Mac mini: /group 1 keep Mac mini',
        'Ask me first: /group 1 ask first']))]
    assert watched.notify() == []
    gateway.status = careful_status()
    assert run(watched.state, '/group 1 keep Mac mini').startswith('Go back to Mac mini? Home VPS pauses now')
    assert hosts.KEEP not in gateway.methods()
    assert run(watched.state, '/group 1 keep Mac mini confirm', message_id='m-2') == (
        'Done. Home VPS paused “Research”; it continues on Mac mini as soon as it’s reachable.')
    assert [params for method, params, _ in gateway.calls if method == hosts.KEEP] == [
        {'room_id': 'mine', 'install_id': 'inst-mac'}]


def test_a_bare_number_is_never_taken_as_a_choice(watched):
    """A Bot may ask a numbered question of its own, so a reply such as "3" is its, never a choice:
    nothing but the gateway's own pending prompts may claim a plain message."""
    from gateway.run_inbound import GatewayInboundMixin
    watched.notify()
    moved_here(watched, proof_kind='evidence')
    assert watched.notify()  # a notice with choices is waiting

    async def nothing(*_args):
        return None
    runner = SimpleNamespace(_hm_update_prompt_reply=lambda *_: None, _hm_clarify_reply=nothing,
                             _hm_slash_confirm_reply=nothing)
    event = SimpleNamespace(allow_gateway_control=True, text='3')
    assert asyncio.run(GatewayInboundMixin._hm_pending_reply_intercepts(runner, event, None, 'key')) is None


def test_a_careful_move_offers_buttons_that_keep_working(watched, monkeypatch):
    buttons = Buttons()
    watched.adapters['default'] = {Platform.TELEGRAM: buttons}
    gateway = watched.state.gateway
    watched.notify()
    moved_here(watched, proof_kind='evidence')
    assert watched.notify() == []
    offer, = buttons.offers
    assert offer.chat_id == 'chat-1' and offer.text == CAREFUL
    assert [name for name, _ in offer.buttons] == ['Keep going', 'Go back to Mac mini', 'Ask me first']
    assert all(len(data.encode()) <= 64 and data.startswith('hg:') for _, data in offer.buttons)  # Telegram's cap
    go, back, ask = (data for _, data in offer.buttons)
    gateway.status = careful_status()
    text, names, confirm = tap(watched, back)
    assert text.startswith('Go back to Mac mini? Home VPS pauses now') and names == ['Go back to Mac mini', 'Cancel']
    assert tap(watched, confirm[1][1])[:2] == (CAREFUL, ['Keep going', 'Go back to Mac mini', 'Ask me first'])
    # A restart, or another prompt opened in the chat meanwhile, changes nothing: the tap finds its notice.
    monkeypatch.setattr(actions, '_LOCKS', {})
    watched.adapters['default'] = {Platform.TELEGRAM: Buttons()}
    real, calls = controls.dispatch_group_control, []

    async def automatic(connection, method, params, **kwargs):
        if method == hosts.AUTOMATIC:
            calls.append(params)
            return {'room_id': params['room_id'], 'automatic': False, 'configuration_seq': 3}
        return await real(connection, method, params, **kwargs)
    monkeypatch.setattr(controls, 'dispatch_group_control', automatic)
    assert tap(watched, ask)[:2] == ('“Research” will ask you before moving.', [])
    assert calls == [{'room_id': 'mine', 'enabled': False}]
    assert tap(watched, go)[:2] == ('Already resolved: “Research” will ask you before moving.', [])
    assert len(calls) == 1


def test_only_the_owners_private_chat_and_person_may_choose(watched):
    buttons = Buttons()
    watched.adapters['default'] = {Platform.TELEGRAM: buttons}
    watched.notify()
    moved_here(watched, proof_kind='evidence')
    watched.notify()
    go = buttons.offers[0].buttons[0][1]
    # A choice from another kind of notice changes nothing.
    assert tap(watched, 'hg:k0:' + go.rsplit(':', 1)[1])[:2] == (CAREFUL, ['Keep going', 'Go back to Mac mini',
                                                                            'Ask me first'])
    assert hosts.KEEP not in watched.state.gateway.methods()
    assert tap(watched, go, user='mallory') is None
    assert tap(watched, go, chat='team') is None
    assert tap(watched, 'hg:go:' + 'x' * 12)[:2] == ('This choice is no longer available.', [])
    assert tap(watched, 'hg:go:../etc') is None and tap(watched, 'ea:once:1') is None
    watched.bot.config.extra['allow_admin_from'].remove('alice')
    buttons.config.extra['allow_admin_from'].remove('alice')
    assert tap(watched, go) is None  # off the Bot's DM admin list
    buttons.config.extra['allow_admin_from'].append('alice')
    granted = access.control_verb(watched.runner)({'action': 'list'}, OWNER)['chats']
    access.control_verb(watched.runner)({'action': 'revoke', 'grant': next(
        c['grant'] for c in granted if c['kind'] == 'private')}, OWNER)
    assert tap(watched, go) is None  # the chat lost its grant


def test_the_computer_the_careful_move_went_to_asks_which_one_keeps_the_group(watched):
    buttons = Buttons()
    watched.adapters['default'] = {Platform.TELEGRAM: buttons}
    gateway = watched.state.gateway
    watched.notify()
    start = time.time() - 900
    conflict = {'hosts': [{**MAC, 'since': start - 3600}, {**VPS, 'since': start + 180}], 'start': start,
                'end': start + 900, 'running_on': VPS}
    gateway.status = hosting('continued_on_two', host=VPS, conflict=conflict,
                             actions=[{'action': 'keep', 'targets': ['inst-vps', 'inst-mac']}])
    watched.notify()
    offer, = buttons.offers
    assert offer.text == (f'“Research” ran on both Home VPS and Mac mini while they couldn’t reach each other '
                          f'({hosts.span(start, start + 900)}). Home VPS is running the group; Mac mini stopped. '
                          'Choose which one to keep. The other’s messages are kept separately.')
    # Keeping the computer still running it is keep going: one tap. Switching takes a second one.
    assert [name for name, _ in offer.buttons] == ['Keep going on Home VPS', 'Keep Mac mini']
    watched.notify()
    assert len(buttons.offers) == 1  # once per incident
    keep_going, keep_other = (data for _, data in offer.buttons)
    text, names, confirm = tap(watched, keep_other)
    assert text == ('Keep Mac mini? “Research” continues on Mac mini. Messages from Home VPS are kept and shown '
                    'separately.')
    assert names == ['Keep Mac mini', 'Cancel'] and hosts.KEEP not in gateway.methods()
    assert tap(watched, confirm[1][1])[1] == ['Keep going on Home VPS', 'Keep Mac mini']
    assert tap(watched, keep_going)[:2] == (
        '“Research” keeps going on Home VPS. Messages from Mac mini are kept and shown separately.', [])
    assert [params for method, params, _ in gateway.calls if method == hosts.KEEP] == [
        {'room_id': 'mine', 'install_id': 'inst-vps'}]
    # From a gateway that doesn't say which one is running, both Keeps take two taps.
    record = {'outcome': None, 'kind': 'conflict', 'view': 'main', 'data': {'hosts': [['a', 'Mac mini'], ['b', 'VPS']]}}
    assert [(label + name, action) for label, name, action, _ in actions.choices(record)] == [
        ('Keep Mac mini', 'k0'), ('Keep VPS', 'k1')]
    # The computer that hosted the group first stays quiet: one notice per incident, not one per computer.
    gateway.status = hosting('continued_on_two', this=MAC, host=MAC, conflict=conflict)
    watched.notify()
    assert len(buttons.offers) == 1


def test_the_gateways_paused_group_incident_offers_to_continue_here(watched):
    buttons = Buttons()
    watched.adapters['default'] = {Platform.TELEGRAM: buttons}
    gateway = watched.state.gateway
    gateway.status = hosting()

    def tell(**data):
        return asyncio.run(slash.GroupChatSlashCommandsMixin._group_chat_notify(
            watched.runner, 'mine', 'host_offline', data))
    assert tell(host='Mac mini', minutes=12) == 1
    offer, = buttons.offers
    assert offer.text == '“Research” is paused: Mac mini has been offline for 12 min.'
    assert [name for name, _ in offer.buttons] == ['Continue on Home VPS']
    text, names, confirm = tap(watched, offer.buttons[0][1])
    assert text.startswith('Continue this group on Home VPS?\n') and 'Reply' not in text
    assert names == ['Continue on Home VPS', 'Cancel']
    assert tap(watched, confirm[1][1])[:2] == ('“Research” is paused: Mac mini has been offline for 12 min.',
                                               ['Continue on Home VPS'])
    text, _, confirm = tap(watched, offer.buttons[0][1])
    assert tap(watched, confirm[0][1])[:2] == (
        '“Research” now continues on Home VPS. 1 task unknown, 2 waiting for Mac mini.', [])
    gateway.status = hosting(actions=[{'action': 'continue', 'targets': ['inst-book']}])  # not this computer
    assert tell() == 1 and len(buttons.offers) == 1
    assert buttons.sent[-1][1] == '“Research” is paused: Mac mini has been offline.'  # told, nothing offered
    watched.adapters['default'] = {Platform.TELEGRAM: watched.bot}
    gateway.status = hosting()
    assert tell(minutes=3) == 1
    assert watched.bot.sent[-1][1] == '\n'.join([
        '“Research” is paused: Mac mini has been offline for 3 min.', '',
        'Continue on Home VPS: /group 1 continue'])
    with pytest.raises(ValueError):
        asyncio.run(slash.GroupChatSlashCommandsMixin._group_chat_notify(watched.runner, 'mine', 'other', {}))


def test_notices_go_to_the_owners_main_channel(watched):
    connect(watched.state, chat='alice')  # the owner's second private chat with the Bot, set as home
    watched.notify()
    watched.home('alice', user_id='alice')
    moved_here(watched)
    assert [chat for chat, _ in watched.notify()] == ['alice']  # the home channel, when it is one of them
    for grant in access.control_verb(watched.runner)({'action': 'list'}, OWNER)['chats']:
        access.control_verb(watched.runner)({'action': 'revoke', 'grant': grant['grant']}, OWNER)
    moved_here(watched, proof_kind='evidence')
    assert watched.notify() == [] and watched.home_sent == [('alice', CAREFUL + '\n\nChoose in Hermes Desktop.')]


def test_without_a_private_chat_only_a_one_to_one_home_channel_hears(watched):
    for grant in access.control_verb(watched.runner)({'action': 'list'}, OWNER)['chats']:
        access.control_verb(watched.runner)({'action': 'revoke', 'grant': grant['grant']}, OWNER)
    watched.home('-100200300', user_id='alice')  # a group: never told group or computer names
    watched.notify()
    moved_here(watched)
    assert watched.notify() == [] and watched.home_sent == []
    watched.home('alice', user_id='alice')
    moved_here(watched)
    watched.notify()
    assert watched.home_sent == [('alice', '“Research” moved to Home VPS because Mac mini went offline. '
                                          'It’s running.')]


def test_only_the_local_accounts_groups_fall_back_to_the_operators_home_channel(watched):
    from gateway import hosted_rooms
    state = watched.state
    dashboard = 'auth:v1:["dashboard","","bob"]'
    state.service.authorize_room(dashboard, 'theirs', create=True)
    hosted_rooms.create_room(state.db.db_path, room_id='theirs', name='Bob’s plans', members=[
        {'member_id': 'ada', 'profile': 'default', 'handle': 'ada'}], authority_gateway_id=state.gateway_id)
    for grant in access.control_verb(watched.runner)({'action': 'list'}, OWNER)['chats']:
        access.control_verb(watched.runner)({'action': 'revoke', 'grant': grant['grant']}, OWNER)
    watched.home('alice', user_id='alice')
    watched.notify()
    moved_here(watched)  # the scripted log is every room's: both groups moved here
    watched.notify()
    assert watched.home_sent == [('alice', '“Research” moved to Home VPS because Mac mini went offline. '
                                          'It’s running.')]


def test_the_paused_group_notice_reaches_the_main_channel_too(watched):
    connect(watched.state, chat='alice')
    refs = asyncio.run(slash.GroupChatSlashCommandsMixin._group_chat_continue_refs(watched.runner, 'mine'))
    assert sorted(chat for _, chat, _, _ in refs) == ['alice', 'chat-1']
    watched.home('alice', user_id='alice')
    refs = asyncio.run(slash.GroupChatSlashCommandsMixin._group_chat_continue_refs(watched.runner, 'mine'))
    assert [(chat, n) for _, chat, _, n in refs] == [('alice', 1)]


def test_notices_skip_shared_chats_copies_and_gateways_without_the_methods(watched, monkeypatch):
    state, gateway = watched.state, watched.state.gateway
    gateway.status = hosting('paused', this=VPS, host=VPS, paused={'reason': 'lost_majority', 'waiting_for': [MAC]})
    saved = ('SELECT key FROM state_meta WHERE substr(key, 1, ?) = ?',
             (len(notices.NOTICE_PREFIX), notices.NOTICE_PREFIX))
    monkeypatch.delitem(controls.GROUP_METHODS, hosts.STATUS)
    assert asyncio.run(notices.notify_all(state.runner)) == 0
    with state.db._read_ctx() as conn:
        assert conn.execute(*saved).fetchall() == []
    monkeypatch.setitem(controls.GROUP_METHODS, hosts.STATUS, 'session:read')
    real = controls.dispatch_group_control

    async def copies(connection, method, params, **kwargs):
        result = await real(connection, method, params, **kwargs)
        if method == 'groups.list':
            result['rooms'] = [{**room, 'copy': True} for room in result['rooms']]
        return result
    monkeypatch.setattr(controls, 'dispatch_group_control', copies)
    assert watched.notify() == [] and hosts.STATUS not in gateway.methods()  # a copy's host reports pauses
    monkeypatch.setattr(controls, 'dispatch_group_control', real)
    assert [chat for chat, _ in watched.notify()] == ['chat-1']  # never the shared chat
    with state.db._read_ctx() as conn:
        assert len(conn.execute(*saved).fetchall()) == 1  # what the owner was told, once per owner


def test_the_notice_watcher_runs_while_the_gateway_does(monkeypatch):
    from gateway.run_startup import GatewayStartupMixin
    assert '_group_chat_notice_watcher' in GatewayStartupMixin._POST_RECONNECT_WATCHERS
    passes = []
    runner = SimpleNamespace(_running=True)

    async def one_pass(target):
        passes.append(target)
        if len(passes) == 1:
            raise RuntimeError('a failed pass is logged, and the next one still runs')
        target._running = False
        return 0
    monkeypatch.setattr(notices, 'notify_all', one_pass)
    asyncio.run(slash.GroupChatSlashCommandsMixin._group_chat_notice_watcher(runner, interval=0))
    assert passes == [runner, runner]


def test_a_backup_computer_reaches_the_group_it_keeps_a_copy_of(watched, monkeypatch):
    """On a backup computer a group is a copy plus its owner's row, written when this computer's
    operator agreed to keep it, and no hosted room. The owner's chats still list it, continue it
    and hear the gateway's paused incident for it."""
    state, gateway = watched.state, watched.state.gateway
    state.service.authorize_room(OWNER, 'copy-1', create=True)  # the owner's row alone, no hosted room
    copy = {'room_id': 'copy-1', 'name': 'Field notes', 'members': MEMBERS, 'authority_gateway_id': 'inst-mac',
            'authority_epoch': 1, 'revision': 0, 'created_at': 0.0, 'updated_at': 0.0, 'latest_seq': 0,
            'copy': True}
    real = controls.dispatch_group_control

    async def with_copy(connection, method, params, **kwargs):
        # The copy reads a backup computer gives the room's owner: listed and readable, never driven here.
        if method == 'groups.list':
            result = await real(connection, method, params, **kwargs)
            return {**result, 'rooms': [*result['rooms'], copy]}
        if params.get('room_id') == 'copy-1' and method == 'groups.state':
            return {'room': copy, 'driver_status': None}
        if params.get('room_id') == 'copy-1' and method == 'groups.log':
            return {'events': [], 'cursor': 0, 'latest_seq': 0, 'has_more': False}
        return await real(connection, method, params, **kwargs)
    monkeypatch.setattr(controls, 'dispatch_group_control', with_copy)
    assert '2. Field notes · 2 Bots · backup copy' in run(state, '/group list')
    gateway.status = hosting()  # its host, Mac mini, is offline, and this computer may continue it
    assert run(state, '/group 2').startswith('Group 2 · Field notes\nHost: Mac mini, offline since ')
    assert run(state, '/group 2 continue').startswith('Continue this group on Home VPS?')
    assert [params['room_id'] for method, params, _ in gateway.calls if method == hosts.PREPARE] == ['copy-1']
    told = asyncio.run(slash.GroupChatSlashCommandsMixin._group_chat_notify(
        state.runner, 'copy-1', 'host_offline', {'host': 'Mac mini', 'minutes': 6}))
    assert told == 1 and watched.bot.sent[-1][1] == '\n'.join([
        '“Field notes” is paused: Mac mini has been offline for 6 min.', '',
        'Continue on Home VPS: /group 2 continue'])
    watched.notify()  # the watcher keeps a cursor for the copy too
    with state.db._read_ctx() as conn:
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (notices._key(OWNER),)).fetchone()
    assert set(json.loads(row[0])) == {'mine', 'copy-1'}
