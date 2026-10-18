"""Which local bots a Desktop group may seat, on a default install with no ``hosted_rooms`` key.

Regression for #139002: Desktop -> Bots -> New Group Chat refused every multi-bot group with a bare
``invalid_params``, because the canonical room service knew only its own profile and the
``hosted_rooms.profiles`` config, not the sibling profiles the same gateway serves. The rule now:
a group seats the profiles this gateway serves, plus configured destinations, and nothing else;
anything else is refused with the member it names.

Every request goes through the Desktop's own path: the host-scoped socket on the launch profile,
routed by ``profile`` to the active profile's real authority and canonical room service. A
connection bound to one profile (``_sibling_for`` refuses it any other) seats that profile plus the
profiles its ``hosted_rooms.profiles`` entries name, as before.
"""
import asyncio
from contextlib import contextmanager
import json
import logging
import os
import threading

import pytest

from tests.gateway.test_session_authorities_multiplex import _reserve_homes
from tests.waiting import wait_until


@contextmanager
def _gateway(tmp_path, monkeypatch, *, served, unserved=(), linked=(), multiplex=True):
    """A running gateway over ``default`` + ``served`` + ``linked``; ``unserved`` profiles exist on
    disk only, and each ``linked`` profile is a ``profiles/<name>`` link to a home elsewhere."""
    from gateway.config import GatewayConfig
    from gateway.control_socket import GatewayControlServer
    from gateway.run import GatewayRunner
    from gateway.run_runtime import initialize_gateway_runtime, recover_gateway_native_sessions
    from gateway.runtime_ownership import process_ownership

    root, homes = _reserve_homes(tmp_path, monkeypatch, names=(*served, *unserved))
    homes = dict(homes)
    for name in linked:
        target = tmp_path / 'elsewhere' / name
        target.mkdir(parents=True)
        homes[name] = _link(target, root / 'profiles' / name)
    for home in homes.values():
        # A default install's profile config: no hosted_rooms key anywhere.
        (home / 'config.yaml').write_text(json.dumps({'model': {'default': 'fixture'}, 'platform_toolsets': {'cli': []}}))
    reserved = [(name, homes[name]) for name in ('default', *served, *linked)]
    config = GatewayConfig()
    config.multiplex_profiles = multiplex
    config._runtime_profile_homes = tuple(reserved)
    process_ownership.reserve([home for _, home in reserved])
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    runner = GatewayRunner(config)

    def call(coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, loop).result(15)
    try:
        call(initialize_gateway_runtime(runner))
        descriptor = runner.session_runtime_descriptor
        descriptor.update(state='ready', capabilities=['session-authority-v1'],
                          api_origin='http://127.0.0.1:1', supervisor='none')
        runner.session_control_server = GatewayControlServer(
            root, verb_handlers={'identify': lambda: descriptor})
        assert call(runner.session_control_server.start())
        call(recover_gateway_native_sessions(runner))
        yield runner, homes, call
    finally:
        for authority in getattr(runner, 'session_authorities', []):
            service = getattr(authority, 'hosted_room_service', None)
            if service is not None:
                assert service.stop(timeout=5)
            authority.db.close()
        if getattr(runner, 'session_control_server', None) is not None:
            call(runner.session_control_server.stop())
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        assert not thread.is_alive()
        loop.close()
        for _, home in reserved:
            process_ownership.release(home)


def _link(target, link):
    """A directory link this host makes without privileges: a symlink, else a Windows junction."""
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        try:
            import _winapi
            _winapi.CreateJunction(str(target), str(link))
        except (ImportError, OSError) as exc:
            pytest.skip(f'this host cannot link a profile directory: {exc}')
    return link


def _desktop(runner, call, scope='host'):
    """The Desktop's host-scoped socket; ``profile`` names the active bot, as the Bots plugin sends it."""
    from gateway.session_controls import AuthorityConnection
    connection = AuthorityConnection(runner.session_authority, object(),
                                     {'user_id': 'desktop', 'profile_scope': scope})

    def request(active, method, **params):
        return call(connection.dispatch({'id': 1, 'method': method, 'params': {'profile': active, **params}}))
    return request


def _members(*profiles):
    """The roster ``createCanonicalGroup`` sends: one local target per bot."""
    return [{'member_id': p, 'profile': p, 'handle': p, 'target': {'kind': 'local', 'profile': p}}
            for p in profiles]


def test_desktop_group_of_served_bots_is_created_and_runs(tmp_path, monkeypatch):
    from gateway import hosted_rooms, session_finite
    from hermes_constants import get_hermes_home

    ran = []

    async def execute(authority, ref, row):
        ran.append((authority.profile_id, str(get_hermes_home())))
        return 'member reply'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    with _gateway(tmp_path, monkeypatch, served=('alpha', 'beta')) as (runner, homes, call):
        desktop = _desktop(runner, call)
        created = desktop('default', 'groups.create', room_id='crew', name='Crew',
                          members=_members('default', 'alpha', 'beta'))
        assert 'result' in created, created
        assert [m['profile'] for m in created['result']['room']['members']] == ['default', 'alpha', 'beta']
        # The active bot may be any served sibling, seating the launch profile in turn.
        pair = desktop('beta', 'groups.create', room_id='pair', name='Pair', members=_members('beta', 'default'))
        assert 'result' in pair, pair

        # A seated sibling's turn runs in its own home, through the same membership the create used.
        sent = desktop('default', 'groups.send', room_id='crew', event_id='hello',
                       payload={'text': '@alpha hello', 'thread_id': 'thread'})
        assert 'result' in sent, sent
        db_path = runner.session_authority.db.db_path
        wait_until(lambda: any(e['kind'] == 'room.activity' and e['payload'].get('status') == 'settled'
                               for e in hosted_rooms.read_events(db_path, room_id='crew')['events']),
                   what='the room settled after alpha answered')
        assert ran == [(str(homes['alpha']), str(homes['alpha']))], [
            e['payload'] for e in hosted_rooms.read_events(db_path, room_id='crew')['events']]


def test_unserving_one_bot_keeps_the_rest_of_its_group_talking(tmp_path, monkeypatch):
    """Greptile on #140187: unserving alpha must not stop a saved default/alpha/beta group; a turn
    for beta still runs, a turn for alpha fails as that member's own turn, and no new group seats it."""
    from gateway import hosted_rooms, session_finite
    from gateway.run_runtime import unserve_profile_runtime

    ran = []

    async def execute(authority, ref, row):
        ran.append(authority.profile_id)
        return 'member reply'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    with _gateway(tmp_path, monkeypatch, served=('alpha', 'beta')) as (runner, homes, call):
        desktop = _desktop(runner, call)
        assert 'result' in desktop('default', 'groups.create', room_id='crew', name='Crew',
                                   members=_members('default', 'alpha', 'beta'))
        assert call(unserve_profile_runtime(runner, homes['alpha']))
        assert runner.session_authorities.for_home(homes['alpha']) is None
        db_path = runner.session_authority.db.db_path

        def settled(count):
            return sum(e['kind'] == 'room.activity' and e['payload'].get('status') == 'settled'
                       for e in hosted_rooms.read_events(db_path, room_id='crew')['events']) >= count
        sent = desktop('default', 'groups.send', room_id='crew', event_id='hello',
                       payload={'text': '@beta hello', 'thread_id': 'thread'})
        assert 'result' in sent, sent
        wait_until(lambda: settled(1), what='the room settled after beta answered')
        assert ran == [str(homes['beta'])], [
            e['payload'] for e in hosted_rooms.read_events(db_path, room_id='crew')['events']]
        # A turn for alpha fails as alpha's own turn, without wedging the room for beta.
        assert 'result' in desktop('default', 'groups.send', room_id='crew', event_id='to-alpha',
                                   payload={'text': '@alpha hello', 'thread_id': 'thread'})
        wait_until(lambda: settled(2), what="the room settled after alpha's turn failed")
        assert 'result' in desktop('default', 'groups.send', room_id='crew', event_id='again',
                                   payload={'text': '@beta again', 'thread_id': 'thread'})
        wait_until(lambda: settled(3), what='the room settled after beta answered again')
        assert ran == [str(homes['beta'])] * 2, ran
        # A new group cannot seat the unserved bot.
        refused = desktop('default', 'groups.create', room_id='again', name='Crew',
                          members=_members('default', 'alpha'))
        assert "profile 'alpha' is not local to this gateway" in refused['error']['data']['detail'], refused


def test_a_bot_this_gateway_does_not_serve_is_refused_by_name(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger='gateway.session_controls')
    # gamma is a real profile on disk, but this gateway does not serve it; ghost does not exist.
    with _gateway(tmp_path, monkeypatch, served=('alpha',), unserved=('gamma',)) as (runner, homes, call):
        desktop = _desktop(runner, call)
        # Both create paths apply the same rule: the room driver's, and the one used while it is stopped.
        for driver in ('running', 'stopped'):
            for refused in ('gamma', 'ghost'):
                reply = desktop('default', 'groups.create', room_id=f'{refused}-{driver}', name='Crew',
                                members=_members('default', refused))
                assert reply['error']['message'] == 'invalid_params', reply
                assert reply['error']['data']['field'] == 'members', reply
                assert f"profile '{refused}' is not local to this gateway" in reply['error']['data']['detail'], reply
            seated = desktop('default', 'groups.create', room_id=f'seated-{driver}', name='Crew',
                             members=_members('default', 'alpha'))
            assert 'result' in seated, seated
            assert runner.session_authority.hosted_room_service.stop(timeout=5)
        assert ("groups.create refused for profile=default: invalid_params field=members "
                "detail=member 1 profile 'gamma' is not local to this gateway") in caplog.text

        # A connection bound to one profile never reaches a sibling, so it cannot seat one either.
        bound = _desktop(runner, call, scope='profile')('default', 'groups.create', room_id='bound', name='Crew',
                                                       members=_members('default', 'alpha'))
        assert bound['error']['data'] == {
            'reason': 'permission_denied', 'field': 'members',
            'detail': "member 1 profile 'alpha': only a host-scoped connection seats a served profile "
                      "without an entry"}, bound
        # Its own hosted_rooms.profiles entry still seats a served profile at its served home, as on main.
        config = homes['default'] / 'config.yaml'
        alpha = runner.session_authorities.require(homes['alpha']).profile_id
        config.write_text(json.dumps({'hosted_rooms': {'profiles': {'alpha': alpha}}}))
        entry = _desktop(runner, call, scope='profile')('default', 'groups.create', room_id='entry', name='Crew',
                                                        members=_members('default', 'alpha'))
        assert 'result' in entry, entry
        config.write_text('{}')
        # A sibling whose serve is being torn down runs nothing, so it is not seated.
        runner.session_authorities.require(homes['alpha']).retiring = True
        retiring = desktop('default', 'groups.create', room_id='retiring', name='Crew',
                           members=_members('default', 'alpha'))
        assert "profile 'alpha' is not local to this gateway" in retiring['error']['data']['detail'], retiring


def test_single_profile_gateway_seats_only_itself_and_configured_profiles(tmp_path, monkeypatch):
    with _gateway(tmp_path, monkeypatch, served=(), unserved=('beta',), multiplex=False) as (runner, homes, call):
        desktop = _desktop(runner, call)
        refused = desktop('default', 'groups.create', room_id='crew', name='Crew', members=_members('default', 'beta'))
        assert refused['error']['data'] == {
            'reason': 'invalid_params', 'field': 'members',
            'detail': "member 1 profile 'beta' is not local to this gateway"}, refused
        # A profile another local gateway owns still joins through its explicit destination entry,
        # from a connection bound to this profile too.
        config = homes['default'] / 'config.yaml'
        config.write_text(json.dumps({'hosted_rooms': {'profiles': {'beta': str(homes['beta'])}}}))
        bound = _desktop(runner, call, scope='profile')
        created = bound('default', 'groups.create', room_id='crew', name='Crew', members=_members('default', 'beta'))
        assert 'result' in created, created


def test_a_bad_hosted_rooms_entry_refuses_only_the_member_naming_it(tmp_path, monkeypatch, caplog):
    from gateway import hosted_rooms, session_finite

    caplog.set_level(logging.WARNING, logger='gateway.session_hosted_members')
    ran = []

    async def execute(authority, ref, row):
        ran.append(authority.profile_id)
        return 'member reply'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    with _gateway(tmp_path, monkeypatch, served=('alpha', 'beta')) as (runner, homes, call):
        desktop, bound = _desktop(runner, call), _desktop(runner, call, scope='profile')
        config = homes['default'] / 'config.yaml'

        def create(request, room_id, *profiles):
            return request('default', 'groups.create', room_id=room_id, name='Crew', members=_members(*profiles))

        def ignored():
            return [r.getMessage() for r in caplog.records if r.name == 'gateway.session_hosted_members']
        # A room that seats helper while its entry names a home of its own.
        config.write_text(json.dumps({'hosted_rooms': {'profiles': {'helper': str(tmp_path.resolve())}}}))
        assert 'result' in create(desktop, 'kept', 'default', 'beta', 'helper')

        # Now alpha's entry moves a served bot, and helper's names beta's home a second time.
        config.write_text(json.dumps({'hosted_rooms': {'profiles': {
            'alpha': str(homes['beta']), 'helper': str(homes['beta'])}}}))
        for attempt in range(2):
            assert 'result' in desktop('default', 'profiles.list', include_sessions=False)
            assert 'result' in create(desktop, f'rest-{attempt}', 'default', 'beta')
            # Desktop seats a served bot at its served home, whatever its entry says.
            assert 'result' in create(desktop, f'served-{attempt}', 'default', 'alpha')
            # A profile-bound connection seats it only through its entry, which must agree.
            assert create(bound, f'moved-{attempt}', 'default', 'alpha')['error']['data'] == {
                'reason': 'permission_denied', 'field': 'hosted_rooms.profiles.alpha',
                'detail': "member 1 profile 'alpha': this gateway serves that profile from another home"}
            assert create(desktop, f'alias-{attempt}', 'default', 'helper')['error']['data'] == {
                'reason': 'invalid_params', 'field': 'hosted_rooms.profiles.helper',
                'detail': "member 1 profile 'helper': names the home already seated as 'beta'"}
        # Each ignored entry is logged once, however often the roster is read.
        assert ignored() == [
            'Ignoring hosted_rooms.profiles.alpha for profile=default: this gateway serves that profile from another home',
            "Ignoring hosted_rooms.profiles.helper for profile=default: names the home already seated as 'beta'"]
        # The room that already seats helper keeps running its other members.
        sent = desktop('default', 'groups.send', room_id='kept', event_id='hello',
                       payload={'text': '@beta hello', 'thread_id': 'thread'})
        assert 'result' in sent, sent
        db_path = runner.session_authority.db.db_path
        wait_until(lambda: any(e['kind'] == 'room.activity' and e['payload'].get('status') == 'settled'
                               for e in hosted_rooms.read_events(db_path, room_id='kept')['events']),
                   what='the room settled after beta answered')
        assert ran == [str(homes['beta'])]

        # An entry fixed and then broken another way is logged again; a bare one does not drop alpha.
        config.write_text(json.dumps({'hosted_rooms': {'profiles': {'alpha': None}}}))
        assert 'result' in create(desktop, 'bare', 'default', 'alpha')
        assert ignored()[-1] == 'Ignoring hosted_rooms.profiles.alpha for profile=default: must map a profile name to a home'
        # A section whose entries are all commented out is empty; another shape is named in the log
        # and leaves every other refusal its own reason.
        config.write_text(json.dumps({'hosted_rooms': None}))
        assert 'result' in create(desktop, 'emptied', 'default', 'alpha')
        config.write_text(json.dumps({'hosted_rooms': {'profiles': ['helper']}}))
        assert 'result' in desktop('default', 'profiles.list', include_sessions=False)
        assert 'result' in create(desktop, 'listed', 'default', 'beta')
        assert create(desktop, 'ghost', 'default', 'ghost')['error']['data'] == {
            'reason': 'invalid_params', 'field': 'members',
            'detail': "member 1 profile 'ghost' is not local to this gateway"}
        assert ignored()[-1] == 'Ignoring hosted_rooms.profiles for profile=default: must map profile names to profile homes'


def test_a_linked_profile_is_refused_by_name_until_groups_support_it(tmp_path, monkeypatch):
    # The member transport names a destination by its resolved home, which for a linked
    # profiles/<name> is outside profiles/: seated, every one of its turns would fail.
    with _gateway(tmp_path, monkeypatch, served=('alpha',), linked=('linked',)) as (runner, _homes, call):
        desktop = _desktop(runner, call)
        refused = desktop('default', 'groups.create', room_id='linked', name='Crew',
                          members=_members('default', 'linked'))
        assert refused['error']['data'] == {
            'reason': 'invalid_params', 'field': 'members',
            'detail': "member 1 profile 'linked': a linked profile home is not supported in groups yet"}, refused
        # Its home never answers to another name: default still seats the rest.
        seated = desktop('default', 'groups.create', room_id='crew', name='Crew', members=_members('default', 'alpha'))
        assert 'result' in seated, seated
