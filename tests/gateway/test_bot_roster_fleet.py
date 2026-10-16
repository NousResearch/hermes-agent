"""Regression for #139227: on a multiplexed gateway, every Bots roster row resolves its own profile's
Bot Chat (the status dot's session id and the row's age), however that chat was created.

Each profile here holds its Bot Chat from a different creation path: a pre-upgrade CLI chat adopted
on its first open (the reporter's fleet), a first ``bot_relay.deliver`` (``gateway/session_bot.py``),
``hermes -p <bot> chat -c "Bot Chat" --create-if-missing``, and the Desktop's own mint. The Desktop
asks ``profiles.list`` of its active bot over its one host-scoped socket, so the other bots' rows
come from that same answer.
"""
from pathlib import Path

import pytest
import pytest_asyncio

from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner
from tests.gateway.test_ticket_profile_scope import _Transport

AUTHOR = {"id": "bot:cto", "name": "cto", "is_bot": True}
LEGACY_ID = '20261006_105832_fa0db9'  # a pre-upgrade CLI session id, kept as chat_id on adoption


@pytest_asyncio.fixture
async def fleet(tmp_path, monkeypatch, request):
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_authorities import authority_for_profile_id, owner_scope
    from gateway.session_controls import AuthorityConnection

    executed = []

    async def execute(authority, ref, row):
        executed.append(ref.session_id)
        return 'pong'

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    root, homes = _reserve_homes(tmp_path, monkeypatch, names=('alpha', 'beta', 'gamma'))
    # An extra served home outside ``profiles/`` (what a linked ``profiles/<name>`` resolves to).
    for name in getattr(request, 'param', ()):
        (tmp_path / 'linked' / name).mkdir(parents=True)
        homes.append((name, tmp_path / 'linked' / name))
    for _, home in homes:
        (home / 'config.yaml').write_text('model:\n  default: fixture\nplatform_toolsets:\n  cli: []\n')
    process_ownership.reserve([home for _, home in homes])
    connections, runner = [], None

    def connect(name, subject='uid:desk', *, scope='profile', operator=True):
        # gateway/run_api.py's redeem → identity → connection binding for a native ticket.
        store = runner.session_ticket_store
        ticket = store.mint(profile_id=str(dict(homes)[name].resolve()), subject=subject,
                            purpose='interactive', scope=scope)
        grant = store.redeem(ticket, profile_id=None, purpose='interactive')
        authority = authority_for_profile_id(runner, grant['profile_id'])
        identity = {'user_id': grant['subject'], 'provider': 'local', 'profile_id': grant['profile_id'],
                    'instance_id': grant['instance_id'], 'capabilities': grant['capabilities'],
                    'native_bootstrap': True, 'profile_scope': grant['scope']}
        with owner_scope(authority):
            connection = AuthorityConnection(authority, _Transport(), identity, operator=operator)
        connections.append(connection)
        return connection

    async def call(connection, method, **params):
        with owner_scope(connection.authority):
            reply = await connection.dispatch({'id': 1, 'method': method, 'params': params})
        assert 'result' in reply, (method, reply)
        return reply['result']

    try:
        runner = _runner(root, homes)
        await initialize_gateway_runtime(runner)
        yield _Fleet(runner, dict(homes), connect, call, executed)
    finally:
        for connection in connections:
            with owner_scope(connection.authority):
                await connection.close()
        for authority in list(getattr(runner, 'session_authorities', []) or []):
            authority.db.close()
        for _, home in homes:
            process_ownership.release(Path(home))


class _Fleet:
    def __init__(self, runner, homes, connect, call, executed):
        self.runner, self.homes, self.connect, self.call, self.executed = runner, homes, connect, call, executed

    def db(self, name):
        from gateway.session_authorities import authority_for_home
        return authority_for_home(self.runner, self.homes[name]).db

    async def bot_chats(self):
        """Mint each bot's Bot Chat the way its creator does; return ``{profile: Bot Chat id}``."""
        from tests.waiting import async_wait_until
        from tools.bot_live_delivery import read_delivery_result
        desktop = self.connect('default', scope='host')
        # default: created by the pre-upgrade CLI, adopted by the Desktop's first open of the bot.
        legacy = self.db('default')
        legacy.create_session(LEGACY_ID, source='cli', model='fixture')
        legacy.set_session_title(LEGACY_ID, 'Bot Chat')
        legacy.append_message(LEGACY_ID, 'assistant', 'kept from before the upgrade')
        found = await self.call(desktop, 'session.list', profile='default', title='Bot Chat',
                                include_hidden=True, limit=50)
        assert [row['session_id'] for row in found['sessions']] == [LEGACY_ID], found
        assert legacy.get_session(LEGACY_ID)['chat_id'] == LEGACY_ID  # the reporter's row shape
        # alpha: no Bot Chat until a teammate's first delivery mints it (gateway/session_bot.py).
        relay = self.connect('alpha')
        delivered = await self.call(relay, 'bot_relay.deliver', id='a' * 32, profile='alpha',
                                    message='status?', author=AUTHOR)
        await async_wait_until(
            lambda: (read_delivery_result(self.homes['alpha'], 'a' * 32) or {}).get('status') == 'settled',
            timeout=30, interval=.05)
        # beta: provisioned by `hermes -p beta chat -c "Bot Chat" --create-if-missing`.
        cli = self.connect('beta')
        provisioned = await self.call(cli, 'session.create', request_id='provision', title='Bot Chat',
                                      source='cli', cwd=str(self.homes['beta']))
        # gamma: an ordinary chat first, then the Desktop's own mint on a bot's first click
        # (canonical-chat.ts), born hidden and newer.
        visible = await self.call(desktop, 'session.create', profile='gamma', request_id='side', source='gui')
        minted = await self.call(desktop, 'session.create', profile='gamma', title='Bot Chat',
                                 hidden=True, follow_profile_config=True)
        self.gamma_chat = visible['stored_session_id']
        return {'default': LEGACY_ID, 'alpha': delivered['session_id'],
                'beta': provisioned['stored_session_id'], 'gamma': minted['stored_session_id']}

    def connections(self):
        # Every AuthorityConnection registers its transport with the authority it binds.
        return sum(len(authority.events) for authority in self.runner.session_authorities.serving())


def _rows(result):
    return {row['name']: row for row in result['profiles']}


@pytest.mark.asyncio
async def test_every_bot_row_resolves_its_own_bot_chat_whichever_bot_is_active(fleet):
    chats = await fleet.bot_chats()
    # "Open recent session" offers the newest chat a picker shows: never a hidden Bot Chat.
    recent = {**chats, 'gamma': fleet.gamma_chat}
    desktop = fleet.connect('default', scope='host')
    for active in chats:  # route the shared socket to every bot once, as a click does
        await fleet.call(desktop, 'session.list', profile=active)
    routed = fleet.connections()
    for active in chats:
        # The Desktop's roster poll: profiles.list of the active bot over the shared socket.
        rows = _rows(await fleet.call(desktop, 'profiles.list', profile=active))
        assert set(rows) == set(chats)
        for name, chat in chats.items():
            canonical = rows[name].get('canonical_session')
            assert canonical is not None, f'{name} row has no Bot Chat while {active} is active'
            assert canonical['resolved_id'] == chat, (active, name, canonical)
            stored = fleet.db(name).get_session(chat)
            assert canonical['last_active'] == (stored['last_activity_at'] or stored['started_at']) > 0
            assert (rows[name].get('last_session') or {}).get('id') == recent[name], (active, name, rows[name])
    assert fleet.connections() == routed, 'a roster poll built connections of its own'
    assert fleet.executed == [chats['alpha']]


@pytest.mark.asyncio
async def test_no_row_shows_a_bot_chat_its_viewer_may_not_read(fleet):
    chats = await fleet.bot_chats()
    # A profile-bound socket (`hermes -p beta chat`) can route nowhere else: its siblings stay
    # discovery metadata, even though each of them holds a Bot Chat.
    rows = _rows(await fleet.call(fleet.connect('beta'), 'profiles.list'))
    assert rows['beta']['canonical_session']['resolved_id'] == chats['beta']
    for name in ('default', 'alpha', 'gamma'):
        assert not {'canonical_session', 'last_session'} & set(rows[name]), rows[name]
    # Another principal without operator access reads no one's Bot Chat, on any row.
    foreign = fleet.connect('default', 'uid:other', scope='host', operator=False)
    for name, row in _rows(await fleet.call(foreign, 'profiles.list')).items():
        assert row.get('canonical_session') is None and row.get('last_session') is None, (name, row)


@pytest.mark.asyncio
async def test_a_sibling_that_cannot_be_read_safely_leaves_only_its_own_row_bare(fleet, monkeypatch, caplog):
    from hermes_state_errors import StateDbReplacedError
    chats = await fleet.bot_chats()

    def replaced(*args, **kwargs):
        raise StateDbReplacedError('state.db was replaced')

    monkeypatch.setattr(fleet.db('alpha'), '_read_ctx', replaced)
    # beta's store closes as an unserve would, between the retiring check and the read.
    fleet.db('beta').close()
    # gamma is being unserved: retirement marks it before its store closes.
    fleet.runner.session_authorities.for_home(fleet.homes['gamma']).retiring = True
    desktop = fleet.connect('default', scope='host')
    for _poll in range(2):
        rows = _rows(await fleet.call(desktop, 'profiles.list'))
        assert rows['default']['canonical_session']['resolved_id'] == chats['default']
        for name in ('alpha', 'beta', 'gamma'):
            assert not {'canonical_session', 'last_session'} & set(rows[name]), rows[name]
    assert fleet.db('beta')._conn is None, 'the roster reopened a store an unserve had closed'
    warned = [r.getMessage() for r in caplog.records if r.levelname == 'WARNING' and 'metadata-only' in r.getMessage()]
    for name in ('alpha', 'beta'):  # once each across both polls; a retiring sibling is no failure
        profile_id = fleet.runner.session_authorities.for_home(fleet.homes[name]).profile_id
        assert sum(profile_id in message for message in warned) == 1, warned
    assert len(warned) == 2, warned


@pytest.mark.asyncio
@pytest.mark.parametrize('fleet', [('delta',)], indirect=True)
async def test_a_row_whose_name_routes_elsewhere_carries_no_bot_chat(fleet):
    # Served as ``delta`` from outside ``profiles/``, it is listed as ``default``: a click on that
    # row routes to the launch profile, so the row must not paint delta's Bot Chat.
    linked = fleet.db('delta')
    linked.create_session('local-delta', source='gui', chat_id='local-delta', user_id='uid:desk')
    linked.set_session_title('local-delta', 'Bot Chat')
    result = await fleet.call(fleet.connect('default', scope='host'), 'profiles.list')
    row, = [row for row in result['profiles'] if row['path'] == str(fleet.homes['delta'])]
    assert row['name'] == 'default'
    assert not {'canonical_session', 'last_session'} & set(row), row
