"""A cold messaging-platform session mounts for a local viewer (TUI-8, NousResearch/hermes-agent#134889).

Before its first post-restart inbound nothing publishes a messaging-platform lane (Telegram, Weixin,
Discord, ...) into the authority, so ``session.resume`` of its row answered ``not_found`` until
someone wrote to the lane again. Before the canonical authority (#106742) a viewer opened any such
row straight from ``state.db``; this restores that for every cold row of the profile that carries a
``session_key`` and a messaging-platform source. The routing entry is the preferred, highest-fidelity
source; a row the route no longer names (``/new``, a pruned entry) is rebuilt from its own
``origin_json``, else its columns. Each fence has its own test: no key, a local/API/unknown source,
a foreign profile, an origin that contradicts the row, a draining gateway.
"""
import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from gateway.authz_mixin import GatewayAuthorizationMixin


class _Runner(GatewayAuthorizationMixin):
    def __init__(self, tmp_path, multiplex=None):
        from gateway.config import GatewayConfig
        from gateway.session import SessionStore
        self.config = GatewayConfig(multiplex_profiles=multiplex)
        self.session_store = SessionStore(tmp_path / 'sessions', self.config)
        self._session_db = self.session_store._db
        self.adapters = {}
        self._draining = False


def _source(platform='telegram', chat_id='4242', profile=None, chat_name=None):
    from gateway.config import Platform
    from gateway.session import SessionSource
    source = SessionSource(platform=Platform(platform), chat_id=chat_id, user_id='u-' + chat_id, chat_type='dm',
                           chat_name=chat_name)
    if profile is not None:
        source.profile = profile
    return source


_ROUTE = object()  # the key the store derives for the source


def _seed(tmp_path, sid, source, *, routed=True, row_origin=_ROUTE, row_source=None, row_key=_ROUTE,
          entry_key=None, entry_source=None, transport_profile=None, multiplex=None):
    """The durable state a previous process left: a session row, and (``routed``) the routing entry
    naming it. ``row_origin`` is the row's ``origin_json`` (default: *source*; None: absent)."""
    from gateway.session import SessionEntry
    runner = _Runner(tmp_path, multiplex)
    store = runner.session_store
    route = store._generate_session_key(source)
    if row_origin is _ROUTE:
        row_origin = json.dumps(source.to_dict())
    runner._session_db.create_session(
        sid, source=row_source or source.platform.value, session_key=route if row_key is _ROUTE else row_key,
        chat_id=source.chat_id, user_id=source.user_id, chat_type=source.chat_type,
        display_name=source.chat_name, origin_json=row_origin, transport_profile=transport_profile)
    if routed:
        now = datetime.now()
        entry_source = source if entry_source is None else entry_source
        entry = SessionEntry(entry_key or route, sid, now, now, origin=entry_source or None,
                             platform=source.platform, transport_profile=transport_profile)
        # The routing index lives in the store's routing home, which is cached per process: write where
        # the cold store will read, not where this test's HERMES_HOME happens to point.
        store._routing_db.save_gateway_routing_entry(
            route, json.dumps(entry.to_dict()), scope=store._routing_scope())
    return route


async def _cold(tmp_path, multiplex=None, profile_name=None):
    """A fresh process: new store, new authority, nothing published."""
    from gateway.session_authorities import SessionAuthorities
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    runner = _Runner(tmp_path, multiplex)
    authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='cold')
    if profile_name is not None:
        runner.session_authorities = SessionAuthorities(tmp_path / 'launch', multiplexed=True)
        runner.session_authorities.add('fixture', authority, name=profile_name)
    actor = Principal('viewer', 'fixture', frozenset({'session:read'}), 'socket')
    return SimpleNamespace(authority=authority, actor=actor, store=runner.session_store)


def _ref(sid, profile_id='fixture'):
    from gateway.session_contract import SessionRef
    return SessionRef(profile_id, sid)


async def _refused(tmp_path, sid, reason='not_found', **cold_args):
    from hermes_state_runtime import RuntimeStoreError
    cold = await _cold(tmp_path, **cold_args)
    with pytest.raises(RuntimeStoreError, match=reason):
        cold.authority.authorize(cold.actor, _ref(sid), 'session:read')
    assert sid not in cold.authority.sessions


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))


@pytest.mark.asyncio
async def test_a_routed_messaging_session_mounts_on_a_cold_authority(tmp_path):
    """The lane is registered by the pre-restart process through the real write path."""
    from gateway.config import Platform
    from gateway.session_authority import initialize_session_authority
    warm = await initialize_session_authority(_Runner(tmp_path), profile_id='fixture', instance_id='warm')
    ref = warm.register(_source())
    cold = await _cold(tmp_path)
    assert ref.session_id not in cold.authority.sessions
    assert cold.authority.db.get_session(ref.session_id)['source'] == 'telegram'

    cold.authority.authorize(cold.actor, ref, 'session:read')

    live = cold.authority.sessions[ref.session_id]
    assert (live.route, live.source.platform, live.source.chat_id) == (
        cold.store._generate_session_key(_source()), Platform.TELEGRAM, '4242')


@pytest.mark.asyncio
async def test_the_routing_entry_is_preferred_over_the_row(tmp_path):
    _seed(tmp_path, 'lane-a', _source(chat_name='row'), entry_source=_source(chat_name='entry'))
    cold = await _cold(tmp_path)
    cold.authority.authorize(cold.actor, _ref('lane-a'), 'session:read')
    assert cold.authority.sessions['lane-a'].source.chat_name == 'entry'


@pytest.mark.asyncio
async def test_a_row_its_route_moved_past_mounts_from_its_own_origin(tmp_path):
    """Design reversal, not a new finding: before #106742 a viewer opened any messaging row, however
    old. The lane's route names a successor after ``/new``; the historic row still mounts, rebuilt
    from its own ``origin_json`` (never the successor's entry)."""
    from gateway.config import Platform
    _seed(tmp_path, 'old', _source(chat_name='from-row'))
    cold = await _cold(tmp_path)
    successor = cold.store.get_or_create_session(_source(), force_new=True)
    assert successor.session_id != 'old'

    cold.authority.authorize(cold.actor, _ref('old'), 'session:read')

    live = cold.authority.sessions['old']
    assert (live.route, live.source.platform, live.source.chat_id, live.source.chat_name) == (
        cold.store._generate_session_key(_source()), Platform.TELEGRAM, '4242', 'from-row')


@pytest.mark.asyncio
async def test_an_entry_that_names_another_key_does_not_stand_for_the_row(tmp_path):
    _seed(tmp_path, 'lane-key', _source(chat_name='row'), entry_source=_source(chat_name='entry'),
          entry_key='agent:main:telegram:dm:elsewhere')
    cold = await _cold(tmp_path)
    cold.authority.authorize(cold.actor, _ref('lane-key'), 'session:read')
    assert cold.authority.sessions['lane-key'].source.chat_name == 'row'


@pytest.mark.asyncio
@pytest.mark.parametrize('row_origin', [None, 'not json', '{"chat_type": "dm"}'])
async def test_a_row_without_a_usable_origin_is_rebuilt_from_its_columns(tmp_path, row_origin):
    from gateway.config import Platform
    _seed(tmp_path, 'cols', _source(chat_name='Family chat'), routed=False, row_origin=row_origin)
    cold = await _cold(tmp_path)
    cold.authority.authorize(cold.actor, _ref('cols'), 'session:read')
    source = cold.authority.sessions['cols'].source
    assert (source.platform, source.chat_id, source.user_id, source.chat_type, source.chat_name) == (
        Platform.TELEGRAM, '4242', 'u-4242', 'dm', 'Family chat')


@pytest.mark.asyncio
async def test_an_entry_without_an_origin_falls_back_to_the_row(tmp_path):
    _seed(tmp_path, 'lane-bare', _source(chat_name='row'), entry_source=False)
    cold = await _cold(tmp_path)
    cold.authority.authorize(cold.actor, _ref('lane-bare'), 'session:read')
    assert cold.authority.sessions['lane-bare'].source.chat_name == 'row'


@pytest.mark.asyncio
async def test_a_row_with_neither_origin_nor_chat_id_cannot_be_rebuilt(tmp_path):
    cold = await _cold(tmp_path)
    cold.authority.db.create_session('hollow', source='telegram', session_key='agent:main:telegram:dm:x')
    from hermes_state_runtime import RuntimeStoreError
    with pytest.raises(RuntimeStoreError, match='not_found'):
        cold.authority.authorize(cold.actor, _ref('hollow'), 'session:read')
    assert 'hollow' not in cold.authority.sessions


@pytest.mark.asyncio
async def test_a_row_without_a_session_key_stays_not_found(tmp_path):
    """A subagent row has no lane to rebuild."""
    _seed(tmp_path, 'child', _source(), routed=False, row_key=None)
    await _refused(tmp_path, 'child')


@pytest.mark.asyncio
@pytest.mark.parametrize('label', ['tui', 'cli', 'api_server', 'cron', 'unregistered_platform'])
async def test_local_api_and_unknown_sources_are_never_mounted_by_route(tmp_path, label):
    """Local rows restore from their private receipt, API rows from their binding; a route or a row
    is not either."""
    _seed(tmp_path, 'other-' + label, _source(), row_source=label)
    await _refused(tmp_path, 'other-' + label)


@pytest.mark.asyncio
async def test_an_origin_that_contradicts_the_row_is_not_trusted(tmp_path):
    _seed(tmp_path, 'mixed', _source(), routed=False, row_origin=json.dumps(_source('discord').to_dict()))
    await _refused(tmp_path, 'mixed')


@pytest.mark.asyncio
async def test_a_foreign_profile_ref_is_still_refused(tmp_path):
    from hermes_state_runtime import RuntimeStoreError
    _seed(tmp_path, 'lane-foreign', _source())
    cold = await _cold(tmp_path)
    with pytest.raises(RuntimeStoreError, match='profile_mismatch'):
        cold.authority.authorize(cold.actor, _ref('lane-foreign', 'other'), 'session:read')
    assert 'lane-foreign' not in cold.authority.sessions


@pytest.mark.asyncio
async def test_a_served_profile_mounts_its_own_lane_with_the_persisted_bot(tmp_path):
    """Under multiplex the receiving bot persisted on the entry is re-pinned, not guessed."""
    from gateway.session_identity import identity_of
    route = _seed(tmp_path, 'lane-alpha', _source(profile='alpha'), transport_profile='alpha', multiplex=True)
    cold = await _cold(tmp_path, multiplex=True, profile_name='alpha')
    assert cold.store._named_profile_for_key(route) == 'alpha'

    cold.authority.authorize(cold.actor, _ref('lane-alpha'), 'session:read')

    live = cold.authority.sessions['lane-alpha']
    assert (live.route, identity_of(live.source).transport_profile) == (route, 'alpha')


@pytest.mark.asyncio
async def test_a_row_rebuilt_from_columns_keeps_its_profile_and_persisted_bot(tmp_path):
    from gateway.session_identity import identity_of
    _seed(tmp_path, 'cols-alpha', _source(profile='alpha'), routed=False, row_origin=None,
          transport_profile='alpha', multiplex=True)
    cold = await _cold(tmp_path, multiplex=True, profile_name='alpha')

    cold.authority.authorize(cold.actor, _ref('cols-alpha'), 'session:read')

    source = cold.authority.sessions['cols-alpha'].source
    assert (source.profile, identity_of(source).transport_profile) == ('alpha', 'alpha')


@pytest.mark.asyncio
async def test_a_lane_of_another_profile_is_not_mounted_by_this_authority(tmp_path):
    _seed(tmp_path, 'lane-beta', _source(profile='alpha'), transport_profile='alpha', multiplex=True)
    await _refused(tmp_path, 'lane-beta', multiplex=True, profile_name='beta')


@pytest.mark.asyncio
async def test_a_draining_gateway_mounts_nothing(tmp_path):
    from hermes_state_runtime import RuntimeStoreError
    _seed(tmp_path, 'lane-drain', _source())
    cold = await _cold(tmp_path)
    cold.authority.runner._draining = True
    with pytest.raises(RuntimeStoreError, match='runtime_draining'):
        cold.authority.authorize(cold.actor, _ref('lane-drain'), 'session:read')
    assert 'lane-drain' not in cold.authority.sessions
