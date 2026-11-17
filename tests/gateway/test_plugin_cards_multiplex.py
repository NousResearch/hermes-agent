"""``ctx.post_card`` on a multiplexed gateway: two served profiles, two live session authorities in
one process. A plugin posts into conversations of its own profile only: by local session key, by
messaging session key and by origin from a sessionless background task. The card's row lands in
the plugin's own authority and never in the other one; the same card id from two plugins is two
cards.
"""
import asyncio
from pathlib import Path

import pytest

import hermes_yaml as yaml
from persistence_assembly import close_stores_under
from tests.gateway.fixtures.storage_handles import storage_handle
from tests.gateway.test_plugin_cards import _CARD, _cards, _no_turns, _post, _serve
from tests.gateway.test_unviewed_completion_notes import _target

pytestmark = pytest.mark.usefixtures('authority_cwd_in_tmp')

PLUGINS = ('cwplug', 'otherplug')


def _plugin(home, name='cwplug'):
    """A plugin of the profile at *home*, with ``conversation.cards`` granted in that home."""
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    config = yaml.safe_load((home / 'config.yaml').read_text()) if (home / 'config.yaml').exists() else {}
    entries = config.setdefault('plugins', {}).setdefault('entries', {})
    entries[name] = {'granted_capabilities': ['conversation.cards']}
    (home / 'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    return PluginContext(PluginManifest(name=name, source='user', key=name), PluginManager(scope_key=str(home)))


def _admitted_cards(authority, session_id):
    """The card admissions in *session_id*'s durable FIFO: ``(plugin_id, card_id)`` per row."""
    rows = authority.persistence.admissions.list(session_id=authority.logical_owner(session_id), pending_only=False)
    descriptors = [row.payload.get('local_automation_v1') or (row.payload.get('native_text_v1') or {}).get('automation')
                   or {} for row in rows]
    return [(d['plugin_card']['plugin_id'], d['plugin_card']['card_id'])
            for d in descriptors if d.get('display_kind') == 'plugin_card']


async def _two_profiles(tmp_path, monkeypatch):
    """A multiplexed runner serving ``default`` (the launch home) and ``beta``, each with its own
    live authority over its own state.db, a live local session, and a Discord bot with one
    committed human DM from user 42 (chat 42 on default's bot, chat 43 on beta's)."""
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner, _profile_runtime_scope
    from gateway.session_authorities import SessionAuthorities
    from gateway.session_authority import SessionAuthority
    from gateway.session_contract import SessionActor
    from gateway.session_ingress_context import native_callback, register_transport_home
    from gateway.session_local import create_local_session
    from plugins.platforms.discord.adapter import DiscordAdapter

    launch = tmp_path / '.hermes'
    beta = launch / 'profiles' / 'beta'
    beta.mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(launch))
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)  # admitted rows stay queued
    monkeypatch.setattr('gateway.session_bot._watch_reply', lambda *args: None)
    runner = GatewayRunner(GatewayConfig(sessions_dir=launch / 'sessions'))
    runner.config.multiplex_profiles = True
    runner.config._runtime_profile_homes = [('default', launch), ('beta', beta)]
    runner.session_authorities = SessionAuthorities(launch, multiplexed=True)
    profiles = {}
    for name, home in (('default', launch), ('beta', beta)):
        handle = storage_handle(home)
        authority = SessionAuthority(runner, profile_id=str(home), instance_id='owner', persistence=handle,
                                     epoch=handle.runtime.begin_epoch(instance_id='owner'))
        runner.session_authorities.add(home, authority, name=name)
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token=f'{name}-token', typing_indicator=False))
        adapter.gateway_runner = runner
        if name == 'default':
            runner.adapters = {Platform.DISCORD: adapter}
            runner._wire_adapter_handlers(adapter)
            register_transport_home(runner, None, launch)
        else:
            runner._profile_adapters = {name: {Platform.DISCORD: adapter}}
            runner._configure_profile_adapter(adapter, name, Platform.DISCORD)
            register_transport_home(runner, name, home)
        profiles[name] = dict(home=home, authority=authority, adapter=adapter)
    runner.session_authority = profiles['default']['authority']
    for name, profile in profiles.items():
        home, authority, adapter = profile['home'], profile['authority'], profile['adapter']
        actor = SessionActor('test-owner', authority.profile_id,
                             frozenset({'session:create', 'session:submit', 'session:read'}), 'fixture')
        with _profile_runtime_scope(home, hydrate_secrets=False):
            ref = create_local_session(authority, actor, dict(request_id='cli', source='cli', model='fixture',
                                                              toolsets=[]))
        profile['ref'], profile['local_key'] = ref, authority.sessions[ref.session_id].route
        chat = '42' if name == 'default' else '43'
        dm = adapter.build_source(chat_id=chat, chat_type='dm', user_id='42')
        human = MessageEvent(text='hello', source=adapter.build_source(chat_id=chat, chat_type='dm', user_id='42',
                                                                        message_id=f'{name}-human'),
                             message_id=f'{name}-human')
        with native_callback(runner, human, home, profile=None if name == 'default' else name):
            receipt = await authority.admit_native(human)
        profile['dm'], profile['dm_key'] = dm, runner.session_store._generate_session_key(dm)
        profile['dm_session'] = receipt.ref.session_id
        profile['plugin'] = _plugin(home)
    assert profiles['default']['dm_key'] != profiles['beta']['dm_key']
    assert profiles['default']['local_key'] != profiles['beta']['local_key']
    _serve(runner, monkeypatch)
    _no_turns(runner)
    return runner, profiles


def _close(profiles):
    for profile in profiles.values():
        close_stores_under(profile['authority'].persistence.scope.profile_dir)


@pytest.mark.asyncio
async def test_two_profiles_local_and_messaging_keys(tmp_path, monkeypatch):
    runner, profiles = await _two_profiles(tmp_path, monkeypatch)
    try:
        for own, other in (('default', 'beta'), ('beta', 'default')):
            plugin = profiles[own]['plugin']
            for kind in ('local_key', 'dm_key'):
                card = {**_CARD, 'id': f'{own}-{kind}'}
                accepted = await _post(plugin, card, session_key=profiles[own][kind])
                assert accepted == {'ok': True, 'card_id': card['id'], 'status': 'queued'}, (own, kind, accepted)
                crossed = await _post(plugin, card, session_key=profiles[other][kind])
                assert crossed['error'] == 'no_conversation', (own, other, kind, crossed)
                assert crossed.get('ok') is False
                # Refused by the ownership check itself, not later by the admission finding no route.
                expected = ("not live in this plugin's profile" if kind == 'local_key'
                            else 'belongs to another profile')
                assert expected in crossed['detail'], (own, other, kind, crossed)
        for name, profile in profiles.items():
            authority = profile['authority']
            # The local card is in the plugin's own profile's live session and drains as a note there.
            await authority._drain(profile['ref'])
            rows = _cards(authority, _target(runner, authority, profile['ref']))
            assert [row['content'] for row in rows] == [f'[cwplug posted a card {name}-local_key]'], (name, rows)
            # The messaging card is one admission in that profile's own DM session, behind the human turn.
            assert _admitted_cards(authority, profile['dm_session']) == [('cwplug', f'{name}-dm_key')], name
        # Nothing of one profile's cards reached the other authority's store: each store holds its own two,
        # and the other profile's sessions hold nothing there.
        for name, profile in profiles.items():
            other = profiles['beta' if name == 'default' else 'default']
            own = (_admitted_cards(profile['authority'], profile['ref'].session_id)
                   + _admitted_cards(profile['authority'], profile['dm_session']))
            assert sorted(own) == [('cwplug', f'{name}-dm_key'), ('cwplug', f'{name}-local_key')], (name, own)
            for session_id in (other['ref'].session_id, other['dm_session']):
                assert _admitted_cards(profile['authority'], session_id) == [], (name, session_id)
    finally:
        _close(profiles)


@pytest.mark.asyncio
async def test_two_profiles_origin_from_a_sessionless_task(tmp_path, monkeypatch):
    """A background task with no session bound names the chat by origin: the card lands in the
    plugin's own profile's DM session. An origin naming the other profile is refused, and the other
    profile's chat is not reachable through the plugin's own bot (no such session there)."""
    _runner, profiles = await _two_profiles(tmp_path, monkeypatch)
    try:
        for name, profile in profiles.items():
            origin = profile['dm'].to_dict()
            if name == 'beta':
                assert origin.get('profile') == 'beta', origin
            result = await _post(profile['plugin'], {**_CARD, 'id': f'{name}-origin'}, origin=origin)
            assert result == {'ok': True, 'card_id': f'{name}-origin', 'status': 'queued'}, (name, result)
            other = 'beta' if name == 'default' else 'default'
            crossed = await _post(profile['plugin'], {**_CARD, 'id': 'cross'}, origin=profiles[other]['dm'].to_dict())
            assert crossed['error'] == 'no_conversation', (name, crossed)
            # Default's origin carries no profile: for beta's plugin it is chat 42 on beta's own bot, a
            # session that does not exist. Beta's origin names its profile, which default's plugin may not target.
            assert ('agent:beta:discord:dm:42' if name == 'beta' else 'not a chat of this plugin') in crossed['detail'], crossed
            renamed = await _post(profile['plugin'], {**_CARD, 'id': 'cross'}, origin={**origin, 'profile': other})
            assert renamed['error'] == 'no_conversation', (name, renamed)
            assert 'not a chat of this plugin' in renamed['detail'], renamed
        for name, profile in profiles.items():
            assert _admitted_cards(profile['authority'], profile['dm_session']) == [('cwplug', f'{name}-origin')], name
    finally:
        _close(profiles)


@pytest.mark.asyncio
async def test_same_card_id_from_two_plugins_is_two_cards(tmp_path, monkeypatch):
    """A card is identified by (plugin, card id): two plugins of one profile posting ``share-1``
    into the same session are two cards, not a duplicate and not a conflict."""
    runner, profiles = await _two_profiles(tmp_path, monkeypatch)
    try:
        profile = profiles['default']
        plugins = [profile['plugin'], _plugin(profile['home'], 'otherplug')]
        results = await asyncio.gather(*(_post(plugin, _CARD, session_key=profile['local_key']) for plugin in plugins))
        assert [result['status'] for result in results] == ['queued', 'queued'], results
        authority = profile['authority']
        await authority._drain(profile['ref'])
        rows = _cards(authority, _target(runner, authority, profile['ref']))
        assert sorted(row['content'] for row in rows) == ['[cwplug posted a card share-1]',
                                                           '[otherplug posted a card share-1]']
        assert sorted(_admitted_cards(authority, profile['ref'].session_id)) == [('cwplug', 'share-1'),
                                                                                  ('otherplug', 'share-1')]
        # Each plugin's replay is still its own duplicate.
        again = await _post(plugins[1], _CARD, session_key=profile['local_key'])
        assert again['status'] == 'duplicate', again
    finally:
        _close(profiles)
