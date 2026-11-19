"""Gateway runtime recovery must not lose routes before platform discovery."""
from contextlib import nullcontext
from pathlib import Path

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize('multiplex', [False, True], ids=['launch-profile', 'secondary-profile'])
async def test_plugin_route_survives_early_restart_scan(tmp_path, monkeypatch, multiplex):
    from gateway.config import GatewayConfig, Platform
    from gateway.platform_registry import PlatformEntry, platform_registry
    from gateway.run import GatewayRunner, _profile_runtime_scope
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session import SessionSource, SessionStore

    home = tmp_path / 'lab'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    name = 'session_restart_secondary' if multiplex else 'session_restart_probe'
    plugin_home = home / 'profiles' / 'secondary' if multiplex else home
    plugin = plugin_home / 'plugins' / name
    plugin.mkdir(parents=True)
    (plugin_home / 'config.yaml').write_text(f'plugins:\n  enabled: [{name}]\n', encoding='utf-8')
    (plugin / 'plugin.yaml').write_text(f'name: {name}\nkind: platform\n', encoding='utf-8')
    (plugin / '__init__.py').write_text(
        'def register(ctx):\n'
        f'    ctx.register_platform({name!r}, "Restart probe", lambda cfg: None, lambda: True)\n',
        encoding='utf-8',
    )
    config = GatewayConfig(multiplex_profiles=multiplex)
    monkeypatch.setattr(config, '_runtime_profile_homes', (('secondary', plugin_home),) if multiplex else (), raising=False)
    profile = 'secondary' if multiplex else None

    def plugin_scope():
        return _profile_runtime_scope(plugin_home, hydrate_secrets=False) if multiplex else nullcontext()

    first = SessionStore(config.sessions_dir, config)
    try:
        # Discard the first process's enum cache and registry entry before the next boot.
        with monkeypatch.context() as seed:
            seed.setattr(Platform, '_value2member_map_', dict(Platform._value2member_map_))
            seed.setattr(Platform, '_member_map_', dict(Platform._member_map_))
            platform_registry.register(PlatformEntry(name, 'Restart probe', lambda cfg: None, lambda: True))
            try:
                with plugin_scope():
                    entry = first.get_or_create_session(SessionSource(Platform(name), 'fixture-chat', profile=profile))
            finally:
                platform_registry.unregister(name)
        key, session_id = entry.session_key, entry.session_id
        first.append_to_transcript(session_id, {'role': 'user', 'content': 'before restart'})
        history = first.load_transcript(session_id)
        assert history[0]['content'] == 'before restart'
        persistence = first.persistence
        assert persistence is not None
        rows = persistence.routing.load_routing_entries(routing_scope=first._routing_scope())
        assert key in rows
        mirror = config.sessions_dir / 'sessions.json'
        mirror_before = mirror.read_bytes()
        # A crash-left FIFO admission makes runtime initialization read the routing index.
        owner = first._persistence_for_key(key)
        assert owner is not None
        admissions = owner.admissions
        epoch = owner.runtime.begin_epoch(instance_id='before-restart')
        admissions.admit(epoch=epoch, principal_id='fixture-user', session_id=session_id,
                         request_id='crash-left', payload={'text': 'in flight'})
        assert admissions.claim(epoch=epoch, session_id=session_id)
    finally:
        first.close_all_db_handles()

    with pytest.raises(ValueError):
        Platform(name)
    process_ownership.reserve([home, plugin_home] if multiplex else [home])
    runner = None
    try:
        runner = GatewayRunner(config)
        await initialize_gateway_runtime(runner)
        # Run the normal registration phase even if boot already registered this platform.
        from hermes_cli.plugins import discover_plugins

        with plugin_scope():
            discover_plugins()
            source = SessionSource(Platform(name), 'fixture-chat', profile=profile)
        store = runner.session_store
        restored = store.lookup_by_session_key(key)
        assert restored is not None, 'early restart scan lost the plugin route'
        assert restored.origin == source
        with plugin_scope():
            assert store.get_or_create_session(source, touch_activity=False).session_id == session_id
        assert store.load_transcript(session_id) == history
        persistence = store.persistence
        assert persistence is not None
        assert persistence.routing.load_routing_entries(routing_scope=store._routing_scope()) == rows
        assert mirror.read_bytes() == mirror_before
    finally:
        if runner is not None:
            for authority in getattr(runner, 'session_authorities', ()):
                authority.store_reference.close()
            runner.session_store.close_all_db_handles()
        process_ownership.release(home)
        if multiplex:
            process_ownership.release(plugin_home)
