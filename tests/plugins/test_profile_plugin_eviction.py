"""Deleted profiles lose plugin callbacks and may discover a fresh plugin at the same home."""
from hermes_cli.plugins import get_plugin_manager
from hermes_cli.plugins_lifecycle import evict_profile_plugins
from agent import shell_hooks, outbound_webhooks


def test_eviction_clears_plugins_and_config_hooks_before_recreation(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_BUNDLED_PLUGINS', str(tmp_path / 'empty'))
    monkeypatch.chdir(tmp_path)
    plugin = tmp_path / 'plugins/owned'
    plugin.mkdir(parents=True)
    (tmp_path / 'config.yaml').write_text('plugins:\n  enabled: [owned]\n')
    (plugin / 'plugin.yaml').write_text('name: owned\nversion: 1.0.0\n')
    source = plugin / '__init__.py'
    source.write_text('def register(ctx):\n    ctx.register_hook("pre_llm_call", lambda **kw: {"context": "old"})\n')
    manager = get_plugin_manager()
    manager.discover_and_load()
    assert manager.invoke_hook('pre_llm_call') == [{'context': 'old'}]
    cfg = {'hooks': {'pre_llm_call': [{'command': 'true'}]}}
    assert shell_hooks.register_from_config(cfg, accept_hooks=True)
    outbound = {'hooks': {'outbound': [{'url': 'https://example.invalid/hook', 'events': ['pre_llm_call']}]}}
    assert outbound_webhooks.register_from_config(outbound)
    evict_profile_plugins(tmp_path)
    assert not manager.invoke_hook('pre_llm_call')
    source.write_text('def register(ctx):\n    ctx.register_hook("pre_llm_call", lambda **kw: {"context": "fresh-profile"})\n')
    fresh = get_plugin_manager()
    assert fresh is not manager
    fresh.discover_and_load()
    assert fresh.invoke_hook('pre_llm_call') == [{'context': 'fresh-profile'}]
    assert shell_hooks.register_from_config(cfg, accept_hooks=True)
    assert outbound_webhooks.register_from_config(outbound)
    evict_profile_plugins(tmp_path)
