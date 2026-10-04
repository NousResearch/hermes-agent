import pytest
from hermes_cli import plugins
from agent.auxiliary_client import _get_auxiliary_task_config
from hermes_cli.main_provider_setup import _all_aux_tasks

@pytest.mark.parametrize('child_first', [True, False])
def test_real_loader_order_and_unloaded_base(tmp_path, monkeypatch, caplog, child_first):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(plugins, 'get_bundled_plugins_dir', lambda: tmp_path / 'empty')
    monkeypatch.setattr(plugins.PluginManager, '_scan_entry_points', lambda self: [])
    for name, dirname, code in [
        ('base', 'z_base' if child_first else 'a_base', "ctx.register_auxiliary_task(key='base_task', display_name='Base', description='d', defaults={'model': 'base-model', 'provider': 'openai', 'base_url': 'https://example.invalid', 'api_key': 'fixture'})"),
        ('child', 'a_child' if child_first else 'z_child', "ctx.register_auxiliary_task(key='my_aux', display_name='Child', description='d', inherit_from='base_task', defaults={'timeout': 12})\n    ctx.register_command('child-command', lambda args: args)")]:
        directory = tmp_path / 'plugins' / dirname
        directory.mkdir(parents=True)
        (directory / 'plugin.yaml').write_text(f'name: {dirname}\nversion: 0.1.0\n')
        (directory / '__init__.py').write_text('def register(ctx):\n    ' + code + '\n')
    names = ['z_base', 'a_child'] if child_first else ['a_base', 'z_child']
    (tmp_path / 'config.yaml').write_text(f'plugins:\n  enabled: {names}\nauxiliary:\n  my_aux:\n    timeout: 99\n')
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, 'get_plugin_manager', lambda: manager)
    monkeypatch.setattr(plugins, '_ensure_plugins_discovered', lambda: manager)
    manager.discover_and_load()
    assert 'child-command' in manager._plugin_commands
    cfg = _get_auxiliary_task_config('my_aux')
    assert cfg['model'] == 'base-model'
    assert cfg['timeout'] == 99
    from gateway.run import _bridge_auxiliary_config_to_env
    _bridge_auxiliary_config_to_env({'my_aux': {'timeout': 99}})
    import os
    assert os.environ['AUXILIARY_MY_AUX_PROVIDER'] == 'openai'
    assert os.environ['AUXILIARY_MY_AUX_API_KEY'] == 'fixture'
    assert manager.unload(names[0])
    assert 'my_aux' not in {key for key, _, _ in _all_aux_tasks()}
    assert _get_auxiliary_task_config('my_aux') == {}
    assert 'base_task' in caplog.text
    manager.unload(names[1])

@pytest.mark.parametrize('self_loop', [True, False])
def test_cycle_is_bounded_and_diagnosed(tmp_path, monkeypatch, caplog, self_loop):
    manager = plugins.PluginManager()
    manager._discovered = True
    monkeypatch.setattr(plugins, '_ensure_plugins_discovered', lambda: manager)
    ctx = plugins.PluginContext(plugins.PluginManifest(name='loop'), manager)
    ctx.register_auxiliary_task(key='a', display_name='A', description='d')
    ctx.register_auxiliary_task(key='b', display_name='B', description='d', inherit_from='a')
    ctx.register_auxiliary_task(key='a', display_name='A', description='d', inherit_from='a' if self_loop else 'b')
    assert _get_auxiliary_task_config('a') == {}
    assert 'cycle' in caplog.text.lower()
    assert 'a' not in {key for key, _, _ in _all_aux_tasks()}
