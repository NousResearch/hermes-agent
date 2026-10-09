"""Remote skill callbacks remain inside the native per-profile plugin host."""
import os
import sys

import hermes_yaml as yaml

from hermes_cli import plugins


def test_skill_source_callbacks_run_in_host(tmp_path, monkeypatch):
    home = tmp_path / 'profile'
    plugin_dir = home / 'plugins' / 'ramprobe'
    plugin_dir.mkdir(parents=True)
    (plugin_dir / 'plugin.yaml').write_text('name: ramprobe\nversion: "1"\n')
    (plugin_dir / '__init__.py').write_text('''
import os

def register(ctx):
    ctx.register_skill_source('catalog',
        list_skills=lambda: [{'name': 'interview-me', 'uri': 'test://skill'}],
        load_skill=lambda uri: {'name': 'interview-me', 'content': str(os.getpid())})
''')
    (home / 'config.yaml').write_text(yaml.safe_dump({
        'plugins': {'enabled': ['ramprobe'], 'isolation': 'host'}}))
    bundled = tmp_path / 'bundled'
    bundled.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(plugins, 'get_bundled_plugins_dir', lambda: bundled)
    manager = plugins.PluginManager()
    manager.discover_and_load()
    try:
        assert manager._plugins['ramprobe'].error is None
        info = manager.list_skill_source_commands()['/interview-me']
        payload = manager.load_skill_source_payload(info)
        assert int(payload['content']) != os.getpid()
        assert not any('ramprobe' in name for name in sys.modules)
        assert not list(home.rglob('SKILL.md'))
    finally:
        manager.unload()
        manager._plugin_host().shutdown()
    assert manager.list_skill_source_commands() == {}
