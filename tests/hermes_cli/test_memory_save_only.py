"""Desktop save-only configuration stays separate from explicit readiness-gated Use."""
import json
from pathlib import Path


def test_declared_save_only_and_explicit_use_are_owner_scoped(tmp_path, monkeypatch):
    import plugins.memory as memory
    from agent import secret_scope
    from hermes_cli.config import atomic_config_write
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
    from starlette.testclient import TestClient

    root = tmp_path / '.hermes'
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(root))
    monkeypatch.setattr(memory, '_MEMORY_PLUGINS_DIR', tmp_path / 'no-bundled')
    monkeypatch.setattr(memory, '_iter_entry_points', lambda: [])
    homes = {'default': root, 'second': root / 'profiles' / 'second'}
    for home in homes.values():
        plugin = home / 'plugins' / 'save-fixture'
        plugin.mkdir(parents=True)
        atomic_config_write(home / 'config.yaml', {'memory': {'provider': 'previous'}})
        (plugin / 'plugin.yaml').write_text('name: save-fixture\nversion: 1.0.0\n')
        (plugin / '__init__.py').write_text('''from agent.memory_provider import MemoryProvider
class Provider(MemoryProvider):
    name = "save-fixture"
    def is_available(self): return True
    def initialize(self, config, hermes_home): pass
    def shutdown(self): pass
    def get_tool_schemas(self): return []
    def get_config_schema(self):
        return [{"key": "workspace", "type": "text", "required": True}]
def register(ctx): ctx.register_memory_provider(Provider())
''')
        (plugin / 'config_schema.py').write_text('''from plugins.memory.config_schema import ProviderConfigSchema, ProviderField
CONFIG_SCHEMA = ProviderConfigSchema(name="save-fixture", label="Fixture", fields=(ProviderField(key="workspace", label="Workspace"),))
''')
    monkeypatch.setattr('hermes_cli.memory_catalog.featured_memory_entries', lambda: [])
    client = TestClient(app, headers={_SESSION_HEADER_NAME: _SESSION_TOKEN})
    old = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        for profile in ('default', 'second', 'default'):
            home = homes[profile]
            other = homes['second' if profile == 'default' else 'default']
            before = (home / 'config.yaml').read_bytes()
            other_before = (other / 'config.yaml').read_bytes()
            params = {'profile': profile, 'surface': 'declared'}
            if not (home / 'save-fixture' / 'config.json').exists():
                assert client.put('/api/memory/provider', params={'profile': profile}, json={'provider': 'save-fixture'}).status_code == 400
            response = client.put('/api/memory/providers/save-fixture/config', params=params,
                                  json={'values': {'workspace': profile}, 'activate': False})
            assert response.status_code == 200, response.text
            assert (home / 'config.yaml').read_bytes() == before
            payload = client.get('/api/memory/providers/save-fixture/config', params=params).json()
            assert payload['supports_save_only'] is True
            assert payload['fields'][0]['value'] == profile
            assert (other / 'config.yaml').read_bytes() == other_before
            result = client.put('/api/memory/provider', params={'profile': profile}, json={'provider': 'save-fixture'})
            assert result.status_code == 200, result.text
            assert client.get('/api/memory', params={'profile': profile}).json()['active'] == 'save-fixture'
        # Older callers keep the historical activate-on-save default.
        from hermes_cli.config import atomic_config_replace
        atomic_config_replace(root / 'config.yaml', {'memory': {'provider': 'previous'}})
        response = client.put('/api/memory/providers/save-fixture/config', params={'profile': 'default', 'surface': 'declared'}, json={'values': {'workspace': 'legacy'}})
        assert response.status_code == 200, response.text
        assert client.get('/api/memory', params={'profile': 'default'}).json()['active'] == 'save-fixture'
    finally:
        secret_scope.set_multiplex_active(old)
