"""Plugin lifecycle contracts plus native-loader composition in full checkouts."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
PLUGIN = ROOT / 'plugins' / 'model-providers' / 'freemaxxing'


@pytest.fixture
def plugin(monkeypatch):
    """Isolated host-API fixture; native host integration is tested below."""
    profiles, runtime = {}, {}
    class Record:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    bindings = {
        'providers': {'register_provider': lambda p: profiles.update({p.name: p})},
        'providers.base': {'ProviderProfile': Record},
        'agent.secret_scope': {'is_multiplex_active': lambda: False,
                               'current_secret_scope': lambda: None,
                               'get_secret': lambda _name: None},
        'hermes_cli.auth': {'PROVIDER_REGISTRY': runtime, 'ProviderConfig': Record,
                           'resolve_nous_runtime_credentials': lambda: {}},
        'hermes_cli.config': {'get_env_value_prefer_dotenv': lambda name: os.environ.get(name, '')},
    }
    for name, values in bindings.items():
        module = types.ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    for key in list(os.environ):
        if key.startswith('FREEMAXXING_') or key.endswith('_API_KEY'):
            monkeypatch.delenv(key, raising=False)
    name = '_freemaxxing_registration_contract'
    spec = importlib.util.spec_from_file_location(name, PLUGIN / '__init__.py',
                                                 submodule_search_locations=[str(PLUGIN)])
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    yield module, profiles, runtime
    module.stop_proxy()
    for key in list(sys.modules):
        if key.startswith(name + '.'):
            sys.modules.pop(key, None)


def test_discovery_is_disabled_and_metadata_only(plugin):
    module, profiles, runtime = plugin
    assert module._listener is None and module.pool.count() == 0
    assert profiles['freemaxxing'].fetch_models() == ['freemaxxing']
    assert list(runtime['freemaxxing'].api_key_env_vars) == []
    assert 'FREEMAXXING_API_KEY' not in os.environ


def test_enabled_plugin_registers_only_its_local_capability(plugin, monkeypatch):
    module, profiles, runtime = plugin
    monkeypatch.setenv('FREEMAXXING_ENABLED', '1')
    monkeypatch.setenv('FREEMAXXING_PORT', '0')
    module._register()
    assert profiles['freemaxxing'].base_url.startswith('http://127.0.0.1:')
    assert os.environ['FREEMAXXING_API_KEY'] == module.local_token()
    assert list(runtime['fm'].api_key_env_vars) == ['FREEMAXXING_API_KEY']
    assert module.pool.count() == 0  # No credentials/catalog during registration.


def test_multiplex_refuses_before_credentials_or_listener(plugin, monkeypatch):
    module, _, runtime = plugin
    monkeypatch.setenv('FREEMAXXING_ENABLED', '1')
    monkeypatch.setattr(sys.modules['agent.secret_scope'], 'is_multiplex_active', lambda: True)
    with pytest.raises(RuntimeError, match='multiplex_profiles'):
        module.ensure_proxy()
    with pytest.raises(RuntimeError, match='multiplex_profiles'):
        module._build_pool()
    assert module._listener is None and module.pool.count() == 0
    assert list(runtime['fm'].api_key_env_vars) == []


def test_unknown_scope_is_not_permission_to_initialize(plugin, monkeypatch):
    module, _, _ = plugin
    def fail():
        raise RuntimeError('scope probe failed')
    monkeypatch.setattr(sys.modules['agent.secret_scope'], 'is_multiplex_active', fail)
    with pytest.raises(RuntimeError):
        module._build_pool()
    assert module.pool.count() == 0


def test_scoped_secret_miss_never_inherits_process_key(plugin, monkeypatch):
    module, _, _ = plugin
    monkeypatch.setenv('OPENROUTER_API_KEY', 'other-profile-secret')
    monkeypatch.setattr(sys.modules['agent.secret_scope'], 'current_secret_scope', lambda: {})
    assert module._resolve_key(('OPENROUTER_API_KEY',)) == ''


def test_keys_do_not_auto_enroll_allowance_accounts(plugin, monkeypatch):
    module, _, _ = plugin
    for provider in ('GROQ', 'GEMINI', 'MISTRAL', 'NOUS'):
        monkeypatch.setenv(provider + '_API_KEY', 'not-free-proof')
    module._build_pool()
    assert [backend.name for backend in module.pool.snapshot()] == ['opencode-free']


def test_explicit_account_enrollment_and_local_boundary(plugin, monkeypatch):
    module, _, _ = plugin
    monkeypatch.setenv('FREEMAXXING_FREE_TIER_PROVIDERS', 'nous-portal,groq,gemini,mistral')
    for provider in ('OPENROUTER', 'GROQ', 'GEMINI', 'MISTRAL'):
        monkeypatch.setenv(provider + '_API_KEY', 'test-only')
    monkeypatch.setenv('FREEMAXXING_LOCAL_BASE_URL', 'http://127.0.0.1:11434/v1')
    monkeypatch.setenv('FREEMAXXING_LOCAL_MODELS', 'local-test')
    module._build_pool()
    assert {b.name for b in module.pool.snapshot()} == {
        'nous-portal', 'openrouter', 'opencode-free', 'groq', 'gemini', 'mistral', 'local'}
    nous = next(b for b in module.pool.snapshot() if b.name == 'nous-portal')
    assert nous.get_cached_models() == [] and nous.api_key == ''


def test_invalid_config_does_not_retire_previous_pool(plugin, monkeypatch):
    module, _, _ = plugin
    module._build_pool()
    previous = module.pool.snapshot()
    monkeypatch.setenv('FREEMAXXING_FREE_TIER_PROVIDERS', 'paid-provider')
    with pytest.raises(ValueError):
        module._build_pool()
    assert module.pool.snapshot() == previous and not previous[0].closed


def test_packaged_moa_preset_is_explicit_and_has_no_paid_slot():
    data = yaml.safe_load((PLUGIN / 'examples' / 'free-moa.yaml').read_text(encoding='utf-8'))
    preset = data['moa']['presets']['freemaxxing']
    assert preset['enabled'] is False and preset['fanout'] == 'user_turn'
    assert all(slot['provider'] == 'freemaxxing' for slot in
               preset['reference_models'] + [preset['aggregator']])
    assert len({s['model'] for s in preset['reference_models']}) == 3


@pytest.mark.skipif(not (ROOT / 'providers' / '__init__.py').is_file(),
                    reason='native Hermes host modules require a complete repository checkout')
@pytest.mark.parametrize('enabled,auth_first', [('0', False), ('1', False), ('1', True)])
def test_native_provider_loader_and_generic_credentials(tmp_path, enabled, auth_first):
    # New interpreter and disposable profile prevent test-collection mutation of
    # the real provider registry. The production loader owns module identity.
    script = '''
import importlib, json, os, sys, urllib.request
import httpx
httpx.Client.send = lambda *a, **k: (_ for _ in ()).throw(AssertionError("no upstream request allowed"))
if os.environ["FM_AUTH_FIRST"] == "1":
    import hermes_cli.auth
from providers import get_provider_profile
p = get_provider_profile("freemaxxing")
assert p is not None
m = sys.modules[type(p).__module__]
assert p.fetch_models() == ["freemaxxing"] and m.pool.count() == 0
if os.environ["FREEMAXXING_ENABLED"] == "1":
    from hermes_cli.auth import resolve_api_key_provider_credentials
    resolved = resolve_api_key_provider_credentials("freemaxxing")
    assert resolved["api_key"] == m.local_token()
    assert resolved["base_url"] == p.base_url
    req = urllib.request.Request(p.base_url + "/models", headers={"Authorization": "Bearer " + m.local_token()})
    with urllib.request.urlopen(req, timeout=3) as response:
        assert json.load(response)["data"][0]["id"] == "freemaxxing"
    assert m.pool.count() == 0
else:
    assert m._listener is None
    assert "FREEMAXXING_API_KEY" not in os.environ
m.stop_proxy()
'''
    env = dict(os.environ)
    for key in list(env):
        if key.startswith('FREEMAXXING_'):
            env.pop(key)
    env.update(HERMES_HOME=str(tmp_path), FREEMAXXING_ENABLED=enabled, FREEMAXXING_PORT='0',
               FM_AUTH_FIRST='1' if auth_first else '0', NO_PROXY='127.0.0.1,localhost')
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, env=env,
                            text=True, encoding='utf-8', capture_output=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
