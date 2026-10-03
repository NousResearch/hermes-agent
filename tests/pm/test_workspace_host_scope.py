"""Local plugin locks preserve core constraints without resolving other hosts."""
import sys
import tomllib

import pytest
from packaging.markers import Marker

from pm.workspace import _generate_pyproject


@pytest.mark.parametrize('host', ['win32', 'linux', 'darwin'])
def test_plugin_generation_intersects_core_support_and_host(tmp_path, monkeypatch, host):
    core, plugin, out = (tmp_path / name for name in ('core', 'plugin', 'out'))
    core.mkdir()
    plugin.mkdir()
    original = ('[project]\nname="fixture-core"\nversion="1"\n'
                'requires-python=">=3.14"\n[tool.uv]\n'
                'environments=["python_version >= \'3.14\'"]\n')
    (core / 'pyproject.toml').write_text(original)
    (plugin / 'plugin.yaml').write_text('name: fixture\npython_dependencies: ["example==1"]\n')
    monkeypatch.setattr(sys, 'platform', host)
    _generate_pyproject([plugin], out, source=core)
    settings = tomllib.loads((out / 'pyproject.toml').read_text())['tool']['uv']
    predicates = [Marker(m) for m in settings['environments']]
    def accepts(platform, version):
        return any(m.evaluate({'sys_platform': platform, 'python_version': version}) for m in predicates)
    assert accepts(host, '3.14')
    assert not accepts(host, '3.13')
    assert not accepts('android', '3.14')
    assert (core / 'pyproject.toml').read_text() == original
    assert not (core / 'uv.lock').exists()


def test_memberless_generation_keeps_release_project_unchanged(tmp_path):
    core = tmp_path / 'core'
    core.mkdir()
    original = '[project]\nname="fixture-core"\nversion="1"\n'
    (core / 'pyproject.toml').write_text(original)
    out = tmp_path / 'out'
    _generate_pyproject([], out, source=core)
    assert (out / 'pyproject.toml').read_text() == original


def test_member_manifest_migration_uses_release_seed_without_touching_live_lock(tmp_path):
    from pm.workspace import lock_and_sync
    core, plugin = tmp_path / 'core', tmp_path / 'plugin'
    core.mkdir()
    plugin.mkdir()
    (core / 'pyproject.toml').write_text('[project]\nname="core"\nversion="1"\n')
    release = b'version=1\n[[package]]\nname="core"\nversion="1"\nsource={virtual="."}\n'
    (core / 'uv.lock').write_bytes(release)
    (plugin / 'pyproject.toml').write_text('[project]\nname="fixture-plugin"\nversion="1"\n')
    live = tmp_path / 'live.lock'
    stale = b'version=1\n[[package]]\nname="old-member"\nversion="1"\nsource={virtual="plugin-deps/removed"}\n'
    live.write_bytes(stale)
    class Environment:
        def sync(self, root, **kwargs):
            assert (root / 'uv.lock').read_bytes() == release
    lock_and_sync([plugin], [], root=tmp_path / 'out', source=core, seed_lock=live,
                  environment=Environment())
    assert live.read_bytes() == stale
    assert (core / 'uv.lock').read_bytes() == release
