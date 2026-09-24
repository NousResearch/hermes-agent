"""Platform adapters and core modules must not prepend the checkout root onto ``sys.path``.

Several importable modules carry a defensive shim so the file still resolves its repo siblings
when loaded outside a process that already has the repo on ``sys.path``. The shim used to be
``sys.path.insert(0, root)``, which made checkout-level modules (``utils``, ``tools``, ``evals``,
...) shadow same-named installed packages for every later import in the process, added a duplicate
entry per adapter import, and prepended an arbitrary directory when the file lived anywhere but a
checkout layout.

The shim is now a guarded append: a real checkout root (``gateway/__init__.py`` marker) is appended
only when absent. In adapters where the shim sits below the first repo import (telegram, whatsapp,
discord) it cannot rescue a bare file load; it still runs in processes where the editable install's
meta-path finder resolves repo modules, which is exactly where it used to pollute.

These tests run imports in a subprocess because ``sys.path`` is process-global.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
ADAPTERS = ["telegram", "whatsapp", "raft", "discord", "slack"]
# The shim precedes the first repo import only in these files; elsewhere it is dead code for the
# standalone scenario (imports above it already require the repo).
LIVE_SHIM = ["raft", "slack"]

_STRIP_FINDER = (
    "sys.meta_path = [f for f in sys.meta_path"
    " if 'editable' not in getattr(f, '__module__', '')]"
)


def _probe(script: str, cwd: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, HERMES_HOME=str(cwd / "hermes_home"))
    return subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True, text=True, cwd=str(cwd), env=env, timeout=120,
    )


def _adapter_file(adapter: str) -> str:
    return str(REPO_ROOT / "plugins" / "platforms" / adapter / "adapter.py")


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_adapter_import_does_not_duplicate_sys_path(adapter, tmp_path):
    """When the root is already importable (every real load path), the shim is a no-op: no
    duplicate entry, no reordering."""
    script = textwrap.dedent(f"""
        import importlib, sys
        repo = {str(REPO_ROOT)!r}
        assert repo not in sys.path
        sys.path.insert(0, repo)
        before = list(sys.path)
        importlib.import_module('plugins.platforms.{adapter}.adapter')
        assert sys.path == before, sys.path
    """)
    result = _probe(script, tmp_path)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_adapter_import_does_not_shadow_installed_packages(adapter, tmp_path):
    """A same-named installed package must still win over the checkout root: the shim appends,
    never prepends. ``evals`` is a repo-root package the editable finder does not map, so the
    import resolves purely by ``sys.path`` order."""
    fake_site = tmp_path / "fake_site"
    (fake_site / "evals").mkdir(parents=True)
    (fake_site / "evals" / "__init__.py").write_text("MARKER = 'installed'\n", encoding="utf-8")
    script = textwrap.dedent(f"""
        import importlib, os, sys
        repo = {str(REPO_ROOT)!r}
        sys.path.insert(0, {str(fake_site)!r})
        sys.path = [p for p in sys.path if os.path.realpath(p or '.') != repo]
        before = list(sys.path)
        importlib.import_module('plugins.platforms.{adapter}.adapter')
        assert sys.path == before + [repo], sys.path
        import evals
        assert getattr(evals, 'MARKER', None) == 'installed', evals.__file__
    """)
    result = _probe(script, tmp_path)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("adapter", LIVE_SHIM)
def test_adapter_standalone_file_load_still_works(adapter, tmp_path):
    """The load-bearing case: file-loaded in a process with neither the checkout on sys.path nor
    the editable finder, the shim's appended root still resolves the ``gateway`` siblings."""
    script = textwrap.dedent(f"""
        import importlib.util, os, sys
        {_STRIP_FINDER}
        repo = {str(REPO_ROOT)!r}
        sys.path = [p for p in sys.path if os.path.realpath(p or '.') != repo]
        spec = importlib.util.spec_from_file_location(
            'plugins.platforms.{adapter}.adapter', {_adapter_file(adapter)!r})
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert sys.path[-1] == repo, sys.path
        import gateway  # resolves through the appended root
    """)
    result = _probe(script, tmp_path)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mod", ["gateway.platforms.base", "gateway.run", "tools.cronjob_tools"])
def test_core_module_import_does_not_mutate_sys_path(mod, tmp_path):
    """The same shim lives in core modules on the adapter import chain; importing them with the
    repo already on sys.path must leave sys.path untouched."""
    script = textwrap.dedent(f"""
        import importlib, sys
        repo = {str(REPO_ROOT)!r}
        sys.path.insert(0, repo)
        before = list(sys.path)
        importlib.import_module({mod!r})
        assert sys.path == before, sys.path
    """)
    result = _probe(script, tmp_path)
    assert result.returncode == 0, result.stderr


def test_real_plugin_loader_does_not_mutate_sys_path(tmp_path):
    """E2E through the production loader: ``PluginManager._load_plugin`` file-loads each
    ``plugins/platforms/<a>/__init__.py`` as ``hermes_plugins.<slug>`` and runs its
    ``register(ctx)``. With the repo on sys.path (every real deployment), loading all five
    migrated adapters must leave sys.path completely untouched."""
    script = textwrap.dedent(f"""
        import sys, tempfile
        repo = {str(REPO_ROOT)!r}
        sys.path.insert(0, repo)
        before = list(sys.path)
        from hermes_cli.plugins import PluginManager
        from hermes_cli.plugins_manifest import PluginManifest
        mgr = PluginManager(scope_key=tempfile.mkdtemp())
        for name in {ADAPTERS!r}:
            m = PluginManifest(name=name + '-platform', kind='platform', source='bundled',
                               path=repo + '/plugins/platforms/' + name, key='platforms/' + name)
            mgr._load_plugin(m)
            loaded = mgr._plugins['platforms/' + name]
            assert loaded.enabled, (name, loaded.error)
        assert sys.path == before, sys.path
    """)
    result = _probe(script, tmp_path)
    assert result.returncode == 0, result.stderr


def test_real_plugin_loader_appends_root_once_when_finder_resolves(tmp_path):
    """E2E with the repo absent from sys.path: imports resolve through the editable finder,
    the adapter shims fire, and across all five adapters plus the core-module chain exactly
    one entry is appended, at the end. This is the path where the old ``insert(0)`` prepended
    five duplicate copies ahead of site-packages."""
    script = textwrap.dedent(f"""
        import os, sys, tempfile
        repo = {str(REPO_ROOT)!r}
        sys.path = [p for p in sys.path if os.path.realpath(p or '.') != repo]
        before = list(sys.path)
        from hermes_cli.plugins import PluginManager
        from hermes_cli.plugins_manifest import PluginManifest
        mgr = PluginManager(scope_key=tempfile.mkdtemp())
        for name in {ADAPTERS!r}:
            m = PluginManifest(name=name + '-platform', kind='platform', source='bundled',
                               path=repo + '/plugins/platforms/' + name, key='platforms/' + name)
            mgr._load_plugin(m)
            loaded = mgr._plugins['platforms/' + name]
            assert loaded.enabled, (name, loaded.error)
        assert sys.path == before + [repo], sys.path
    """)
    result = _probe(script, tmp_path)
    assert result.returncode == 0, result.stderr
