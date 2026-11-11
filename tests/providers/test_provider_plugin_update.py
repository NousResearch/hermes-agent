"""An updated ``$HERMES_HOME`` provider plugin reloads whole in the running process.

Regression for #153300: a provider plugin was imported once per process and never again, so an update
reached a running gateway only through files the plugin imported lazily (inside ``create_client``),
which then ran against siblings still loaded from the old tree ("cannot import name ..."). The update
surfaces meanwhile said nothing of the plugin needed a restart.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import textwrap
from pathlib import Path

import pytest

_INIT = textwrap.dedent("""
    from providers import register_provider
    from providers.base import ProviderProfile
    from .setup import old_fn  # imported at load, like the DirectSDK plugin's directsdk_setup

    class Profile(ProviderProfile):
        def create_client(self, **kw):
            from .client import Client  # imported on first use, like `from .directsdk import Client`
            return Client(**kw)

    register_provider(Profile(name="demo-provider", aliases=("demo-alias",),
                              base_url="process://demo-provider", auth_type="external_process"))
    {extra}
""")


def _write(plugin: Path, version: int) -> None:
    """v1 has ``old_fn`` and a second profile; v2 adds ``new_fn`` (which v2's client needs) and drops it."""
    plugin.mkdir(parents=True, exist_ok=True)
    (plugin / "plugin.yaml").write_text(f"name: demo-provider\nkind: model-provider\nversion: {version}.0\n")
    (plugin / "setup.py").write_text("def old_fn():\n    return 'v1'\n" + ("def new_fn():\n    return 'v2'\n"
                                                                          if version == 2 else ""))
    fn = "new_fn" if version == 2 else "old_fn"
    (plugin / "client.py").write_text(
        f"from .setup import {fn} as fn\nclass Client:\n    def __init__(self, **kw):\n        self.version = fn()\n")
    extra = ('register_provider(ProviderProfile(name="demo-extra", base_url="process://demo-extra", '
             'auth_type="external_process"))') if version == 1 else ""
    (plugin / "__init__.py").write_text(_INIT.format(extra=extra))
    (plugin / ".hermes-catalog.json").write_text(json.dumps({"catalog_name": "demo", "sha": f"{version}" * 40}))
    # Same-size source rewritten within one second would reuse its bytecode; the updater drops it too.
    shutil.rmtree(plugin / "__pycache__", ignore_errors=True)


def _reset_providers():
    import providers

    providers._REGISTRY.clear()
    providers._ALIASES.clear()
    providers._HOME_LAYERS.clear()
    providers._PROVIDER_LIST_CACHE = None
    providers._discovered = False
    for mod in [m for m in sys.modules if m.startswith("_hermes_user_provider")]:
        del sys.modules[mod]


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _reset_providers()
    yield tmp_path
    _reset_providers()


def _forced_sweep(home: Path):
    """What the gateway's ``reload-plugins`` verb runs after an install / enable / update."""
    from hermes_cli.plugins import PluginManager

    manager = PluginManager()
    manager.discover_and_load(force=True)
    return manager


def test_an_updated_provider_plugin_reloads_whole_after_the_forced_sweep(home):
    from providers import get_provider_profile

    plugin = home / "plugins" / "demo-provider"
    _write(plugin, 1)
    before = get_provider_profile("demo-provider")
    assert before.create_client().version == "v1" and get_provider_profile("demo-extra") is not None

    _write(plugin, 2)
    _forced_sweep(home)

    assert get_provider_profile("demo-provider").create_client().version == "v2"
    assert get_provider_profile("demo-alias") is get_provider_profile("demo-provider")
    # An agent still holding the old profile object gets the new package too, never a mix of both.
    assert before.create_client().version == "v2"
    assert get_provider_profile("demo-extra") is None  # a profile the new version dropped is gone


def test_the_update_hint_says_the_provider_reloaded_or_why_it_failed(home):
    from hermes_cli.plugins_activation import activation_hint, find_activation, plugin_activation_summary
    from providers import get_provider_profile

    plugin = home / "plugins" / "demo-provider"
    _write(plugin, 1)
    get_provider_profile("demo-provider")

    def hint_after_update() -> str:
        manager = _forced_sweep(home)
        key = next(k for k in manager._plugins if k.endswith("demo-provider"))
        summary = find_activation([plugin_activation_summary(manager, key)], "demo-provider")
        return activation_hint({"gateway_reloaded": True, "activation": summary})

    _write(plugin, 2)
    assert "model provider (new requests use this version)" in hint_after_update()

    (plugin / ".hermes-catalog.json").unlink()  # a git install: the files' mtimes are the identity
    (plugin / "setup.py").write_text("raise RuntimeError('broken release')\n")
    later = max(p.stat().st_mtime_ns for p in plugin.rglob("*.py")) + 10**9
    for p in plugin.rglob("*.py"):
        os.utime(p, ns=(later, later))
    assert "failed to load in the running gateway: RuntimeError: broken release" in hint_after_update()
