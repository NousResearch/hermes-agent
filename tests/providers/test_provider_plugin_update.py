"""An updated ``$HERMES_HOME`` provider plugin is swapped in whole by the forced plugin sweep.

Regression for #153300: a provider plugin was imported once per process and never again, so an update reached
a running gateway only through files the plugin imported lazily (inside ``create_client``), which then ran
against siblings still loaded from the old tree ("cannot import name ..."). The update surfaces meanwhile said
nothing of the plugin needed a restart.
"""

from __future__ import annotations

import os
import shutil
import sys
import textwrap
from pathlib import Path

import pytest

_INIT = textwrap.dedent("""
    from providers import register_provider
    from providers.base import ProviderProfile
    from .setup import fn  # imported at load, like the DirectSDK plugin's directsdk_setup

    class Profile(ProviderProfile):
        def create_client(self, **kw):
            from .client import Client  # imported on first use, like `from .directsdk import Client`
            return Client(**kw)

    register_provider(Profile(name="demo-provider", aliases=ALIASES,
                              base_url="process://demo-provider", auth_type="external_process"))
""")


def _write(plugin: Path, version: int, *, extra: bool = False, broken: bool = False) -> None:
    """One release of the demo plugin; ``extra`` also registers a second profile and a second alias, ``broken``
    raises at import after loading its ``setup`` submodule."""
    plugin.mkdir(parents=True, exist_ok=True)
    (plugin / "plugin.yaml").write_text(f"name: demo-provider\nkind: model-provider\nversion: {version}.0\n")
    (plugin / "setup.py").write_text(f"def fn():\n    return 'v{version}'\n")
    # A client file from one release running against another release's ``setup`` reports "mixed".
    (plugin / "client.py").write_text(
        f"from .setup import fn\nclass Client:\n    def __init__(self, **kw):\n"
        f"        self.version = fn() if fn() == 'v{version}' else 'mixed'\n")
    tail = ""
    if extra:
        tail += ('register_provider(ProviderProfile(name="demo-extra", base_url="process://demo-extra", '
                 'auth_type="external_process"))\n')
    if broken:
        tail += "raise RuntimeError('broken release')\n"
    aliases = ("demo-alias", "demo-old") if extra else ("demo-alias",)
    (plugin / "__init__.py").write_text(f"ALIASES = {aliases!r}\n" + _INIT + tail)
    # Every release gets newer files than the last, as an update writes them.
    stamp = (version + 1) * 10**12
    for path in plugin.rglob("*"):
        if path.is_file():
            os.utime(path, ns=(stamp, stamp))
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
def plugin(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _reset_providers()
    yield tmp_path / "plugins" / "demo-provider"
    _reset_providers()


def _forced_sweep():
    """What the gateway's ``reload-plugins`` verb runs after an install / enable / update."""
    from hermes_cli.plugins import PluginManager

    manager = PluginManager()
    manager.discover_and_load(force=True)
    return manager


@pytest.mark.parametrize("isolation", ["in-process", "host"])
def test_an_update_serves_new_sessions_whole_while_old_clients_stay_on_their_version(plugin, isolation):
    # Under ``host`` the demo profile overrides create_client, so it is one of the profiles the plugin host
    # hands back to the gateway process: the same lazy import, through the hosted path.
    from providers import get_provider_profile

    (plugin.parent.parent / "config.yaml").write_text(f"plugins:\n  isolation: {isolation}\n", encoding="utf-8")
    _write(plugin, 1, extra=True)
    before = get_provider_profile("demo-provider")  # loaded, but its lazy ``client`` not imported yet
    assert get_provider_profile("demo-extra") is not None and get_provider_profile("demo-old") is before

    _write(plugin, 2)
    _forced_sweep()

    assert get_provider_profile("demo-provider").create_client().version == "v2"
    assert get_provider_profile("demo-alias") is get_provider_profile("demo-provider")
    # A profile and an alias the new release dropped are gone.
    assert get_provider_profile("demo-extra") is None and get_provider_profile("demo-old") is None
    # An agent still holding the old profile imports its lazy file for the first time now: still v1, no mix.
    assert before.create_client().version == "v1"


def test_a_broken_update_keeps_the_previous_version_and_says_so(plugin):
    from hermes_cli.plugins_activation import activation_hint, find_activation, plugin_activation_summary
    from providers import get_provider_profile
    from tui_gateway.contracts.tools_mcp_plugins import PluginActivation

    _write(plugin, 1)
    get_provider_profile("demo-provider")

    def update_to(version: int, **kw) -> dict:
        _write(plugin, version, **kw)
        manager = _forced_sweep()
        key = next(k for k in manager._plugins if k.endswith("demo-provider"))
        summary = find_activation([plugin_activation_summary(manager, key)], "demo-provider")
        PluginActivation.model_validate(summary)  # the shape Desktop / TUI clients receive
        return {"gateway_reloaded": True, "activation": summary}

    assert "model provider (new sessions use this version)" in activation_hint(update_to(2))

    hint = activation_hint(update_to(3, broken=True))
    assert "failed to load (RuntimeError: broken release)" in hint and "previous version keeps serving" in hint
    assert get_provider_profile("demo-provider").create_client().version == "v2"

    # The fixed release after it loads clean: nothing the broken one half-imported is reused.
    assert "new sessions use this version" in activation_hint(update_to(4))
    assert get_provider_profile("demo-provider").create_client().version == "v4"


def test_snapshot_carries_the_plugin_code_but_not_its_test_tree(plugin, monkeypatch):
    """The snapshot prefix is ~220 characters, so on Windows a plugin whose ``tests/`` adds long names blows the
    260-character MAX_PATH and cannot be staged at all ("Could not stage provider plugin", #159414). A test tree
    is never imported, so it must not be copied — the same directories the fingerprint already skips — and the
    plugin has to still load without it."""
    import providers.plugin_reload as reload
    from providers import get_provider_profile

    monkeypatch.setattr(reload, "_snapshot_root", None)
    _write(plugin, 1)
    deep_rel = Path("tests") / "directsdk" / ("test_" + "directsdk_streaming_" * 2)
    deep_rel_file = deep_rel / ("test_a_very_long_descriptive_file_name_" * 2 + ".py")
    (plugin / deep_rel).mkdir(parents=True)
    (plugin / deep_rel_file).write_text("x = 1\n")
    (plugin / "evals" / "cases").mkdir(parents=True)
    (plugin / "evals" / "cases" / "streaming.yaml").write_text("id: 1\n")
    assert len(str(deep_rel_file)) > 40  # the headroom 260 minus the ~220-character snapshot prefix

    snapshot = reload._snapshot(plugin, "_hermes_user_provider_demo")

    assert (snapshot / "__init__.py").is_file() and (snapshot / "plugin.yaml").is_file()
    assert not (snapshot / "tests").exists()
    assert not (snapshot / "evals").exists()
    _forced_sweep()  # the plugin still loads, test tree gone
    assert get_provider_profile("demo-provider").create_client().version == "v1"
    assert reload._snapshot(plugin, "_hermes_user_provider_demo") == snapshot  # frozen copy is reused

