"""Plugin toolset keys written into a home's config must be the ones THAT home can resolve.

``hermes_cli/tools_config.py`` writes plugin toolset names into ``platform_toolsets.<platform>`` and
records them in ``known_plugin_toolsets.<platform>``, using ``_get_plugin_toolset_keys()``. Those keys
have to belong to the home the config is being written FOR, not to whichever home the calling process
loaded its plugins from: a save made for home B by a process whose own home has plugin X used to be
able to put X's key into B's list, after which every run in B warned ``Unknown toolsets: <x>`` (B has
no such plugin, so nothing can resolve the name). ``hermes_cli/plugins_cmd.py`` writes the same pair
when a plugin is enabled, so both writers are covered here.

Real temp homes, real ``config.yaml``, a real (synthetic) directory plugin — no mocks.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest
import yaml

from hermes_constants import reset_hermes_home_override, set_hermes_home_override

_PLUGIN_DIR = "probeplugin"
_PLUGIN_KEY = "probeplugin"
_TOOLSET_KEY = "probe_ts"


def _write_plugin(home: Path) -> None:
    """A minimal directory plugin that registers one tool under toolset ``probe_ts``."""
    plugin = home / "plugins" / _PLUGIN_DIR
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        f"name: {_PLUGIN_KEY}\nversion: 1.0.0\ndescription: probe plugin\n"
        f"provides_tools:\n  - probe_tool\n",
        encoding="utf-8")
    (plugin / "__init__.py").write_text(
        '"""Probe plugin: registers one tool in its own toolset."""\n'
        "\n"
        "def register(ctx):\n"
        "    async def _handler(**kwargs):\n"
        "        return 'ok'\n"
        "    ctx.register_tool(\n"
        "        name='probe_tool',\n"
        "        toolset='probe_ts',\n"
        "        schema={'name': 'probe_tool', 'description': 'probe',\n"
        "                'parameters': {'type': 'object', 'properties': {}}},\n"
        "        handler=_handler,\n"
        "    )\n",
        encoding="utf-8")


@pytest.fixture
def two_homes(tmp_path, monkeypatch):
    """Home A (the process home, plugin installed and enabled) and home B (no plugin dir)."""
    home_a = tmp_path / "A"
    home_b = tmp_path / "B"
    for home in (home_a, home_b):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(
            "platform_toolsets:\n  cli:\n    - hermes-cli\n", encoding="utf-8")
    _write_plugin(home_a)
    (home_a / "config.yaml").write_text(
        "platform_toolsets:\n  cli:\n    - hermes-cli\n"
        f"plugins:\n  enabled:\n    - {_PLUGIN_KEY}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    return home_a, home_b


@contextlib.contextmanager
def _under(home: Path):
    token = set_hermes_home_override(str(home))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _read_config(home: Path) -> dict:
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}


def _saved_plugin_names(config: dict) -> set:
    """Every plugin toolset name the config lists or records, for any platform."""
    known = config.get("known_plugin_toolsets") or {}
    listed = set()
    for entries in (config.get("platform_toolsets") or {}).values():
        listed.update(str(e) for e in (entries or []))
    for entries in known.values():
        listed.update(str(e) for e in (entries or []))
    return listed


def _toggle_and_save() -> None:
    """What the dashboard toolset toggle / ``hermes tools`` do for one toolset: resolve, flip, save."""
    from hermes_cli.tools_config import _get_platform_tools, _save_platform_tools
    from hermes_cli.config import load_config

    config = load_config()
    enabled = set(_get_platform_tools(config, "cli", include_default_mcp_servers=False))
    enabled.add("memory")  # stand-in for the toolset the visitor actually toggled
    _save_platform_tools(config, "cli", enabled)


def test_save_for_another_home_does_not_stamp_the_callers_plugin_toolset(two_homes):
    """A save for home B writes B's own plugin keys, never the caller's."""
    home_a, home_b = two_homes
    from hermes_cli.plugins import discover_plugins, get_plugin_toolset_keys_nowait

    # Warm the site under the process home: A's plugin is live, so its key is in play.
    with _under(home_a):
        discover_plugins()
        assert _TOOLSET_KEY in get_plugin_toolset_keys_nowait()

    with _under(home_b):
        _toggle_and_save()

    # B cannot resolve the plugin (no ``<B>/plugins/probeplugin``), so its name must not be written.
    saved_in_b = _saved_plugin_names(_read_config(home_b))
    assert _TOOLSET_KEY not in saved_in_b, (
        "a platform_toolsets save for home B carried the CALLER home's plugin toolset "
        f"{_TOOLSET_KEY!r}; B has no such plugin and would warn 'Unknown toolsets' on every run")

    # Positive control: the same save for A -- whose plugin really is installed -- does keep the key.
    with _under(home_a):
        _toggle_and_save()
    assert _TOOLSET_KEY in _read_config(home_a)["platform_toolsets"]["cli"]


def test_enabling_a_plugin_does_not_stamp_it_into_a_home_without_it(two_homes):
    """The plugin-enable writer records the toolset only where the plugin is actually installed."""
    home_a, home_b = two_homes
    from hermes_cli.plugins import discover_plugins
    from hermes_cli.plugins_cmd import dashboard_set_agent_plugin_enabled

    with _under(home_a):
        discover_plugins()

    with _under(home_b):
        result = dashboard_set_agent_plugin_enabled(_PLUGIN_KEY, enabled=True)

    assert result.get("ok") is False, (
        "enabling a plugin that lives only in the caller's home must not report success")
    config_b = _read_config(home_b)
    assert _PLUGIN_KEY not in (config_b.get("plugins") or {}).get("enabled", []), (
        "home B's plugins.enabled gained a plugin it does not have")
    assert _TOOLSET_KEY not in _saved_plugin_names(config_b)
