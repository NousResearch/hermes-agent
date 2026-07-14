"""Discovery contract for the opt-in research protocol plugin."""

import pytest
import yaml

from hermes_cli.plugins import PluginManager


PLUGIN_KEY = "research-protocol"


def _configure_plugins(hermes_home, plugins):
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": plugins}),
        encoding="utf-8",
    )


def _discover(tmp_path, monkeypatch, plugins=None):
    hermes_home = tmp_path / "hermes-home"
    if plugins is not None:
        _configure_plugins(hermes_home, plugins)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_BUNDLED_PLUGINS", raising=False)
    manager = PluginManager()
    manager.discover_and_load()
    return manager._plugins[PLUGIN_KEY]


def _assert_no_runtime_registrations(loaded):
    assert loaded.tools_registered == []
    assert loaded.hooks_registered == []
    assert loaded.middleware_registered == []
    assert loaded.commands_registered == []


def test_research_protocol_is_discovered_but_not_enabled_by_default(
    tmp_path, monkeypatch
):
    """The bundled plugin is visible to discovery but remains opt-in."""
    loaded = _discover(tmp_path, monkeypatch)

    assert loaded.manifest.name == PLUGIN_KEY
    assert loaded.manifest.key == PLUGIN_KEY
    assert loaded.manifest.kind == "standalone"
    assert loaded.enabled is False
    assert loaded.module is None
    assert "not enabled in config" in (loaded.error or "")
    _assert_no_runtime_registrations(loaded)


def test_research_protocol_explicit_enable_loads_no_runtime_behavior(
    tmp_path, monkeypatch
):
    """The exact public key opts in, but the PR 0 entry point is a no-op."""
    loaded = _discover(tmp_path, monkeypatch, {"enabled": [PLUGIN_KEY]})

    assert loaded.enabled is True
    assert loaded.error is None
    _assert_no_runtime_registrations(loaded)


@pytest.mark.parametrize("enabled", [None, "", {}, [], PLUGIN_KEY])
def test_research_protocol_malformed_or_empty_enabled_fails_closed(
    tmp_path, monkeypatch, enabled
):
    """Only an explicit list containing the exact plugin key enables code."""
    loaded = _discover(tmp_path, monkeypatch, {"enabled": enabled})

    assert loaded.enabled is False
    assert loaded.module is None
    assert "not enabled in config" in (loaded.error or "")
    _assert_no_runtime_registrations(loaded)


def test_research_protocol_disabled_list_has_priority(tmp_path, monkeypatch):
    """An explicit deny wins even if the same plugin is allowlisted."""
    loaded = _discover(
        tmp_path,
        monkeypatch,
        {"enabled": [PLUGIN_KEY], "disabled": [PLUGIN_KEY]},
    )

    assert loaded.enabled is False
    assert loaded.module is None
    assert "disabled via config" in (loaded.error or "")
    _assert_no_runtime_registrations(loaded)


def test_pr0_register_entry_point_cannot_touch_any_plugin_surface():
    """The PR 0 entry point must remain a true no-op on every context API."""
    from plugins.research_protocol import register

    class RejectEveryContextAccess:
        def __getattr__(self, name):
            pytest.fail(f"register() accessed plugin context surface {name!r}")

    assert register(RejectEveryContextAccess()) is None
