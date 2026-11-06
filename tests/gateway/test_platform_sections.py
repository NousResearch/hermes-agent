"""gateway.config_loader.build_platform_sections / PlatformSections.finish: the loader's platform steps,
shared by load_yaml_layer, the dashboard's builtin_platform_status and relay_explicitly_disabled."""

from __future__ import annotations

import pytest
import hermes_yaml as yaml

from gateway import config_loader
from gateway.config import Platform
from gateway.platform_registry import PlatformEntry, platform_registry


def _forget(name: str) -> None:
    """Make *name* unknown: not registered and no cached ``Platform`` pseudo-member (unregister keeps it)."""
    platform_registry.unregister(name)
    Platform._value2member_map_.pop(name, None)
    Platform._member_map_.pop(name.upper(), None)


def _register(name: str) -> None:
    platform_registry.register(PlatformEntry(
        name=name, label=name, adapter_factory=lambda cfg: None, check_fn=lambda: True, source="plugin"), scope=None)


@pytest.fixture
def cold_plugin(request, monkeypatch, tmp_path):
    """A user-installed platform plugin that only becomes a Platform once discovery registers it."""
    from hermes_cli import plugins

    name = f"cold_{request.node.name[:40]}".lower()

    def discover(*_args, **_kwargs):
        if not platform_registry.is_registered(name):
            _register(name)

    _forget(name)
    monkeypatch.setattr(plugins, "discover_plugins", discover)
    yield name
    _forget(name)


def _first_load_require_mention(home, name) -> object:
    assert name not in Platform._value2member_map_  # the load below is the plugin's first
    gw_data: dict = {}
    config_loader.load_yaml_layer(home, gw_data)
    return gw_data["platforms"][name]["extra"]["require_mention"]


def test_a_cold_plugins_gateway_block_is_authored_on_the_first_load(cold_plugin, tmp_path):
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        cold_plugin: {"require_mention": True},
        "gateway": {cold_plugin: {"extra": {"require_mention": False}}},
    }), encoding="utf-8")

    assert _first_load_require_mention(tmp_path, cold_plugin) is False  # as on every later load


def test_an_administrator_pin_on_a_cold_plugin_holds_on_the_first_load(
        cold_plugin, monkeypatch, tmp_path, tmp_path_factory):
    from hermes_cli import managed_scope

    managed = tmp_path_factory.mktemp("managed")
    (managed / "config.yaml").write_text(yaml.safe_dump({
        "gateway": {cold_plugin: {"extra": {"require_mention": False}}}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope.invalidate_managed_cache()
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({cold_plugin: {"require_mention": True}}), encoding="utf-8")
    try:
        assert _first_load_require_mention(tmp_path, cold_plugin) is False
    finally:
        managed_scope.invalidate_managed_cache()


@pytest.fixture
def other_plugin(monkeypatch):
    name = "sections_other_plugin"
    _forget(name)
    _register(name)
    yield name
    _forget(name)


def _relay_sections(cfg):
    sections = config_loader.build_platform_sections(cfg, {}, names={"relay"})
    sections.finish([Platform.RELAY])
    return sections


def test_names_keep_other_platforms_blocks_out(other_plugin):
    sections = _relay_sections({"platforms": {"relay": {"enabled": False}, "telegram": {"extra": None}},
                                "gateway": {other_plugin: {"enabled": True}}})

    assert set(sections.platforms) == {"relay"}
    assert other_plugin not in Platform._value2member_map_  # no process-wide pseudo-member

