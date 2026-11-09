"""The end of a hosted boot that passes every check: the registries freeze, one-way.

Real imports, a temp ``HERMES_HOME``, and the stub sandbox and storage providers installed through
their real discovery (a plugin dir under the home). The freeze is process-global and irreversible,
so the freezing test is the LAST one in this file and nothing after it touches a registry.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli import hosted_mode
import hermes_cli.plugins as plugins_mod
import registry_freeze
from hermes_constants import get_hermes_home
from tests.hermes_cli.test_hosted_mode import (
    SANDBOX_PLUGIN, STORAGE_PLUGIN, _clean_hosted_config, _curated, _install_plugin, _write_config,
)


@pytest.fixture
def verdict(monkeypatch):
    monkeypatch.setattr(hosted_mode, "_VERDICT", {"hosted": None})


@pytest.fixture
def clean_hosted(verdict, monkeypatch, tmp_path):
    """The deployment of the passing-boot test, with an empty bundled-plugins dir so the boot
    judges the home's own stubs."""
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: bundled)
    _write_config(_clean_hosted_config())
    _install_plugin("stub-sandbox", SANDBOX_PLUGIN, manifest={"name": "stub-sandbox",
                                                              "version": "1.0"})
    _curated(get_hermes_home() / "plugins" / "stub-sandbox")
    _install_plugin("stub-storage", STORAGE_PLUGIN)


def test_a_local_boot_never_freezes_the_registries(verdict):
    """hosted false: finish_hosted_boot does nothing, and a tool can still register."""
    assert hosted_mode.finish_hosted_boot() is None
    assert registry_freeze.registries_frozen() is False
    from tools.registry import registry

    registry.register(name="late_local_tool", toolset="probe",
                      schema={"name": "late_local_tool", "description": "x", "parameters": {}},
                      handler=lambda args, **kw: "{}")
    registry.deregister("late_local_tool")


def test_a_pending_platform_loader_blocks_the_freeze_until_the_boot_drops_it(clean_hosted, monkeypatch):
    """A deferred platform loader nobody looked up would be lost by the freeze (its registration
    would be refused afterwards); the hosted boot drops the pending ones first. Without the drop
    the freeze refuses, naming the platform."""
    from gateway.platform_registry import PlatformRegistry, platform_registry

    seen = []
    platform_registry.register_deferred("lazyplat", lambda: seen.append("loaded"))

    hosted_mode.assert_hosted_boot(env={"PATH": "/usr/bin"})
    assert registry_freeze.registries_frozen() is False
    monkeypatch.setattr(PlatformRegistry, "drop_pending_deferred", lambda self: [])
    with pytest.raises(registry_freeze.RegistriesIncomplete, match="lazyplat"):
        hosted_mode.finish_hosted_boot()
    assert seen == []  # it was never materialised


def test_a_hosted_boot_that_passes_every_check_freezes_the_registries(clean_hosted):
    """The passing boot: every check green through the real discovery, then the freeze — after
    which every registration entry point refuses, and reads keep working."""
    import providers
    from hermes_cli.plugins import get_plugin_manager
    from tools.registry import registry

    hosted_mode.assert_hosted_boot(env={"PATH": "/usr/bin", "HOME": "/home/op", "LC_ALL": "C"})
    assert registry_freeze.registries_frozen() is False
    hosted_mode.finish_hosted_boot()
    assert registry_freeze.registries_frozen() is True

    with pytest.raises(registry_freeze.RegistryFrozen):
        registry.register(name="too_late", toolset="probe",
                          schema={"name": "too_late", "description": "x", "parameters": {}},
                          handler=lambda args, **kw: "{}")
    with pytest.raises(registry_freeze.RegistryFrozen):
        providers.register_provider(SimpleNamespace())  # the refusal fires before the payload
    with pytest.raises(registry_freeze.RegistryFrozen):
        get_plugin_manager().register_approval_transport(
            "too-late", lambda: True, plugin_id="test")
    assert sorted(p.name for p in providers.list_providers())  # reads keep working
