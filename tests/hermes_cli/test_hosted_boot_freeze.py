"""The end of a hosted boot that passes every check: the registries freeze, one-way.

Real imports, a temp ``HERMES_HOME``, and the stub sandbox and storage providers installed through
their real discovery (a plugin dir under the home). The freeze is process-global and irreversible,
so the freezing test is the LAST one in this file and nothing after it touches a registry.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import hosted_mode
import hermes_cli.plugins as plugins_mod
import registry_freeze
from hermes_constants import get_hermes_home
from tests.hermes_cli import test_hosted_mode as boot_tests
from tests.hermes_cli.test_hosted_mode import (
    SANDBOX_PLUGIN, STORAGE_PLUGIN, _clean_hosted_config, _curated, _install_plugin, _write_config,
)


@pytest.fixture(autouse=True)
def _shipped_catalog(monkeypatch):
    """The catalog entries the deployment image ships (the same state the boot-profile tests
    curate into): the trust checks read an install's tier and reviewed pin from the shipped
    entry, never from the record's own claims."""
    from hermes_cli import plugin_catalog as pc_cat

    boot_tests._SHIPPED_CATALOG.clear()
    monkeypatch.setattr(pc_cat, "get_catalog_entry",
                        lambda name, catalog_dir=None: boot_tests._SHIPPED_CATALOG.get(name))
    yield boot_tests._SHIPPED_CATALOG
    boot_tests._SHIPPED_CATALOG.clear()


@pytest.fixture
def verdict(monkeypatch):
    monkeypatch.setattr(hosted_mode, "_VERDICT", {"hosted": None})


@pytest.fixture
def clean_hosted(verdict, monkeypatch, tmp_path):
    """The deployment of the passing-boot test, with an empty bundled-plugins dir so the boot
    judges the home's own stubs; the storage stub installs in the session-storage kind's bundled
    dir (a hosted boot resolves named providers only from the bundled tree)."""
    from plugins import session_storage

    bundled = tmp_path / "bundled"
    (bundled / "session_storage").mkdir(parents=True)
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: bundled)
    monkeypatch.setattr(session_storage, "_PLUGINS_DIR", bundled / "session_storage")
    _write_config(_clean_hosted_config())
    _install_plugin("stub-sandbox", SANDBOX_PLUGIN, manifest={"name": "stub-sandbox",
                                                              "version": "1.0"})
    _curated(get_hermes_home() / "plugins" / "stub-sandbox")
    storage = session_storage._PLUGINS_DIR / "stub-storage"
    storage.mkdir(parents=True)
    (storage / "__init__.py").write_text(STORAGE_PLUGIN, encoding="utf-8")


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


def test_finish_refuses_when_a_core_tool_did_not_register(clean_hosted, monkeypatch):
    """Builtin tool discovery is mandatory and synchronous before the freeze: a core tool that
    never registered (its module failed to import, or discovery was left to the first turn's
    lazy import) refuses the boot instead of freezing an incomplete registry."""
    from hermes_cli.hosted_mode import HostedBootRefused
    from tools.registry import registry

    real_names = registry.get_all_tool_names
    monkeypatch.setattr(registry, "get_all_tool_names",
                        lambda: [name for name in real_names() if name != "web_search"])
    with pytest.raises(HostedBootRefused, match="web_search"):
        hosted_mode.finish_hosted_boot()
    assert registry_freeze.registries_frozen() is False  # refused before anything froze


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


_FINISH_PROBE = '''
import json

from hermes_cli import hosted_mode

assert hosted_mode.is_hosted() is True
hosted_mode.finish_hosted_boot()
import model_tools  # the finish imported it already; the core set is registered
from tools.registry import registry

from registry_freeze import RegistryFrozen, registries_frozen

result = {"frozen": registries_frozen(),
          "core": {"terminal", "read_file"} <= set(registry.get_all_tool_names())}
try:
    registry.register(name="too_late", toolset="probe",
                      schema={"name": "too_late", "description": "x", "parameters": {}},
                      handler=lambda args, **kw: "{}")
    result["refused"] = False
except RegistryFrozen:
    result["refused"] = True
print("PROBE " + json.dumps(result))
'''


def test_a_fresh_hosted_finish_registers_the_core_tools_and_freezes(tmp_path):
    """A fresh process runs a hosted finish, then imports the tool layer: the core tools are
    registered (builtin discovery was synchronous during the finish, not a lazy first-turn
    import the freeze would have refused), and a registration after the freeze raises instead
    of being swallowed by a discovery path."""
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("hosted:\n  enabled: true\n", encoding="utf-8")
    env = {**os.environ, "HERMES_HOME": str(home),
           "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    probe = subprocess.run([sys.executable, "-c", _FINISH_PROBE], env=env, check=False,
                            capture_output=True, text=True, timeout=300)
    assert probe.returncode == 0, probe.stderr
    (line,) = [line for line in probe.stdout.splitlines() if line.startswith("PROBE ")]
    assert json.loads(line[len("PROBE "):]) == {"frozen": True, "core": True, "refused": True}
