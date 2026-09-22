"""Runtime-aware regression coverage for platform plugin discovery."""

from hermes_cli.plugins import PluginManager
from hermes_cli.plugins_discovery import gate_manifest
from hermes_cli.plugins_manifest import PluginManifest


def _platform(*, source: str) -> PluginManifest:
    return PluginManifest(name="example", source=source, kind="platform")


def test_enabled_third_party_platform_defers_outside_gateway():
    """Enabled third-party adapters stay lazy outside the gateway."""
    third_party = _platform(source="user")

    assert gate_manifest(third_party, set(), {"example"}).action == "defer"


def test_enabled_third_party_platform_loads_at_gateway_startup():
    """Gateway startup eagerly registers enabled third-party platform adapters."""
    third_party = _platform(source="user")

    assert gate_manifest(third_party, set(), {"example"}, defer_platforms=False).action == "load_now"


def test_bundled_platform_remains_deferred_outside_gateway():
    """The existing bundled non-gateway lazy-load behavior is unchanged."""
    bundled = _platform(source="bundled")

    assert gate_manifest(bundled, set(), None).action == "defer"


def test_gateway_discovery_reloads_a_previously_deferred_manager(monkeypatch):
    """A process that becomes a gateway upgrades its prior lazy platform discovery once."""
    manager = PluginManager()
    calls = []

    monkeypatch.setattr(manager, "_discover_and_load_inner", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(manager, "_refresh_secret_sources_after_discovery", lambda: None)

    manager.discover_and_load()
    manager.discover_and_load(defer_platforms=False)

    assert calls == [{}, {"defer_platforms": False}]
