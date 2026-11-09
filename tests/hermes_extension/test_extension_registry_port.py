"""The user extension registry port as a plugin kind, and the local backend's pin and file.

``resolve_extension_registry(name)`` picks a backend by explicit name (no caller is wired to it
yet): ``"local"`` is the per-profile ``extension_registry.json``; another name loads a provider
through real discovery (``$HERMES_HOME/plugins/<name>/``), and a missing one is an error, never a
fallback. The local backend serves only the local tenant, its own profile and the local owner,
and its file is not read by local plugin loading: local installs keep today's in-process plugins.
"""
import json
import textwrap
from pathlib import Path

import pytest

from agent.runtime_scope import bind_profile_scope, bind_state
from hermes_extension_registry_port import (
    KINDS, ExtensionRegistry, ExtensionRegistryProvider, RegistryAdminRefused, RegistryPinRefused,
    resolve_extension_registry,
)
from hermes_extension_registry_port_local import (
    REGISTRY_FILE, LocalExtensionRegistry, LocalExtensionRegistryProvider,
)
from hermes_scope import LOCAL_OWNER, LOCAL_TENANT, Principal, Scope, push_state

_PLUGIN = '''
from hermes_extension_registry_port import ExtensionRegistryProvider

class Probe(ExtensionRegistryProvider):
    @property
    def name(self):
        return "probe"

    def open(self, home=None):
        raise NotImplementedError

    def assert_isolating(self):
        raise RuntimeError("probe role bypasses row security")


def register(ctx):
    ctx.register_extension_registry(Probe())
'''


def test_local_names_the_local_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name in ("local", ""):
        provider = resolve_extension_registry(name)
        assert isinstance(provider, LocalExtensionRegistryProvider)
        provider.assert_isolating()


def test_a_provider_that_does_not_check_isolation_cannot_be_built():
    class Unchecked(ExtensionRegistryProvider):
        @property
        def name(self):
            return "unchecked"

        def open(self, home=None):
            raise NotImplementedError

    with pytest.raises(TypeError, match="assert_isolating"):
        Unchecked()


def test_a_named_provider_loads_through_discovery_and_runs_its_isolation_check(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin = tmp_path / "plugins" / "probe"
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(textwrap.dedent(_PLUGIN))
    provider = resolve_extension_registry("probe")
    assert isinstance(provider, ExtensionRegistryProvider) and provider.name == "probe"
    with pytest.raises(RuntimeError, match="bypasses row security"):
        provider.assert_isolating()


def test_an_unavailable_named_provider_is_an_error_not_the_local_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin = tmp_path / "plugins" / "offline"
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(textwrap.dedent(_PLUGIN).replace(
        "    def assert_isolating(self):", "    def is_available(self):\n        return False\n\n    def assert_isolating(self):"))
    with pytest.raises(LookupError, match="not available"):
        resolve_extension_registry("offline")


def test_a_missing_named_provider_is_an_error_not_the_local_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with pytest.raises(LookupError, match="postgres"):
        resolve_extension_registry("postgres")


@pytest.fixture
def registry(tmp_path):
    return LocalExtensionRegistryProvider().open(tmp_path)


def test_the_local_backend_records_rows_in_its_own_file(registry, tmp_path):
    """The registry is one small file under the profile home — separate from ``state.db`` and
    from the ``plugins/`` tree today's in-process loading reads, which this backend never
    touches: nothing installs a plugin here, and plugin loading never consults this file."""
    with bind_profile_scope(tmp_path):
        row = registry.install(name="weather", kind="plugin", version="1.0",
                               content_hash="hash-1", manifest={"tools": ["weather"]},
                               scan_result={"passed": True})
        assert row["name"] == "weather" and row["enabled"] is True and row["level"] == "owner"
        assert registry.set_agent_enabled("weather", True)
        assert registry.agent_enabled("weather") is True
        assert [r["name"] for r in registry.enabled_extensions()] == ["weather"]
    data = json.loads((tmp_path / REGISTRY_FILE).read_text(encoding="utf-8"))
    assert [e["name"] for e in data["entries"]] == ["weather"]
    assert data["switches"] == {"weather": True}
    # the files today's local plugin loading reads are untouched by the registry
    assert not (tmp_path / "plugins").exists()


def test_a_reinstall_replaces_the_entry_and_keeps_the_switch(registry, tmp_path):
    with bind_profile_scope(tmp_path):
        registry.install(name="weather", kind="plugin", version="1.0", content_hash="h1", manifest={})
        registry.set_agent_enabled("weather", True)
        registry.install(name="weather", kind="plugin", version="1.1", content_hash="h2", manifest={})
        row = registry.get("weather")
        assert row["version"] == "1.1" and row["content_hash"] == "h2"
        assert registry.agent_enabled("weather") is True  # the agent's choice survives an update


def test_the_local_backend_refuses_tenant_wide_entries(registry, tmp_path):
    """The registry's local exception: a local install is one owner's, with nothing to share
    tenant-wide, so the local backend keeps owner-wide entries as the local owner's and
    refuses the tenant level outright — whatever scope asks for it, provisioning included.
    An entry never sits at the agent level by design (a malformed one is refused below); only
    the per-agent switch is the agent's row."""
    with bind_profile_scope(tmp_path):
        with pytest.raises(ValueError):
            registry.install(name="org", kind="plugin", version="1", content_hash="h",
                             manifest={}, level="tenant")
    assert not (tmp_path / REGISTRY_FILE).exists()  # and nothing was written


def test_removal_takes_the_switch_with_it(registry, tmp_path):
    with bind_profile_scope(tmp_path):
        registry.install(name="weather", kind="plugin", version="1.0", content_hash="h", manifest={})
        registry.set_agent_enabled("weather", True)
        assert registry.remove("weather") is True
        assert registry.get("weather") is None
        assert registry.agent_enabled("weather") is False
        assert registry.remove("weather") is False  # and a missing one answers the same


@pytest.mark.parametrize("bad", [
    {"name": " ", "kind": "plugin", "version": "1", "content_hash": "h", "manifest": {}},
    {"name": "x", "kind": "theme", "version": "1", "content_hash": "h", "manifest": {}},
    {"name": "x", "kind": "plugin", "version": "", "content_hash": "h", "manifest": {}},
    {"name": "x", "kind": "plugin", "version": "1", "content_hash": "", "manifest": {}},
    {"name": "x", "kind": "plugin", "version": "1", "content_hash": "h", "manifest": "no"},
    {"name": "x", "kind": "plugin", "version": "1", "content_hash": "h", "manifest": {},
     "scan_result": "no"},
    {"name": "x", "kind": "plugin", "version": "1", "content_hash": "h", "manifest": {}, "level": "agent"},
], ids=["name", "kind", "version", "content_hash", "manifest", "scan_result", "level"])
def test_a_malformed_entry_is_refused_before_anything_is_written(registry, tmp_path, bad):
    with bind_profile_scope(tmp_path):
        with pytest.raises(ValueError):
            registry.install(**bad)
    assert not (tmp_path / REGISTRY_FILE).exists()


@pytest.mark.parametrize("scope", [
    # A hosted tenant naming this very path as its agent is still not this local registry's.
    Scope("acme", "{home}", Principal("u"), owner=LOCAL_OWNER, can_administer=True),
    # Nor is another profile of this machine, whoever its scope says owns it.
    Scope(LOCAL_TENANT, "{home}/elsewhere", LOCAL_OWNER),
    # Nor a local scope that names another owner: the local registry serves the local person.
    Scope(LOCAL_TENANT, "{home}", LOCAL_OWNER, owner=Principal("rival"), can_administer=True),
], ids=["another_tenant", "another_profile", "another_owner"])
def test_the_local_backend_refuses_any_pin_but_its_own_profile(registry, tmp_path, scope):
    with bind_profile_scope(tmp_path):
        registry.install(name="weather", kind="plugin", version="1.0", content_hash="h", manifest={})
    pinned = Scope(scope.tenant, scope.agent.format(home=tmp_path), scope.actor,
                   owner=scope.owner, can_administer=scope.can_administer)
    with bind_state(pinned) if not pinned.is_local else push_state(scope=pinned):
        for call in (lambda: registry.get("weather"), lambda: registry.list(),
                     lambda: registry.install(name="x", kind="plugin", version="1",
                                              content_hash="h", manifest={}),
                     lambda: registry.set_agent_enabled("weather", True)):
            with pytest.raises(RegistryPinRefused):
                call()
    with bind_profile_scope(tmp_path):  # the refusal changed nothing
        assert registry.get("weather") is not None


def test_with_no_scope_bound_the_pin_is_the_ambient_profile(registry, tmp_path, monkeypatch):
    with bind_profile_scope(tmp_path):
        registry.install(name="weather", kind="plugin", version="1.0", content_hash="h", manifest={})
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert registry.get("weather")["name"] == "weather"  # a local caller with no scope, as today
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "other"))
    with pytest.raises(RegistryPinRefused):
        registry.get("weather")


def test_a_scope_that_may_not_administer_is_refused_the_same_whatever_it_names(registry, tmp_path):
    with bind_profile_scope(tmp_path):
        registry.install(name="weather", kind="plugin", version="1.0", content_hash="h", manifest={})
    no_admin = Scope(LOCAL_TENANT, str(tmp_path), LOCAL_OWNER, owner=LOCAL_OWNER,
                     can_administer=False)
    with push_state(scope=no_admin):
        assert registry.get("weather")["name"] == "weather"  # reads still work
        assert registry.list() and registry.enabled_extensions() == []
        for call in (lambda: registry.install(name="x", kind="plugin", version="1",
                                              content_hash="h", manifest={}),
                     lambda: registry.set_enabled("weather", False),
                     lambda: registry.set_agent_enabled("weather", True),
                     lambda: registry.remove("weather")):
            with pytest.raises(RegistryAdminRefused):
                call()
        # and the refusal never says whether the entry exists
        with pytest.raises(RegistryAdminRefused) as present:
            registry.remove("weather")
        with pytest.raises(RegistryAdminRefused) as absent:
            registry.remove("absent")
        assert str(present.value) == str(absent.value)
    with bind_profile_scope(tmp_path):  # nothing changed
        assert registry.get("weather")["enabled"] is True
        assert registry.agent_enabled("weather") is False


def test_a_relative_profile_home_is_refused_at_open(tmp_path):
    with pytest.raises(RegistryPinRefused, match="absolute"):
        LocalExtensionRegistry(Path("relative/home"))


def test_the_kinds_are_the_user_extension_kinds():
    assert KINDS == frozenset({"plugin", "mcp_server"})
