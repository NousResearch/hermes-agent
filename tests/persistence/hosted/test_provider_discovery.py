"""``persistence.provider`` names a provider beyond the built-in SQLite: a bundled ``plugins/persistence/<name>/``
package or an installed distribution's ``hermes_agent.persistence`` entry point. Resolution is by explicit name only:
an unknown, failing or doubly-claimed name refuses, and nothing ever falls back to SQLite."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

import persistence_assembly
import plugins.persistence as registry
from hermes_scope import Principal, Scope, pop_state, push_state

_PLUGIN = '''
from types import SimpleNamespace

CLOSED = []
BUILT = []


class FakeStore:
    def __init__(self, backend, scope):
        self.backend, self.scope = backend, scope
        self.capabilities = backend.capabilities

    def ready(self):
        return None


class FakeBackend:
    def __init__(self, stores, read_only, read_patience_s):
        self.stores, self.read_only, self.read_patience_s = dict(stores), read_only, read_patience_s
        self.capabilities = SimpleNamespace(scope_pinning={pinning!r}, physical_admin=False)
        BUILT.append(self)
{isolating}
    def open(self, scope):
        return FakeStore(self, scope)

    def close(self):
        CLOSED.append(self)

{failure}
def register(ctx):
    ctx.register_persistence_backend(
        lambda stores, *, read_only, read_patience_s: FakeBackend(stores, read_only, read_patience_s))
'''

_ISOLATING = {
    "pass": "    def assert_isolating(self):\n        return None\n",
    "raise": "    def assert_isolating(self):\n        from persistence import IsolationUnsupported\n"
             "        raise IsolationUnsupported('role bypasses row security')\n",
    "absent": "",
    "not-callable": "    assert_isolating = None\n",
}


def plugin_source(*, pinning: bool = False, isolating: str = "absent", failure: str = "") -> str:
    """The source of a tiny fake persistence plugin; ``failure`` is extra module-level code (say, a raise)."""
    return _PLUGIN.format(pinning=pinning, isolating=_ISOLATING[isolating], failure=failure)


@pytest.fixture
def bundled_root(tmp_path, monkeypatch):
    """The bundled persistence plugin root, moved to an empty temp dir; plugin modules loaded from it are dropped."""
    root = tmp_path / "bundled-persistence"
    root.mkdir()
    monkeypatch.setattr(registry, "_PERSISTENCE_PLUGINS_DIR", root)
    before = set(sys.modules)
    yield root
    for name in set(sys.modules) - before:
        if name.startswith("plugins.persistence."):
            sys.modules.pop(name, None)


@pytest.fixture
def entry_points(monkeypatch):
    """Installed ``hermes_agent.persistence`` entry points: append ``(name, loaded object)`` pairs."""
    installed: list[tuple[str, object]] = []

    def fake_entry_points(**selection):
        eps = [SimpleNamespace(name=name, group="hermes_agent.persistence", value=f"fake_{name}:plugin",
                               load=(lambda obj=obj: obj() if callable(obj) and not hasattr(obj, "register") else obj))
               for name, obj in installed]
        group = selection.get("group")
        return [ep for ep in eps if group is None or ep.group == group]

    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    return installed


def add_bundled(root: Path, name: str, source: str) -> Path:
    plugin_dir = root / name
    plugin_dir.mkdir()
    (plugin_dir / "__init__.py").write_text(textwrap.dedent(source), encoding="utf-8")
    return plugin_dir


def plugin_module(name: str):
    return sys.modules[f"plugins.persistence.{name}"]


def configured_home(tmp_path: Path, provider: str, *, name: str = "target") -> Path:
    home = tmp_path / name
    home.mkdir()
    (home / "config.yaml").write_text(f"persistence:\n  provider: {provider}\n", encoding="utf-8")
    return home


def entry_point_plugin(source: str) -> SimpleNamespace:
    """An entry point's loaded object: a module-like namespace executed from ``source``."""
    namespace: dict = {}
    exec(compile(textwrap.dedent(source), "<entry-point plugin>", "exec"), namespace)
    return SimpleNamespace(**{k: v for k, v in namespace.items() if not k.startswith("__")})


def hosted_scope() -> Scope:
    return Scope("tenant-a", "agent-a", Principal("member-a"), owner=Principal("member-a"), can_administer=True)


def test_a_bundled_provider_resolves_by_name_and_backs_the_profiles_handles(tmp_path, bundled_root, entry_points):
    add_bundled(bundled_root, "fakedir", plugin_source())
    home = configured_home(tmp_path, "fakedir")
    assert persistence_assembly._provider_for_home(home) == "fakedir"
    scope = persistence_assembly.profile_scope(home=home)
    with persistence_assembly.open_persistence(scope) as handle:
        handle.ready()
        backend = plugin_module("fakedir").BUILT[-1]
        assert handle.capabilities is backend.capabilities
        assert backend.stores == {scope.hermes_home_key: Path(scope.hermes_home_key) / "state.db"}
        assert (backend.read_only, backend.read_patience_s) == (False, None)
    assert plugin_module("fakedir").CLOSED == [backend]
    with persistence_assembly.open_persistence_reader(home, patience_s=1.5) as reader:
        reader.ready()
        assert (plugin_module("fakedir").BUILT[-1].read_only, plugin_module("fakedir").BUILT[-1].read_patience_s) == (True, 1.5)


def test_an_entry_point_provider_resolves_by_name(tmp_path, bundled_root, entry_points):
    plugin = entry_point_plugin(plugin_source())
    entry_points.append(("fakeep", plugin))
    home = configured_home(tmp_path, "fakeep")
    backend = persistence_assembly._assemble(persistence_assembly._provider_for_home(home), {"k": tmp_path / "x"})
    assert plugin.BUILT == [backend] and backend.stores == {"k": tmp_path / "x"}


def test_the_built_in_sqlite_needs_no_discovery(tmp_path, bundled_root, entry_points, monkeypatch):
    from persistence.sqlite import SQLiteBackend

    add_bundled(bundled_root, "sqlite", plugin_source())
    entry_points.append(("sqlite", entry_point_plugin(plugin_source())))
    for home in (tmp_path / "unconfigured", configured_home(tmp_path, "sqlite")):
        home.mkdir(exist_ok=True)
        provider = persistence_assembly._provider_for_home(home)
        backend = persistence_assembly._assemble(provider, {"k": home / "state.db"})
        assert provider == "sqlite" and isinstance(backend, SQLiteBackend)
        backend.close()


def test_an_unknown_provider_is_refused_with_its_name(tmp_path, bundled_root, entry_points):
    add_bundled(bundled_root, "other", plugin_source())
    entry_points.append(("another", entry_point_plugin(plugin_source())))
    home = configured_home(tmp_path, "missing")
    with pytest.raises(ValueError, match="unknown persistence provider 'missing'"):
        persistence_assembly._provider_for_home(home)
    with pytest.raises(ValueError, match="unknown persistence provider 'missing'"):
        persistence_assembly._assemble("missing", {"k": tmp_path / "x"})


def test_a_name_claimed_by_a_directory_and_an_entry_point_is_refused(tmp_path, bundled_root, entry_points):
    add_bundled(bundled_root, "twice", plugin_source())
    entry_points.append(("twice", entry_point_plugin(plugin_source())))
    with pytest.raises(ValueError, match="persistence provider 'twice' .*more than once"):
        persistence_assembly._provider_for_home(configured_home(tmp_path, "twice"))
    with pytest.raises(ValueError, match="persistence provider 'twice' .*more than once"):
        persistence_assembly._assemble("twice", {"k": tmp_path / "x"})


def test_a_name_claimed_by_two_entry_points_is_refused(tmp_path, bundled_root, entry_points):
    entry_points.extend([("twin", entry_point_plugin(plugin_source())), ("twin", entry_point_plugin(plugin_source()))])
    with pytest.raises(ValueError, match="persistence provider 'twin' .*more than once"):
        persistence_assembly._assemble("twin", {"k": tmp_path / "x"})


_REDEFINED = "\n\ndef register(ctx):\n    {body}\n"


@pytest.mark.parametrize("source", [
    plugin_source(failure="raise RuntimeError('driver missing')\n"),
    plugin_source() + _REDEFINED.format(body="raise RuntimeError('driver missing')"),
    plugin_source() + _REDEFINED.format(body="pass"),
    plugin_source().replace("def register(ctx):", "def not_register(ctx):"),
], ids=["import-raises", "register-raises", "registers-nothing", "no-register"])
def test_a_provider_that_fails_to_load_is_refused_never_replaced_by_sqlite(tmp_path, bundled_root, entry_points,
                                                                          source):
    add_bundled(bundled_root, "broken", source)
    home = configured_home(tmp_path, "broken")
    scope = persistence_assembly.profile_scope(home=home)
    with persistence_assembly.open_persistence(scope) as handle:
        with pytest.raises(ValueError, match="persistence provider 'broken' failed to load"):
            handle.ready()
    assert not (home / "state.db").exists()


def test_an_entry_point_that_fails_to_load_is_refused(tmp_path, bundled_root, entry_points):
    def explode():
        raise ImportError("no driver")

    entry_points.append(("brokenep", explode))
    with pytest.raises(ValueError, match="persistence provider 'brokenep' failed to load"):
        persistence_assembly._assemble("brokenep", {"k": tmp_path / "x"})


def test_user_plugin_directories_are_never_consulted(tmp_path, bundled_root, entry_points, monkeypatch):
    from hermes_constants import get_hermes_home

    from plugins import plugin_loader

    user_plugins = get_hermes_home() / "plugins"
    (user_plugins / "persistence").mkdir(parents=True)
    add_bundled(user_plugins / "persistence", "userland", plugin_source())
    add_bundled(user_plugins, "userland2", plugin_source())

    def consulted():
        raise AssertionError("user plugins dir consulted")

    monkeypatch.setattr(plugin_loader, "user_plugins_dir", consulted)
    for name in ("userland", "userland2"):
        with pytest.raises(ValueError, match=f"unknown persistence provider '{name}'"):
            persistence_assembly._provider_for_home(configured_home(tmp_path, name, name=f"home-{name}"))
    assert not [module for module in sys.modules if "userland" in module]


@pytest.mark.parametrize("config_text", ["persistence: [unclosed\n", "- just\n- a list\n"])
def test_an_unreadable_foreign_config_is_sqlite_locally_and_refused_when_hosted(tmp_path, config_text):
    home = tmp_path / "foreign"
    home.mkdir()
    (home / "config.yaml").write_text(config_text, encoding="utf-8")
    assert persistence_assembly._provider_for_home(home) == "sqlite"
    token = push_state(scope=hosted_scope())
    try:
        assert persistence_assembly.bound_hosted_scope() is not None
        with pytest.raises(ValueError, match="config.yaml"):
            persistence_assembly._provider_for_home(home)
    finally:
        pop_state(token)
    assert persistence_assembly._provider_for_home(home) == "sqlite"


def test_a_config_the_process_cannot_open_is_refused_when_hosted(tmp_path, monkeypatch):
    home = tmp_path / "foreign"
    home.mkdir()
    (home / "config.yaml").write_text("persistence:\n  provider: sqlite\n", encoding="utf-8")
    import utils

    def unreadable(_fh):
        raise PermissionError("denied")

    monkeypatch.setattr(utils, "fast_safe_load", unreadable)
    assert persistence_assembly._provider_for_home(home) == "sqlite"
    token = push_state(scope=hosted_scope())
    try:
        with pytest.raises(ValueError, match="config.yaml"):
            persistence_assembly._provider_for_home(home)
    finally:
        pop_state(token)


def test_sqlite_only_helpers_refuse_for_a_held_plugin_provider(tmp_path, bundled_root, entry_points):
    from persistence import CapabilityUnsupported

    add_bundled(bundled_root, "fakedir", plugin_source())
    scope = persistence_assembly.profile_scope(home=configured_home(tmp_path, "fakedir"))
    with persistence_assembly.open_persistence(scope, eager=True):
        assert persistence_assembly.store_location(scope) is None
        with pytest.raises(CapabilityUnsupported):
            persistence_assembly.storage_health(scope)
        with pytest.raises(CapabilityUnsupported):
            persistence_assembly.store_exists(scope)
    sqlite_scope = persistence_assembly.profile_scope(home=configured_home(tmp_path, "sqlite", name="local"))
    assert persistence_assembly.store_location(sqlite_scope) == Path(sqlite_scope.hermes_home_key) / "state.db"
    assert persistence_assembly.store_exists(sqlite_scope) is False


def test_store_location_of_a_home_not_on_disk_reads_no_config(tmp_path, monkeypatch):
    """A home that does not exist yet holds no config naming another provider: the file-level helpers answer for the
    built-in store without loading (or creating) the process config, as they did before providers were named."""
    missing = tmp_path / "root" / "profiles" / "research"
    monkeypatch.setenv("HERMES_HOME", str(missing))
    scope = persistence_assembly.profile_scope(home=missing)
    assert persistence_assembly.store_location(scope) == missing / "state.db"
    assert not missing.exists()


def test_the_registry_never_constructs_the_built_in():
    """The storage layer's owner (``persistence_assembly``) constructs SQLite; the plugin registry imports no backend."""
    with pytest.raises(ValueError, match="built in"):
        registry.backend_factory("sqlite")
