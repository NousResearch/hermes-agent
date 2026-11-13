"""PluginContext's explicit ``register_<category>_provider`` methods each run their own row of the
registrar table: same registry, kind, docstring and signature as the generated methods they replace."""

import importlib
import inspect

import pytest

from hermes_agent.hermes_cli import plugins_loader
from hermes_agent.hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from hermes_agent.hermes_cli.plugins_registrars import _SCOPED_PROVIDER_REGISTRARS, make_scoped_provider_registrar
from hermes_agent.hermes_constants import get_hermes_home


def _provider_for(base_ref: str, name: str):
    module, attr = base_ref.split(":")
    base = getattr(importlib.import_module(module), attr)
    cls = type(f"Probe{attr}", (base,), {"name": name, "prefix": f"{name}:"})
    cls.__abstractmethods__ = frozenset()
    return cls()


@pytest.mark.parametrize("row", _SCOPED_PROVIDER_REGISTRARS, ids=lambda row: row[0])
def test_each_registrar_registers_into_its_rows_registry(row):
    method, kind, registry_mod, base_ref = row[:4]
    name = "probe" + kind.replace("_", "")
    provider = _provider_for(base_ref, name)
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="probe", source="bundled"), manager)
    registry = importlib.import_module(registry_mod)

    handle = getattr(ctx, method)(provider)
    try:
        assert handle is not None and handle.kind == kind
        assert registry.snapshot_registration(name, scope=manager.scope_key) is provider
    finally:
        if handle is not None:
            handle.dispose()
    assert registry.snapshot_registration(name, scope=manager.scope_key) is None


@pytest.mark.parametrize(
    "row", [r for r in _SCOPED_PROVIDER_REGISTRARS if r[6].get("admit")], ids=lambda row: row[0]
)
def test_admission_is_judged_under_the_managers_home(row, tmp_path, monkeypatch):
    method, kind, _registry_mod, base_ref, *_rest, options = row
    module, attr = options["admit"].split(":")
    homes = []
    monkeypatch.setattr(
        importlib.import_module(module), attr, lambda manifest: homes.append(get_hermes_home()) or "refused"
    )
    owner_home = tmp_path / "owner-home"
    owner_home.mkdir()
    manager = PluginManager(scope_key=str(owner_home))
    ctx = PluginContext(PluginManifest(name="probe", source="user"), manager)

    assert getattr(ctx, method)(_provider_for(base_ref, "probeadmit" + kind.replace("_", ""))) is None
    assert homes == [manager.home_path] and manager.home_path != get_hermes_home()


def test_explicit_registrars_keep_the_row_docstring_and_parameter():
    for name, *_rest, doc, options in _SCOPED_PROVIDER_REGISTRARS:
        method = getattr(PluginContext, name)
        param = options.get("param", "provider")
        assert method.__doc__ == doc, name
        assert method.__qualname__ == f"PluginContext.{name}"
        assert str(inspect.signature(method)) == f"(self, {param})", name


def test_explicit_registrars_still_ignore_calls_after_an_abandoned_load(caplog):
    ctx = PluginContext.__new__(PluginContext)
    ctx._load_abandoned = True
    ctx.manifest = PluginManifest(name="late")
    for name, *_ in _SCOPED_PROVIDER_REGISTRARS:
        assert getattr(ctx, name)(object()) is None
    assert sum("after its load timed out" in r.getMessage() for r in caplog.records) == len(
        _SCOPED_PROVIDER_REGISTRARS
    )


@pytest.mark.parametrize("row", _SCOPED_PROVIDER_REGISTRARS, ids=lambda row: row[0])
def test_make_scoped_provider_registrar_still_builds_a_working_method(row, monkeypatch):
    """Main's factory stays importable and builds the same method main's setattr loop installed."""
    method, kind, registry_mod, base_ref, *_rest, doc, options = row
    built = make_scoped_provider_registrar(*row)
    assert (built.__name__, built.__qualname__, built.__doc__) == (method, f"PluginContext.{method}", doc)
    assert str(inspect.signature(built)) == f"(self, {options.get('param', 'provider')})"
    name = "probebuilt" + kind.replace("_", "")
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="probe", source="bundled"), manager)
    registry = importlib.import_module(registry_mod)
    real, transactions = plugins_loader.replacement_coordinator, []

    class RecordingCoordinator:
        def __getattr__(self, attr):
            return getattr(real, attr)

        def transaction(self):
            transactions.append(method)
            return real.transaction()

    monkeypatch.setattr(plugins_loader, "replacement_coordinator", RecordingCoordinator())
    provider = _provider_for(base_ref, name)
    handle = built(ctx, provider)
    try:
        assert handle is not None and handle.kind == kind and transactions == [method]
        assert registry.snapshot_registration(name, scope=manager.scope_key) is provider
    finally:
        if handle is not None:
            handle.dispose()
    assert registry.snapshot_registration(name, scope=manager.scope_key) is None
