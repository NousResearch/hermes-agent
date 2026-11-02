"""After a plugin's load times out (#108139), nothing its abandoned worker calls on ``ctx`` may land:
every registrar is a no-op, including the ones a ``PluginContext`` mixin contributes."""

from __future__ import annotations

from types import SimpleNamespace

from hermes_cli import plugins
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _is_registrar(name: str) -> bool:
    return name.startswith("register_") or name in {"subscribe", "on_unload"}


def test_every_registrar_is_ignored_after_an_abandoned_load():
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="late", key="late", source="user"), manager)
    ctx._abandon_load()
    registrars = sorted(name for name in dir(PluginContext)
                        if _is_registrar(name) and callable(getattr(PluginContext, name)))
    # Mixin-defined registrars are part of the surface the guard has to cover.
    assert {"register_tool", "register_hook", "register_review_destination"} <= set(registrars)

    unguarded = []
    for name in registrars:
        try:
            if getattr(ctx, name)() is not None:  # the guard returns before the registrar runs
                unguarded.append(name)
        except TypeError:  # the registrar itself ran and asked for its arguments
            unguarded.append(name)

    assert unguarded == []
    assert manager._ownership_ledger == {}


def test_registrars_a_mixin_contributes_are_guarded_too():
    calls = []

    class Mixin:
        def register_probe(self, value):
            calls.append(value)
            return value

    class Context(Mixin):
        manifest = SimpleNamespace(name="probe")

        def register_own(self, value):
            calls.append(value)
            return value

    plugins._ignore_registrars_after_abandoned_load(Context)
    live, abandoned = Context(), Context()
    abandoned._load_abandoned = True

    assert (live.register_probe("a"), live.register_own("b")) == ("a", "b")
    assert (abandoned.register_probe("c"), abandoned.register_own("d")) == (None, None)
    assert calls == ["a", "b"]
