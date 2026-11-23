"""After a plugin's load times out, nothing its abandoned worker calls on ``ctx`` may land: every
registrar is a no-op, including the ones a ``PluginContext`` mixin contributes
(``register_idle_task`` on ``PluginIdleTaskCtxMixin``)."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from hermes_cli import plugins_load_guard
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from hermes_cli.plugins_idle_tasks import registered_idle_tasks, run_plugin_idle_tasks


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _is_registrar(name: str) -> bool:
    return name.startswith("register_") or name in {"subscribe", "on_unload"}


def _abandoned():
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="late", key="late", source="user"), manager)
    ctx._abandon_load()
    return manager, ctx


def test_every_registrar_is_ignored_after_an_abandoned_load(home):
    manager, ctx = _abandoned()
    registrars = sorted(name for name in dir(PluginContext)
                        if _is_registrar(name) and callable(getattr(PluginContext, name)))
    # Mixin-defined registrars are part of the surface the guard has to cover.
    assert {"register_tool", "register_hook", "subscribe", "on_unload", "register_idle_task"} <= set(registrars)

    unguarded = []
    for name in registrars:
        try:
            if getattr(ctx, name)() is not None:  # the guard returns before the registrar runs
                unguarded.append(name)
        except TypeError:  # the registrar itself ran and asked for its arguments
            unguarded.append(name)

    assert unguarded == []
    assert manager._ownership_ledger == {}


def test_an_abandoned_load_registers_no_idle_task_and_the_poll_runs_nothing(home):
    manager, ctx = _abandoned()
    calls = []

    assert ctx.register_idle_task("late", calls.append, interval_hours=1, first_run="immediate") is None

    assert registered_idle_tasks(manager) == {}
    assert run_plugin_idle_tasks(idle_for_seconds=10 * 3600.0, manager=manager, now=datetime.now(UTC)) == {}
    assert calls == []


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

    plugins_load_guard.ignore_registrars_after_abandoned_load(Context)
    live, abandoned = Context(), Context()
    abandoned._load_abandoned = True

    assert (live.register_probe("a"), live.register_own("b")) == ("a", "b")
    assert (abandoned.register_probe("c"), abandoned.register_own("d")) == (None, None)
    assert calls == ["a", "b"]
