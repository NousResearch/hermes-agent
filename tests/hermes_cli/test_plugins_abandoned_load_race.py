"""A registrar that was already running when its plugin load was abandoned must not stay live.

The loader marks the context abandoned and sweeps the ledger at the deadline, but a ``register_*`` call
that passed the abandoned-load guard just before it can still land afterwards: it is missing from the
sweep's snapshot and the registrar hands the plugin a live handle into a load that failed (#149326).
These cases drive the exact interleaving deterministically — the abandonment happens inside the write
the registrar was making — instead of racing two threads.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


def _context(manager: PluginManager, name: str = "late-registrar") -> PluginContext:
    return PluginContext(PluginManifest(name=name, key=name, source="user"), manager)


class _AbandonAfterAppend(list):
    """A manager registry list whose append models the deadline firing mid-registration."""

    def __init__(self, on_append: Callable[[], None]) -> None:
        super().__init__()
        self._on_append = on_append

    def append(self, item: object) -> None:
        super().append(item)
        self._on_append()


def _passthrough(payload=None):
    return payload


def test_registrar_running_when_the_load_is_abandoned_leaves_nothing_live():
    manager = PluginManager()
    ctx = _context(manager)
    middleware = _passthrough

    manager._middleware["llm_execution"] = _AbandonAfterAppend(ctx._abandon_load)
    handle = ctx.register_middleware("llm_execution", middleware)

    assert ctx._load_abandoned
    assert handle is None
    assert "llm_execution" not in manager._middleware
    assert manager._registration_order == []
    assert manager._ownership_ledger == {}


def test_subscription_that_lands_after_abandonment_is_removed():
    manager = PluginManager()
    ctx = _context(manager, "late-subscriber")
    real_subscribe = manager._subscribe_event

    def subscribe_then_abandon(owner: str, event: str, callback: Callable) -> None:
        real_subscribe(owner, event, callback)
        ctx._abandon_load()

    manager._subscribe_event = subscribe_then_abandon
    ctx.subscribe("late-subscriber:ping", lambda payload: None)

    assert ctx._load_abandoned
    assert manager._subscriptions == {}


def test_live_registration_is_untouched():
    manager = PluginManager()
    ctx = _context(manager, "live")
    middleware = _passthrough

    handle = ctx.register_middleware("llm_execution", middleware)

    assert handle is not None and handle.active
    assert middleware in manager._middleware["llm_execution"]
    assert [r.plugin_key for r in manager._registration_order] == ["live"]


def test_two_sweeps_of_one_plugin_release_each_handle_once():
    manager = PluginManager()
    ctx = _context(manager, "double-sweep")
    released: list[str] = []
    ctx._track("probe", "one", lambda: released.append("one"))
    ctx._track("probe", "two", lambda: released.append("two"))

    manager._dispose_plugin_registrations("double-sweep")
    # The abandoned load's own worker reaches the same sweep; remove-then-release makes it a no-op.
    manager._dispose_plugin_registrations("double-sweep")

    assert sorted(released) == ["one", "two"]
    assert manager._registration_order == []
