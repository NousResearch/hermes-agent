"""Gateway-independent draining of restart-safe cron deliveries."""

from contextlib import contextmanager
from types import SimpleNamespace

import cron.scheduler as scheduler
import gateway.run as gateway_run


class _OneTickStopEvent:
    def __init__(self):
        self.waited = False

    def is_set(self):
        return self.waited

    def wait(self, timeout=None):
        self.waited = True
        return True


def test_gateway_housekeeping_drains_cron_delivery_with_live_adapters(monkeypatch):
    adapters = {"discord": object()}
    loop = object()
    calls = []
    monkeypatch.setattr(
        scheduler,
        "drain_delivery_queue",
        lambda live_adapters, live_loop: calls.append((live_adapters, live_loop)),
        raising=False,
    )

    gateway_run._start_gateway_housekeeping(
        _OneTickStopEvent(), adapters=adapters, loop=loop, interval=0
    )

    assert calls == [(adapters, loop)]


def test_gateway_housekeeping_drains_cron_delivery_without_connected_adapters(monkeypatch):
    adapters = {}
    loop = object()
    calls = []
    monkeypatch.setattr(
        scheduler,
        "drain_delivery_queue",
        lambda live_adapters, live_loop: calls.append((live_adapters, live_loop)),
        raising=False,
    )

    gateway_run._start_gateway_housekeeping(
        _OneTickStopEvent(), adapters=adapters, loop=loop, interval=0
    )

    assert calls == [(adapters, loop)]


def test_multiplex_housekeeping_scopes_primary_and_drains_each_profile(
    tmp_path, monkeypatch
):
    root_adapters = {}
    secondary_adapters = {}
    runner = SimpleNamespace(
        config=SimpleNamespace(multiplex_profiles=True),
        adapters=root_adapters,
        _profile_adapters={"secondary": secondary_adapters},
    )
    root_home = tmp_path / "root"
    secondary_home = tmp_path / "secondary"
    calls = []

    monkeypatch.setattr(gateway_run, "get_hermes_home", lambda: root_home)

    monkeypatch.setattr(
        gateway_run,
        "_handoff_watch_scopes",
        lambda _runner: [(None, None), ("secondary", secondary_home)],
    )

    @contextmanager
    def fake_scope(home):
        calls.append(("scope", home))
        yield

    monkeypatch.setattr(gateway_run, "_profile_runtime_scope", fake_scope)
    monkeypatch.setattr(
        scheduler,
        "drain_delivery_queue",
        lambda adapters, loop: calls.append(("drain", adapters)),
    )

    gateway_run._start_gateway_housekeeping(
        _OneTickStopEvent(),
        adapters=root_adapters,
        loop=object(),
        interval=0,
        runner=runner,
    )

    assert calls == [
        ("scope", root_home),
        ("drain", root_adapters),
        ("scope", secondary_home),
        ("drain", secondary_adapters),
    ]


def test_multiplex_housekeeping_uses_primary_routes_for_credentialless_satellite(
    tmp_path, monkeypatch
):
    root_adapters = {"slack": object()}
    secondary_home = tmp_path / "secondary"
    runner = SimpleNamespace(
        config=SimpleNamespace(multiplex_profiles=True),
        adapters=root_adapters,
        _profile_adapters={"secondary": {}},
    )
    calls = []
    routed = object()

    monkeypatch.setattr(
        gateway_run,
        "_handoff_watch_scopes",
        lambda _runner: [(None, None), ("secondary", secondary_home)],
    )

    @contextmanager
    def fake_scope(_home):
        yield

    class FakeSharedRouteAdapters:
        def __new__(cls, adapters, routes):
            calls.append(("routed", adapters, routes))
            return routed

    monkeypatch.setattr(gateway_run, "_profile_runtime_scope", fake_scope)
    monkeypatch.setattr(scheduler, "SharedRouteAdapters", FakeSharedRouteAdapters)
    monkeypatch.setattr(
        scheduler,
        "_primary_profile_routes_for_current_home",
        lambda: ["route-to-secondary"],
    )
    monkeypatch.setattr(
        scheduler,
        "drain_delivery_queue",
        lambda adapters, _loop: calls.append(("drain", adapters)),
    )

    gateway_run._drain_restart_safe_cron_deliveries(
        root_adapters, object(), runner
    )

    assert calls == [
        ("drain", root_adapters),
        ("routed", root_adapters, ["route-to-secondary"]),
        ("drain", routed),
    ]
