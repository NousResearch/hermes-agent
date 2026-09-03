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


def test_multiplex_housekeeping_drains_each_profile_with_its_adapters(
    tmp_path, monkeypatch
):
    root_adapters = {}
    secondary_adapters = {}
    runner = SimpleNamespace(
        config=SimpleNamespace(multiplex_profiles=True),
        adapters=root_adapters,
        _profile_adapters={"secondary": secondary_adapters},
    )
    secondary_home = tmp_path / "secondary"
    calls = []

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
        ("drain", root_adapters),
        ("scope", secondary_home),
        ("drain", secondary_adapters),
    ]
