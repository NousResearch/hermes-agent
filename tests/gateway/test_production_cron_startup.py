from __future__ import annotations

import threading

import pytest

from cron.scheduler_provider import InProcessCronScheduler
from gateway.production_cron_startup import (
    ProductionCronActivationGate,
    ProductionCronStartupError,
)


@pytest.fixture(autouse=True)
def _ticker_heartbeats(monkeypatch: pytest.MonkeyPatch) -> list[bool | None]:
    from cron import jobs

    observed: list[bool | None] = []
    monkeypatch.setattr(
        jobs,
        "record_ticker_heartbeat",
        lambda success=None: observed.append(success),
    )
    return observed


def _gate(
    *,
    provider: InProcessCronScheduler | None = None,
    stop_event: threading.Event | None = None,
) -> ProductionCronActivationGate:
    return ProductionCronActivationGate(
        provider=provider or InProcessCronScheduler(),
        stop_event=stop_event or threading.Event(),
        adapters={"exact": object()},
        loop=object(),
        interval=1,
    )


def test_gate_parks_without_entering_provider_then_activates(
    monkeypatch: pytest.MonkeyPatch,
    _ticker_heartbeats: list[bool | None],
) -> None:
    entered = threading.Event()
    observed: dict[str, object] = {}

    def fake_start(self, stop_event, *, adapters=None, loop=None, interval=60):
        observed.update(
            stop_event=stop_event,
            adapters=adapters,
            loop=loop,
            interval=interval,
        )
        entered.set()
        stop_event.wait(1)

    monkeypatch.setattr(InProcessCronScheduler, "start", fake_start)
    gate = _gate()
    try:
        gate.prepare(timeout=1)
        assert gate.parked_heartbeats >= 2
        assert _ticker_heartbeats == [None]
        assert gate.thread is not None and gate.thread.is_alive()
        assert entered.is_set() is False
        assert gate.activated is False

        gate.activate(timeout=1)
        assert entered.wait(1)
        assert gate.activated is True
        assert observed == {
            "stop_event": gate.stop_event,
            "adapters": gate.adapters,
            "loop": gate.loop,
            "interval": 1,
        }
    finally:
        gate.stop()
        if gate.thread is not None:
            gate.thread.join(timeout=1)
    assert gate.finished is True


def test_stop_before_activation_never_enters_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    monkeypatch.setattr(
        InProcessCronScheduler,
        "start",
        lambda *_args, **_kwargs: entered.set(),
    )
    gate = _gate()
    gate.prepare(timeout=1)
    gate.stop()
    assert gate.thread is not None
    gate.thread.join(timeout=1)
    assert gate.finished is True
    assert entered.is_set() is False
    with pytest.raises(
        ProductionCronStartupError,
        match="already activated|stop was requested",
    ):
        gate.activate(timeout=1)


def test_gate_rejects_non_exact_provider() -> None:
    class DerivedProvider(InProcessCronScheduler):
        pass

    with pytest.raises(
        ProductionCronStartupError,
        match="exact built-in provider",
    ):
        _gate(provider=DerivedProvider())


def test_gate_state_transitions_are_one_shot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        InProcessCronScheduler,
        "start",
        lambda _self, stop_event, **_kwargs: stop_event.wait(1),
    )
    gate = _gate()
    try:
        gate.prepare(timeout=1)
        with pytest.raises(ProductionCronStartupError, match="already prepared"):
            gate.prepare(timeout=1)
        gate.activate(timeout=1)
        with pytest.raises(ProductionCronStartupError, match="already activated"):
            gate.activate(timeout=1)
    finally:
        gate.stop()
        if gate.thread is not None:
            gate.thread.join(timeout=1)


@pytest.mark.parametrize("timeout", [0, -1, None, "1"])
def test_gate_rejects_invalid_timeouts(timeout) -> None:
    gate = _gate()
    with pytest.raises(ValueError, match="timeout"):
        gate.prepare(timeout=timeout)


def test_gate_surfaces_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = threading.Event()

    def fail_start(*_args, **_kwargs):
        failed.set()
        raise RuntimeError("boom")

    monkeypatch.setattr(InProcessCronScheduler, "start", fail_start)
    gate = _gate()
    gate.prepare(timeout=1)
    # The activation acknowledgement and provider failure can race.  Once the
    # provider has failed, a second lifecycle operation must surface it rather
    # than treating the dead thread as healthy.
    try:
        try:
            gate.activate(timeout=1)
        except ProductionCronStartupError:
            pass
        assert failed.wait(1)
        assert gate.thread is not None
        gate.thread.join(timeout=1)
        assert gate.finished is True
        with pytest.raises(ProductionCronStartupError):
            gate.activate(timeout=1)
    finally:
        gate.stop()
