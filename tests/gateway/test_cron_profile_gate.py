"""The gateway ticker stands down for a profile whose OWN gateway process ticks it.

The tick set is every profile regardless of ``gateway.multiplex_profiles``, but that flag also
selects the deployment where each profile runs its own gateway (the s6 per-profile services). With
no gate, profile B's gateway and the launch gateway both tick B's store: the per-home
``cron/.tick.lock`` stops a simultaneous double-run, but which process wins is a race, and when
the launch gateway wins, B's delivery leaves through ``SharedRouteAdapters``/fail-closed instead
of B's own live adapters.

The gate must compare against THIS process's pid: the launch gateway holds its own home's
``gateway.pid`` and publishes every served profile in ``served_profiles``, so a bare liveness
answer says "running" for every home it serves and would stand cron down host-wide.
"""

from __future__ import annotations

import asyncio
import os
import types

import gateway.run as gw_run
from gateway.status import GatewayLiveness


def _liveness(monkeypatch, liveness):
    from gateway import status as gateway_status

    monkeypatch.setattr(gateway_status, "resolve_gateway_liveness", lambda **_kw: liveness)


class TestCronProfileGate:
    def test_stands_down_for_a_profile_with_its_own_gateway(self, tmp_path, monkeypatch):
        """Another process owns this home's gateway -> this ticker must not visit it."""
        _liveness(monkeypatch, GatewayLiveness(running=True, pid=os.getpid() + 1, source="pid"))
        assert gw_run._cron_profile_gate("b", tmp_path / "home-b") is False

    def test_keeps_ticking_a_home_this_process_serves(self, tmp_path, monkeypatch):
        """Our OWN pid answering (launch home, or a served profile via the multiplexer rung) must
        never stand this ticker down — that would stop cron for every profile at once."""
        for source in ("pid", "multiplexer", "runtime_status"):
            _liveness(monkeypatch, GatewayLiveness(running=True, pid=os.getpid(), source=source))
            assert gw_run._cron_profile_gate("b", tmp_path / "home-b") is True, source

    def test_keeps_ticking_a_home_with_no_gateway(self, tmp_path, monkeypatch):
        _liveness(monkeypatch, GatewayLiveness(running=False, pid=None, source="none"))
        assert gw_run._cron_profile_gate("b", tmp_path / "home-b") is True

    def test_probe_failure_ticks_rather_than_silently_dropping_cron(self, tmp_path, monkeypatch):
        """A raising probe must not be read as "another gateway owns it"."""
        from gateway import status as gateway_status

        def _boom(**_kw):
            raise RuntimeError("probe exploded")

        monkeypatch.setattr(gateway_status, "resolve_gateway_liveness", _boom)
        assert gw_run._cron_profile_gate("b", tmp_path / "home-b") is True


def test_gateway_passes_a_profile_gate_to_the_cron_ticker(tmp_path, monkeypatch):
    """Wiring: an unpassed gate gates nothing. Capture what the gateway hands the ticker."""
    from cron import scheduler_provider

    home_b = tmp_path / "home-b"
    home_b.mkdir()
    captured: dict = {}

    class _Thread:
        def __init__(self, target, args=(), kwargs=None, stop_event=None):
            captured.update(kwargs or {})

        def start(self):
            pass

    monkeypatch.setattr(
        gw_run, "_cron_tick_profile_homes", lambda _cfg: [("default", tmp_path), ("b", home_b)])
    monkeypatch.setattr(
        scheduler_provider, "scheduler_for_profile_mode",
        lambda _p, multiplex_profiles=False: scheduler_provider.InProcessCronScheduler())
    from cron import scheduler_thread

    monkeypatch.setattr(scheduler_thread, "SupervisedTickerThread", _Thread)
    monkeypatch.setattr(gw_run, "_start_gateway_housekeeping_thread", lambda *_a, **_k: None,
                        raising=False)

    runner = types.SimpleNamespace(
        config={}, adapters={}, _profile_adapters=None, _primary_profile_name="default",
        _draining=False, _external_drain_active=False)

    async def _go():
        return gw_run._start_gateway_start_cron_and_housekeeping(runner)

    asyncio.run(_go())

    gate = captured.get("profile_gate")
    assert callable(gate), "the gateway must pass a profile_gate to the cron ticker"
    # And the gate it passes is the stand-down one, not a pass-everything stub.
    from gateway import status as gateway_status

    monkeypatch.setattr(
        gateway_status, "resolve_gateway_liveness",
        lambda **_kw: GatewayLiveness(running=True, pid=os.getpid() + 1, source="pid"))
    assert gate("b", home_b) is False
