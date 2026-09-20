"""Signal-driven stops under launchd must fit the live ``ExitTimeOut``.

launchd's per-user (gui) domain clamps ``ExitTimeOut`` (measured 60s on
macOS 26: plist 215 -> live 60). A gateway configured with a longer
``restart_drain_timeout`` drains past that budget and is SIGKILLed mid
SQLite teardown — the unclean-exit half of the state.db corruption class.
The gateway therefore reads the live value at boot and caps only the
signal-driven stop drain (in-band SIGUSR1 restarts and --replace
takeovers are not launchd-timed and keep the configured drain).
"""

from __future__ import annotations

from types import SimpleNamespace

from gateway.restart import (
    LAUNCHD_STOP_CLEANUP_RESERVE_S,
    effective_stop_drain_timeout,
    effective_stop_watchdog_delay,
    resolve_launchd_capped_drain,
)
from gateway.shutdown_watchdog import resolve_shutdown_watchdog_delay


def test_capped_drain_fits_inside_launchd_budget_minus_reserve():
    # The incident shape: configured 180s, launchd clamps to 60s.
    assert resolve_launchd_capped_drain(180.0, 60.0) == 60.0 - LAUNCHD_STOP_CLEANUP_RESERVE_S
    # Never extends a short drain; no launchd budget leaves the configured drain alone.
    assert resolve_launchd_capped_drain(20.0, 60.0) == 20.0
    assert resolve_launchd_capped_drain(180.0, None) == 180.0


def _runner(*, drain: float, launchd: float | None, by_signal: bool):
    return SimpleNamespace(
        _restart_drain_timeout=drain, _launchd_exit_timeout_s=launchd, _stop_requested_by_signal=by_signal,
    )


def test_effective_drain_capped_only_for_signal_stops_under_launchd():
    signal_stop = _runner(drain=180.0, launchd=60.0, by_signal=True)
    assert effective_stop_drain_timeout(signal_stop) == 50.0
    # In-band restart (SIGUSR1 → after-turn → stop()) is not launchd-timed.
    assert effective_stop_drain_timeout(_runner(drain=180.0, launchd=60.0, by_signal=False)) == 180.0
    # Not launchd-owned (systemd, s6, foreground): configured drain stands.
    assert effective_stop_drain_timeout(_runner(drain=180.0, launchd=None, by_signal=True)) == 180.0
    # The thread watchdog (drain + grace) must also fire before launchd's SIGKILL.
    leash = resolve_shutdown_watchdog_delay(effective_stop_drain_timeout(signal_stop))
    assert effective_stop_watchdog_delay(signal_stop, leash) < 60.0 < leash
