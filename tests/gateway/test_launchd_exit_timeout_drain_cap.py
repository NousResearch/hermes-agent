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

import asyncio
import signal
from types import SimpleNamespace

import pytest

from gateway.restart import (
    LAUNCHD_LABEL_ENV,
    LAUNCHD_STOP_CLEANUP_RESERVE_S,
    effective_stop_drain_timeout,
    effective_stop_watchdog_delay,
    is_gateway_supervisor_process,
    launchd_service_label,
    read_launchd_exit_timeout_s,
    resolve_launchd_capped_drain,
)
from gateway.shutdown_watchdog import resolve_shutdown_watchdog_delay

_CAPPED = 60.0 - LAUNCHD_STOP_CLEANUP_RESERVE_S


def test_capped_drain_fits_inside_launchd_budget_minus_reserve():
    # The incident shape: configured 180s, launchd clamps to 60s.
    assert resolve_launchd_capped_drain(180.0, 60.0) == _CAPPED
    # Never extends a short drain; no launchd budget leaves the configured drain alone.
    assert resolve_launchd_capped_drain(20.0, 60.0) == 20.0
    assert resolve_launchd_capped_drain(180.0, None) == 180.0


def _direct(label):
    """launchd's own job process: the label sits in XPC_SERVICE_NAME."""
    return {"XPC_SERVICE_NAME": label}


def _grandchild(label):
    """Under the generated plist the gateway is a grandchild of the stderr-timestamp wrapper: launchd
    stamps XPC_SERVICE_NAME only on the wrapper, the grandchild reads "0" and finds the job label only
    in the wrapper's HERMES_LAUNCHD_LABEL re-export."""
    return {"XPC_SERVICE_NAME": "0", LAUNCHD_LABEL_ENV: label}


@pytest.mark.parametrize(
    "platform, environ, expected",
    [
        ("darwin", _direct("ai.hermes.gateway"), _CAPPED),
        ("darwin", _grandchild("ai.hermes.gateway"), _CAPPED),
        # App-coalition label (IDE integrated terminal) is not our job: no budget, drain unchanged —
        # whichever variable carries it.
        ("darwin", _direct("application.com.example.ide.123"), 180.0),
        ("darwin", _grandchild("application.com.example.ide.123"), 180.0),
        # No label at all (foreground start, or a wrapper older than the forward) is not launchd-owned.
        ("darwin", {"XPC_SERVICE_NAME": "0"}, 180.0),
        # launchd is darwin-only (same predicate as control_socket): a leaked label elsewhere is ignored.
        ("linux", _direct("ai.hermes.gateway"), 180.0),
        ("linux", _grandchild("ai.hermes.gateway"), 180.0),
    ],
    ids=["darwin-direct", "darwin-grandchild", "darwin-direct-app", "darwin-grandchild-app",
         "darwin-unlabelled", "linux-direct", "linux-grandchild"],
)
def test_launchd_reader_yields_a_budget_only_for_hermes_jobs(platform, environ, expected):
    """End to end from the process environment to the stop drain: the reader sizes the drain to the
    live ExitTimeOut only for an ``ai.hermes`` job on darwin, seen directly or through the wrapper's
    re-export. Platform is data, not the host (the launchctl call is faked)."""
    fake_run = lambda *a, **k: SimpleNamespace(returncode=0, stdout="exit timeout = 60\n")  # noqa: E731
    budget = read_launchd_exit_timeout_s(environ=environ, uid=501, run=fake_run, platform=platform)
    assert resolve_launchd_capped_drain(180.0, budget) == expected


def test_forwarded_label_marks_grandchild_supervised_for_every_reader():
    """One seam: the restart route and the control-socket declaration see the same launchd identity
    the drain cap does, so a grandchild is not 'manual' to one reader and 'launchd' to another."""
    grandchild_env = _grandchild("ai.hermes.gateway")
    assert launchd_service_label(grandchild_env, platform="darwin") == "ai.hermes.gateway"
    assert is_gateway_supervisor_process(grandchild_env) is True
    assert is_gateway_supervisor_process({"XPC_SERVICE_NAME": "0"}) is False
    assert is_gateway_supervisor_process(_grandchild("application.com.x.1")) is False


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


@pytest.mark.parametrize("takeover", [False, True])
def test_sigterm_handler_marks_stop_as_signal_driven_unless_planned_takeover(monkeypatch, takeover):
    import gateway.run as run_mod
    import gateway.shutdown_forensics as forensics
    import gateway.status as status

    monkeypatch.setattr(status, "consume_takeover_marker_for_self", lambda: takeover)
    monkeypatch.setattr(status, "consume_planned_stop_marker_for_self", lambda: False)
    monkeypatch.setattr(forensics, "snapshot_shutdown_context", lambda *a, **k: None)
    monkeypatch.setattr(run_mod.asyncio, "create_task", lambda coro: coro.close())
    runner = _runner(drain=180.0, launchd=60.0, by_signal=False)
    runner.stop = lambda: asyncio.sleep(0)
    run_mod._start_gateway_make_shutdown_signal_handler(runner, [False])(signal.SIGTERM)
    assert runner._stop_requested_by_signal is (not takeover)
    assert effective_stop_drain_timeout(runner) == (180.0 if takeover else 50.0)
