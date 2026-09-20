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

import subprocess

import pytest

from gateway.restart import (
    LAUNCHD_STOP_CLEANUP_RESERVE_S,
    launchd_service_label,
    parse_launchd_exit_timeout,
    read_launchd_exit_timeout_s,
    resolve_launchd_capped_drain,
)
from gateway.shutdown_watchdog import (
    DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S,
    resolve_shutdown_watchdog_delay,
)

_PRINT_OUTPUT = """\
ai.hermes.gateway-aegis = {
\tactive count = 1
\tpath = /Users/x/Library/LaunchAgents/ai.hermes.gateway-aegis.plist
\tstate = running

\tprogram = /bin/sh
\targuments = {
\t\t/bin/sh
\t\t-lc
\t\texec hermes gateway run
\t}

\tdefault environment = {
\t\tPATH => /usr/bin:/bin:/usr/sbin:/sbin
\t}

\tenvironment = {
\t\tXPC_SERVICE_NAME => ai.hermes.gateway-aegis
\t}

\tdomain = gui/501 [100010]
\tminimum runtime = 10
\texit timeout = 60
\truns = 3
\tpid = 83601
\tsuccessive crashes = 0
"""


# ---------------------------------------------------------------------------
# parse / read
# ---------------------------------------------------------------------------


def test_parse_exit_timeout_from_launchctl_print():
    assert parse_launchd_exit_timeout(_PRINT_OUTPUT) == 60.0


@pytest.mark.parametrize("text", ["", None, "state = running\n", "exit timeout = abc\n"])
def test_parse_exit_timeout_absent_returns_none(text):
    assert parse_launchd_exit_timeout(text) is None


def test_parse_exit_timeout_ignores_lookalike_keys():
    # "minimum runtime" and other "= N" lines must not be mistaken for it.
    assert parse_launchd_exit_timeout("\tminimum runtime = 10\n\tpid = 5\n") is None


def test_service_label_from_xpc_env():
    assert launchd_service_label({"XPC_SERVICE_NAME": "ai.hermes.gateway"}) == "ai.hermes.gateway"
    assert launchd_service_label({"XPC_SERVICE_NAME": "0"}) is None
    assert launchd_service_label({"XPC_SERVICE_NAME": ""}) is None
    assert launchd_service_label({}) is None


def test_read_exit_timeout_queries_gui_domain_for_label():
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        assert kwargs.get("timeout"), "launchctl probe must be bounded"
        return subprocess.CompletedProcess(argv, 0, stdout=_PRINT_OUTPUT, stderr="")

    value = read_launchd_exit_timeout_s(
        environ={"XPC_SERVICE_NAME": "ai.hermes.gateway-aegis"}, uid=501, run=fake_run
    )
    assert value == 60.0
    assert calls == [["launchctl", "print", "gui/501/ai.hermes.gateway-aegis"]]


def test_read_exit_timeout_uses_system_domain_for_root():
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout=_PRINT_OUTPUT, stderr="")

    read_launchd_exit_timeout_s(
        environ={"XPC_SERVICE_NAME": "ai.hermes.gateway"}, uid=0, run=fake_run
    )
    assert calls == [["launchctl", "print", "system/ai.hermes.gateway"]]


def test_read_exit_timeout_not_launchd_owned_skips_probe():
    def fake_run(*_a, **_k):  # pragma: no cover - must not be reached
        raise AssertionError("launchctl must not run when XPC_SERVICE_NAME is unset")

    assert read_launchd_exit_timeout_s(environ={}, uid=501, run=fake_run) is None
    assert (
        read_launchd_exit_timeout_s(environ={"XPC_SERVICE_NAME": "0"}, uid=501, run=fake_run)
        is None
    )


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError("launchctl"),
        subprocess.TimeoutExpired(cmd="launchctl", timeout=5),
        OSError("boom"),
    ],
)
def test_read_exit_timeout_fails_open_on_probe_errors(failure):
    def fake_run(*_a, **_k):
        raise failure

    assert (
        read_launchd_exit_timeout_s(
            environ={"XPC_SERVICE_NAME": "ai.hermes.gateway"}, uid=501, run=fake_run
        )
        is None
    )


def test_read_exit_timeout_fails_open_on_nonzero_rc():
    def fake_run(argv, **_k):
        return subprocess.CompletedProcess(argv, 113, stdout="", stderr="Could not find service")

    assert (
        read_launchd_exit_timeout_s(
            environ={"XPC_SERVICE_NAME": "ai.hermes.gateway"}, uid=501, run=fake_run
        )
        is None
    )


# ---------------------------------------------------------------------------
# resolve_launchd_capped_drain
# ---------------------------------------------------------------------------


def test_capped_drain_fits_inside_launchd_budget_minus_reserve():
    # The incident shape: configured 180s, launchd clamps to 60s.
    assert resolve_launchd_capped_drain(180.0, 60.0) == 60.0 - LAUNCHD_STOP_CLEANUP_RESERVE_S


def test_capped_drain_never_extends_a_short_drain():
    assert resolve_launchd_capped_drain(20.0, 60.0) == 20.0
    assert resolve_launchd_capped_drain(0.0, 60.0) == 0.0


def test_capped_drain_no_launchd_budget_returns_configured():
    assert resolve_launchd_capped_drain(180.0, None) == 180.0
    assert resolve_launchd_capped_drain(180.0, 0.0) == 180.0
    assert resolve_launchd_capped_drain(180.0, -5.0) == 180.0


def test_capped_drain_tiny_budget_clamps_to_zero_not_negative():
    assert resolve_launchd_capped_drain(180.0, 5.0) == 0.0


def test_capped_drain_tolerates_garbage_inputs():
    assert resolve_launchd_capped_drain("nope", 60.0) == 0.0  # type: ignore[arg-type]
    assert resolve_launchd_capped_drain(180.0, "nope") == 180.0  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# GatewayRunner wiring
# ---------------------------------------------------------------------------


def _runner(*, drain: float, launchd: float | None, by_signal: bool):
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._restart_drain_timeout = drain
    runner._launchd_exit_timeout_s = launchd
    runner._stop_requested_by_signal = by_signal
    return runner


def test_effective_drain_capped_only_for_signal_stops_under_launchd():
    assert _runner(drain=180.0, launchd=60.0, by_signal=True)._effective_stop_drain_timeout() == 50.0
    # In-band restart (SIGUSR1 → after-turn → stop()) is not launchd-timed.
    assert _runner(drain=180.0, launchd=60.0, by_signal=False)._effective_stop_drain_timeout() == 180.0
    # Not launchd-owned (systemd, s6, foreground): configured drain stands.
    assert _runner(drain=180.0, launchd=None, by_signal=True)._effective_stop_drain_timeout() == 180.0


def test_effective_drain_getattr_guarded_for_bare_doubles():
    from types import SimpleNamespace

    from gateway.restart import effective_stop_drain_timeout
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._restart_drain_timeout = 45.0
    # No _stop_requested_by_signal / _launchd_exit_timeout_s set at all.
    assert runner._effective_stop_drain_timeout() == 45.0
    # Shutdown-path suites drive _stop_impl from non-GatewayRunner doubles.
    assert effective_stop_drain_timeout(SimpleNamespace(_restart_drain_timeout=45.0)) == 45.0
    assert (
        effective_stop_drain_timeout(
            SimpleNamespace(
                _restart_drain_timeout=180.0,
                _stop_requested_by_signal=True,
                _launchd_exit_timeout_s=60.0,
            )
        )
        == 50.0
    )


def test_load_launchd_exit_timeout_warns_when_drain_exceeds_budget(monkeypatch, caplog):
    import logging

    from gateway import run as run_mod

    monkeypatch.setattr(run_mod, "read_launchd_exit_timeout_s", lambda: 60.0)
    monkeypatch.setenv("XPC_SERVICE_NAME", "ai.hermes.gateway-test")
    with caplog.at_level(logging.INFO, logger=run_mod.logger.name):
        assert run_mod.GatewayRunner._load_launchd_exit_timeout(180.0) == 60.0
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "misconfiguration must be loud at boot"
    msg = warnings[0].getMessage()
    assert "180s" in msg and "60s" in msg and "ai.hermes.gateway-test" in msg


def test_load_launchd_exit_timeout_quiet_when_drain_fits(monkeypatch, caplog):
    import logging

    from gateway import run as run_mod

    monkeypatch.setattr(run_mod, "read_launchd_exit_timeout_s", lambda: 60.0)
    with caplog.at_level(logging.INFO, logger=run_mod.logger.name):
        assert run_mod.GatewayRunner._load_launchd_exit_timeout(30.0) == 60.0
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_load_launchd_exit_timeout_none_when_not_launchd(monkeypatch):
    from gateway import run as run_mod

    monkeypatch.setattr(run_mod, "read_launchd_exit_timeout_s", lambda: None)
    assert run_mod.GatewayRunner._load_launchd_exit_timeout(180.0) is None


def test_cron_leash_under_launchd_cannot_exceed_exit_timeout():
    """The cron drain floor (#82161) is clamped to the launchd budget too.

    Without the clamp the cron ceiling is watchdog(drain+grace) - reserve,
    which for a capped 50s drain is 100s — past launchd's 60s SIGKILL.
    """
    from gateway.restart import CRON_DRAIN_CLEANUP_RESERVE_S, resolve_cron_drain_budget

    drain = resolve_launchd_capped_drain(180.0, 60.0)  # 50
    leash = min(resolve_shutdown_watchdog_delay(drain), 60.0)
    budget = resolve_cron_drain_budget(drain, 600.0, watchdog_delay=leash, elapsed=0.0)
    assert budget == max(drain, 60.0 - CRON_DRAIN_CLEANUP_RESERVE_S)
    assert budget <= 60.0
    # Sanity: the un-clamped leash would have blown the budget.
    unclamped = resolve_cron_drain_budget(
        drain, 600.0, watchdog_delay=resolve_shutdown_watchdog_delay(drain), elapsed=0.0
    )
    assert unclamped == drain + DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S - CRON_DRAIN_CLEANUP_RESERVE_S
    assert unclamped > 60.0


def test_signal_handler_marks_stop_as_signal_driven():
    """SIGTERM/SIGINT handler flags the runner before scheduling stop().

    The handler is a closure inside the gateway main; this is a structural
    contract check that both halves of the cap exist — the producer in the
    signal handler and the consumer in the stop path — so neither can be
    dropped in a refactor without this test going red.
    """
    import inspect

    from gateway import run as run_mod

    src = inspect.getsource(run_mod)
    assert src.count("runner._stop_requested_by_signal = True") == 1
    # The drain budget is consumed in the stop path (``_stop_impl``).
    assert "effective_stop_drain_timeout(self)" in inspect.getsource(
        run_mod.GatewayRunner._stop_impl
    )
