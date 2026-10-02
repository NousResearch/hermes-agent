"""A port-binding platform that cannot bind at startup must take the gateway DOWN, loudly.

A messaging platform that fails to connect degrades service and a human notices within one message.
A platform that binds a port is a contract surface: the only thing an external prober can see is the
port, and an unbound port on a live process is indistinguishable from a hung host. The gateway used
to park the failed binder, keep serving its messaging siblings and report ``degraded`` — while
``systemctl is-active`` said ``active`` and a provisioner polling ``curl 127.0.0.1:<port>`` waited
out its full 20-minute timeout. Now such a failure exits 78 so the supervisor parks the unit as
``failed``.

Covered here:
  1. the real adapter against a real squatted port still classifies EADDRINUSE fatal/non-retryable;
  2. the startup gate exits 78 with a healthy sibling connected;
  3. a NON-port-binding fatal failure still stays alive and degraded (the old contract);
  4. the fatal set is DERIVED from gateway.config, not from a hand-written list;
  5. the bind retry budget is wall-clock, so a port freed after the old ~3s window still binds.
"""
import asyncio
import errno
import socket
import time

import pytest

from gateway.config import (
    PORT_BINDING_CONDITIONAL_MODES,
    PORT_BINDING_PLATFORM_VALUES,
    GatewayConfig,
    Platform,
    PlatformConfig,
)
from gateway.platforms import api_server as api_server_mod
from gateway.platforms.api_server import APIServerAdapter
from gateway.restart import GATEWAY_FATAL_CONFIG_EXIT_CODE
from gateway.run import GatewayRunner
from gateway.status import flush_runtime_status, read_runtime_status

# >= 16 chars: a short key trips the weak-key guard, which fails connect() for a DIFFERENT reason
# and would make every assertion below vacuous.
_API_KEY = "9f3c1d7b52a84e0c6b1f49d2a7c30e85"


def _squat_port() -> tuple[socket.socket, int]:
    """Hold a real ephemeral port the way a not-yet-dead predecessor gateway does."""
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(5)
    return sock, sock.getsockname()[1]


def _adapter(port: int) -> APIServerAdapter:
    return APIServerAdapter(
        PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": port, "key": _API_KEY})
    )


def _runner(tmp_path, monkeypatch) -> GatewayRunner:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig(
        platforms={
            Platform.API_SERVER: PlatformConfig(enabled=True, extra={"port": 8642}),
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="***"),
        },
        sessions_dir=tmp_path / "sessions",
    )
    return GatewayRunner(config)


# --------------------------------------------------------------------------------------------
# 1. Real socket: the bind failure really is fatal + non-retryable
# --------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_real_port_conflict_is_fatal_and_non_retryable(monkeypatch):
    """The regression's ground truth, against a real bound socket and the real adapter."""
    monkeypatch.setattr(api_server_mod, "_BIND_RETRY_BUDGET_SECONDS", 0.3)  # don't burn 30s in CI
    squatter, port = _squat_port()
    adapter = _adapter(port)
    try:
        assert await adapter.connect() is False
        assert adapter.has_fatal_error is True
        assert adapter.fatal_error_code == "api_server_port_in_use"
        assert adapter.fatal_error_retryable is False, (
            "non-retryable is what keeps the reconnect watcher from leaking fds forever (#52132)"
        )
    finally:
        await adapter.disconnect()
        squatter.close()


# --------------------------------------------------------------------------------------------
# 2. The startup gate: fatal even though a sibling connected
# --------------------------------------------------------------------------------------------


def test_unbound_port_binder_exits_78_even_with_a_connected_sibling(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    message = (
        "Port 8642 already in use. Set platforms.api_server.port in config.yaml to a different "
        "value, then `/platform resume api_server`."
    )

    must_exit = runner._start_handle_no_connections(
        connected_count=1,
        enabled_platform_count=2,
        startup_retryable_errors=[],
        startup_nonretryable_errors=[f"api_server: {message}"],
        startup_nonretryable_details=[(Platform.API_SERVER, {"port": 8642}, message)],
    )

    assert must_exit is True
    assert runner._exit_code == GATEWAY_FATAL_CONFIG_EXIT_CODE
    flush_runtime_status()
    state = read_runtime_status()
    assert state["gateway_state"] == "startup_failed"
    assert state["gateway_state"] != "running"
    assert state["gateway_state"] != "degraded", (
        "degraded keeps the process up; a dark port needs the supervisor to park the unit"
    )


def test_unbound_port_binder_is_fatal_even_next_to_a_retryable_peer(tmp_path, monkeypatch):
    """Mixed fatal + transient with nothing connected: the retryable peer's chance to recover does
    not buy enough to stay up with a listener that will never listen."""
    runner = _runner(tmp_path, monkeypatch)
    must_exit = runner._start_handle_no_connections(
        connected_count=0,
        enabled_platform_count=2,
        startup_retryable_errors=["telegram: TimedOut"],
        startup_nonretryable_errors=["api_server: Port 8642 already in use."],
        startup_nonretryable_details=[(Platform.API_SERVER, {"port": 8642}, "Port 8642 already in use.")],
    )
    assert must_exit is True
    assert runner._exit_code == GATEWAY_FATAL_CONFIG_EXIT_CODE
    flush_runtime_status()
    assert read_runtime_status()["gateway_state"] == "startup_failed"


# --------------------------------------------------------------------------------------------
# 3. Non-regression: a messaging platform's fatal failure still only degrades
# --------------------------------------------------------------------------------------------


def test_non_port_binding_fatal_failure_still_stays_alive_and_degraded(tmp_path, monkeypatch):
    """WhatsApp never paired while Telegram serves: staying up is correct and must not change.

    This is the behaviour the counter-reasoning in _start_handle_no_connections protects; the port
    binder rule is a carve-out for unobservable listeners, not a general "fatal means exit".
    """
    runner = _runner(tmp_path, monkeypatch)
    message = "WhatsApp is not paired. Run `hermes whatsapp pair`."

    must_exit = runner._start_handle_no_connections(
        connected_count=1,
        enabled_platform_count=2,
        startup_retryable_errors=[],
        startup_nonretryable_errors=[f"whatsapp: {message}"],
        startup_nonretryable_details=[(Platform.WHATSAPP, {}, message)],
    )

    assert must_exit is False
    assert runner._exit_code != GATEWAY_FATAL_CONFIG_EXIT_CODE
    assert runner._serving_state() == "degraded"


# --------------------------------------------------------------------------------------------
# 4. Derive-check: the fatal set comes from gateway.config, never from a literal
# --------------------------------------------------------------------------------------------


def test_fatal_classification_is_derived_from_port_binding_platform_values():
    """Every configured port binder must classify fatal, and non-binders must not.

    Computed by calling the production helper over the production set: a literal tuple in
    run_startup.py (``{"api_server", "webhook"}``) would pass a hand-written test and silently skip
    every platform added to PORT_BINDING_PLATFORM_VALUES afterwards.
    """
    classify = GatewayRunner._startup_fatal_port_binder_reasons

    for value in sorted(PORT_BINDING_PLATFORM_VALUES):
        platform = Platform(value)
        mode = PORT_BINDING_CONDITIONAL_MODES.get(value)
        extra = {"connection_mode": mode} if mode else {}
        assert classify([(platform, extra, "boom")]) == [f"{value}: boom"], (
            f"{value} binds a port and must be fatal at startup"
        )

    non_binders = [p for p in Platform if p.value not in PORT_BINDING_PLATFORM_VALUES]
    assert non_binders, "sanity: there must be platforms that do not bind a port"
    for platform in non_binders:
        assert classify([(platform, {}, "boom")]) == [], (
            f"{platform.value} does not bind a port and must only degrade"
        )

    # Conditional binders follow the connection mode, exactly like platform_binds_port does.
    for value, mode in PORT_BINDING_CONDITIONAL_MODES.items():
        other_mode = "websocket" if mode != "websocket" else "webhook"
        assert classify([(Platform(value), {"connection_mode": other_mode}, "boom")]) == [], (
            f"{value} in {other_mode} mode binds no port, so it must only degrade"
        )


# --------------------------------------------------------------------------------------------
# 5. Bind retry budget is wall-clock, not 5 attempts
# --------------------------------------------------------------------------------------------


@pytest.fixture
def fast_sleep(monkeypatch):
    """Replace the retry sleeps with near-instant ones and record every attempt's requested delay."""
    real_sleep = asyncio.sleep
    delays: list[float] = []

    async def _sleep(delay, *args, **kwargs):
        if delay and api_server_mod._BIND_RETRY_INITIAL_SLEEP <= delay <= api_server_mod._BIND_RETRY_MAX_SLEEP:
            # Only the bind loop's own backoff; unrelated background sleeps pass through untouched.
            delays.append(delay)
            return await real_sleep(0.001)
        return await real_sleep(delay, *args, **kwargs)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    return delays


@pytest.mark.asyncio
async def test_bind_retries_past_five_attempts_within_the_budget(monkeypatch, fast_sleep):
    """The old schedule gave up after 5 attempts / ~3s. The budget keeps trying until it expires."""
    monkeypatch.setattr(api_server_mod, "_BIND_RETRY_BUDGET_SECONDS", 3.0)
    squatter, port = _squat_port()
    adapter = _adapter(port)
    started = time.monotonic()
    try:
        assert await adapter.connect() is False
        assert len(fast_sleep) > 5, (
            f"expected the retry loop to outlive the old 5-attempt cap, got {len(fast_sleep)}"
        )
        assert max(fast_sleep) <= api_server_mod._BIND_RETRY_MAX_SLEEP
        assert adapter.fatal_error_code == "api_server_port_in_use"
        assert time.monotonic() - started < 15, "the budget must bound the wall clock"
    finally:
        await adapter.disconnect()
        squatter.close()


@pytest.mark.asyncio
async def test_port_freed_after_the_old_window_still_binds(monkeypatch, fast_sleep):
    """The losing side of a restart race self-heals instead of leaving the port dark forever."""
    monkeypatch.setattr(api_server_mod, "_BIND_RETRY_BUDGET_SECONDS", 5.0)
    squatter, port = _squat_port()
    real_start = api_server_mod.start_tcp_site
    attempts = {"n": 0}

    async def _counting_start(runner, host, prt, *, log_tag):
        attempts["n"] += 1
        # Release the port only after the old fixed 5-attempt schedule would have given up.
        if attempts["n"] == 7:
            squatter.close()
        if attempts["n"] < 7:
            raise OSError(errno.EADDRINUSE, "address already in use")
        return await real_start(runner, host, prt, log_tag=log_tag)

    monkeypatch.setattr(api_server_mod, "start_tcp_site", _counting_start)
    adapter = _adapter(port)
    try:
        assert await adapter.connect() is True
        assert attempts["n"] == 7, "must have retried past the old 5-attempt cap"
        assert adapter.has_fatal_error is False
    finally:
        await adapter.disconnect()
        squatter.close()
