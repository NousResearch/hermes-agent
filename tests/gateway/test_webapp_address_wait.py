"""A specific ``webapp.host`` an interface is given after the gateway starts is waited for, briefly.

A tailnet (Tailscale/Headscale/WireGuard) address, or a DHCP lease, is assigned a moment after its daemon
reports ready, so a gateway that starts with ``webapp.host`` set to it can lose the first bind with
EADDRNOTAVAIL. Without a retry the gateway keeps running with the Webapp off and nothing ever tries again:
no crash, so no ``Restart=`` policy fires. The wait is bounded, and every other bind failure still reports
at once.
"""
import asyncio
import errno
import logging
import types

import pytest

from gateway import run_api, run_runtime_listener
from gateway.control_socket import GatewayControlServer


def _api(failures):
    """A stand-in for ``gateway.run_api`` whose first starts raise *failures* in order."""
    calls = []

    async def start_gateway_api(runner, **kwargs):
        calls.append(kwargs)
        if failures:
            raise failures.pop(0)
        return "handle"

    return types.SimpleNamespace(start_gateway_api=start_gateway_api,
                                 ListenerAddressError=run_api.ListenerAddressError), calls


def _start(api, *, webapp="mount", host="100.64.0.5", port=9119, wait_seconds=5.0):
    return asyncio.run(run_runtime_listener._start_gateway_api_waiting_for_address(
        api, object(), host=host, port=port, webapp=webapp, wait_seconds=wait_seconds))


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(run_runtime_listener, "_WEBAPP_ADDRESS_POLL_SECONDS", 0.01)
    return monkeypatch


def test_an_address_assigned_after_the_first_bind_is_waited_for(fast):
    api, calls = _api([OSError(errno.EADDRNOTAVAIL, "address not assigned to an interface")])
    assert _start(api) == "handle"
    assert len(calls) == 2, "the bind was retried once the address appeared"
    assert calls[0] == {"host": "100.64.0.5", "port": 9119, "webapp": "mount"}


def test_a_gateway_without_the_webapp_keeps_its_own_listener_call(fast):
    api, calls = _api([])
    assert _start(api, webapp=None, host="127.0.0.1", port=0) == "handle"
    assert calls == [{}], "main's listener is started with no address of its own"


def test_a_gateway_without_the_webapp_does_not_wait_on_eaddrnotavail(fast, caplog):
    sleeps = []

    async def record_sleep(delay):
        sleeps.append(delay)

    fast.setattr(run_runtime_listener.asyncio, "sleep", record_sleep)
    api, calls = _api([OSError(errno.EADDRNOTAVAIL, "address not assigned to an interface")])
    with caplog.at_level(logging.WARNING, logger="gateway.run_runtime"):
        with pytest.raises(OSError) as caught:
            _start(api, webapp=None, host="127.0.0.1", port=0)
    assert caught.value.errno == errno.EADDRNOTAVAIL
    assert calls == [{}]
    assert sleeps == []
    assert not any("not assigned to any interface yet" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost", "0.0.0.0", "::"])
def test_loopback_and_wildcard_webapp_hosts_do_not_wait_on_eaddrnotavail(fast, caplog, host):
    sleeps = []

    async def record_sleep(delay):
        sleeps.append(delay)

    fast.setattr(run_runtime_listener.asyncio, "sleep", record_sleep)
    api, calls = _api([OSError(errno.EADDRNOTAVAIL, "Cannot assign requested address")])
    with caplog.at_level(logging.WARNING, logger="gateway.run_runtime"):
        with pytest.raises(OSError) as caught:
            _start(api, host=host, port=9119, webapp="mount")
    assert caught.value.errno == errno.EADDRNOTAVAIL
    assert len(calls) == 1
    assert sleeps == []
    assert not any("not assigned to any interface yet" in r.getMessage() for r in caplog.records)


def test_ipv6_loopback_eaddrnotavail_falls_back_immediately_without_retrying(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "webapp:\n  enabled: true\n  host: '::1'\n  port: 9119\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(run_runtime_listener, "_WEBAPP_ADDRESS_POLL_SECONDS", 0.01)

    calls = []

    async def start(runner, *, host="127.0.0.1", port=0, webapp=None):
        calls.append(host)
        if host == "::1":
            raise OSError(errno.EADDRNOTAVAIL, "Cannot assign requested address")
        return types.SimpleNamespace(api_origin="http://127.0.0.1:50001")

    monkeypatch.setattr(run_api, "start_gateway_api", start)
    runner = types.SimpleNamespace(
        session_runtime_descriptor={},
        session_authorities=[],
        session_control_server=GatewayControlServer(home),
    )
    asyncio.run(run_runtime_listener.start_gateway_runtime_api(runner, address_wait=5.0))
    assert calls == ["::1", "127.0.0.1"]
    assert runner.webapp_status["state"] == "unavailable"


def test_the_wait_is_bounded_and_the_last_error_is_raised(fast):
    api, calls = _api([OSError(errno.EADDRNOTAVAIL, "address not assigned to an interface")] * 100)
    with pytest.raises(OSError) as caught:
        _start(api, wait_seconds=0.05)
    assert caught.value.errno == errno.EADDRNOTAVAIL
    assert len(calls) > 1, "it retried before giving up"


def test_zero_disables_the_wait(fast):
    api, calls = _api([OSError(errno.EADDRNOTAVAIL, "address not assigned to an interface")])
    with pytest.raises(OSError):
        _start(api, wait_seconds=0.0)
    assert len(calls) == 1, "one attempt, then the usual 'Webapp unavailable' report"


def test_a_busy_port_is_not_waited_for(fast):
    api, calls = _api([OSError(errno.EADDRINUSE, "Address already in use")])
    with pytest.raises(OSError) as caught:
        _start(api)
    assert caught.value.errno == errno.EADDRINUSE
    assert len(calls) == 1, "another process holding the port does not clear up by waiting"


def test_the_loopback_companion_failure_is_not_waited_for(fast):
    """A specific address's loopback socket fails for its own reasons (no IPv6 loopback, say)."""
    api, calls = _api([run_api.ListenerAddressError(
        "::1:9119", OSError(errno.EADDRNOTAVAIL, "Cannot assign requested address"))])
    with pytest.raises(run_api.ListenerAddressError):
        _start(api)
    assert len(calls) == 1


def test_start_gateway_runtime_api_waits_for_late_webapp_host_and_warns_once(monkeypatch, tmp_path, caplog):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "webapp:\n  enabled: true\n  host: 100.64.0.5\n  port: 9119\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(run_runtime_listener, "_WEBAPP_ADDRESS_POLL_SECONDS", 0.01)

    webapp_calls = []
    remaining_failures = [OSError(errno.EADDRNOTAVAIL, "address not assigned to an interface")] * 3

    async def start(runner, *, host="127.0.0.1", port=0, webapp=None):
        if webapp is not None:
            webapp_calls.append((host, port))
            if remaining_failures:
                raise remaining_failures.pop(0)
            return types.SimpleNamespace(api_origin=f"http://127.0.0.1:{port}")
        return types.SimpleNamespace(api_origin="http://127.0.0.1:50001")

    monkeypatch.setattr(run_api, "start_gateway_api", start)
    runner = types.SimpleNamespace(
        session_runtime_descriptor={},
        session_authorities=[],
        session_control_server=GatewayControlServer(home),
    )

    with caplog.at_level(logging.WARNING, logger="gateway.run_runtime"):
        asyncio.run(run_runtime_listener.start_gateway_runtime_api(runner, address_wait=5.0))

    assert runner.webapp_status == {
        "state": "serving",
        "url": "http://127.0.0.1:9119",
        "reason": None,
        "address": "http://100.64.0.5:9119",
    }
    assert len(webapp_calls) == 4
    retry_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "not assigned to any interface yet" in r.getMessage()
    ]
    assert len(retry_warnings) == 1


@pytest.mark.parametrize(("err_no", "msg"), [
    (errno.EADDRINUSE, "Address already in use"),
    (errno.EACCES, "Permission denied"),
])
def test_non_eaddrnotavail_oserrors_fail_immediately_without_sleeping(fast, caplog, err_no, msg):
    sleeps = []

    async def record_sleep(delay):
        sleeps.append(delay)

    fast.setattr(run_runtime_listener.asyncio, "sleep", record_sleep)
    api, calls = _api([OSError(err_no, msg)])
    with caplog.at_level(logging.WARNING, logger="gateway.run_runtime"):
        with pytest.raises(OSError) as caught:
            _start(api)
    assert caught.value.errno == err_no
    assert len(calls) == 1
    assert sleeps == []
    assert not any("not assigned to any interface yet" in r.getMessage() for r in caplog.records)


def test_zero_wait_does_not_sleep_or_warn(fast, caplog):
    sleeps = []

    async def record_sleep(delay):
        sleeps.append(delay)

    fast.setattr(run_runtime_listener.asyncio, "sleep", record_sleep)
    api, calls = _api([OSError(errno.EADDRNOTAVAIL, "address not assigned to an interface")])
    with caplog.at_level(logging.WARNING, logger="gateway.run_runtime"):
        with pytest.raises(OSError) as caught:
            _start(api, wait_seconds=0.0)
    assert caught.value.errno == errno.EADDRNOTAVAIL
    assert len(calls) == 1
    assert sleeps == []
    assert not any("not assigned to any interface yet" in r.getMessage() for r in caplog.records)


def test_cancellation_during_wait_sleep_propagates(monkeypatch):
    monkeypatch.setattr(run_runtime_listener, "_WEBAPP_ADDRESS_POLL_SECONDS", 10.0)
    real_sleep = asyncio.sleep

    async def run_and_cancel():
        entered_sleep = asyncio.Event()

        async def tracked_sleep(delay):
            entered_sleep.set()
            await real_sleep(delay)

        monkeypatch.setattr(run_runtime_listener.asyncio, "sleep", tracked_sleep)
        api, calls = _api([OSError(errno.EADDRNOTAVAIL, "address not assigned to an interface")] * 5)
        task = asyncio.create_task(run_runtime_listener._start_gateway_api_waiting_for_address(
            api, object(), host="100.64.0.5", port=9119, webapp="mount", wait_seconds=5.0))
        await asyncio.wait_for(entered_sleep.wait(), timeout=2.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(calls) == 1

    asyncio.run(run_and_cancel())


def test_cancellation_during_bind_propagates_without_retrying(fast):
    sleeps = []

    async def record_sleep(delay):
        sleeps.append(delay)

    fast.setattr(run_runtime_listener.asyncio, "sleep", record_sleep)
    api, calls = _api([asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        _start(api)
    assert len(calls) == 1
    assert sleeps == []


def test_operator_stop_during_wait_aborts_without_rebinding_or_private_fallback(monkeypatch, tmp_path):
    import signal

    from gateway import run_bootstrap, shutdown_forensics, status
    from gateway.run import GatewayRunner

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "webapp:\n  enabled: true\n  host: 192.0.2.77\n  port: 9119\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(run_runtime_listener, "_WEBAPP_ADDRESS_POLL_SECONDS", 5.0)
    monkeypatch.setattr(status, "consume_takeover_marker_for_self", lambda: False)
    monkeypatch.setattr(shutdown_forensics, "snapshot_shutdown_context", lambda sig: None)

    async def run_scenario():
        shutdown_event = asyncio.Event()
        completed = asyncio.Event()
        runner = types.SimpleNamespace(
            session_runtime_descriptor={},
            session_authorities=[],
            session_control_server=GatewayControlServer(home),
            _stop_task=None,
            _shutdown_event=shutdown_event,
            _draining=False,
            _restart_requested=False,
            _stop_requested_by_signal=False,
            _planned_takeover_shutdown=False,
        )

        async def teardown(self):
            self._draining = True
            self._shutdown_event.set()
            completed.set()

        monkeypatch.setattr(GatewayRunner, "_stop_impl", teardown)
        runner.stop = types.MethodType(GatewayRunner.stop, runner)
        stop_handler = run_bootstrap._start_gateway_make_shutdown_signal_handler(runner, [False])

        calls = []

        async def start(r, *, host="127.0.0.1", port=0, webapp=None):
            calls.append((host, completed.is_set()))
            if len(calls) == 1:
                asyncio.get_running_loop().call_soon(stop_handler, signal.SIGINT)
                raise OSError(errno.EADDRNOTAVAIL, "address not assigned to an interface")
            return types.SimpleNamespace(api_origin="http://127.0.0.1:50001")

        monkeypatch.setattr(run_api, "start_gateway_api", start)
        await asyncio.wait_for(run_runtime_listener.start_gateway_runtime_api(runner, address_wait=5.0), timeout=2.0)
        assert completed.is_set(), "operator stop ran while the address wait was active"
        assert calls == [("192.0.2.77", False)], "no retry and no private fallback after operator stop"
        assert runner.session_api is None
        assert runner.webapp_status["state"] != "serving"
        assert "api_origin" not in runner.session_runtime_descriptor

    asyncio.run(run_scenario())


def test_restart_requested_during_wait_aborts_without_binding(monkeypatch):
    monkeypatch.setattr(run_runtime_listener, "_WEBAPP_ADDRESS_POLL_SECONDS", 0.01)

    async def run_scenario():
        runner = types.SimpleNamespace(_restart_requested=False)
        calls = []

        async def start(r, **kwargs):
            calls.append(kwargs)
            r._restart_requested = True
            raise OSError(errno.EADDRNOTAVAIL, "address not assigned to an interface")

        api = types.SimpleNamespace(start_gateway_api=start, ListenerAddressError=run_api.ListenerAddressError)
        result = await run_runtime_listener._start_gateway_api_waiting_for_address(
            api, runner, host="100.64.0.5", port=9119, webapp="mount", wait_seconds=5.0
        )
        assert result is None
        assert len(calls) == 1

    asyncio.run(run_scenario())


@pytest.mark.parametrize(("failure", "binds", "closed"), [
    (None, 1, 1),
    (OSError(errno.EADDRINUSE, "Address already in use"), 2, 1),
    (SystemExit("the sign-in gate refused"), 1, 0),
], ids=["webapp-bound", "private-fallback-bound", "webapp-refused"])
def test_a_stop_during_a_bind_publishes_no_listener(monkeypatch, tmp_path, failure, binds, closed):
    """A stop that begins while a listener binds: nothing is published after teardown, no fallback starts."""
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "webapp:\n  enabled: true\n  host: 100.64.0.5\n  port: 9119\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    calls, stopped = [], []

    async def start(r, *, host="127.0.0.1", port=0, webapp=None):
        calls.append(host)
        if webapp is not None and failure is not None:
            if isinstance(failure, SystemExit):
                r._stop_requested_by_signal = True  # the stop lands while the gate refuses
            raise failure
        r._stop_requested_by_signal = True  # the stop lands while this bind succeeds
        return types.SimpleNamespace(api_origin="http://127.0.0.1:50001")

    async def stop(handle):
        stopped.append(handle)

    monkeypatch.setattr(run_api, "start_gateway_api", start)
    monkeypatch.setattr(run_api, "stop_gateway_api", stop)
    runner = types.SimpleNamespace(
        session_runtime_descriptor={},
        session_authorities=[],
        session_control_server=GatewayControlServer(home),
    )
    asyncio.run(run_runtime_listener.start_gateway_runtime_api(runner))
    assert len(calls) == binds
    assert len(stopped) == closed
    assert runner.session_api is None
    assert "api_origin" not in runner.session_runtime_descriptor


def test_unified_startup_abort_handles_restart_flags_and_minimal_test_doubles():
    from gateway.run_common import abort_startup_if_shutdown_requested, startup_should_abort

    async def run_scenario():
        shutdown_event = asyncio.Event()
        stop_calls = []

        async def stop(*, restart=False, detached_restart=False, service_restart=False):
            stop_calls.append((restart, detached_restart, service_restart))
            shutdown_event.set()

        runner = types.SimpleNamespace(
            _stop_task=None,
            _shutdown_event=shutdown_event,
            _restart_requested=True,
            _restart_detached=False,
            _restart_via_service=True,
            stop=stop,
        )
        assert startup_should_abort(runner) is True
        assert await abort_startup_if_shutdown_requested(runner) is True
        assert stop_calls == [(True, False, True)]
        assert shutdown_event.is_set()

        # Runner double without optional attributes (_shutdown_event, _restart_detached) still
        # receives the three restart flags through the real GatewayRunner.stop signature.
        minimal_stops = []

        class MinimalRunner:
            _restart_requested = True
            _restart_via_service = True

            async def stop(
                self, *, restart: bool = False, detached_restart: bool = False, service_restart: bool = False
            ):
                minimal_stops.append((restart, detached_restart, service_restart))

        minimal = MinimalRunner()
        assert startup_should_abort(minimal) is True
        assert await abort_startup_if_shutdown_requested(minimal) is True
        assert minimal_stops == [(True, False, True)]

    asyncio.run(run_scenario())


@pytest.mark.parametrize(("signal_initiated", "expected_verdict"), [(False, True), (True, False)])
def test_bootstrap_listener_and_runner_aborts_before_keepalive_and_runner_start(
    monkeypatch, signal_initiated, expected_verdict
):
    from gateway import run_bootstrap

    async def run_scenario():
        shutdown_event = asyncio.Event()
        called_runner = []
        stop_calls = []

        async def fake_start_api(r, **kwargs):
            r._stop_requested_by_signal = True
            r.session_api = None

        async def fake_run_runner(r, sig):
            called_runner.append(True)

        async def stop(*, restart=False, detached_restart=False, service_restart=False):
            stop_calls.append((restart, detached_restart, service_restart))
            shutdown_event.set()

        monkeypatch.setattr(run_runtime_listener, "start_gateway_runtime_api", fake_start_api)
        monkeypatch.setattr(run_bootstrap, "_start_gateway_run_runner", fake_run_runner)
        runner = types.SimpleNamespace(
            _stop_task=None,
            _shutdown_event=shutdown_event,
            _stop_requested_by_signal=False,
            _restart_requested=False,
            _restart_detached=False,
            _restart_via_service=False,
            should_exit_with_failure=False,
            exit_code=None,
            exit_reason=None,
            _gateway_health_export_runtime=None,
            stop=stop,
            wait_for_shutdown=shutdown_event.wait,
        )
        verdict, ws_preload, unclean = await run_bootstrap._start_gateway_listener_and_runner(
            runner, [signal_initiated], idle_exit=False
        )
        assert (verdict, ws_preload, unclean) == (expected_verdict, None, None)
        assert stop_calls == [(False, False, False)]
        assert shutdown_event.is_set()
        assert called_runner == []

    asyncio.run(run_scenario())


@pytest.mark.parametrize(("idle_exit", "webapp_binds", "state"), [(True, 1, "unavailable"), (False, 3, "serving")],
                         ids=["client-idle-exit-start", "supervised"])
def test_only_a_supervised_gateway_waits_for_the_address(monkeypatch, tmp_path, idle_exit, webapp_binds, state):
    """A TUI or ``chat -q`` client waits on its own idle-exit gateway with a hard start budget (Ink: 30 s)."""
    from gateway import lifecycle_ledger, run_bootstrap
    from hermes_cli import nous_auth_keepalive

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "webapp:\n  enabled: true\n  host: 100.64.0.5\n  port: 9119\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(run_runtime_listener, "_WEBAPP_ADDRESS_POLL_SECONDS", 0.01)
    monkeypatch.setattr(lifecycle_ledger, "record_startup", lambda: None)
    monkeypatch.setattr(nous_auth_keepalive, "start_nous_auth_keepalive", lambda: None)
    monkeypatch.setattr(run_api, "preload_session_socket", lambda: None)
    monkeypatch.setattr(run_bootstrap, "preload_stop_path_modules", lambda: None)

    async def run_runner(r, sig):
        return None

    monkeypatch.setattr(run_bootstrap, "_start_gateway_run_runner", run_runner)
    binds = []

    async def start(r, *, host="127.0.0.1", port=0, webapp=None):
        if webapp is not None:
            binds.append(host)
            if len(binds) < 3:  # the address arrives on the third try
                raise OSError(errno.EADDRNOTAVAIL, "address not assigned to an interface")
        return types.SimpleNamespace(api_origin="http://127.0.0.1:50001")

    monkeypatch.setattr(run_api, "start_gateway_api", start)
    runner = types.SimpleNamespace(
        session_runtime_descriptor={},
        session_authorities=[],
        session_control_server=GatewayControlServer(home),
    )
    asyncio.run(run_bootstrap._start_gateway_listener_and_runner(runner, [False], idle_exit=idle_exit))
    assert len(binds) == webapp_binds
    assert runner.webapp_status["state"] == state
