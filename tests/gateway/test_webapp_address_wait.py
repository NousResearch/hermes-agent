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

from gateway import run_api, run_runtime
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


def _start(api, *, webapp="mount", host="100.64.0.5", port=9119):
    return asyncio.run(run_runtime._start_gateway_api_waiting_for_address(
        api, object(), host=host, port=port, webapp=webapp))


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(run_runtime, "_WEBAPP_ADDRESS_POLL_SECONDS", 0.01)
    monkeypatch.setenv("HERMES_WEBAPP_ADDRESS_WAIT", "5")
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

    fast.setattr(run_runtime.asyncio, "sleep", record_sleep)
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

    fast.setattr(run_runtime.asyncio, "sleep", record_sleep)
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
    monkeypatch.setenv("HERMES_WEBAPP_ADDRESS_WAIT", "5")
    monkeypatch.setattr(run_runtime, "_WEBAPP_ADDRESS_POLL_SECONDS", 0.01, raising=False)

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
    asyncio.run(run_runtime.start_gateway_runtime_api(runner))
    assert calls == ["::1", "127.0.0.1"]
    assert runner.webapp_status["state"] == "unavailable"


def test_the_wait_is_bounded_and_the_last_error_is_raised(fast):
    fast.setenv("HERMES_WEBAPP_ADDRESS_WAIT", "0.05")
    api, calls = _api([OSError(errno.EADDRNOTAVAIL, "address not assigned to an interface")] * 100)
    with pytest.raises(OSError) as caught:
        _start(api)
    assert caught.value.errno == errno.EADDRNOTAVAIL
    assert len(calls) > 1, "it retried before giving up"


def test_zero_disables_the_wait(fast):
    fast.setenv("HERMES_WEBAPP_ADDRESS_WAIT", "0")
    api, calls = _api([OSError(errno.EADDRNOTAVAIL, "address not assigned to an interface")])
    with pytest.raises(OSError):
        _start(api)
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


def test_a_garbage_override_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("HERMES_WEBAPP_ADDRESS_WAIT", "soon")
    assert run_runtime._webapp_address_wait_seconds() == run_runtime.WEBAPP_ADDRESS_WAIT_SECONDS


def test_no_override_uses_the_default(monkeypatch):
    monkeypatch.delenv("HERMES_WEBAPP_ADDRESS_WAIT", raising=False)
    assert run_runtime._webapp_address_wait_seconds() == run_runtime.WEBAPP_ADDRESS_WAIT_SECONDS


def test_start_gateway_runtime_api_waits_for_late_webapp_host_and_warns_once(monkeypatch, tmp_path, caplog):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "webapp:\n  enabled: true\n  host: 100.64.0.5\n  port: 9119\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_WEBAPP_ADDRESS_WAIT", "5")
    monkeypatch.setattr(run_runtime, "_WEBAPP_ADDRESS_POLL_SECONDS", 0.01, raising=False)

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
        asyncio.run(run_runtime.start_gateway_runtime_api(runner))

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

    fast.setattr(run_runtime.asyncio, "sleep", record_sleep)
    api, calls = _api([OSError(err_no, msg)])
    with caplog.at_level(logging.WARNING, logger="gateway.run_runtime"):
        with pytest.raises(OSError) as caught:
            _start(api)
    assert caught.value.errno == err_no
    assert len(calls) == 1
    assert sleeps == []
    assert not any("not assigned to any interface yet" in r.getMessage() for r in caplog.records)


def test_zero_wait_does_not_sleep_or_warn(fast, caplog):
    fast.setenv("HERMES_WEBAPP_ADDRESS_WAIT", "0")
    sleeps = []

    async def record_sleep(delay):
        sleeps.append(delay)

    fast.setattr(run_runtime.asyncio, "sleep", record_sleep)
    api, calls = _api([OSError(errno.EADDRNOTAVAIL, "address not assigned to an interface")])
    with caplog.at_level(logging.WARNING, logger="gateway.run_runtime"):
        with pytest.raises(OSError) as caught:
            _start(api)
    assert caught.value.errno == errno.EADDRNOTAVAIL
    assert len(calls) == 1
    assert sleeps == []
    assert not any("not assigned to any interface yet" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf"])
def test_non_finite_override_falls_back_to_default(monkeypatch, caplog, raw):
    monkeypatch.setenv("HERMES_WEBAPP_ADDRESS_WAIT", raw)
    with caplog.at_level(logging.WARNING, logger="gateway.run_runtime"):
        assert run_runtime._webapp_address_wait_seconds() == run_runtime.WEBAPP_ADDRESS_WAIT_SECONDS
    assert any("HERMES_WEBAPP_ADDRESS_WAIT" in r.getMessage() for r in caplog.records)


def test_negative_override_clamps_to_zero_without_waiting(fast):
    fast.setenv("HERMES_WEBAPP_ADDRESS_WAIT", "-5")
    sleeps = []

    async def record_sleep(delay):
        sleeps.append(delay)

    fast.setattr(run_runtime.asyncio, "sleep", record_sleep)
    assert run_runtime._webapp_address_wait_seconds() == 0.0
    api, calls = _api([OSError(errno.EADDRNOTAVAIL, "address not assigned to an interface")])
    with pytest.raises(OSError) as caught:
        _start(api)
    assert caught.value.errno == errno.EADDRNOTAVAIL
    assert len(calls) == 1
    assert sleeps == []


def test_cancellation_during_wait_sleep_propagates(monkeypatch):
    monkeypatch.setenv("HERMES_WEBAPP_ADDRESS_WAIT", "5")
    monkeypatch.setattr(run_runtime, "_WEBAPP_ADDRESS_POLL_SECONDS", 10.0)
    real_sleep = asyncio.sleep

    async def run_and_cancel():
        entered_sleep = asyncio.Event()

        async def tracked_sleep(delay):
            entered_sleep.set()
            await real_sleep(delay)

        monkeypatch.setattr(run_runtime.asyncio, "sleep", tracked_sleep)
        api, calls = _api([OSError(errno.EADDRNOTAVAIL, "address not assigned to an interface")] * 5)
        task = asyncio.create_task(run_runtime._start_gateway_api_waiting_for_address(
            api, object(), host="100.64.0.5", port=9119, webapp="mount"))
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

    fast.setattr(run_runtime.asyncio, "sleep", record_sleep)
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
    monkeypatch.setenv("HERMES_WEBAPP_ADDRESS_WAIT", "5")
    monkeypatch.setattr(run_runtime, "_WEBAPP_ADDRESS_POLL_SECONDS", 5.0, raising=False)
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
            _startup_should_abort=lambda: shutdown_event.is_set(),
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
        await asyncio.wait_for(run_runtime.start_gateway_runtime_api(runner), timeout=2.0)
        assert completed.is_set(), "operator stop ran while the address wait was active"
        assert calls == [("192.0.2.77", False)], "no retry and no private fallback after operator stop"
        assert runner.session_api is None
        assert runner.webapp_status["state"] != "serving"
        assert "api_origin" not in runner.session_runtime_descriptor

    asyncio.run(run_scenario())


def test_restart_requested_during_wait_aborts_without_binding(monkeypatch):
    monkeypatch.setenv("HERMES_WEBAPP_ADDRESS_WAIT", "5")
    monkeypatch.setattr(run_runtime, "_WEBAPP_ADDRESS_POLL_SECONDS", 0.01)

    async def run_scenario():
        runner = types.SimpleNamespace(_restart_requested=False)
        calls = []

        async def start(r, **kwargs):
            calls.append(kwargs)
            r._restart_requested = True
            raise OSError(errno.EADDRNOTAVAIL, "address not assigned to an interface")

        api = types.SimpleNamespace(start_gateway_api=start, ListenerAddressError=run_api.ListenerAddressError)
        result = await run_runtime._start_gateway_api_waiting_for_address(
            api, runner, host="100.64.0.5", port=9119, webapp="mount"
        )
        assert result is None
        assert len(calls) == 1

    asyncio.run(run_scenario())


def test_bootstrap_abort_before_runner_awaits_stop_and_returns_exit_verdict():
    from gateway import run_bootstrap

    async def run_scenario(signal_initiated: bool):
        shutdown_event = asyncio.Event()
        stop_calls = []

        async def stop(*, restart=False, detached_restart=False, service_restart=False):
            stop_calls.append((restart, detached_restart, service_restart))
            shutdown_event.set()

        runner = types.SimpleNamespace(
            _stop_task=None,
            _shutdown_event=shutdown_event,
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
        verdict = await run_bootstrap._abort_startup_before_runner(runner, [signal_initiated])
        assert stop_calls == [(False, False, False)]
        assert shutdown_event.is_set()
        return verdict

    assert asyncio.run(run_scenario(False)) is True
    assert asyncio.run(run_scenario(True)) is False
