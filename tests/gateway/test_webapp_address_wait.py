"""A specific ``webapp.host`` an interface is given after the gateway starts is waited for, briefly.

A tailnet (Tailscale/Headscale/WireGuard) address, or a DHCP lease, is assigned a moment after its daemon
reports ready, so a gateway that starts with ``webapp.host`` set to it can lose the first bind with
EADDRNOTAVAIL. Without a retry the gateway keeps running with the Webapp off and nothing ever tries again:
no crash, so no ``Restart=`` policy fires. The wait is bounded, and every other bind failure still reports
at once.
"""
import asyncio
import errno
import types

import pytest

from gateway import run_api, run_runtime


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
    api, calls = _api([OSError(errno.EADDRNOTAVAIL, "address not assigned to an interface")])
    assert _start(api, webapp=None, host="127.0.0.1", port=0) == "handle"
    assert calls == [{}, {}], "main's listener is started with no address of its own"


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
