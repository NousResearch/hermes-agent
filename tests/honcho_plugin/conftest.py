"""Package-level isolation for Honcho unit tests.

Contract (B8): unit tests in this package make ZERO network requests, even
when a live local Honcho is reachable and ambient production config exists.
A real incident proved unique fixtures are not enough - test messages were
written into the production workspace. The guard below turns any connection
attempt into a hard failure; individually marked tests may opt out with
@pytest.mark.allow_network.
"""

import socket
import threading

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "allow_network: opt-in marker for tests that intentionally touch the network",
    )
    config.addinivalue_line(
        "markers",
        "expect_network_attempts: test asserts on blocked attempts itself",
    )


@pytest.fixture
def network_attempts():
    """Recorded connection attempts (visible to tests for assertions)."""
    return []


@pytest.fixture(autouse=True)
def _no_network(request, monkeypatch, network_attempts):
    """Fail any test in this package that attempts a real socket connection.

    Recording + teardown assert (not just raising) catches attempts even when
    intermediate code swallows the exception (the async writer retries and
    logs errors instead of propagating them).
    """
    if request.node.get_closest_marker("allow_network"):
        yield
        return

    def _blocked_connect(self, address, *args, **kwargs):
        network_attempts.append(address)
        raise RuntimeError(f"network disabled in honcho unit tests: {address!r}")

    def _blocked_create_connection(address, *args, **kwargs):
        network_attempts.append(address)
        raise RuntimeError(f"network disabled in honcho unit tests: {address!r}")

    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)
    monkeypatch.setattr(socket, "create_connection", _blocked_create_connection)
    yield
    leftover = list(network_attempts)
    if request.node.get_closest_marker("expect_network_attempts"):
        return
    assert not leftover, (
        f"unit test attempted network connections: {leftover} - "
        "inject a fake client/factory before constructing the manager"
    )


@pytest.fixture(autouse=True)
def _no_leaked_writer_threads():
    """Every async manager must be shut down by the test that created it."""
    yield
    leaked = [
        t for t in threading.enumerate()
        if t.name == "honcho-async-writer" and t.is_alive()
    ]
    assert not leaked, (
        f"leaked honcho-async-writer threads: {len(leaked)} - "
        "call manager.shutdown() (use the make_manager fixture)"
    )
