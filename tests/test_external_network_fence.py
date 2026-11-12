"""No test reaches a host off this machine unless it is a ``live`` canary (``tests/conftest.py``).

A test home has no Nous identity, so a hosted connector action used to mint a real guest on
the production Portal: seconds per test, red when the Portal was slow, and a new production
guest on every CI run. Every other live host had the same shape: building an agent fetched
OpenRouter / models.dev / model-catalog metadata, and verdicts followed whatever came back.
"""

import socket

import pytest


def test_first_connector_action_in_a_test_home_mints_no_portal_guest():
    from hermes_cli import anon_auth
    from hermes_agent.tools.connectors.gateway.config import ensure_guest_identity

    failure = ensure_guest_identity()

    assert failure and failure.startswith("Hosted connectors could not be set up.")
    assert not anon_auth.has_guest()


@pytest.mark.parametrize("host", [
    "portal.nousresearch.com", "inference-api.nousresearch.com", b"WELCOME-API.NOUSRESEARCH.COM.",
    "telemetry.staging-nousresearch.com", "openrouter.ai", "models.dev", "raw.githubusercontent.com",
    "example.com", b"pypi.org",
])
def test_external_hosts_are_refused_at_the_name_lookup(host):
    with pytest.raises(socket.gaierror, match="test fence"):
        socket.getaddrinfo(host, 443)


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "::1", "169.254.169.254", None])
def test_loopback_and_address_literals_still_resolve(host):
    assert socket.getaddrinfo(host, 80, type=socket.SOCK_STREAM)


@pytest.mark.parametrize("host", ["LOCALHOST.", "api.localhost", "[::1]", socket.gethostname()])
def test_names_of_this_machine_are_not_fenced(host):
    try:
        socket.getaddrinfo(host, 80, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:  # the host's own resolver may not know them; the fence must not refuse
        assert "test fence" not in str(exc)


@pytest.fixture(scope="module")
def _lookup_outside_any_test():
    """Runs before any test-scoped fixture: where a per-test fence had not been installed yet,
    and where a thread a previous test left running does its lookups."""
    try:
        socket.getaddrinfo("openrouter.ai", 443)
    except socket.gaierror as exc:
        return str(exc)
    return "resolved"


def test_lookups_outside_a_test_are_fenced_too(_lookup_outside_any_test):
    assert "test fence" in _lookup_outside_any_test
