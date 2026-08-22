"""SIGUSR2 faulthandler dump must not chain to the default kill (#84373)."""

from __future__ import annotations

import inspect

from gateway.run import GatewayRunner


def test_gateway_sigusr2_registration_does_not_chain():
    src = inspect.getsource(GatewayRunner.start)
    assert "faulthandler.register" in src
    # Prefer explicit non-destructive registration.
    assert "chain=False" in src
    # Guard against reintroducing the destructive default.
    assert "chain=True" not in src
