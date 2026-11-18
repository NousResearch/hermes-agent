"""Pinned (hosted) scopes: a backend declaring ``scope_pinning`` serves each ``(tenant, agent)`` its own rows only; any other backend refuses a pin at ``open``. Every case runs on every registered backend: the pinning cases where the backend declares it, the refusal case where it does not, never a skip."""

from __future__ import annotations

import pytest


def test_backend_without_pinning_refuses_a_pinned_scope(backend_name: str, backend) -> None:
    """A backend not declaring scope_pinning refuses a pinned ProfileScope at open with ScopePinRefused, before touching a store."""
    pytest.fail("not implemented yet")


def test_pinned_handles_see_only_their_own_tenant(backend_name: str, backend) -> None:
    """Tenant B's pinned handle never sees tenant A's sessions or messages, and a foreign id reads as missing."""
    pytest.fail("not implemented yet")


def test_pinning_an_agent_never_widens_the_tenant(backend_name: str, backend) -> None:
    """Same tenant, two agents: each agent's handle sees only its own rows."""
    pytest.fail("not implemented yet")


def test_cross_tenant_write_is_refused_or_invisible(backend_name: str, backend) -> None:
    """A write through B's handle naming A's session id is refused as missing and leaves A's rows unchanged."""
    pytest.fail("not implemented yet")


def test_assert_isolating_matches_declared_pinning(backend_name: str, backend) -> None:
    """assert_isolating passes on a backend declaring scope_pinning and raises IsolationUnsupported on any other."""
    pytest.fail("not implemented yet")

