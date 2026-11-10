"""The relay guard: a missing egress module or parent package is absence; a failure inside it is not."""
from __future__ import annotations

import pytest

from hermes_agent.gateway.relay.egress import _is_missing_gateway_relay


@pytest.mark.parametrize("name", ["hermes_agent.gateway", "hermes_agent.gateway.relay", "hermes_agent.gateway.relay.egress"])
def test_relay_guard_reads_a_missing_target_or_parent_as_absence(name):
    assert _is_missing_gateway_relay(ModuleNotFoundError(name=name)) is True


@pytest.mark.parametrize("exc", [
    ModuleNotFoundError(name="hermes_agent.gateway.relay.egress.dependency"),
    ModuleNotFoundError(name="hermes_agent.gateway.relay.egress_dependency"),
    ModuleNotFoundError(name="requests"),
    ModuleNotFoundError(name=None),
    ImportError(name="hermes_agent.gateway.relay.egress"),
])
def test_relay_guard_reads_anything_else_as_a_fault(exc):
    assert _is_missing_gateway_relay(exc) is False
