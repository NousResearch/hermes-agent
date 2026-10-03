"""A degraded gateway is serving, on the fallback path too.

``degraded`` is a whole-life serving state: the gateway runs with one platform
parked. ``gateway.status`` learned that in b97b0c6bc9, but
``build_gateway_health_snapshot`` keeps a local approximation for callers that
cannot import the gateway package, and that copy still accepted ``running``
alone -- so the exported ``busy``/``drainable`` signals flipped depending on
whether the import happened to succeed.
"""

from __future__ import annotations

import sys

import pytest

from agent.monitoring.gateway_health import build_gateway_health_snapshot


def _signals(state: str, *, active_agents: int = 2) -> dict[str, float]:
    snapshot = build_gateway_health_snapshot(
        {"gateway_state": state, "active_agents": active_agents},
        gateway_running=True, profile="default", install_id="install-1", version="1.0.0",
    )
    return {
        metric.name: metric.value
        for metric in snapshot.metrics
        if metric.name in ("hermes.gateway.busy", "hermes.gateway.drainable")
    }


@pytest.fixture
def no_gateway_status(monkeypatch):
    """Make ``import gateway.status`` fail so the local approximation is used."""
    monkeypatch.setitem(sys.modules, "gateway.status", None)


@pytest.mark.parametrize("state", ["running", "degraded"])
def test_serving_states_are_busy_and_drainable(state):
    assert _signals(state) == {"hermes.gateway.busy": 1, "hermes.gateway.drainable": 1}


@pytest.mark.parametrize("state", ["running", "degraded"])
def test_fallback_agrees_with_gateway_status(state, no_gateway_status):
    # The NAS drain gate reads these metrics: a degraded gateway with in-flight
    # turns reported idle+undrainable would be drained out from under them.
    assert _signals(state) == {"hermes.gateway.busy": 1, "hermes.gateway.drainable": 1}


def test_fallback_idle_serving_gateway_is_drainable_but_not_busy(no_gateway_status):
    assert _signals("degraded", active_agents=0) == {
        "hermes.gateway.busy": 0, "hermes.gateway.drainable": 1,
    }


@pytest.mark.parametrize("state", ["starting", "draining", "stopping", "stopped"])
def test_fallback_non_serving_states_stay_idle(state, no_gateway_status):
    assert _signals(state) == {"hermes.gateway.busy": 0, "hermes.gateway.drainable": 0}
