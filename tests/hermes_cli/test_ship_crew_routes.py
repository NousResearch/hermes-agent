from __future__ import annotations

import pytest

from hermes_cli.ship_crew_routes import (
    RoutePolicyError,
    RouteRegistry,
    RouteRequest,
    select_route,
)


def _registry(calibrated: bool = True):
    route = {
        "provider": "configured-provider",
        "model": "approved-model",
        "executor": "hermes-native",
        "mode": "medium",
        "reasoning_effort": "medium",
        "quota_domain": "provider-a",
        "write_scope": "read-only",
        "profiles": ["captain", "navigator", "pirate"],
        "complexity_tiers": ["T1", "T2"],
        "output_classes": ["O0", "O1", "O2"],
        "fallback_routes": [],
    }
    if calibrated:
        route["calibration"] = {
            "route_id": "readonly-medium",
            "executor": "hermes-native",
            "model": "approved-model",
            "mode": "medium",
            "authenticated": True,
            "protocol_valid": True,
            "acceptance_fixture": True,
            "receipt_ref": "/tmp/calibration.json",
        }
    return RouteRegistry.from_mapping({"schema_version": "crew-route-registry/v1", "routes": {"readonly-medium": route}})


def test_t0_is_deterministic_and_uncalibrated_routes_are_inactive():
    registry = _registry(calibrated=False)
    selection = select_route(RouteRequest("navigator", "T0", "O0", "read-only", "lite"), registry)
    assert selection.effective_route_id is None
    with pytest.raises(RoutePolicyError, match="no_route"):
        select_route(RouteRequest("navigator", "T1", "O0", "read-only", "lite"), registry)


def test_exact_calibrated_route_selects_and_persists_mode_semantics():
    selection = select_route(RouteRequest("navigator", "T1", "O1", "read-only", "standard"), _registry())
    assert selection.effective_route_id == "readonly-medium"
    assert selection.executor == "hermes-native"
    assert selection.mode == "medium"
    assert selection.reasoning_effort == "medium"


def test_profile_write_scope_and_excluded_codex_fail_closed():
    registry = _registry()
    write_route = dict(registry.routes["readonly-medium"].__dict__)
    write_route["write_scope"] = "worktree"
    write_route.pop("calibration")
    write_route["calibration"] = {
        "route_id": "readonly-medium", "executor": "hermes-native", "model": "approved-model", "mode": "medium",
        "authenticated": True, "protocol_valid": True, "acceptance_fixture": True, "receipt_ref": "/tmp/calibration.json",
    }
    registry = RouteRegistry.from_mapping({"schema_version": "crew-route-registry/v1", "routes": {"readonly-medium": write_route}})
    with pytest.raises(RoutePolicyError, match="no_route"):
        select_route(RouteRequest("pirate", "T1", "O1", "worktree", "standard"), registry)

    excluded = dict(write_route)
    excluded["model"] = "Codex GPT-5.6"
    excluded["calibration"] = {**excluded["calibration"], "model": "Codex GPT-5.6"}
    with pytest.raises(RoutePolicyError, match="no_route"):
        select_route(RouteRequest("captain", "T1", "O1", "read-only", "lite"), RouteRegistry.from_mapping({"schema_version": "crew-route-registry/v1", "routes": {"readonly-medium": excluded}}))


def test_fallback_cycles_and_unknown_routes_reject():
    raw = {
        "schema_version": "crew-route-registry/v1",
        "routes": {
            "a": {"provider": "p", "model": "m", "executor": "e", "mode": "x", "quota_domain": "q", "write_scope": "read-only", "profiles": ["captain"], "complexity_tiers": ["T1"], "output_classes": ["O0"], "fallback_routes": ["b"]},
            "b": {"provider": "p", "model": "m2", "executor": "e", "mode": "x", "quota_domain": "q", "write_scope": "read-only", "profiles": ["captain"], "complexity_tiers": ["T1"], "output_classes": ["O0"], "fallback_routes": ["a"]},
        },
    }
    with pytest.raises(RoutePolicyError, match="fallback_cycle"):
        RouteRegistry.from_mapping(raw)

    raw["routes"]["b"]["fallback_routes"] = ["missing"]
    with pytest.raises(RoutePolicyError, match="fallback_unknown"):
        RouteRegistry.from_mapping(raw)
