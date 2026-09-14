"""Protected contract tests for the Kanban versioned routing contract.

These tests use only synthetic in-memory mappings.  The module under test is
intentionally inactive and has no dispatcher, gateway, filesystem, or network
imports.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from typing import Any

LANES = (
    "planning",
    "research",
    "coding",
    "review",
    "security",
    "ops",
)


def _api() -> Any:
    return importlib.import_module("hermes_cli.kanban_route_contract")


def _contract(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "routing_contract_version": 1,
        "default_assignee": "",
        "orchestrator_profile": "kanban-coordinator",
        "lanes": {
            "planning": {"profile": "kanban-planning", "execution": "read_only"},
            "research": {"profile": "kanban-research", "execution": "read_only"},
            "coding": {"profile": "kanban-coding", "execution": "remote_isolated"},
            "review": {"profile": "kanban-review", "execution": "remote_isolated"},
            "security": {"profile": "kanban-security", "execution": "remote_isolated"},
            "ops": {"profile": "kanban-ops", "execution": "remote_approval_gated"},
        },
    }
    values.update(overrides)
    return values


def _parse(raw: object) -> Any:
    module = _api()
    return module.RouteContract.from_mapping(raw)


def test_missing_malformed_and_unsupported_contract_fail_closed() -> None:
    module = _api()
    bad_contracts = [
        None,
        {},
        "not-a-mapping",
        _contract(routing_contract_version=2),
        _contract(routing_contract_version=1.0),
        _contract(routing_contract_version=True),
        _contract(orchestrator_profile=""),
        _contract(default_assignee="kanban-coding"),
        _contract(unknown_key="x"),
    ]
    for raw in bad_contracts:
        contract = _parse(raw)
        assert contract.valid is False, raw
        decision = module.resolve_route(contract, task_id="t_0123456789abcdef")
        assert decision.status == "blocked", raw
        assert decision.profile is None, raw


def test_lane_map_failures_fail_closed() -> None:
    bad_lane_maps = [
        None,
        "x",
        {},
        {lane: {"profile": f"p-{lane}", "execution": "read_only"} for lane in LANES if lane != "ops"},
        dict(
            {lane: {"profile": f"p-{lane}", "execution": "read_only"} for lane in LANES},
            extra_lane={"profile": "p-extra", "execution": "read_only"},
        ),
        dict(
            {lane: {"profile": f"p-{lane}", "execution": "read_only"} for lane in LANES},
            ops={"profile": "p-ops", "execution": "local"},
        ),
        dict(
            {lane: {"profile": f"p-{lane}", "execution": "read_only"} for lane in LANES},
            ops={"profile": "p-ops"},
        ),
        dict(
            {lane: {"profile": f"p-{lane}", "execution": "read_only"} for lane in LANES},
            ops={"profile": "p-ops", "execution": "read_only", "extra": 1},
        ),
        dict(
            {lane: {"profile": f"p-{lane}", "execution": "read_only"} for lane in LANES},
            ops={"profile": "", "execution": "read_only"},
        ),
        dict(
            {lane: {"profile": f"p-{lane}", "execution": "read_only"} for lane in LANES},
            ops="not-a-mapping",
        ),
        dict(
            {lane: {"profile": f"p-{lane}", "execution": "read_only"} for lane in LANES},
            ops={"profile": "p-ops", "execution": 7},
        ),
        dict(
            {lane: {"profile": f"p-{lane}", "execution": "read_only"} for lane in LANES},
            ops={"profile": "P-Ops", "execution": "read_only"},
        ),
        dict(
            {lane: {"profile": f"p-{lane}", "execution": "read_only"} for lane in LANES},
            ops={"profile": " p-ops", "execution": "read_only"},
        ),
    ]
    for lanes in bad_lane_maps:
        contract = _parse(_contract(lanes=lanes))
        assert contract.valid is False, lanes


def test_profile_separation_is_enforced() -> None:
    for lane_a, lane_b in (
        ("coding", "review"),
        ("coding", "security"),
        ("review", "security"),
        ("coding", "ops"),
        ("review", "ops"),
        ("security", "ops"),
    ):
        base = {lane: {"profile": f"p-{lane}", "execution": "read_only"} for lane in LANES}
        base[lane_b]["profile"] = f"p-{lane_a}"  # type: ignore[index]
        contract = _parse(_contract(lanes=base))
        assert contract.valid is False, (lane_a, lane_b)


def test_valid_contract_parses_and_hashes_deterministically() -> None:
    contract = _parse(_contract())
    assert contract.valid is True
    assert contract.routing_contract_version == 1
    assert contract.default_assignee == ""
    assert contract.hash == _parse(_contract()).hash
    assert isinstance(contract.hash, str) and len(contract.hash) == 64

    reordered = _contract()
    reordered_lanes = dict(reversed(list(reordered["lanes"].items())))  # type: ignore[call-overload]
    reordered["lanes"] = reordered_lanes
    assert _parse(reordered).hash == contract.hash

    changed = _contract()
    changed["lanes"]["coding"]["profile"] = "kanban-coding-2"  # type: ignore[index]
    assert _parse(changed).hash != contract.hash

    for mutation in (
        {"orchestrator_profile": "kanban-coordinator-2"},
        {
            "lanes": dict(
                _contract()["lanes"],  # type: ignore[call-overload]
                ops={"profile": "p-ops", "execution": "remote_isolated"},
            )
        },
    ):
        mutated = _contract() | mutation
        assert _parse(mutated).hash != contract.hash


def test_hash_is_exact_canonical_sha256_construction() -> None:
    contract = _parse(_contract())
    expected_input = {
        "routing_contract_version": 1,
        "default_assignee": "",
        "orchestrator_profile": "kanban-coordinator",
        "lanes": {
            lane: {"profile": f"kanban-{lane}", "execution": spec["execution"]}
            for lane, spec in _contract()["lanes"].items()  # type: ignore[union-attr]
        },
    }
    expected = hashlib.sha256(
        json.dumps(expected_input, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert contract.hash == expected


def test_parsed_contract_is_deeply_immutable() -> None:
    import dataclasses

    contract = _parse(_contract())
    assert contract.valid is True
    for mutate in (
        lambda: contract.lanes.__setitem__(  # type: ignore[attr-defined]
            "coding", {"profile": "x", "execution": "read_only"}
        ),
        lambda: contract.lanes["coding"].__setitem__("profile", "x"),  # type: ignore[attr-defined]
        lambda: contract.lanes.__delitem__("coding"),  # type: ignore[attr-defined]
    ):
        try:
            mutate()
        except (TypeError, AttributeError):
            pass
        else:
            raise AssertionError("contract internals were mutable")
    # A derived copy via dataclasses.replace must not affect the original.
    derived = dataclasses.replace(contract, hash="0" * 64)
    assert derived.hash == "0" * 64
    assert contract.hash != derived.hash
    assert contract.valid is True
    assert contract.hash == _parse(_contract()).hash


def test_explicit_assignee_is_authoritative() -> None:
    module = _api()
    contract = _parse(_contract())
    decision = module.resolve_route(
        contract,
        task_id="t_0123456789abcdef",
        explicit_assignee="kanban-review",
        lane="coding",
    )
    assert decision.status == "routed"
    assert decision.profile == "kanban-review"
    assert decision.routing_contract_version == 1
    assert decision.routing_contract_hash == contract.hash


def test_lane_resolution_persists_contract_identity() -> None:
    module = _api()
    contract = _parse(_contract())
    decision = module.resolve_route(
        contract,
        task_id="t_0123456789abcdef",
        explicit_assignee=None,
        lane="coding",
    )
    assert decision.status == "routed"
    assert decision.profile == "kanban-coding"
    assert decision.routing_contract_version == contract.routing_contract_version
    assert decision.routing_contract_hash == contract.hash


def test_no_route_holds_in_triage_without_default_fallback() -> None:
    module = _api()
    contract = _parse(_contract())
    decision = module.resolve_route(
        contract,
        task_id="t_0123456789abcdef",
        explicit_assignee=None,
        lane=None,
    )
    assert decision.status == "triage"
    assert decision.profile is None
    assert "diagnostic" in decision.reason

    unknown_lane = module.resolve_route(
        contract,
        task_id="t_0123456789abcdef",
        explicit_assignee=None,
        lane="does-not-exist",
    )
    assert unknown_lane.status == "triage"
    assert unknown_lane.profile is None
    assert "diagnostic" in unknown_lane.reason


def test_unknown_explicit_assignee_does_not_fallback_to_lane() -> None:
    module = _api()
    contract = _parse(_contract())
    decision = module.resolve_route(
        contract,
        task_id="t_0123456789abcdef",
        explicit_assignee="",
        lane="coding",
    )
    # An invalid (empty) explicit assignee is not authoritative and must not
    # silently fall back to a lane resolution either.
    assert decision.status == "triage"
    assert decision.profile is None
