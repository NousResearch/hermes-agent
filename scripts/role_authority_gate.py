#!/usr/bin/env python3
"""Small deterministic authority gate for semantic role transitions.

The gate is an explicit reusable API and a JSON CLI. It is intentionally not
wired into every Ares runtime or publication consumer yet; callers must opt in
until those integration paths are separately authorized and tested.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence


REGISTRY_PATH = Path(__file__).resolve().parents[1] / "docs/role-contracts/role-contracts.json"
UNCONNECTED_CONSUMER_NOTE = (
    "This gate is not connected to every Ares runtime or publication path; "
    "consumers must call the explicit API or CLI until integration is separately verified."
)


class Role(StrEnum):
    SUPERVISOR = "role.ares_supervisor"
    EXPLORER = "role.explorer"
    PUBLIC = "role.public_evidence_editor"
    DATA_EVIDENCE = "role.data_evidence"


class Action(StrEnum):
    PUBLICATION_READY = "publication_ready"
    RECONCILIATION = "reconciliation"
    PROMOTION = "promotion"
    FIV_PROMOTION = "fiv_promotion"
    RUNTIME_AUTHORITY = "runtime_authority"


class GateInputError(ValueError):
    """Raised when a request is malformed or uses the wrong role/action pair."""


@dataclass(frozen=True)
class AuthorityRequest:
    role: Role
    action: Action
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class GateResult:
    allowed: bool
    code: str
    reason: str

    @classmethod
    def allow(cls) -> "GateResult":
        return cls(True, "allowed", "authority transition accepted")


_DEFAULT_NON_VERIFYING = frozenset({"blocked", "unknown", "superseded"})


def _canonical_registry() -> Mapping[str, Any]:
    try:
        value = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateInputError(f"canonical role registry unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise GateInputError("canonical role registry must be an object")
    validator_path = Path(__file__).resolve().with_name("validate_role_contracts.py")
    spec = importlib.util.spec_from_file_location("_role_contract_validator", validator_path)
    if spec is None or spec.loader is None:
        raise GateInputError("role registry validator unavailable")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    errors = validator.validate_registry(value)
    if errors:
        raise GateInputError("canonical role registry invalid: " + "; ".join(errors))
    return value


def _role_contract(registry: Mapping[str, Any], role: Role) -> Mapping[str, Any]:
    for item in registry.get("roles", []):
        if isinstance(item, dict) and item.get("role_id") == role.value:
            return item
    raise GateInputError(f"canonical role contract missing: {role.value}")


def _string_list(payload: Mapping[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise GateInputError(f"payload.{key} must be a list of non-empty strings")
    return value


def _require_action_role(request: AuthorityRequest, expected_role: Role) -> None:
    if request.role is not expected_role:
        raise GateInputError(f"{request.action.value} requires {expected_role.value}")


def _publication_gate(payload: Mapping[str, Any]) -> GateResult:
    blockers = _string_list(payload, "claim_blockers") + _string_list(payload, "evidence_blockers")
    if blockers:
        return GateResult(False, "blocking_evidence", "publication_ready requires no claim or evidence blockers")
    return GateResult.allow()


def _explorer_gate(payload: Mapping[str, Any], registry: Mapping[str, Any]) -> GateResult:
    preservation = payload.get("preservation")
    reference = payload.get("preserved_artifact_ref")
    policy = _role_contract(registry, Role.EXPLORER).get("dissent_policy", {})
    required_preservation = policy.get("preservation", "preserved_artifact")
    if preservation != required_preservation or not isinstance(reference, str) or not reference.strip():
        return GateResult(
            False,
            "dissent_not_preserved",
            "Explorer reconciliation must preserve dissent and retain a preserved artifact reference",
        )
    return GateResult.allow()


def _data_evidence_gate(payload: Mapping[str, Any], registry: Mapping[str, Any]) -> GateResult:
    promotion = payload.get("promotion")
    if not isinstance(promotion, str) or not promotion.strip():
        raise GateInputError("payload.promotion must be a non-empty string")
    authority = _role_contract(registry, Role.DATA_EVIDENCE).get("authority", {})
    allowed = authority.get("can_promote", [])
    if promotion not in allowed:
        return GateResult(False, "promotion_forbidden", f"Data/Evidence cannot promote {promotion}")
    return GateResult.allow()


def _fiv_gate(payload: Mapping[str, Any], registry: Mapping[str, Any]) -> GateResult:
    stages = payload.get("stages")
    if not isinstance(stages, list) or not all(isinstance(stage, dict) for stage in stages):
        raise GateInputError("payload.stages must be a list of stage objects")
    expected = ("finding", "implementation", "verification")
    states = set(registry.get("evidence_states", ())) or set(_DEFAULT_NON_VERIFYING)
    if len(stages) != len(expected) or tuple(stage.get("kind") for stage in stages) != expected:
        return GateResult(False, "fiv_not_verifiable", "F→I→V promotion requires finding, implementation, and verification stages")
    if any(stage.get("evidence_state") not in states for stage in stages):
        return GateResult(False, "fiv_not_verifiable", "F→I→V stages must use canonical evidence states")
    verification_state = stages[2].get("evidence_state")
    if verification_state in _DEFAULT_NON_VERIFYING:
        return GateResult(False, "fiv_not_verifiable", "F→I→V promotion is blocked by non-verifying verification state")
    return GateResult.allow()


def _runtime_authority_gate() -> GateResult:
    return GateResult(
        False,
        "runtime_authority_not_granted",
        "runtime authority is outside this semantic gate and is not granted by it",
    )


def _evaluate(request: AuthorityRequest, registry: Mapping[str, Any]) -> GateResult:
    if not isinstance(request, AuthorityRequest):
        raise GateInputError("request must be an AuthorityRequest")
    if request.action is Action.PUBLICATION_READY:
        _require_action_role(request, Role.PUBLIC)
        return _publication_gate(request.payload)
    if request.action is Action.RECONCILIATION:
        _require_action_role(request, Role.EXPLORER)
        return _explorer_gate(request.payload, registry)
    if request.action is Action.PROMOTION:
        _require_action_role(request, Role.DATA_EVIDENCE)
        return _data_evidence_gate(request.payload, registry)
    if request.action is Action.FIV_PROMOTION:
        _require_action_role(request, Role.SUPERVISOR)
        return _fiv_gate(request.payload, registry)
    if request.action is Action.RUNTIME_AUTHORITY:
        _require_action_role(request, Role.SUPERVISOR)
        return _runtime_authority_gate()
    raise GateInputError(f"unsupported action: {request.action}")


def evaluate(request: AuthorityRequest) -> GateResult:
    """Evaluate a request against the validated canonical registry."""
    return _evaluate(request, _canonical_registry())


def _evaluate_for_tests(request: AuthorityRequest, registry: Mapping[str, Any]) -> GateResult:
    """Test-only seam; production callers must use evaluate()."""
    return _evaluate(request, registry)


def _request_from_json(value: Any) -> AuthorityRequest:
    if not isinstance(value, dict) or not isinstance(value.get("payload"), dict):
        raise GateInputError("request JSON must contain role, action, and object payload")
    try:
        role = Role(value["role"])
        action = Action(value["action"])
    except (KeyError, ValueError) as exc:
        raise GateInputError("request JSON has an unknown role or action") from exc
    return AuthorityRequest(role=role, action=action, payload=value["payload"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path, help="JSON file containing one AuthorityRequest")
    args = parser.parse_args(argv)
    try:
        request = _request_from_json(json.loads(args.request.read_text(encoding="utf-8")))
        result = evaluate(request)
    except (OSError, json.JSONDecodeError, GateInputError) as exc:
        print(json.dumps({"allowed": False, "code": "invalid_request", "reason": str(exc)}))
        return 2
    print(json.dumps({"allowed": result.allowed, "code": result.code, "reason": result.reason, "consumer_note": UNCONNECTED_CONSUMER_NOTE}))
    return 0 if result.allowed else 1


if __name__ == "__main__":
    raise SystemExit(main())
