"""Structured Kanban ready-contract validation.

The ready contract is the machine-readable admission SSOT for dispatchable
Kanban tasks. Markdown card bodies may render the same information for humans,
but strict ready admission must validate this structure instead of accepting
marker-only prose.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

READY_CONTRACT_SCHEMA = "kanban_ready_contract.v1"

_REQUIRED_SEQUENCE_FIELDS = (
    "scope",
    "non_goals",
    "acceptance_criteria",
    "done_criteria",
    "verification_requirements",
)
_REQUIRED_MAPPING_FIELDS = (
    "repo_lane_truth",
    "routing_verdict",
    "authority_boundary",
    "risk_flags",
    "dependencies_blockers",
    "review_package_expectation",
)
_REQUIRED_TEXT_FIELDS = ("goal", "end_state")


@dataclass(frozen=True)
class ReadyContractValidation:
    accepted: bool
    reason_codes: list[str] = field(default_factory=list)
    ready_contract: dict[str, Any] | None = None

    def to_gate_payload(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason_codes": list(self.reason_codes),
            "ready_contract": self.ready_contract,
        }


def _text(value: Any) -> str:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return ""
    return str(value).strip()


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_nonempty_sequence(value: Any) -> list[Any]:
    if isinstance(value, list):
        return [item for item in value if item not in (None, "", [], {})]
    if isinstance(value, tuple):
        return [item for item in value if item not in (None, "", [], {})]
    return []


def _profile_exists(name: str, checker: Callable[[str], bool] | None) -> bool:
    if not name or checker is None:
        return True
    try:
        return bool(checker(name))
    except Exception:
        return False


def render_ready_contract_markdown(contract: Mapping[str, Any]) -> str:
    """Render a validated-ish ready contract into canonical human-facing body text."""
    data = _as_mapping(contract)

    def lines_for(value: Any) -> list[str]:
        seq = _as_nonempty_sequence(value)
        if not seq:
            return ["- <none>"]
        return [f"- {item}" for item in seq]

    sections: list[str] = [
        "# Structured Kanban Ready Contract",
        "",
        f"schema: {_text(data.get('schema')) or READY_CONTRACT_SCHEMA}",
        "",
        "## Goal",
        _text(data.get("goal")) or "<missing>",
        "",
        "## End state",
        _text(data.get("end_state")) or "<missing>",
        "",
        "## Scope",
        *lines_for(data.get("scope")),
        "",
        "## Non-goals",
        *lines_for(data.get("non_goals")),
        "",
        "## Acceptance criteria",
        *lines_for(data.get("acceptance_criteria")),
        "",
        "## Done criteria",
        *lines_for(data.get("done_criteria")),
        "",
        "## Verification requirements",
        *lines_for(data.get("verification_requirements")),
        "",
        "## Structured contract JSON",
        "```json",
    ]
    import json

    sections.append(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))
    sections.extend(["```", ""])
    return "\n".join(sections)


def validate_ready_contract(
    value: Any,
    *,
    goal_mode: bool = False,
    assignee: str | None = None,
    profile_exists: Callable[[str], bool] | None = None,
) -> ReadyContractValidation:
    """Validate a structured ready contract for strict ready admission."""

    contract = _as_mapping(value)
    if not contract:
        return ReadyContractValidation(False, ["missing_structured_ready_contract"], None)

    reason_codes: list[str] = []
    if _text(contract.get("schema")) != READY_CONTRACT_SCHEMA:
        reason_codes.append("invalid_ready_contract_schema")

    for field_name in _REQUIRED_TEXT_FIELDS:
        if not _text(contract.get(field_name)):
            reason_codes.append(f"missing_ready_contract_{field_name}")

    for field_name in _REQUIRED_SEQUENCE_FIELDS:
        if not _as_nonempty_sequence(contract.get(field_name)):
            reason_codes.append(f"missing_ready_contract_{field_name}")

    for field_name in _REQUIRED_MAPPING_FIELDS:
        if not _as_mapping(contract.get(field_name)):
            reason_codes.append(f"missing_ready_contract_{field_name}")

    routing = _as_mapping(contract.get("routing_verdict"))
    contract_assignee = _text(routing.get("assignee"))
    effective_assignee = _text(assignee) or contract_assignee
    if not _text(routing.get("verdict")):
        reason_codes.append("missing_ready_contract_routing_verdict")
    if not effective_assignee:
        reason_codes.append("missing_ready_contract_assignee")
    elif not _profile_exists(effective_assignee, profile_exists):
        reason_codes.append("unknown_ready_contract_assignee")

    authority = _as_mapping(contract.get("authority_boundary"))
    if not _as_nonempty_sequence(authority.get("forbidden")):
        reason_codes.append("missing_ready_contract_forbidden_actions")

    repo_lane = _as_mapping(contract.get("repo_lane_truth"))
    if not (_text(repo_lane.get("repository")) or _text(repo_lane.get("repo"))):
        reason_codes.append("missing_ready_contract_repository")
    if not _text(repo_lane.get("branch")):
        reason_codes.append("missing_ready_contract_branch")

    reviewer_loop = _as_mapping(contract.get("reviewer_loop"))
    reviewer_required = reviewer_loop.get("required") is True
    reviewer_profile = _text(reviewer_loop.get("reviewer_profile"))
    if reviewer_required:
        if not goal_mode:
            reason_codes.append("ready_contract_reviewer_loop_requires_goal_mode")
        if not reviewer_profile:
            reason_codes.append("missing_ready_contract_reviewer_profile")
        elif not _profile_exists(reviewer_profile, profile_exists):
            reason_codes.append("unknown_ready_contract_reviewer_profile")

    return ReadyContractValidation(
        accepted=not reason_codes,
        reason_codes=list(dict.fromkeys(reason_codes)),
        ready_contract=contract,
    )
