"""HTR identity helpers — stable prefixed IDs without external services."""

from __future__ import annotations

import re
import secrets
from datetime import datetime, timezone
from typing import Literal

IdKind = Literal[
    "goal",
    "run",
    "task",
    "attempt",
    "event",
    "tool_call",
    "artifact",
    "verification",
    "heal_cycle",
    "approval",
    "reconciliation",
    "marker_disposition",
    "marker_disposition_approval",
    "marker_disposition_claim",
    "marker_disposition_attempt",
    "recovery_run_request",
    "recovery_run_approval",
    "recovery_run_claim",
    "recovery_run_attempt",
    "bounded_action_proposal",
    "bounded_action_review_decision",
    "bounded_action_escalation",
    "project",
]

ID_PREFIXES: dict[IdKind, str] = {
    "goal": "goal_",
    "run": "run_",
    "task": "task_",
    "attempt": "att_",
    "event": "evt_",
    "tool_call": "tc_",
    "artifact": "art_",
    "verification": "ver_",
    "heal_cycle": "heal_",
    "approval": "apr_",
    "reconciliation": "rcn_",
    "marker_disposition": "mdp_",
    "marker_disposition_approval": "mda_",
    "marker_disposition_claim": "mdc_",
    "marker_disposition_attempt": "mat_",
    "recovery_run_request": "rcr_",
    "recovery_run_approval": "rap_",
    "recovery_run_claim": "rcl_",
    "recovery_run_attempt": "rat_",
    "bounded_action_proposal": "bar_",
    "bounded_action_review_decision": "brd_",
    "bounded_action_escalation": "bes_",
    "project": "prj_",
}

_ID_BODY_RE = re.compile(
    r"^(goal|run|task|att|evt|tc|art|ver|heal|apr|rcn|mdp|mda|mdc|mat|rcr|rap|rcl|rat|bar|brd|bes|prj)_(\d{8})_([a-f0-9]{6})$"
)


def _utc_date_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _random_suffix(length: int = 6) -> str:
    return secrets.token_hex(length // 2)


def generate_id(kind: IdKind) -> str:
    """Return a new ID: ``{prefix}{YYYYMMDD}_{random_hex}``."""
    prefix = ID_PREFIXES[kind]
    return f"{prefix}{_utc_date_stamp()}_{_random_suffix()}"


def parse_id(value: str) -> tuple[str, str, str] | None:
    """Parse an ID into ``(prefix_token, date_stamp, random_suffix)``."""
    match = _ID_BODY_RE.match(value)
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3)


def validate_id(value: str, kind: IdKind) -> bool:
    """Return True when *value* matches the expected prefix and format."""
    parsed = parse_id(value)
    if parsed is None:
        return False
    prefix_token, _, _ = parsed
    expected = ID_PREFIXES[kind].rstrip("_")
    return prefix_token == expected


def require_id(value: str, kind: IdKind) -> None:
    """Raise ValueError when *value* does not match the expected prefix and format."""
    if not validate_id(value, kind):
        raise ValueError(f"invalid {kind} id: {value!r}")


def new_goal_id() -> str:
    return generate_id("goal")


def new_run_id() -> str:
    return generate_id("run")


def new_task_id() -> str:
    return generate_id("task")


def new_attempt_id() -> str:
    return generate_id("attempt")


def new_event_id() -> str:
    return generate_id("event")


def new_tool_call_id() -> str:
    return generate_id("tool_call")


def new_artifact_id() -> str:
    return generate_id("artifact")


def new_verification_id() -> str:
    return generate_id("verification")


def new_heal_cycle_id() -> str:
    return generate_id("heal_cycle")


def new_approval_id() -> str:
    return generate_id("approval")


def new_reconciliation_case_id() -> str:
    return generate_id("reconciliation")


def generate_reconciliation_case_id() -> str:
    return new_reconciliation_case_id()


def generate_marker_disposition_id() -> str:
    return generate_id("marker_disposition")


def generate_marker_disposition_approval_id() -> str:
    return generate_id("marker_disposition_approval")


def generate_marker_disposition_claim_id() -> str:
    return generate_id("marker_disposition_claim")


def generate_marker_disposition_attempt_id() -> str:
    return generate_id("marker_disposition_attempt")


def generate_recovery_request_id() -> str:
    return generate_id("recovery_run_request")


def generate_recovery_approval_id() -> str:
    return generate_id("recovery_run_approval")


def generate_recovery_claim_id() -> str:
    return generate_id("recovery_run_claim")


def generate_recovery_attempt_id() -> str:
    return generate_id("recovery_run_attempt")


def generate_successor_run_id() -> str:
    return new_run_id()


def generate_bounded_action_proposal_id() -> str:
    return generate_id("bounded_action_proposal")


def generate_bounded_action_review_decision_id() -> str:
    return generate_id("bounded_action_review_decision")


def generate_bounded_action_escalation_id() -> str:
    return generate_id("bounded_action_escalation")


def generate_project_id() -> str:
    return generate_id("project")
