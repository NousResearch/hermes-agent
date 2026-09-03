"""HTR workspace path contract under ``~/.hermes/runs/``."""

from __future__ import annotations

from pathlib import Path

from htr.ids import validate_id


def _validate_path_component(value: str, name: str) -> str:
    if not value or value in {".", ".."}:
        raise ValueError(f"invalid {name}: empty or reserved")
    if "/" in value or "\\" in value or ".." in value:
        raise ValueError(f"invalid {name}: path traversal not allowed")
    return value


def default_runs_root() -> Path:
    """Return the default HTR runs root (``HERMES_HOME/runs`` when available)."""
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home() / "runs"
    except Exception:
        return Path.home() / ".hermes" / "runs"


def runs_root(base_dir: Path | None = None) -> Path:
    return Path(base_dir) if base_dir is not None else default_runs_root()


def run_root(run_id: str, base_dir: Path | None = None) -> Path:
    _validate_path_component(run_id, "run_id")
    if not validate_id(run_id, "run"):
        raise ValueError(f"invalid run_id format: {run_id!r}")
    return runs_root(base_dir) / run_id


def run_manifest_path(run_id: str, base_dir: Path | None = None) -> Path:
    return run_root(run_id, base_dir) / "run_manifest.json"


def task_events_path(run_id: str, base_dir: Path | None = None) -> Path:
    return run_root(run_id, base_dir) / "task_events.jsonl"


def approvals_path(run_id: str, base_dir: Path | None = None) -> Path:
    """Legacy bootstrap placeholder only — not authoritative (Task 24)."""
    return run_root(run_id, base_dir) / "approvals.jsonl"


CONTROL_DIR_NAME = ".control"
APPROVALS_DIR_NAME = "approvals"


def control_root(base_dir: Path | None = None) -> Path:
    return runs_root(base_dir) / CONTROL_DIR_NAME


def control_approvals_root(base_dir: Path | None = None) -> Path:
    return control_root(base_dir) / APPROVALS_DIR_NAME


def approval_control_dir(approval_id: str, base_dir: Path | None = None) -> Path:
    _validate_path_component(approval_id, "approval_id")
    if not validate_id(approval_id, "approval"):
        raise ValueError(f"invalid approval_id format: {approval_id!r}")
    return control_approvals_root(base_dir) / approval_id


def approval_issue_path(approval_id: str, base_dir: Path | None = None) -> Path:
    return approval_control_dir(approval_id, base_dir) / "issue.json"


def approval_revoke_path(approval_id: str, base_dir: Path | None = None) -> Path:
    return approval_control_dir(approval_id, base_dir) / "revoke.json"


def approval_claim_path(approval_id: str, base_dir: Path | None = None) -> Path:
    return approval_control_dir(approval_id, base_dir) / "claim.json"


def approval_outcome_path(approval_id: str, base_dir: Path | None = None) -> Path:
    return approval_control_dir(approval_id, base_dir) / "outcome.json"


RECONCILIATION_DIR_NAME = "reconciliation"


def control_reconciliation_root(base_dir: Path | None = None) -> Path:
    return control_root(base_dir) / RECONCILIATION_DIR_NAME


def reconciliation_case_dir(case_id: str, base_dir: Path | None = None) -> Path:
    _validate_path_component(case_id, "case_id")
    if not validate_id(case_id, "reconciliation"):
        raise ValueError(f"invalid case_id format: {case_id!r}")
    return control_reconciliation_root(base_dir) / case_id


def reconciliation_open_path(case_id: str, base_dir: Path | None = None) -> Path:
    return reconciliation_case_dir(case_id, base_dir) / "open.json"


def reconciliation_observation_path(case_id: str, base_dir: Path | None = None) -> Path:
    return reconciliation_case_dir(case_id, base_dir) / "observation.json"


def reconciliation_decision_path(case_id: str, base_dir: Path | None = None) -> Path:
    return reconciliation_case_dir(case_id, base_dir) / "decision.json"


MARKER_DISPOSITIONS_DIR_NAME = "marker_dispositions"


def control_marker_dispositions_root(base_dir: Path | None = None) -> Path:
    return control_root(base_dir) / MARKER_DISPOSITIONS_DIR_NAME


def marker_disposition_dir(disposition_id: str, base_dir: Path | None = None) -> Path:
    _validate_path_component(disposition_id, "disposition_id")
    if not validate_id(disposition_id, "marker_disposition"):
        raise ValueError(f"invalid disposition_id format: {disposition_id!r}")
    return control_marker_dispositions_root(base_dir) / disposition_id


def marker_disposition_request_path(disposition_id: str, base_dir: Path | None = None) -> Path:
    return marker_disposition_dir(disposition_id, base_dir) / "request.json"


def marker_disposition_issue_path(disposition_id: str, base_dir: Path | None = None) -> Path:
    return marker_disposition_dir(disposition_id, base_dir) / "issue.json"


def marker_disposition_revoke_path(disposition_id: str, base_dir: Path | None = None) -> Path:
    return marker_disposition_dir(disposition_id, base_dir) / "revoke.json"


def marker_disposition_claim_path(disposition_id: str, base_dir: Path | None = None) -> Path:
    return marker_disposition_dir(disposition_id, base_dir) / "claim.json"


def marker_disposition_attempt_path(disposition_id: str, base_dir: Path | None = None) -> Path:
    return marker_disposition_dir(disposition_id, base_dir) / "attempt.json"


def marker_disposition_outcome_path(disposition_id: str, base_dir: Path | None = None) -> Path:
    return marker_disposition_dir(disposition_id, base_dir) / "outcome.json"


RECOVERY_RUNS_DIR_NAME = "recovery_runs"


def control_recovery_runs_root(base_dir: Path | None = None) -> Path:
    return control_root(base_dir) / RECOVERY_RUNS_DIR_NAME


def recovery_run_control_dir(recovery_request_id: str, base_dir: Path | None = None) -> Path:
    _validate_path_component(recovery_request_id, "recovery_request_id")
    if not validate_id(recovery_request_id, "recovery_run_request"):
        raise ValueError(f"invalid recovery_request_id format: {recovery_request_id!r}")
    return control_recovery_runs_root(base_dir) / recovery_request_id


def recovery_run_request_path(recovery_request_id: str, base_dir: Path | None = None) -> Path:
    return recovery_run_control_dir(recovery_request_id, base_dir) / "request.json"


def recovery_run_issue_path(recovery_request_id: str, base_dir: Path | None = None) -> Path:
    return recovery_run_control_dir(recovery_request_id, base_dir) / "issue.json"


def recovery_run_revoke_path(recovery_request_id: str, base_dir: Path | None = None) -> Path:
    return recovery_run_control_dir(recovery_request_id, base_dir) / "revoke.json"


def recovery_run_claim_path(recovery_request_id: str, base_dir: Path | None = None) -> Path:
    return recovery_run_control_dir(recovery_request_id, base_dir) / "claim.json"


def recovery_run_attempt_path(recovery_request_id: str, base_dir: Path | None = None) -> Path:
    return recovery_run_control_dir(recovery_request_id, base_dir) / "attempt.json"


def recovery_run_outcome_path(recovery_request_id: str, base_dir: Path | None = None) -> Path:
    return recovery_run_control_dir(recovery_request_id, base_dir) / "outcome.json"


def recovery_origin_path(run_id: str, base_dir: Path | None = None) -> Path:
    return run_root(run_id, base_dir) / "recovery_origin.json"


BOUNDED_ACTIONS_DIR_NAME = "bounded_actions"
PUBLICATION_COORD_DIR_NAME = "_publication_coord"
SUCCESSOR_COORD_DIR_NAME = "_successor_coord"


def control_bounded_actions_root(base_dir: Path | None = None) -> Path:
    return control_root(base_dir) / BOUNDED_ACTIONS_DIR_NAME


def publication_coord_dir(base_dir: Path | None = None) -> Path:
    return control_bounded_actions_root(base_dir) / PUBLICATION_COORD_DIR_NAME


def bounded_action_publication_coord_dir(base_dir: Path | None = None) -> Path:
    return publication_coord_dir(base_dir)


def successor_coord_dir(base_dir: Path | None = None) -> Path:
    return control_bounded_actions_root(base_dir) / SUCCESSOR_COORD_DIR_NAME


def bounded_action_successor_coord_root(base_dir: Path | None = None) -> Path:
    return successor_coord_dir(base_dir)


def bounded_action_successor_coord_dir(successor_run_id: str, base_dir: Path | None = None) -> Path:
    _validate_path_component(successor_run_id, "successor_run_id")
    if not validate_id(successor_run_id, "run"):
        raise ValueError(f"invalid successor_run_id format: {successor_run_id!r}")
    return bounded_action_successor_coord_root(base_dir) / successor_run_id


def bounded_action_case_dir(proposal_id: str, base_dir: Path | None = None) -> Path:
    _validate_path_component(proposal_id, "proposal_id")
    if not validate_id(proposal_id, "bounded_action_proposal"):
        raise ValueError(f"invalid proposal_id format: {proposal_id!r}")
    return control_bounded_actions_root(base_dir) / proposal_id


def bounded_action_proposal_path(proposal_id: str, base_dir: Path | None = None) -> Path:
    return bounded_action_case_dir(proposal_id, base_dir) / "proposal.json"


def bounded_action_review_decision_path(proposal_id: str, base_dir: Path | None = None) -> Path:
    return bounded_action_case_dir(proposal_id, base_dir) / "review_decision.json"


def bounded_action_escalation_path(proposal_id: str, base_dir: Path | None = None) -> Path:
    return bounded_action_case_dir(proposal_id, base_dir) / "escalation.json"


def reports_dir(run_id: str, base_dir: Path | None = None) -> Path:
    return run_root(run_id, base_dir) / "reports"


def tasks_dir(run_id: str, base_dir: Path | None = None) -> Path:
    return run_root(run_id, base_dir) / "tasks"


def task_dir(run_id: str, task_id: str, base_dir: Path | None = None) -> Path:
    _validate_path_component(task_id, "task_id")
    if not validate_id(task_id, "task"):
        raise ValueError(f"invalid task_id format: {task_id!r}")
    return tasks_dir(run_id, base_dir) / task_id


def task_card_path(run_id: str, task_id: str, base_dir: Path | None = None) -> Path:
    return task_dir(run_id, task_id, base_dir) / "task_card.json"


def task_status_path(run_id: str, task_id: str, base_dir: Path | None = None) -> Path:
    return task_dir(run_id, task_id, base_dir) / "task_status.json"


def attempts_dir(run_id: str, task_id: str, base_dir: Path | None = None) -> Path:
    return task_dir(run_id, task_id, base_dir) / "attempts"


def attempt_dir(
    run_id: str,
    task_id: str,
    attempt_id: str,
    base_dir: Path | None = None,
) -> Path:
    _validate_path_component(attempt_id, "attempt_id")
    if not validate_id(attempt_id, "attempt"):
        raise ValueError(f"invalid attempt_id format: {attempt_id!r}")
    return attempts_dir(run_id, task_id, base_dir) / attempt_id


def attempt_status_path(
    run_id: str,
    task_id: str,
    attempt_id: str,
    base_dir: Path | None = None,
) -> Path:
    return attempt_dir(run_id, task_id, attempt_id, base_dir) / "attempt_status.json"


def artifact_manifest_path(
    run_id: str,
    task_id: str,
    attempt_id: str,
    base_dir: Path | None = None,
) -> Path:
    return attempt_dir(run_id, task_id, attempt_id, base_dir) / "artifact_manifest.json"


def tool_calls_path(
    run_id: str,
    task_id: str,
    attempt_id: str,
    base_dir: Path | None = None,
) -> Path:
    return attempt_dir(run_id, task_id, attempt_id, base_dir) / "tool_calls.jsonl"


def input_dir(
    run_id: str,
    task_id: str,
    attempt_id: str,
    base_dir: Path | None = None,
) -> Path:
    return attempt_dir(run_id, task_id, attempt_id, base_dir) / "input"


def working_dir(
    run_id: str,
    task_id: str,
    attempt_id: str,
    base_dir: Path | None = None,
) -> Path:
    return attempt_dir(run_id, task_id, attempt_id, base_dir) / "working"


def output_dir(
    run_id: str,
    task_id: str,
    attempt_id: str,
    base_dir: Path | None = None,
) -> Path:
    return attempt_dir(run_id, task_id, attempt_id, base_dir) / "output"


def result_json_path(
    run_id: str,
    task_id: str,
    attempt_id: str,
    base_dir: Path | None = None,
) -> Path:
    return output_dir(run_id, task_id, attempt_id, base_dir) / "result.json"


def artifacts_dir(
    run_id: str,
    task_id: str,
    attempt_id: str,
    base_dir: Path | None = None,
) -> Path:
    return attempt_dir(run_id, task_id, attempt_id, base_dir) / "artifacts"


def logs_dir(
    run_id: str,
    task_id: str,
    attempt_id: str,
    base_dir: Path | None = None,
) -> Path:
    return attempt_dir(run_id, task_id, attempt_id, base_dir) / "logs"


def verification_dir(
    run_id: str,
    task_id: str,
    attempt_id: str,
    base_dir: Path | None = None,
) -> Path:
    return attempt_dir(run_id, task_id, attempt_id, base_dir) / "verification"


def heal_dir(
    run_id: str,
    task_id: str,
    attempt_id: str,
    base_dir: Path | None = None,
) -> Path:
    return attempt_dir(run_id, task_id, attempt_id, base_dir) / "heal"


HTR_META_DIR_NAME = ".htr"
PROJECT_REGISTRY_DIR_NAME = "project_registry"
PROJECT_REGISTRY_PROJECTS_DIR_NAME = "projects"
PROJECT_REGISTRY_RECORD_NAME = "record.json"
PROJECT_REGISTRY_LOCK_NAME = "registry.lock"


def _hermes_home_path(hermes_home: Path | None = None) -> Path:
    if hermes_home is not None:
        return Path(hermes_home)
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home()
    except Exception:
        return Path.home() / ".hermes"


def project_registry_root(hermes_home: Path | None = None) -> Path:
    """Return ``{HERMES_HOME}/.htr/project_registry`` — above per-project runs trees."""
    return _hermes_home_path(hermes_home) / HTR_META_DIR_NAME / PROJECT_REGISTRY_DIR_NAME


def project_registry_lock_path(hermes_home: Path | None = None) -> Path:
    return project_registry_root(hermes_home) / PROJECT_REGISTRY_LOCK_NAME


def project_registry_projects_root(hermes_home: Path | None = None) -> Path:
    return project_registry_root(hermes_home) / PROJECT_REGISTRY_PROJECTS_DIR_NAME


def project_record_dir(project_id: str, hermes_home: Path | None = None) -> Path:
    _validate_path_component(project_id, "project_id")
    if not validate_id(project_id, "project"):
        raise ValueError(f"invalid project_id format: {project_id!r}")
    return project_registry_projects_root(hermes_home) / project_id


def project_record_path(project_id: str, hermes_home: Path | None = None) -> Path:
    return project_record_dir(project_id, hermes_home) / PROJECT_REGISTRY_RECORD_NAME
