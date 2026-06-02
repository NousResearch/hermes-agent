"""Kanban-native closeout verifier.

This module owns the review lifecycle above raw Kanban worker execution:
``worker_done`` records executor completion, ``review_ready`` proves the work is
ready for human review, and ``closed`` records final close authority.  It is
intentionally Linear-free and fail-closed; Linear Done mutation, PR merge, and
gateway restart/reload are outside this surface.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Optional, cast

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_drift_audit

CLOSEOUT_EVIDENCE_SCHEMA = "kanban_closeout_evidence.v1"
DONE_CRITERIA_LEDGER_SCHEMA = "kanban_done_criteria_ledger.v1"
WORKER_EVIDENCE_SCHEMA = "kanban_worker_evidence.v1"
VERIFIER_RESULT_SCHEMA = "kanban_verifier_result.v1"
VALID_CLOSEOUT_PHASES = ("worker_done", "review_ready", "closed")
_SUCCESSFUL_CHECK_CONCLUSIONS = {"success", "neutral", "skipped"}
_FAILED_CHECK_CONCLUSIONS = {
    "failure",
    "failed",
    "error",
    "cancelled",
    "canceled",
    "timed_out",
    "action_required",
}
_TERMINAL_CHECK_STATUSES = {"completed", "complete", "success"}
_APPROVAL_DECISIONS = {"approved", "close_approved", "accepted", "ship"}
_DATA_MOUNT_PREFIX = "/mnt/hermes-data/"
_HARDBLOCK_RESIDUE_KINDS = {
    "archive_backup",
    "build_cache",
    "completed_workspace",
    "node_modules",
    "obsolete_branch",
    "pr_worktree",
    "task_tmp",
    "workspace_archive",
    "workspace_backup",
}
_PASSING_RESIDUE_DISPOSITIONS = {"none", "cleared", "moved", "retained"}
_UNDISPOSED_RESIDUE_DISPOSITIONS = {"", "needs_decision", "pending", "unknown", "unaccounted"}


@dataclass(frozen=True)
class CloseoutVerification:
    """Result of a fail-closed closeout transition verification."""

    allowed: bool
    target_phase: str
    blockers: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "target_phase": self.target_phase,
            "blockers": list(self.blockers),
            "reason": self.reason,
            "evidence": self.evidence,
        }


def _text(value: Any) -> str:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return ""
    return str(value).strip()


def _lower(value: Any) -> str:
    return _text(value).lower().replace("-", "_").replace(" ", "_")


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _collapse_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", _text(value)).strip()


def _normalize_string_list(value: Any) -> list[str]:
    return sorted({_collapse_ws(item) for item in _as_list(value) if _collapse_ws(item)})


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def normalize_done_criteria_ledger(ledger: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a canonical Done Criteria Ledger with a stable criteria hash.

    The hash intentionally covers criteria meaning and authority boundaries,
    not incidental formatting/order noise or mutable task metadata.
    """

    raw = _as_mapping(ledger)
    normalized: dict[str, Any] = {
        "schema": _text(raw.get("schema")) or DONE_CRITERIA_LEDGER_SCHEMA,
        "task_id": _text(raw.get("task_id")),
        "public_id": _text(raw.get("public_id")),
        "source": _as_mapping(raw.get("source")),
        "version": raw.get("version"),
        "criteria": [],
        "forbidden_actions": _normalize_string_list(raw.get("forbidden_actions")),
        "refinement_required": raw.get("refinement_required") is True,
    }

    for item in _as_list(raw.get("criteria")):
        if not isinstance(item, Mapping):
            normalized["criteria"].append({"invalid": True, "raw": item})
            continue
        criterion = _as_mapping(item)
        normalized["criteria"].append(
            {
                "id": _collapse_ws(criterion.get("id")),
                "text": _collapse_ws(criterion.get("text")),
                "source_section": _collapse_ws(criterion.get("source_section")),
                "required_evidence_types": _normalize_string_list(criterion.get("required_evidence_types")),
                "deterministic_checks": _normalize_string_list(criterion.get("deterministic_checks")),
                "authority_boundary": _collapse_ws(criterion.get("authority_boundary")),
                "ambiguous": criterion.get("ambiguous") is True,
            }
        )
    normalized["criteria"].sort(key=lambda c: (_text(c.get("id")), _text(c.get("text"))))

    hash_material = {
        "schema": DONE_CRITERIA_LEDGER_SCHEMA,
        "criteria": [
            {
                "id": c.get("id"),
                "text": c.get("text"),
                "required_evidence_types": c.get("required_evidence_types"),
                "deterministic_checks": c.get("deterministic_checks"),
                "authority_boundary": c.get("authority_boundary"),
                "ambiguous": c.get("ambiguous"),
            }
            for c in normalized["criteria"]
        ],
        "forbidden_actions": normalized["forbidden_actions"],
    }
    normalized["criteria_hash"] = "sha256:" + hashlib.sha256(_canonical_json(hash_material).encode("utf-8")).hexdigest()
    return normalized


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _resolve_repo_root(repo_path: str | Path) -> Optional[Path]:
    candidate = Path(repo_path).expanduser().resolve(strict=False)
    if candidate.is_file():
        candidate = candidate.parent
    result = _run_git(candidate, "rev-parse", "--show-toplevel")
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip()).resolve(strict=False)


def _current_git_facts(repo_path: str | Path | None) -> dict[str, Any]:
    if not repo_path:
        return {}
    repo = _resolve_repo_root(repo_path)
    if repo is None:
        return {"repo_path": str(repo_path), "repo_resolved": False}

    facts: dict[str, Any] = {"repo_path": str(repo), "repo_resolved": True}
    head = _run_git(repo, "rev-parse", "HEAD")
    if head.returncode == 0:
        facts["head_sha"] = head.stdout.strip()
    status = _run_git(repo, "status", "--short")
    if status.returncode == 0:
        facts["status_short"] = status.stdout
        facts["worktree_clean"] = not bool(status.stdout.strip())
    return facts


def _pr_selector(evidence: Mapping[str, Any]) -> str | None:
    pr = _as_mapping(evidence.get("pr"))
    for key in ("number", "url"):
        value = _text(pr.get(key))
        if value:
            return value
    return None


def _run_gh_pr_view(repo_path: str | Path | None, selector: str | None) -> dict[str, Any]:
    if not repo_path or not shutil.which("gh"):
        return {}
    repo = _resolve_repo_root(repo_path)
    if repo is None:
        return {}

    cmd = [
        "gh",
        "pr",
        "view",
        *([selector] if selector else []),
        "--json",
        "number,url,state,isDraft,headRefOid,mergeCommit,statusCheckRollup",
    ]
    result = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, check=False, timeout=30)
    if result.returncode != 0:
        return {"provider_error": result.stderr.strip() or result.stdout.strip() or "gh pr view failed"}
    try:
        data = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return {"provider_error": "gh pr view returned invalid JSON"}
    if not isinstance(data, dict):
        return {"provider_error": "gh pr view returned non-object JSON"}
    return {
        "number": data.get("number"),
        "url": data.get("url"),
        "state": data.get("state"),
        "is_draft": data.get("isDraft"),
        "head_sha": data.get("headRefOid"),
        "merge_commit_sha": _text(_as_mapping(data.get("mergeCommit")).get("oid")),
        "live": True,
        "provider": "gh_pr_view",
        "checks": data.get("statusCheckRollup") or [],
    }


def collect_live_closeout_evidence(
    evidence: Mapping[str, Any] | None = None,
    *,
    repo_path: str | Path | None = None,
    live_pr_provider: Callable[[Mapping[str, Any], str | Path | None], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Merge caller evidence with live local git/PR facts.

    ``live_pr_provider`` is injectable so tests and future GitHub adapters can
    supply live PR state without shelling out.  When omitted, the helper uses
    ``gh pr view`` opportunistically; verification still fails closed if no live
    PR facts are available.
    """

    merged: dict[str, Any] = copy.deepcopy(dict(evidence or {}))
    merged.setdefault("schema", CLOSEOUT_EVIDENCE_SCHEMA)
    merged["observed_at"] = int(time.time())

    git_facts = _current_git_facts(repo_path)
    if git_facts:
        existing_git = _as_mapping(merged.get("git"))
        existing_git.update(git_facts)
        merged["git"] = existing_git

    live_pr: Mapping[str, Any] = {}
    if live_pr_provider is not None:
        live_pr = live_pr_provider(merged, repo_path) or {}
    else:
        live_pr = _run_gh_pr_view(repo_path, _pr_selector(merged))

    if live_pr:
        live_pr_dict = dict(live_pr)
        existing_pr = _as_mapping(merged.get("pr"))
        checks = live_pr_dict.pop("checks", None)
        existing_pr.update(live_pr_dict)
        merged["pr"] = existing_pr
        if checks is not None and "checks" not in merged:
            merged["checks"] = checks

    return merged


def _has_worker_evidence(evidence: Mapping[str, Any]) -> bool:
    worker_evidence = _as_mapping(evidence.get("worker_evidence"))
    if worker_evidence:
        return True
    direct = (
        _text(evidence.get("summary"))
        or _text(evidence.get("proof"))
        or _text(evidence.get("verification"))
    )
    if direct:
        return True
    work = _as_mapping(evidence.get("evidence"))
    return bool(
        _text(work.get("summary"))
        or _text(work.get("proof"))
        or _as_list(work.get("tests_run"))
        or _as_list(work.get("changed_files"))
    )


def _cleanup_proven(evidence: Mapping[str, Any]) -> tuple[bool, str | None]:
    cleanup = _as_mapping(evidence.get("cleanup"))
    proof = _text(cleanup.get("proof")) or _text(evidence.get("cleanup_proof"))
    if not proof:
        return False, "missing_cleanup_proof"

    git = _as_mapping(evidence.get("git"))
    for value in (cleanup.get("worktree_clean"), git.get("worktree_clean")):
        if value is False:
            return False, "dirty_worktree"
    status_short = _text(git.get("status_short"))
    if status_short:
        return False, "dirty_worktree"

    # Require structured evidence — prose-only "cleaned up" is insufficient.
    artifacts_removed = cleanup.get("artifacts_removed")
    worktree_retained = cleanup.get("worktree_retained")
    if artifacts_removed is None and worktree_retained is None:
        return False, "missing_structured_cleanup_evidence"
    if worktree_retained is True:
        if not _text(cleanup.get("retained_reason")):
            return False, "retained_worktree_missing_reason"
        if not _has_ttl_or_revisit(cleanup):
            return False, "retained_worktree_missing_ttl"

    return True, None


def _has_ttl_or_revisit(item: Mapping[str, Any]) -> bool:
    return bool(
        _text(item.get("ttl"))
        or _text(item.get("revisit_at"))
        or _text(item.get("expires_at"))
    )


def _is_on_data_mount(path: Any) -> bool:
    value = _text(path)
    return value == "/mnt/hermes-data" or value.startswith(_DATA_MOUNT_PREFIX)


def _residue_blockers(evidence: Mapping[str, Any]) -> list[str]:
    if "residue" not in evidence:
        return ["missing_residue_evidence"]
    residue = evidence.get("residue")
    if not isinstance(residue, Mapping):
        return ["invalid_residue_evidence"]

    blockers: list[str] = []
    summary = _text(residue.get("summary"))
    items = residue.get("items")
    if items is None:
        items_list: list[Any] = []
    elif isinstance(items, list):
        items_list = items
    else:
        return ["invalid_residue_evidence"]

    if not summary and not items_list and "db_backups" not in residue:
        blockers.append("invalid_residue_evidence")

    for raw_item in items_list:
        if not isinstance(raw_item, Mapping):
            blockers.append("invalid_residue_evidence")
            continue
        item = _as_mapping(raw_item)
        kind = _lower(item.get("kind"))
        disposition = _lower(item.get("disposition")) or ("none" if kind == "none" else "")
        if disposition in _UNDISPOSED_RESIDUE_DISPOSITIONS:
            blockers.append("undisposed_residue")
            continue
        if disposition not in _PASSING_RESIDUE_DISPOSITIONS:
            blockers.append("invalid_residue_evidence")
            continue
        if disposition == "retained":
            if not _text(item.get("reason")):
                blockers.append("retained_residue_missing_reason")
            if not _has_ttl_or_revisit(item):
                blockers.append("retained_residue_missing_ttl")
        if disposition == "moved" and kind in _HARDBLOCK_RESIDUE_KINDS:
            if not _is_on_data_mount(item.get("destination")):
                blockers.append("moved_residue_not_on_data_mount")

    db_backups = residue.get("db_backups")
    if db_backups is not None:
        if not isinstance(db_backups, Mapping):
            blockers.append("invalid_residue_evidence")
        else:
            db = _as_mapping(db_backups)
            count = db.get("count")
            has_any = bool(count or _text(db.get("retention")) or _has_ttl_or_revisit(db))
            if has_any and not (_text(db.get("retention")) and _has_ttl_or_revisit(db)):
                blockers.append("db_backup_retention_uncapped")

    return list(dict.fromkeys(blockers))


def _extract_pr_candidates(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for key in ("pr_candidates", "pull_requests", "prs"):
        for item in _as_list(evidence.get(key)):
            if isinstance(item, Mapping):
                candidates.append(dict(item))
    pr = _as_mapping(evidence.get("pr"))
    if pr:
        candidates.append(pr)
    return candidates


def _check_pr(evidence: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    candidates = _extract_pr_candidates(evidence)
    if len(candidates) != 1:
        return ["missing_live_pr" if not candidates else "ambiguous_pr_evidence"]

    pr = candidates[0]
    if pr.get("live") is not True:
        blockers.append("missing_live_pr")
    if _lower(pr.get("state")) not in {"open", "opened"}:
        blockers.append("pr_not_open")
    if pr.get("is_draft") is True or pr.get("draft") is True:
        blockers.append("pr_is_draft")

    pr_head = _text(pr.get("head_sha") or pr.get("headRefOid") or pr.get("head_oid"))
    git_head = _text(_as_mapping(evidence.get("git")).get("head_sha"))
    if not pr_head:
        blockers.append("missing_pr_head_sha")
    elif git_head and pr_head != git_head:
        blockers.append("stale_pr")
    return blockers


def _check_closed_pr(evidence: Mapping[str, Any]) -> list[str]:
    """Verify PR evidence for the ``closed`` phase — PR must be MERGED/CLOSED.

    Unlike ``_check_pr`` (which requires OPEN for ``review_ready``), this
    gate only accepts a merged or closed PR because final Done should not
    happen while a review surface is still open.
    """
    blockers: list[str] = []
    candidates = _extract_pr_candidates(evidence)
    if len(candidates) != 1:
        return ["missing_pr" if not candidates else "ambiguous_pr_evidence"]

    pr = candidates[0]
    if pr.get("live") is not True:
        blockers.append("missing_live_pr")
    state = _lower(pr.get("state"))
    if state in {"open", "opened"}:
        blockers.append("pr_not_merged")
    elif state not in {"merged", "closed"}:
        blockers.append("pr_unrecognized_state")

    pr_head = _text(pr.get("head_sha") or pr.get("headRefOid") or pr.get("head_oid"))
    merge_commit = _text(pr.get("merge_commit_sha") or pr.get("mergeCommitSha"))
    if not merge_commit:
        merge_commit = _text(_as_mapping(pr.get("mergeCommit")).get("oid"))
    git_head = _text(_as_mapping(evidence.get("git")).get("head_sha"))
    if not pr_head:
        blockers.append("missing_pr_head_sha")
    elif git_head and pr_head != git_head and merge_commit != git_head:
        blockers.append("stale_pr")
    return blockers


def _normalize_check(item: Any) -> tuple[str, str, str]:
    if not isinstance(item, Mapping):
        return ("check", "", "")
    name = _text(
        item.get("name")
        or item.get("context")
        or item.get("workflowName")
        or item.get("__typename")
    ) or "check"
    status = _lower(item.get("status") or item.get("state"))
    conclusion = _lower(item.get("conclusion") or item.get("outcome"))
    return name, status, conclusion


def _check_statuses(evidence: Mapping[str, Any]) -> list[str]:
    checks = _as_list(evidence.get("checks") or _as_mapping(evidence.get("pr")).get("checks"))
    if not checks:
        return ["missing_checks"]

    failed: list[str] = []
    pending: list[str] = []
    unknown: list[str] = []
    for item in checks:
        name, status, conclusion = _normalize_check(item)
        if conclusion in _FAILED_CHECK_CONCLUSIONS or status in _FAILED_CHECK_CONCLUSIONS:
            failed.append(name)
        elif conclusion in _SUCCESSFUL_CHECK_CONCLUSIONS:
            continue
        elif status and status not in _TERMINAL_CHECK_STATUSES:
            pending.append(name)
        else:
            unknown.append(name)
    blockers: list[str] = []
    if failed:
        blockers.append("failed_checks")
    if pending:
        blockers.append("pending_checks")
    if unknown:
        blockers.append("ambiguous_check_evidence")
    return blockers


def _verifier_pass_present(evidence: Mapping[str, Any]) -> bool:
    result = _as_mapping(evidence.get("verifier_result"))
    if result:
        return _lower(result.get("verdict")) == "pass"
    verdict = evidence.get("verifier_verdict") or evidence.get("verifier")
    if isinstance(verdict, Mapping):
        return _lower(verdict.get("verdict") or verdict.get("status")) == "pass"
    return _lower(verdict) == "pass"


def _governed_reviewer_loop_enabled(evidence: Mapping[str, Any]) -> bool:
    return bool(
        evidence.get("require_done_criteria_ledger") is True
        or evidence.get("require_worker_evidence_contract") is True
        or evidence.get("require_verifier_result_contract") is True
        or isinstance(evidence.get("done_criteria_ledger"), Mapping)
        or isinstance(evidence.get("worker_evidence"), Mapping)
        or isinstance(evidence.get("verifier_result"), Mapping)
        or isinstance(evidence.get("reviewer_loop"), Mapping)
    )


def _validate_done_criteria_ledger(evidence: Mapping[str, Any]) -> tuple[list[str], dict[str, Any] | None]:
    raw = evidence.get("done_criteria_ledger")
    if not isinstance(raw, Mapping):
        return (["missing_done_criteria_ledger"] if _governed_reviewer_loop_enabled(evidence) else []), None
    if _text(raw.get("schema")) != DONE_CRITERIA_LEDGER_SCHEMA:
        return ["invalid_done_criteria_ledger_schema"], normalize_done_criteria_ledger(raw)
    ledger = normalize_done_criteria_ledger(raw)
    blockers: list[str] = []
    criteria = ledger.get("criteria") or []
    if not criteria:
        blockers.append("empty_done_criteria")
    if ledger.get("refinement_required") is True:
        blockers.append("refinement_required")
    if not ledger.get("forbidden_actions"):
        blockers.append("missing_forbidden_actions")
    for criterion in criteria:
        if criterion.get("invalid"):
            blockers.append("invalid_done_criteria")
            continue
        if not _text(criterion.get("id")) or not _text(criterion.get("text")):
            blockers.append("invalid_done_criteria")
        if criterion.get("ambiguous") is True:
            blockers.append("refinement_required")
        if not _text(criterion.get("authority_boundary")):
            blockers.append("missing_authority_boundary")
        required_evidence_types = set(_as_list(criterion.get("required_evidence_types")))
        requires_deterministic = bool(required_evidence_types & {"test", "db_query", "screenshot", "no_code_proof"})
        if requires_deterministic and not _as_list(criterion.get("deterministic_checks")):
            blockers.append("missing_deterministic_checks")
    if isinstance(evidence, dict):
        cast(MutableMapping[str, Any], evidence)["done_criteria_ledger"] = ledger
    return list(dict.fromkeys(blockers)), ledger


def _criterion_ids(ledger: Mapping[str, Any] | None) -> list[str]:
    if not ledger:
        return []
    return [_text(c.get("id")) for c in _as_list(ledger.get("criteria")) if isinstance(c, Mapping) and _text(c.get("id"))]


def _validate_worker_evidence_contract(evidence: Mapping[str, Any], ledger: Mapping[str, Any] | None) -> list[str]:
    if not _governed_reviewer_loop_enabled(evidence):
        return []
    worker = _as_mapping(evidence.get("worker_evidence"))
    if not worker:
        return ["missing_worker_evidence_contract"] if evidence.get("require_worker_evidence_contract") is True else []
    blockers: list[str] = []
    if _text(worker.get("schema")) != WORKER_EVIDENCE_SCHEMA:
        blockers.append("invalid_worker_evidence_schema")
    expected_hash = _text((ledger or {}).get("criteria_hash"))
    if expected_hash and _text(worker.get("criteria_hash")) != expected_hash:
        blockers.append("stale_worker_criteria_hash")
    per_criterion = worker.get("per_criterion") if isinstance(worker.get("per_criterion"), Mapping) else {}
    ids = _criterion_ids(ledger)
    if ids and not per_criterion:
        blockers.append("missing_worker_per_criterion_evidence")
    for cid in ids:
        item = _as_mapping(per_criterion.get(cid))
        if not item or _lower(item.get("claim")) not in {"satisfied", "not_applicable"}:
            blockers.append("missing_worker_per_criterion_evidence")
        if not (_as_list(item.get("evidence_refs")) or _text(item.get("notes"))):
            blockers.append("missing_worker_per_criterion_evidence")
    if worker.get("authority_boundary_confirmed") is not True:
        blockers.append("worker_authority_boundary_unconfirmed")
    if _as_list(worker.get("forbidden_actions_performed")):
        blockers.append("forbidden_action_performed")
    return list(dict.fromkeys(blockers))


def _validate_verifier_result_contract(evidence: Mapping[str, Any], ledger: Mapping[str, Any] | None) -> list[str]:
    if not _governed_reviewer_loop_enabled(evidence):
        return []
    result = _as_mapping(evidence.get("verifier_result"))
    if not result:
        return ["missing_verifier_result_contract"] if evidence.get("require_verifier_result_contract") is True else []
    blockers: list[str] = []
    if _text(result.get("schema")) != VERIFIER_RESULT_SCHEMA:
        blockers.append("invalid_verifier_result_schema")
    expected_hash = _text((ledger or {}).get("criteria_hash"))
    if expected_hash and _text(result.get("criteria_hash")) != expected_hash:
        blockers.append("stale_verifier_criteria_hash")
    verdict = _lower(result.get("verdict"))
    if verdict != "pass":
        blockers.append("verifier_result_not_pass")
    per_criterion = result.get("per_criterion") if isinstance(result.get("per_criterion"), Mapping) else {}
    for cid in _criterion_ids(ledger):
        item = _as_mapping(per_criterion.get(cid))
        if not item or _lower(item.get("verdict")) != "pass":
            blockers.append("criterion_verifier_not_pass")
    if result.get("authority_boundary_ok") is not True:
        blockers.append("verifier_authority_boundary_failed")
    return list(dict.fromkeys(blockers))


def _reviewer_loop_attempt(evidence: Mapping[str, Any]) -> tuple[int, int]:
    loop = _as_mapping(evidence.get("reviewer_loop"))
    try:
        attempt = int(loop.get("attempt") or _as_mapping(evidence.get("verifier_result")).get("verification_attempt") or 1)
    except (TypeError, ValueError):
        attempt = 1
    try:
        max_attempts = int(loop.get("max_attempts") or loop.get("max_verification_attempts") or 3)
    except (TypeError, ValueError):
        max_attempts = 3
    return max(1, attempt), max(1, max_attempts)


_REVIEWER_FAIL_REMEDIATION_BLOCKERS = {
    "verifier_result_not_pass",
    "criterion_verifier_not_pass",
    "missing_verifier_pass",
}

_STRUCTURAL_REMEDIATION_BLOCKERS = {
    "ambiguous_check_evidence",
    "missing_checks",
    "invalid_residue_evidence",
    "missing_residue_evidence",
}

_REMEDIATION_SCHEMA = "kanban_remediation_request.v1"


def _make_remediation_request(
    *,
    source_phase: str,
    blockers: list[str],
    goal: str,
    attempt: int,
    max_attempts: int,
    kind: str,
    required_outputs: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema": _REMEDIATION_SCHEMA,
        "allowed": True,
        "source_phase": source_phase,
        "kind": kind,
        "blockers": list(dict.fromkeys(blockers)),
        "remediation_goal": goal,
        "required_outputs": required_outputs or [],
        "retry_allowed": True,
        "attempt": attempt,
        "next_attempt": attempt + 1,
        "max_attempts": max_attempts,
        "dispatcher_owned": True,
    }


def _remediation_candidate(evidence: Mapping[str, Any], blockers: list[str] | None = None) -> dict[str, Any] | None:
    blocker_set = set(blockers or [])
    result = _as_mapping(evidence.get("verifier_result"))
    verdict = _lower(result.get("verdict"))
    attempt, max_attempts = _reviewer_loop_attempt(evidence)

    if verdict == "fail" and result.get("retry_allowed") is True:
        if blockers is not None and (blocker_set - _REVIEWER_FAIL_REMEDIATION_BLOCKERS):
            return None
        goal = _text(result.get("remediation_goal"))
        if not goal:
            return None
        if attempt >= max_attempts:
            return {
                "allowed": False,
                "blocker": "remediation_attempts_exhausted",
                "attempt": attempt,
                "max_attempts": max_attempts,
            }
        return _make_remediation_request(
            source_phase="review_ready",
            blockers=blockers or ["verifier_result_not_pass"],
            goal=goal,
            attempt=attempt,
            max_attempts=max_attempts,
            kind="verifier_fail",
            required_outputs=[
                "updated worker_evidence per failed criterion",
                "updated tests/proof referenced by verifier_result.missing_evidence",
                "resubmitted verifier_result for the existing task/PR",
            ],
        )

    structural_blockers = [b for b in (blockers or []) if b in _STRUCTURAL_REMEDIATION_BLOCKERS]
    if verdict == "pass" and structural_blockers and blocker_set <= _STRUCTURAL_REMEDIATION_BLOCKERS:
        if attempt >= max_attempts:
            return {
                "allowed": False,
                "blocker": "remediation_attempts_exhausted",
                "attempt": attempt,
                "max_attempts": max_attempts,
            }
        goal_parts: list[str] = []
        if "ambiguous_check_evidence" in structural_blockers or "missing_checks" in structural_blockers:
            goal_parts.append(
                "Add checks[] evidence where every check has a machine-readable terminal conclusion "
                "such as success, neutral, or skipped."
            )
        if "invalid_residue_evidence" in structural_blockers or "missing_residue_evidence" in structural_blockers:
            goal_parts.append(
                "Add residue evidence as an object with summary or items[], and ensure every residue "
                "item has an accepted disposition plus reason/ttl when retained."
            )
        goal = " ".join(goal_parts) or "Repair structural review_ready closeout evidence and resubmit."
        return _make_remediation_request(
            source_phase="review_ready",
            blockers=structural_blockers,
            goal=goal,
            attempt=attempt,
            max_attempts=max_attempts,
            kind="structural_closeout_evidence",
            required_outputs=[
                "checks[] with terminal conclusions",
                "residue object with summary or items[]",
                "updated closeout_evidence package for review_ready",
            ],
        )

    return None


def _review_package_work(evidence: Mapping[str, Any]) -> dict[str, Any]:
    work = _as_mapping(evidence.get("evidence"))
    metadata = _as_mapping(evidence.get("metadata"))
    merged: dict[str, Any] = {}
    merged.update(metadata)
    merged.update(work)
    for key in ("changed_files", "artifact_refs", "proof", "verification"):
        if key in evidence and key not in merged:
            merged[key] = evidence[key]
    return merged


def _review_package_changed_files(evidence: Mapping[str, Any]) -> list[Any]:
    work = _review_package_work(evidence)
    if "changed_files" not in work:
        return []
    return _as_list(work.get("changed_files"))


def _no_pr_reason(evidence: Mapping[str, Any]) -> str:
    exception = _as_mapping(evidence.get("no_pr_exception"))
    return (
        _text(evidence.get("no_pr_reason"))
        or _text(evidence.get("no_pr_review_reason"))
        or _text(exception.get("reason"))
    )


def _has_no_pr_artifact_or_proof(evidence: Mapping[str, Any]) -> bool:
    work = _review_package_work(evidence)
    return bool(
        _text(work.get("proof"))
        or _text(work.get("summary"))
        or _text(work.get("verification"))
        or _as_list(work.get("artifact_refs"))
        or _as_list(work.get("artifacts"))
        or _as_list(evidence.get("artifact_refs"))
        or _as_list(evidence.get("artifacts"))
    )


def _authority_boundary_confirmed(evidence: Mapping[str, Any]) -> bool:
    value = evidence.get("authority_boundary_confirmed")
    if value is None:
        value = evidence.get("boundaries_confirmed")
    return value is True


def _review_package_blockers(evidence: Mapping[str, Any]) -> list[str]:
    """Enforce the v1 two-way review package rule.

    Diff changed files -> a live PR is required.  No changed files -> no-PR
    evidence is a normal path, not an exception, but it still needs an explicit
    reason plus some artifact/proof so the review surface cannot become prose.
    """

    blockers: list[str] = []
    changed_files = _review_package_changed_files(evidence)
    if changed_files:
        blockers.extend(_check_pr(evidence))
        if isinstance(evidence, dict):
            mutable_evidence = cast(MutableMapping[str, Any], evidence)
            mutable_evidence.setdefault(
                "review_package",
                {"schema": "kanban_review_package.v1", "kind": "pr_required", "changed_files": changed_files},
            )
    else:
        if not _no_pr_reason(evidence):
            blockers.append("missing_no_pr_reason")
        if not _has_no_pr_artifact_or_proof(evidence):
            blockers.append("missing_no_pr_artifact_or_proof")
        if isinstance(evidence, dict):
            mutable_evidence = cast(MutableMapping[str, Any], evidence)
            mutable_evidence.setdefault(
                "review_package",
                {"schema": "kanban_review_package.v1", "kind": "no_pr_evidence", "changed_files": []},
            )
    if not _authority_boundary_confirmed(evidence):
        blockers.append("missing_authority_boundary_confirmation")
    return blockers


def _review_ready_blockers(evidence: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    ledger_blockers, ledger = _validate_done_criteria_ledger(evidence)
    blockers.extend(ledger_blockers)
    blockers.extend(_validate_worker_evidence_contract(evidence, ledger))
    blockers.extend(_validate_verifier_result_contract(evidence, ledger))
    if not _has_worker_evidence(evidence):
        blockers.append("missing_worker_evidence")
    if not _verifier_pass_present(evidence):
        blockers.append("missing_verifier_pass")
    blockers.extend(_review_package_blockers(evidence))
    blockers.extend(_check_statuses(evidence))
    blockers.extend(_residue_blockers(evidence))
    cleanup_ok, cleanup_blocker = _cleanup_proven(evidence)
    if not cleanup_ok and cleanup_blocker:
        blockers.append(cleanup_blocker)
    if evidence.get("ambiguous") is True:
        blockers.append("ambiguous_evidence")
    remediation = _remediation_candidate(evidence, list(dict.fromkeys(blockers)))
    if remediation and remediation.get("allowed") is False:
        blockers.append(_text(remediation.get("blocker")) or "remediation_blocked")
    # Preserve order while de-duplicating.
    return list(dict.fromkeys(blockers))


def _closed_blockers(evidence: Mapping[str, Any]) -> list[str]:
    """Fail-closed verification specifically for the ``closed`` terminal phase.

    This is intentionally stricter than ``_review_ready_blockers`` because
    final Done requires post-merge cleanup proof, not just pre-review cleanup.
    The PR must be MERGED/CLOSED (or a documented no-PR exception must be
    present) because closing a task while its review surface is still open is
    an operational lie per the RALPLAN lifecycle ladder.
    """
    blockers: list[str] = []

    no_pr_exception = _as_mapping(evidence.get("no_pr_exception"))
    if no_pr_exception and _text(no_pr_exception.get("policy")) and _text(no_pr_exception.get("reason")):
        # Documented no-PR exception — skip the PR check entirely.
        pass
    else:
        blockers.extend(_check_closed_pr(evidence))

    if not _has_worker_evidence(evidence):
        blockers.append("missing_worker_evidence")
    if not _verifier_pass_present(evidence):
        blockers.append("missing_verifier_pass")
    blockers.extend(_check_statuses(evidence))

    cleanup_ok, cleanup_blocker = _cleanup_proven(evidence)
    if not cleanup_ok and cleanup_blocker:
        blockers.append(cleanup_blocker)

    blockers.extend(_residue_blockers(evidence))

    if evidence.get("ambiguous") is True:
        blockers.append("ambiguous_evidence")

    return list(dict.fromkeys(blockers))


def _approval_present(evidence: Mapping[str, Any]) -> bool:
    approval = _as_mapping(evidence.get("approval"))
    if not approval:
        return False
    if approval.get("approved") is True:
        return bool(_text(approval.get("approved_by") or approval.get("by")))
    decision = _lower(approval.get("decision"))
    return decision in _APPROVAL_DECISIONS and bool(_text(approval.get("approved_by") or approval.get("by")))


def _no_pr_exception_present(evidence: Mapping[str, Any]) -> bool:
    exception = _as_mapping(evidence.get("no_pr_exception"))
    return bool(_text(exception.get("policy")) and _text(exception.get("reason")))


def _no_pr_review_ready_exception_blockers(evidence: Mapping[str, Any]) -> list[str] | None:
    """Return blockers for a review_ready no-PR exception, or None when absent.

    Review-ready normally requires a live open PR.  A no-PR path is intentionally
    narrower than the final-close exception: it is only for review packages whose
    own evidence says no repository diff is expected (for example no-code smoke
    fixtures).  Policy and reason alone are not enough, otherwise any operator
    note could silently bypass the review surface.
    """

    exception = _as_mapping(evidence.get("no_pr_exception"))
    if not exception:
        return None

    blockers: list[str] = []
    if not _text(exception.get("policy")):
        blockers.append("missing_no_pr_exception_policy")
    if not _text(exception.get("reason")):
        blockers.append("missing_no_pr_exception_reason")

    work = _as_mapping(evidence.get("evidence"))
    expectation = _text(exception.get("review_package_expectation")) or _text(
        work.get("review_package_expectation")
    )
    if not expectation:
        blockers.append("missing_no_pr_review_package_expectation")

    changed_files_expected = exception.get("changed_files_expected")
    changed_files = _as_list(work.get("changed_files"))
    if changed_files_expected is not False and changed_files:
        blockers.append("no_pr_exception_has_changed_files")
    elif changed_files_expected is not False:
        blockers.append("missing_no_pr_changed_files_not_expected")

    return blockers


def verify_closeout_transition(
    target_phase: str,
    evidence: Mapping[str, Any] | None = None,
    *,
    current_phase: str | None = None,
    repo_path: str | Path | None = None,
    live_pr_provider: Callable[[Mapping[str, Any], str | Path | None], Mapping[str, Any]] | None = None,
) -> CloseoutVerification:
    """Verify a Kanban closeout phase transition without mutating state."""

    target = _lower(target_phase)
    if target not in VALID_CLOSEOUT_PHASES:
        raise ValueError(f"target_phase must be one of {list(VALID_CLOSEOUT_PHASES)}")

    normalized = collect_live_closeout_evidence(
        evidence,
        repo_path=repo_path,
        live_pr_provider=live_pr_provider,
    )
    normalized["target_phase"] = target
    if current_phase:
        normalized["previous_phase"] = current_phase

    blockers: list[str] = []
    if target == "worker_done":
        if current_phase == "closed":
            blockers.append("already_closed")
        if not _has_worker_evidence(normalized):
            blockers.append("missing_worker_evidence")
    elif target == "review_ready":
        if current_phase is None:
            blockers.append("review_ready_requires_worker_done")
        elif current_phase not in {"worker_done", "review_ready"}:
            blockers.append("invalid_review_ready_source_phase")
        blockers.extend(_review_ready_blockers(normalized))
    else:  # closed
        for key in ("drift_audit", "sustained_drift_audit"):
            blockers.extend(kanban_drift_audit.closeout_blocks_from_audit(normalized.get(key)))
        has_exception = _no_pr_exception_present(normalized)
        if current_phase != "review_ready" and not has_exception:
            blockers.append("closed_requires_review_ready")
        if not _approval_present(normalized) and not has_exception:
            blockers.append("missing_close_approval")
        blockers.extend(_closed_blockers(normalized))

    blockers = list(dict.fromkeys(blockers))
    normalized["verification"] = {
        "allowed": not blockers,
        "blockers": blockers,
        "linear_done_mutated": False,
        "gateway_restarted_or_reloaded": False,
        "pr_merged": False,
        "stored_in_tasks_metadata_blob": False,
    }
    return CloseoutVerification(
        allowed=not blockers,
        target_phase=target,
        blockers=blockers,
        evidence=normalized,
        reason="closeout_verified" if not blockers else "closeout_blocked_fail_closed",
    )


def transition_task_closeout(
    conn: Any,
    task_id: str,
    target_phase: str,
    evidence: Mapping[str, Any] | None = None,
    *,
    repo_path: str | Path | None = None,
    live_pr_provider: Callable[[Mapping[str, Any], str | Path | None], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify and persist a task closeout transition when allowed."""

    task = kb.get_task(conn, task_id)
    if task is None:
        return {
            "status": "blocked",
            "reason": "unknown_task",
            "task_id": task_id,
            "target_phase": _lower(target_phase),
            "blockers": ["unknown_task"],
            "side_effects": {"kanban_task_written": False, "linear_done_mutated": False},
        }

    verification = verify_closeout_transition(
        target_phase,
        evidence,
        current_phase=task.review_phase,
        repo_path=repo_path or task.workspace_path,
        live_pr_provider=live_pr_provider,
    )
    if not verification.allowed:
        existing_closeout = _as_mapping(task.closeout_evidence)
        existing_remediation = _as_mapping(existing_closeout.get("remediation_request"))
        if existing_remediation and verification.target_phase == "review_ready":
            return {
                "status": "remediation_requested",
                "reason": "reviewer_fail_remediation_already_queued",
                "task_id": task_id,
                "current_phase": task.review_phase,
                "target_phase": verification.target_phase,
                "blockers": verification.blockers,
                "remediation": existing_remediation,
                "evidence": existing_closeout,
                "side_effects": {"kanban_task_written": False, "linear_done_mutated": False},
            }
        remediation = _remediation_candidate(verification.evidence, verification.blockers) if verification.target_phase == "review_ready" else None
        if remediation and remediation.get("allowed") is True and "remediation_attempts_exhausted" not in verification.blockers:
            patched_evidence = copy.deepcopy(verification.evidence)
            patched_evidence["last_reviewer_result"] = _as_mapping(patched_evidence.get("verifier_result"))
            patched_evidence["worker_done_candidate"] = {"status": "rejected", "attempt": remediation.get("attempt")}
            loop = _as_mapping(patched_evidence.get("reviewer_loop"))
            loop.update(
                {
                    "enabled": True,
                    "attempt": remediation.get("next_attempt"),
                    "max_attempts": remediation.get("max_attempts"),
                    "last_remediation_goal": remediation.get("remediation_goal"),
                }
            )
            patched_evidence["reviewer_loop"] = loop
            patched_evidence["remediation_request"] = remediation
            try:
                with kb.write_txn(conn):
                    conn.execute(
                        """
                        UPDATE tasks
                           SET status = 'ready',
                               assignee = COALESCE(assignee, ?),
                               review_phase = NULL,
                               closeout_evidence = ?,
                               claim_lock = NULL,
                               claim_expires = NULL,
                               worker_pid = NULL,
                               completed_at = NULL,
                               last_failure_error = NULL,
                               consecutive_failures = 0
                         WHERE id = ?
                           AND status != 'archived'
                        """,
                        (
                            task.assignee or "arisu",
                            kb._json_dumps_dict(patched_evidence, "closeout_evidence"),
                            task_id,
                        ),
                    )
                    kb._append_event(
                        conn,
                        task_id,
                        "verifier_result",
                        {
                            "target_phase": verification.target_phase,
                            "verdict": "FAIL",
                            "reason": verification.reason,
                            "reason_codes": list(verification.blockers),
                            "blockers": list(verification.blockers),
                            "review_ready_input_eligible": False,
                            "allowed": False,
                        },
                    )
                    kb._append_event(conn, task_id, "remediation_requested", remediation)
            except Exception:
                pass
            updated = kb.get_task(conn, task_id)
            if updated and updated.status == "ready":
                return {
                    "status": "remediation_requested",
                    "reason": "reviewer_fail_remediation_queued",
                    "task_id": task_id,
                    "current_phase": task.review_phase,
                    "target_phase": verification.target_phase,
                    "blockers": verification.blockers,
                    "remediation": remediation,
                    "evidence": patched_evidence,
                    "side_effects": {"kanban_task_written": True, "linear_done_mutated": False},
                }
        try:
            with kb.write_txn(conn):
                kb._append_event(
                    conn,
                    task_id,
                    "verifier_result",
                    {
                        "target_phase": verification.target_phase,
                        "verdict": "FAIL" if verification.blockers else "BLOCKED",
                        "reason": verification.reason,
                        "reason_codes": list(verification.blockers),
                        "blockers": list(verification.blockers),
                        "review_ready_input_eligible": False,
                        "allowed": False,
                    },
                )
        except Exception:
            # Closeout verification must remain fail-closed even if the
            # observability event cannot be written. The caller still receives
            # the blocked verifier result below.
            pass
        return {
            "status": "blocked",
            "reason": verification.reason,
            "task_id": task_id,
            "current_phase": task.review_phase,
            "target_phase": verification.target_phase,
            "blockers": verification.blockers,
            "evidence": verification.evidence,
            "side_effects": {"kanban_task_written": False, "linear_done_mutated": False},
        }

    written = kb.apply_closeout_transition(
        conn,
        task_id,
        review_phase=verification.target_phase,
        closeout_evidence=verification.evidence,
    )
    if not written:
        return {
            "status": "blocked",
            "reason": "closeout_transition_write_failed",
            "task_id": task_id,
            "current_phase": task.review_phase,
            "target_phase": verification.target_phase,
            "blockers": ["closeout_transition_write_failed"],
            "evidence": verification.evidence,
            "side_effects": {"kanban_task_written": False, "linear_done_mutated": False},
        }
    updated = kb.get_task(conn, task_id)
    return {
        "status": "transitioned",
        "reason": verification.reason,
        "task_id": task_id,
        "previous_phase": task.review_phase,
        "review_phase": updated.review_phase if updated else verification.target_phase,
        "task_status": updated.status if updated else None,
        "blockers": [],
        "evidence": verification.evidence,
        "side_effects": {
            "kanban_task_written": True,
            "linear_done_mutated": False,
            "gateway_restarted_or_reloaded": False,
            "pr_merged": False,
        },
    }
