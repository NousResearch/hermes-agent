"""Kanban-native closeout verifier.

This module owns the review lifecycle above raw Kanban worker execution:
``worker_done`` records executor completion, ``review_ready`` proves the work is
ready for human review, and ``closed`` records final close authority.  It is
intentionally Linear-free and fail-closed; Linear Done mutation, PR merge, and
gateway restart/reload are outside this surface.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_drift_audit

CLOSEOUT_EVIDENCE_SCHEMA = "kanban_closeout_evidence.v1"
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
        "number,url,state,isDraft,headRefOid,statusCheckRollup",
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
    git_head = _text(_as_mapping(evidence.get("git")).get("head_sha"))
    if not pr_head:
        blockers.append("missing_pr_head_sha")
    elif git_head and pr_head != git_head:
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
    verdict = evidence.get("verifier_verdict") or evidence.get("verifier")
    if isinstance(verdict, Mapping):
        return _lower(verdict.get("verdict") or verdict.get("status")) == "pass"
    return _lower(verdict) == "pass"


def _review_ready_blockers(evidence: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not _has_worker_evidence(evidence):
        blockers.append("missing_worker_evidence")
    if not _verifier_pass_present(evidence):
        blockers.append("missing_verifier_pass")
    no_pr_exception_blockers = _no_pr_review_ready_exception_blockers(evidence)
    if no_pr_exception_blockers is None:
        blockers.extend(_check_pr(evidence))
    else:
        blockers.extend(no_pr_exception_blockers)
    blockers.extend(_check_statuses(evidence))
    blockers.extend(_residue_blockers(evidence))
    cleanup_ok, cleanup_blocker = _cleanup_proven(evidence)
    if not cleanup_ok and cleanup_blocker:
        blockers.append(cleanup_blocker)
    if evidence.get("ambiguous") is True:
        blockers.append("ambiguous_evidence")
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
