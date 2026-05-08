"""Validation-contract and handoff linting for Hermes Kanban boards.

The rules here are intentionally conservative: they flag missing mission
validation contracts, missing assertion coverage, weak completion handoffs,
user-visible QA without live evidence, promissory known-failure notes, and
repo-touching completions without clean git/commit evidence.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Any, Iterable, Optional


_IMPLEMENTER_PROFILES = {
    "backend-eng",
    "frontend-eng",
    "fullstack-eng",
    "data-eng",
    "devops-eng",
    "docs-writer",
}
_GATE_PROFILES = {"reviewer", "qa-eng", "security-eng", "release-manager"}
_QA_PROFILES = {"qa-eng"}
_REPO_TOUCHING_PROFILES = _IMPLEMENTER_PROFILES | {"security-eng", "release-manager"}

_VC_RE = re.compile(r"\bVC-\d{3,}\b", re.IGNORECASE)
_GITHUB_ISSUE_URL_RE = re.compile(r"https://github\.com/[^\s)]+/issues/\d+", re.IGNORECASE)
_GITHUB_ISSUE_REF_RE = re.compile(r"(?:issue\s*)?#\d+\b", re.IGNORECASE)
_KNOWN_FAILURE_RE = re.compile(
    r"known\s+(?:pre[- ]existing|unrelated)\s+failure|pre[- ]existing\s+failure|issue\s+(?:will\s+be|to\s+be)\s+created",
    re.IGNORECASE,
)
_CREATED_PROMISE_RE = re.compile(r"issue\s+(?:will\s+be|to\s+be)\s+created", re.IGNORECASE)
_USER_VISIBLE_RE = re.compile(
    r"\b(ui|ux|frontend|browser|page|screen|route|modal|form|button|user-visible|live-app|playwright|e2e|screenshot)\b",
    re.IGNORECASE,
)


@dataclass
class ValidationFinding:
    task_id: str
    severity: str
    code: str
    title: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def _extract_vc_ids(*values: Any) -> set[str]:
    found: set[str] = set()
    for value in values:
        for match in _VC_RE.findall(_stringify(value)):
            found.add(match.upper())
    return found


def _metadata(run: Any) -> dict[str, Any]:
    if not run:
        return {}
    return _as_dict(getattr(run, "metadata", None))


def _summary(run: Any) -> str:
    return (getattr(run, "summary", None) or "") if run else ""


def _has_numeric_exit_codes(commands: list[Any]) -> bool:
    if not commands:
        return False
    for cmd in commands:
        if not isinstance(cmd, dict):
            return False
        if not str(cmd.get("command") or "").strip():
            return False
        exit_code = cmd.get("exit_code")
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            return False
    return True


def _is_repo_touching(task: Any, meta: dict[str, Any]) -> bool:
    assignee = (getattr(task, "assignee", None) or "").strip()
    workspace_kind = getattr(task, "workspace_kind", None)
    if workspace_kind == "worktree":
        return True
    if assignee in _REPO_TOUCHING_PROFILES and any(
        meta.get(k)
        for k in ("changed_files", "commit", "branch", "pr_url", "tests_run", "verification_commands")
    ):
        return True
    return False


def _task_is_user_visible(task: Any, meta: dict[str, Any]) -> bool:
    blob = "\n".join(
        [
            getattr(task, "title", "") or "",
            getattr(task, "body", "") or "",
            _stringify(meta.get("changed_files")),
            _stringify(meta.get("acceptance_criteria_checked")),
            _stringify(meta.get("validation_assertions_checked")),
        ]
    )
    return bool(_USER_VISIBLE_RE.search(blob))


def _known_failure_is_tracked(text: str, meta: dict[str, Any]) -> bool:
    if _GITHUB_ISSUE_URL_RE.search(text):
        return True
    issue_fields = [
        meta.get("issue_url"),
        meta.get("known_failure_issue_url"),
        meta.get("tracking_issue_url"),
        meta.get("tracked_issue_url"),
        meta.get("issue_number"),
        meta.get("known_failure_issue_number"),
        meta.get("tracking_issue_number"),
    ]
    if any(issue_fields):
        return True
    tracked = meta.get("known_failures_tracked") or meta.get("tracked_known_failures")
    if isinstance(tracked, list) and tracked:
        return True
    return False


def _validate_contract_task(task: Any) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    body = getattr(task, "body", None) or ""
    title = getattr(task, "title", None) or ""
    lower_blob = f"{title}\n{body}".lower()
    looks_like_root = any(
        term in lower_blob
        for term in (
            "mission",
            "root",
            "workplan",
            "prd",
            "architecture improvement loop",
            "issue series",
            "multi-card",
        )
    )
    if not looks_like_root:
        return findings
    vc_ids = _extract_vc_ids(body)
    if "validation_contract" not in lower_blob and not vc_ids:
        findings.append(ValidationFinding(
            getattr(task, "id"),
            "error",
            "missing_validation_contract",
            "Mission/root card lacks a validation contract",
            "Non-trivial mission/workplan/root cards should define atomic VC-* assertions before implementation fan-out.",
        ))
    if "uncovered_assertions" not in lower_blob and vc_ids:
        findings.append(ValidationFinding(
            getattr(task, "id"),
            "warning",
            "missing_coverage_ledger",
            "Validation assertions lack an explicit coverage ledger",
            "Include uncovered_assertions/coverage_complete so dispatchers can see whether every VC-* has owner/evidence coverage.",
        ))
    return findings


def _validate_ready_task(task: Any) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    assignee = (getattr(task, "assignee", None) or "").strip()
    status = getattr(task, "status", None)
    if status not in {"todo", "ready", "running"}:
        return findings
    body = getattr(task, "body", None) or ""
    vc_ids = _extract_vc_ids(body)
    if assignee in _IMPLEMENTER_PROFILES and not vc_ids:
        findings.append(ValidationFinding(
            getattr(task, "id"),
            "warning",
            "implementation_missing_vc_assertions",
            "Implementation card does not cite VC-* assertions",
            "Implementation cards should list validation_assertions_to_satisfy or otherwise cite parent VC-* IDs.",
        ))
    if assignee in _GATE_PROFILES and not vc_ids:
        findings.append(ValidationFinding(
            getattr(task, "id"),
            "warning",
            "gate_missing_vc_assertions",
            "Gate card does not cite VC-* assertions",
            "Review/QA/security/release cards should list validation_assertions_to_verify or otherwise cite parent VC-* IDs.",
        ))
    return findings


def _validate_completion_handoff(task: Any, latest_run: Any) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    tid = getattr(task, "id")
    meta = _metadata(latest_run)
    summary = _summary(latest_run)
    text = "\n".join([getattr(task, "result", None) or "", summary, _stringify(meta)])
    assignee = (getattr(task, "assignee", None) or "").strip()

    repo_touching = _is_repo_touching(task, meta)
    gate_owner = assignee in _GATE_PROFILES
    if repo_touching or gate_owner:
        commands = _as_list(meta.get("commands_run"))
        # Back-compat: older workers used tests_run as strings. Warn, don't fail, when tests exist but commands_run is absent.
        if not _has_numeric_exit_codes(commands):
            severity = "error" if repo_touching else "warning"
            findings.append(ValidationFinding(
                tid,
                severity,
                "missing_commands_run_exit_codes",
                "Completion handoff lacks commands_run with numeric exit_code",
                "Repo-touching and gate-owning cards should record each verification command with a numeric exit_code.",
            ))

    if repo_touching:
        if not str(meta.get("commit") or "").strip():
            findings.append(ValidationFinding(
                tid,
                "error",
                "repo_completion_missing_commit",
                "Repo-touching completion lacks commit SHA",
                "Repo-writing cards should commit intended changes and report the commit SHA before completion.",
            ))
        git_status = str(meta.get("git_status") or "").lower()
        if not git_status or not git_status.startswith("clean"):
            findings.append(ValidationFinding(
                tid,
                "error",
                "repo_completion_not_clean",
                "Repo-touching completion does not report clean git status",
                "Repo-writing cards must complete clean or explicitly block with the dirty files/reason.",
            ))

    failed = meta.get("validation_assertions_failed") or meta.get("assertions_failed") or meta.get("failed_assertions")
    if failed:
        has_followup = any(
            meta.get(k)
            for k in (
                "corrective_cards",
                "recommended_remediation_cards",
                "bugs_found",
                "issue_url",
                "tracking_issue_url",
                "human_decision_needed",
                "issue_creation_blocked",
            )
        )
        if not has_followup:
            findings.append(ValidationFinding(
                tid,
                "error",
                "failed_assertions_without_followup",
                "Failed validation assertions lack corrective routing",
                "If validation_assertions_failed/assertions_failed is non-empty, include corrective card IDs, issue URL, human_decision_needed, or issue_creation_blocked.",
            ))

    if assignee in _QA_PROFILES and _task_is_user_visible(task, meta):
        mode = str(meta.get("qa_evidence_mode") or "").strip()
        routes = _as_list(meta.get("urls_or_routes_checked"))
        artifacts = _as_list(meta.get("screenshots_or_artifacts")) or _as_list(meta.get("evidence"))
        if mode in {"", "blocked"} or not routes or not artifacts:
            findings.append(ValidationFinding(
                tid,
                "warning",
                "user_visible_qa_missing_live_evidence",
                "User-visible QA lacks live/e2e/manual evidence fields",
                "QA handoff should include qa_evidence_mode, urls_or_routes_checked, and screenshots/logs/artifacts unless explicitly blocked.",
            ))

    if _KNOWN_FAILURE_RE.search(text):
        if _CREATED_PROMISE_RE.search(text) or not _known_failure_is_tracked(text, meta):
            findings.append(ValidationFinding(
                tid,
                "error",
                "known_failure_not_verified_tracked",
                "Known/pre-existing failure is not linked to a verified issue",
                "Do not leave promissory known-failure notes. Create/search/verify a GitHub issue and include its URL/number, or mark issue_creation_blocked and do not call release-ready.",
            ))
    return findings


def validate_tasks(tasks: Iterable[Any], runs_by_task: Optional[dict[str, list[Any]]] = None) -> list[ValidationFinding]:
    runs_by_task = runs_by_task or {}
    findings: list[ValidationFinding] = []
    for task in tasks:
        findings.extend(_validate_contract_task(task))
        findings.extend(_validate_ready_task(task))
        latest_run = None
        runs = runs_by_task.get(getattr(task, "id", ""), [])
        if runs:
            latest_run = runs[-1]
        if latest_run is not None:
            findings.extend(_validate_completion_handoff(task, latest_run))
    return findings


def summarize_findings(findings: Iterable[ValidationFinding]) -> dict[str, int]:
    summary = {"error": 0, "warning": 0}
    for finding in findings:
        summary[finding.severity] = summary.get(finding.severity, 0) + 1
    return summary
