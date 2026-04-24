from __future__ import annotations

import json
import re
import subprocess
import time
from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


AI_SIGNATURE = "🤖 This was written by an AI coding agent"


class PRWorkflowStatus(str, Enum):
    setup = "setup"
    implementing = "implementing"
    pr_created = "pr_created"
    ci_waiting = "ci_waiting"
    ci_failed = "ci_failed"
    review_waiting = "review_waiting"
    changes_requested = "changes_requested"
    fix_in_progress = "fix_in_progress"
    approved = "approved"
    ready_to_merge = "ready_to_merge"
    stuck = "stuck"
    closed = "closed"


class PRCIStatus(str, Enum):
    unknown = "unknown"
    pending = "pending"
    passing = "passing"
    failing = "failing"


class PRReviewStatus(str, Enum):
    unknown = "unknown"
    waiting = "waiting"
    approved = "approved"
    changes_requested = "changes_requested"


ALLOWED_PR_WORKFLOW_TRANSITIONS: dict[PRWorkflowStatus, set[PRWorkflowStatus]] = {
    PRWorkflowStatus.setup: {PRWorkflowStatus.implementing, PRWorkflowStatus.stuck, PRWorkflowStatus.closed},
    PRWorkflowStatus.implementing: {PRWorkflowStatus.pr_created, PRWorkflowStatus.stuck, PRWorkflowStatus.closed},
    PRWorkflowStatus.pr_created: {
        PRWorkflowStatus.ci_waiting,
        PRWorkflowStatus.review_waiting,
        PRWorkflowStatus.approved,
        PRWorkflowStatus.stuck,
        PRWorkflowStatus.closed,
    },
    PRWorkflowStatus.ci_waiting: {
        PRWorkflowStatus.ci_failed,
        PRWorkflowStatus.review_waiting,
        PRWorkflowStatus.approved,
        PRWorkflowStatus.ready_to_merge,
        PRWorkflowStatus.stuck,
        PRWorkflowStatus.closed,
    },
    PRWorkflowStatus.ci_failed: {PRWorkflowStatus.fix_in_progress, PRWorkflowStatus.stuck, PRWorkflowStatus.closed},
    PRWorkflowStatus.review_waiting: {
        PRWorkflowStatus.changes_requested,
        PRWorkflowStatus.approved,
        PRWorkflowStatus.stuck,
        PRWorkflowStatus.closed,
    },
    PRWorkflowStatus.changes_requested: {PRWorkflowStatus.fix_in_progress, PRWorkflowStatus.stuck, PRWorkflowStatus.closed},
    PRWorkflowStatus.fix_in_progress: {
        PRWorkflowStatus.ci_waiting,
        PRWorkflowStatus.review_waiting,
        PRWorkflowStatus.approved,
        PRWorkflowStatus.stuck,
        PRWorkflowStatus.closed,
    },
    PRWorkflowStatus.approved: {PRWorkflowStatus.changes_requested, PRWorkflowStatus.ready_to_merge, PRWorkflowStatus.stuck, PRWorkflowStatus.closed},
    PRWorkflowStatus.ready_to_merge: {PRWorkflowStatus.changes_requested, PRWorkflowStatus.stuck, PRWorkflowStatus.closed},
    PRWorkflowStatus.stuck: {PRWorkflowStatus.implementing, PRWorkflowStatus.fix_in_progress, PRWorkflowStatus.closed},
    PRWorkflowStatus.closed: set(),
}


class PRWorkflowRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    status: PRWorkflowStatus = PRWorkflowStatus.setup
    pr_number: Optional[int] = None
    pr_url: Optional[str] = None
    branch: str
    base_branch: Optional[str] = None
    worktree_path: str
    ci_status: PRCIStatus = PRCIStatus.unknown
    review_status: PRReviewStatus = PRReviewStatus.unknown
    local_verification_passed: Optional[bool] = None
    last_error: Optional[str] = None
    stuck_reason: Optional[str] = None
    closed_reason: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    @field_validator("task_id", "branch", "worktree_path")
    @classmethod
    def _required_text(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("field is required")
        return value


class CheckSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: str
    status: str
    conclusion: Optional[str] = None
    url: Optional[str] = None
    summary: Optional[str] = None


class ReviewSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewer: str
    state: str
    submitted_at: str = ""
    body: str = ""
    url: Optional[str] = None
    is_bot: bool = False


class RemediationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str
    title: str
    summary: str
    task_contract: dict[str, Any]
    ai_signature_required: bool = True


class PRWorkflowLoopResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record: PRWorkflowRecord
    stop: bool = False
    reason: str = "continue"
    ready_to_merge: bool = False
    remediation_requests: list[RemediationRequest] = Field(default_factory=list)


class PrPollResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: PRWorkflowStatus
    repo: str
    pr_number: int
    pr_url: str
    branch: str
    ci_status: PRCIStatus = PRCIStatus.unknown
    review_status: PRReviewStatus = PRReviewStatus.unknown
    local_verification_passed: Optional[bool] = None
    ready_to_merge: bool = False
    remediation_requests: list[RemediationRequest] = Field(default_factory=list)
    failing_checks: list[CheckSignal] = Field(default_factory=list)
    pending_checks: list[CheckSignal] = Field(default_factory=list)
    review_feedback: list[ReviewSignal] = Field(default_factory=list)
    approvals: list[ReviewSignal] = Field(default_factory=list)


def _now(now: Optional[float]) -> float:
    return time.time() if now is None else now


def create_pr_workflow(*, task_id: str, branch: str, worktree_path: str, base_branch: str | None = None, now: float | None = None, **extra: Any) -> PRWorkflowRecord:
    if extra:
        payload = {"task_id": task_id, "branch": branch, "worktree_path": worktree_path, **extra}
        return PRWorkflowRecord.model_validate(payload)
    ts = _now(now)
    return PRWorkflowRecord(task_id=task_id, branch=branch, worktree_path=worktree_path, base_branch=base_branch, created_at=ts, updated_at=ts)


def _coerce_status(status: PRWorkflowStatus | str) -> PRWorkflowStatus:
    return status if isinstance(status, PRWorkflowStatus) else PRWorkflowStatus(str(status))


def _coerce_ci(status: PRCIStatus | str | None, fallback: PRCIStatus) -> PRCIStatus:
    if status is None:
        return fallback
    return status if isinstance(status, PRCIStatus) else PRCIStatus(str(status))


def _coerce_review(status: PRReviewStatus | str | None, fallback: PRReviewStatus) -> PRReviewStatus:
    if status is None:
        return fallback
    return status if isinstance(status, PRReviewStatus) else PRReviewStatus(str(status))


def _is_reachable_transition(source: PRWorkflowStatus, target: PRWorkflowStatus) -> bool:
    if source == target:
        return True
    seen = {source}
    queue = list(ALLOWED_PR_WORKFLOW_TRANSITIONS[source])
    while queue:
        current = queue.pop(0)
        if current == target:
            return True
        if current in seen:
            continue
        seen.add(current)
        queue.extend(next_status for next_status in ALLOWED_PR_WORKFLOW_TRANSITIONS[current] if next_status not in seen)
    return False


class _PollTransitionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pr_number: Optional[int] = None
    pr_url: Optional[str] = None
    ci_status: PRCIStatus = PRCIStatus.unknown
    review_status: PRReviewStatus = PRReviewStatus.unknown
    local_verification_passed: Optional[bool] = None
    stuck_reason: Optional[str] = None
    closed_reason: Optional[str] = None


def _validate_poll_transition(record: PRWorkflowRecord, poll: PrPollResult) -> None:
    if not _is_reachable_transition(record.status, poll.status):
        raise ValueError(f"invalid PR workflow transition: {record.status.value} -> {poll.status.value}")
    payload = _PollTransitionPayload(
        pr_number=poll.pr_number,
        pr_url=poll.pr_url,
        ci_status=poll.ci_status,
        review_status=poll.review_status,
        local_verification_passed=poll.local_verification_passed,
        closed_reason="closed by GitHub" if poll.status == PRWorkflowStatus.closed else None,
    )
    if poll.status == PRWorkflowStatus.pr_created and (not payload.pr_number or not payload.pr_url):
        raise ValueError("pr_created requires pr_number and pr_url")
    if poll.status == PRWorkflowStatus.ci_failed and payload.ci_status != PRCIStatus.failing:
        raise ValueError("ci_failed requires failing CI status")
    if poll.status == PRWorkflowStatus.approved and payload.review_status != PRReviewStatus.approved:
        raise ValueError("approved requires approved review status")
    if poll.status == PRWorkflowStatus.changes_requested and payload.review_status != PRReviewStatus.changes_requested:
        raise ValueError("changes_requested requires changes_requested review status")
    if poll.status == PRWorkflowStatus.ready_to_merge:
        if not (
            payload.pr_number
            and payload.pr_url
            and payload.ci_status == PRCIStatus.passing
            and payload.review_status == PRReviewStatus.approved
            and payload.local_verification_passed is True
        ):
            raise ValueError("ready_to_merge requires PR identity, passing CI, approved review, and local verification")
    if poll.status == PRWorkflowStatus.closed and not payload.closed_reason:
        raise ValueError("closed requires closed_reason")


def transition_pr_workflow(
    record: PRWorkflowRecord,
    new_status: PRWorkflowStatus | str,
    *,
    now: float | None = None,
    pr_number: int | None = None,
    pr_url: str | None = None,
    ci_status: PRCIStatus | str | None = None,
    review_status: PRReviewStatus | str | None = None,
    local_verification_passed: bool | None = None,
    last_error: str | None = None,
    stuck_reason: str | None = None,
    closed_reason: str | None = None,
) -> PRWorkflowRecord:
    target = _coerce_status(new_status)
    if target not in ALLOWED_PR_WORKFLOW_TRANSITIONS[record.status]:
        raise ValueError(f"invalid PR workflow transition: {record.status.value} -> {target.value}")

    next_pr_number = pr_number if pr_number is not None else record.pr_number
    next_pr_url = pr_url if pr_url is not None else record.pr_url
    next_ci = _coerce_ci(ci_status, record.ci_status)
    next_review = _coerce_review(review_status, record.review_status)
    next_local = local_verification_passed if local_verification_passed is not None else record.local_verification_passed

    if target == PRWorkflowStatus.pr_created and (not next_pr_number or not next_pr_url):
        raise ValueError("pr_created requires pr_number and pr_url")
    if target == PRWorkflowStatus.ci_failed and next_ci != PRCIStatus.failing:
        raise ValueError("ci_failed requires failing CI status")
    if target == PRWorkflowStatus.approved and next_review != PRReviewStatus.approved:
        raise ValueError("approved requires approved review status")
    if target == PRWorkflowStatus.changes_requested and next_review != PRReviewStatus.changes_requested:
        raise ValueError("changes_requested requires changes_requested review status")
    if target == PRWorkflowStatus.ready_to_merge:
        if not (next_pr_number and next_pr_url and next_ci == PRCIStatus.passing and next_review == PRReviewStatus.approved and next_local is True):
            raise ValueError("ready_to_merge requires PR identity, passing CI, approved review, and local verification")
    if target == PRWorkflowStatus.stuck and not str(stuck_reason or record.stuck_reason or "").strip():
        raise ValueError("stuck requires stuck_reason")
    if target == PRWorkflowStatus.closed and not str(closed_reason or record.closed_reason or "").strip():
        raise ValueError("closed requires closed_reason")

    return record.model_copy(
        update={
            "status": target,
            "pr_number": next_pr_number,
            "pr_url": next_pr_url,
            "ci_status": next_ci,
            "review_status": next_review,
            "local_verification_passed": next_local,
            "last_error": last_error if last_error is not None else record.last_error,
            "stuck_reason": stuck_reason if stuck_reason is not None else record.stuck_reason,
            "closed_reason": closed_reason if closed_reason is not None else record.closed_reason,
            "updated_at": _now(now),
        }
    )


def append_ai_signature(body: str) -> str:
    body = str(body or "").strip()
    if not body:
        return AI_SIGNATURE
    body = body.replace(AI_SIGNATURE, "").strip()
    return f"{body}\n\n{AI_SIGNATURE}"


_SECRET_PATTERNS = [
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{8,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{12,}"),
    re.compile(r"xox[baprs]-[^\s]+"),
    re.compile(r"(?i)(token|api[_-]?key|secret|password)\s*[=:]\s*[^\s,;]+"),
]


def redact_sensitive_text(text: str) -> str:
    redacted = str(text or "")
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


class GhCliError(RuntimeError):
    pass


class GhCliNotInstalledError(GhCliError):
    pass


class GhCliAuthError(GhCliError):
    pass


class GhCliCommandError(GhCliError):
    pass


class GhCliProtocolError(GhCliError):
    pass


class GhPrPollingAdapter:
    PR_JSON_FIELDS = "number,title,url,state,isDraft,reviewDecision,mergeable,headRefName,headRefOid,baseRefName,statusCheckRollup"

    def __init__(self, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run, *, timeout: int = 15) -> None:
        self.runner = runner
        self.timeout = timeout

    def _run(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            completed = self.runner(cmd, capture_output=True, text=True, timeout=self.timeout)
        except FileNotFoundError as exc:
            raise GhCliNotInstalledError("GitHub CLI (gh) is not installed. Install gh, then run `gh auth login`.") from exc
        if completed.returncode != 0:
            stderr = redact_sensitive_text(completed.stderr or completed.stdout or "")
            if "not logged" in stderr.lower() or "auth" in stderr.lower():
                raise GhCliAuthError("GitHub CLI is not authenticated for github.com. Run `gh auth login`.")
            raise GhCliCommandError(f"gh command failed: {stderr.strip() or completed.returncode}")
        return completed

    def ensure_auth(self) -> None:
        self._run(["gh", "auth", "status", "--hostname", "github.com"])

    def poll_pull_request(self, repo: str, pr_number: int, *, local_verification_passed: bool | None = None) -> PrPollResult:
        repo = str(repo or "").strip()
        if not repo or not pr_number:
            raise ValueError("repo and pr_number are required")
        self.ensure_auth()
        pr = self._load_json([
            "gh",
            "pr",
            "view",
            str(pr_number),
            "--repo",
            repo,
            "--json",
            self.PR_JSON_FIELDS,
        ])
        reviews = self._load_json(["gh", "api", f"repos/{repo}/pulls/{pr_number}/reviews?per_page=100"])
        if not isinstance(pr, dict) or not isinstance(reviews, list):
            raise GhCliProtocolError("unexpected gh JSON shape")
        return self._build_poll_result(repo, int(pr_number), pr, reviews, local_verification_passed=local_verification_passed)

    def _load_json(self, cmd: list[str]) -> Any:
        completed = self._run(cmd)
        try:
            return json.loads(completed.stdout or "null")
        except json.JSONDecodeError as exc:
            raise GhCliProtocolError("gh returned invalid JSON") from exc

    def _build_poll_result(self, repo: str, pr_number: int, pr: dict[str, Any], reviews: list[dict[str, Any]], *, local_verification_passed: bool | None) -> PrPollResult:
        checks = [_normalize_check(item) for item in pr.get("statusCheckRollup") or []]
        failing_checks = [check for check in checks if check.conclusion in {"FAILURE", "ERROR", "TIMED_OUT", "CANCELLED", "ACTION_REQUIRED", "STARTUP_FAILURE"} or check.status in {"FAILURE", "ERROR"}]
        pending_checks = [check for check in checks if check.status in {"QUEUED", "IN_PROGRESS", "PENDING", "REQUESTED", "WAITING"}]
        latest_reviews = _latest_human_reviews(reviews)
        review_feedback = [r for r in latest_reviews if r.state == "CHANGES_REQUESTED" or (r.state == "COMMENTED" and r.body.strip())]
        approvals = [r for r in latest_reviews if r.state == "APPROVED"]

        ci_status = PRCIStatus.failing if failing_checks else PRCIStatus.pending if pending_checks else PRCIStatus.passing
        if any(r.state == "CHANGES_REQUESTED" for r in review_feedback):
            review_status = PRReviewStatus.changes_requested
        elif approvals or pr.get("reviewDecision") == "APPROVED":
            review_status = PRReviewStatus.approved
        elif review_feedback or pr.get("reviewDecision") in {"REVIEW_REQUIRED", "CHANGES_REQUESTED"}:
            review_status = PRReviewStatus.waiting
        else:
            review_status = PRReviewStatus.unknown

        remediation: list[RemediationRequest] = []
        status = PRWorkflowStatus.pr_created
        ready = False
        if pr.get("state") != "OPEN":
            status = PRWorkflowStatus.closed
        elif pr.get("isDraft"):
            status = PRWorkflowStatus.review_waiting
        elif failing_checks:
            status = PRWorkflowStatus.ci_failed
            remediation.append(_ci_remediation_request(repo, pr_number, pr, failing_checks))
        elif review_feedback:
            status = PRWorkflowStatus.changes_requested
            remediation.append(_review_remediation_request(repo, pr_number, pr, review_feedback))
        elif pending_checks:
            status = PRWorkflowStatus.ci_waiting
        elif review_status == PRReviewStatus.approved:
            status = PRWorkflowStatus.approved
            if pr.get("mergeable") == "MERGEABLE" and ci_status == PRCIStatus.passing and local_verification_passed is True:
                status = PRWorkflowStatus.ready_to_merge
                ready = True
        else:
            status = PRWorkflowStatus.review_waiting

        return PrPollResult(
            status=status,
            repo=repo,
            pr_number=pr_number,
            pr_url=str(pr.get("url") or ""),
            branch=str(pr.get("headRefName") or ""),
            ci_status=ci_status,
            review_status=review_status,
            local_verification_passed=local_verification_passed,
            ready_to_merge=ready,
            remediation_requests=remediation,
            failing_checks=failing_checks,
            pending_checks=pending_checks,
            review_feedback=review_feedback,
            approvals=approvals,
        )


def _normalize_check(item: dict[str, Any]) -> CheckSignal:
    kind = str(item.get("__typename") or "StatusContext")
    name = str(item.get("name") or item.get("context") or "unknown")
    status = str(item.get("status") or item.get("state") or "UNKNOWN").upper()
    conclusion = item.get("conclusion")
    conclusion = str(conclusion).upper() if conclusion is not None else None
    return CheckSignal(name=name, kind=kind, status=status, conclusion=conclusion, url=item.get("detailsUrl") or item.get("targetUrl"), summary=item.get("description"))


def _latest_human_reviews(reviews: list[dict[str, Any]]) -> list[ReviewSignal]:
    by_reviewer: dict[str, ReviewSignal] = {}
    for item in reviews:
        user = item.get("user") or {}
        is_bot = str(user.get("type") or "").lower() == "bot"
        reviewer = str(user.get("login") or "unknown")
        signal = ReviewSignal(
            reviewer=reviewer,
            state=str(item.get("state") or "").upper(),
            submitted_at=str(item.get("submitted_at") or ""),
            body=str(item.get("body") or ""),
            url=item.get("html_url"),
            is_bot=is_bot,
        )
        if signal.is_bot:
            continue
        previous = by_reviewer.get(reviewer)
        if previous is None or signal.submitted_at >= previous.submitted_at:
            by_reviewer[reviewer] = signal
    return list(by_reviewer.values())


def _base_contract(repo: str, pr_number: int, pr: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "required_skills": ["github-pr-workflow", "test-driven-development", "verification-before-completion"],
        "required_tools": ["read_file", "search_files", "patch", "terminal"],
        "must_do": [
            "preserve PR context and branch metadata",
            "run local verification before reporting remediation complete",
            "include an AI disclosure on any external GitHub content",
        ],
        "must_not_do": ["do not auto-merge the PR", "do not store GitHub tokens", "do not impersonate a human reviewer"],
        "context": {
            "source": source,
            "repo": repo,
            "pr_number": pr_number,
            "pr_url": pr.get("url"),
            "branch": pr.get("headRefName"),
            "ai_signature_required": True,
            "scheduler_ready": True,
            "task_tool_available": True,
        },
    }


def _ci_remediation_request(repo: str, pr_number: int, pr: dict[str, Any], failing_checks: list[CheckSignal]) -> RemediationRequest:
    names = ", ".join(check.name for check in failing_checks)
    contract = _base_contract(repo, pr_number, pr, "gh_polling_adapter")
    contract.update(
        {
            "task": f"Investigate and remediate failing GitHub checks for PR #{pr_number} in {repo}.",
            "expected_outcome": "A verified local remediation for the failing CI checks without merging the PR.",
        }
    )
    contract["context"]["failing_checks"] = [check.model_dump(mode="json") for check in failing_checks]
    return RemediationRequest(reason="ci_failure", title=f"Remediate CI failures for PR #{pr_number}", summary=f"Failing checks: {names}", task_contract=contract)


def _review_remediation_request(repo: str, pr_number: int, pr: dict[str, Any], feedback: list[ReviewSignal]) -> RemediationRequest:
    contract = _base_contract(repo, pr_number, pr, "gh_polling_adapter")
    contract.update(
        {
            "task": f"Address human review feedback for PR #{pr_number} in {repo}.",
            "expected_outcome": "A verified local remediation or response plan for the latest human review feedback without merging the PR.",
        }
    )
    contract["context"]["review_feedback"] = [item.model_dump(mode="json") for item in feedback]
    return RemediationRequest(reason="review_feedback", title=f"Address review feedback for PR #{pr_number}", summary=f"{len(feedback)} human review signal(s) require follow-up", task_contract=contract)


def update_pr_workflow_from_poll(record: PRWorkflowRecord, poll: PrPollResult, *, now: float | None = None) -> PRWorkflowRecord:
    _validate_poll_transition(record, poll)
    updates = {
        "status": poll.status,
        "pr_number": poll.pr_number,
        "pr_url": poll.pr_url,
        "branch": poll.branch or record.branch,
        "ci_status": poll.ci_status,
        "review_status": poll.review_status,
        "local_verification_passed": poll.local_verification_passed,
        "closed_reason": "closed by GitHub" if poll.status == PRWorkflowStatus.closed and not record.closed_reason else record.closed_reason,
        "updated_at": _now(now),
    }
    return record.model_copy(update=updates)


def _final_review_remediation(repo: str, pr_number: int, pr_url: str, branch: str) -> RemediationRequest:
    contract = _base_contract(repo, pr_number, {"url": pr_url, "headRefName": branch}, "work_with_pr_loop")
    contract.update(
        {
            "task": f"Run final/Cubic review gate for PR #{pr_number} in {repo}.",
            "expected_outcome": "A verified final review result before marking the PR ready to merge.",
        }
    )
    contract["context"]["blocked_reason"] = "final review gate has not passed"
    return RemediationRequest(
        reason="final_review_required",
        title=f"Run final review for PR #{pr_number}",
        summary="PR has approved review, passing CI, and local verification, but final-review/Cubic gate is still required.",
        task_contract=contract,
    )


def evaluate_pr_workflow_loop(
    record: PRWorkflowRecord,
    poll: PrPollResult,
    *,
    round_index: int,
    max_rounds: int = 3,
    final_review_required: bool = False,
    final_review_passed: bool | None = None,
    now: float | None = None,
) -> PRWorkflowLoopResult:
    """Evaluate the Wave 6 work-with-PR loop without auto-merging.

    The loop returns remediation requests that the Wave 5 task tool/scheduler
    can convert into persistent tasks when an explicit caller chooses to enqueue
    them. It does not auto-merge or silently mutate task state by itself.
    """
    updated = update_pr_workflow_from_poll(record, poll, now=now)
    remediation = list(poll.remediation_requests)
    if round_index >= max_rounds and poll.status not in {PRWorkflowStatus.ready_to_merge, PRWorkflowStatus.closed}:
        reason = f"Reached maximum PR workflow rounds ({max_rounds}) before all gates passed."
        return PRWorkflowLoopResult(
            record=updated.model_copy(update={"status": PRWorkflowStatus.stuck, "stuck_reason": reason, "updated_at": _now(now)}),
            stop=True,
            reason="max_rounds_exceeded",
            ready_to_merge=False,
            remediation_requests=remediation,
        )
    if poll.ready_to_merge:
        if final_review_required and final_review_passed is not True:
            pending = updated.model_copy(update={"status": PRWorkflowStatus.approved, "updated_at": _now(now)})
            remediation.append(_final_review_remediation(poll.repo, poll.pr_number, poll.pr_url, poll.branch))
            return PRWorkflowLoopResult(
                record=pending,
                stop=False,
                reason="final_review_required",
                ready_to_merge=False,
                remediation_requests=remediation,
            )
        return PRWorkflowLoopResult(record=updated, stop=True, reason="ready_to_merge", ready_to_merge=True)
    if remediation:
        return PRWorkflowLoopResult(record=updated, stop=False, reason="remediation_required", remediation_requests=remediation)
    return PRWorkflowLoopResult(record=updated, stop=False, reason=poll.status.value, remediation_requests=remediation)


DISALLOWED_PR_ACTIONS = {"merge", "auto_merge", "delete_branch", "close", "enable_auto_merge"}
