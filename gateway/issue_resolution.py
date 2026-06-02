"""GitHub issue resolution lane for Hermes Gateway."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import sqlite3
import time
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


logger = logging.getLogger(__name__)

DEFAULT_GUARDIAN_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_LOCAL_MODEL = "openai/qwen3-35b-uncensored"
DEFAULT_DECOMPOSE_MODEL = "qwen3-35b-uncensored"
DEFAULT_CLOUD_MODEL = "openrouter/deepseek/deepseek-v4-flash"
DEFAULT_REVIEWER_TIER2_MODEL = "openai/gemma4-26b-a4b"
DEFAULT_REVIEWER_TIER3_MODEL = "openrouter/deepseek/deepseek-v4-pro"
DEFAULT_CODER_TIER1_MODEL = "openrouter/deepseek/deepseek-v4-flash"
DEFAULT_CODER_TIER2_MODEL = "openai/qwen-3.6-35b"
DEFAULT_KYBER_ENV = Path.home() / "kyberm0nk" / ".env"
DEFAULT_AIDER_BIN = Path.home() / "aider" / ".venv" / "bin" / "aider"
DEFAULT_STATE_DB = get_hermes_home() / "issue_resolution.db"
MASTER_PLAN_LABEL = "master-plan"
MASTER_PLAN_HEADING = "# Master Project Plan"
ISSUE_RUN_MAX_ATTEMPTS = 3
ISSUE_RUN_RETRY_DELAYS_SECONDS = (60, 300)
ISSUE_RUN_IDLE_POLL_SECONDS = 5.0
DEFAULT_ALLOWED_ISSUE_REPOS = ("m0nklabs/cryptotrader",)
REVIEW_FINDINGS_MAX_FIX_ATTEMPTS = 2
REVIEW_TAG_MAX_PARSE_ATTEMPTS = 2
STRICT_PROTECTED_PUSH_BRANCHES = frozenset({"master", "main"})
DIRECT_MASTER_PUSH_ERROR = (
    "ERROR: Direct pushes to master are strictly forbidden by operator flip."
)
REVIEW_TAG_STATES = {"ready_for_merge", "review_findings", "review_inconclusive"}
REVIEW_TAG_NEXT_ACTIONS = {
    "coding_subagent",
    "rerun_reviewer",
    "tier2_review",
    "tier3_review",
    "ready_for_merge",
}
ISSUE_AUTO_MERGE_ENV = "HERMES_ISSUE_AUTO_MERGE_ENABLED"
MANAGED_REPO_POLICIES = {
    "m0nklabs/cryptotrader": {
        "name": "CryptoTrader",
        "protected_branches": ("master", "main"),
        "allowed_dirty_prefixes": (
            ".aider.chat.history.md",
            ".aider.input.history",
            ".aider.tags.cache.v4/",
        ),
    }
}
ISSUE_BRANCH_REPO_SLUGS = {
    "m0nklabs/cryptotrader": "cryptotrader",
}
PR_BODY_VALIDATION_PLACEHOLDER = (
    "Validation is pending. Hermes must update this section before merge after the "
    "local coder and reviewer finish their checks."
)


class AiderRole(str, Enum):
    """Supported Aider execution roles for issue resolution."""

    LOCAL_CODER = "local_coder"
    CLOUD_REVIEWER = "cloud_reviewer"


class ModelProvider(str, Enum):
    """Model routing provider used by PR-manager tiers."""

    OPENROUTER = "openrouter"
    GUARDIAN = "guardian"


@dataclass(frozen=True)
class ModelTier:
    """One ordered reviewer or coder model tier."""

    name: str
    provider: ModelProvider
    model: str
    purpose: str


@dataclass(frozen=True)
class ReviewSuggestionStats:
    """Suggestion counters extracted from internal and Copilot review metadata."""

    internal_suggestions_count: int
    copilot_suggestions_count: int
    copilot_review_detected: bool
    complex_findings_detected: bool

    @property
    def total_suggestions_count(self) -> int:
        """Return the total actionable suggestion count."""
        return self.internal_suggestions_count + self.copilot_suggestions_count


@dataclass(frozen=True)
class PRManagerDecision:
    """Next PR-manager action after evaluating review signals."""

    next_action: str
    reviewer_tier: ModelTier | None = None
    coder_tier: ModelTier | None = None
    ready_comment: str | None = None


REVIEWER_TIERS: tuple[ModelTier, ...] = (
    ModelTier(
        name="tier1",
        provider=ModelProvider.OPENROUTER,
        model=DEFAULT_CLOUD_MODEL,
        purpose="fast high-confidence review",
    ),
    ModelTier(
        name="tier2",
        provider=ModelProvider.GUARDIAN,
        model=DEFAULT_REVIEWER_TIER2_MODEL,
        purpose="local deep review",
    ),
    ModelTier(
        name="tier3",
        provider=ModelProvider.OPENROUTER,
        model=DEFAULT_REVIEWER_TIER3_MODEL,
        purpose="cloud escalation review",
    ),
)
CODER_TIERS: tuple[ModelTier, ...] = (
    ModelTier(
        name="tier1",
        provider=ModelProvider.OPENROUTER,
        model=DEFAULT_CODER_TIER1_MODEL,
        purpose="cheap review-fix coder",
    ),
    ModelTier(
        name="tier2",
        provider=ModelProvider.GUARDIAN,
        model=DEFAULT_CODER_TIER2_MODEL,
        purpose="local hard-fix coder",
    ),
)


class IssueRunStatus(str, Enum):
    """Persisted issue run states."""

    QUEUED = "queued"
    RUNNING = "running"
    EXPANDED = "expanded"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class IssueRunType(str, Enum):
    """Issue run types handled by the lane."""

    ISSUE = "issue"
    MASTER = "master"
    SUB_ISSUE = "sub_issue"


class ReviewFindingsRetry(RuntimeError):
    """Reviewer found actionable findings that should requeue coder work."""


class ReviewTagParseError(RuntimeError):
    """Reviewer output did not contain a valid kyber routing tag."""


class ReviewLoopCircuitBreaker(RuntimeError):
    """Reviewer findings repeated too many times for one issue run."""


class ReviewMergeGateError(RuntimeError):
    """Reviewer output or PR state did not satisfy automatic merge gates."""


@dataclass(frozen=True)
class IssueResolutionRequest:
    """Normalized request to resolve one GitHub issue."""

    repo: str
    issue_number: int
    workdir: Path
    branch: str | None = None
    kanban_task_id: str | None = None
    kanban_board: str | None = None


@dataclass(frozen=True)
class IssueSelectionRequest:
    """Normalized request to select the next open issue in a repository."""

    repo: str
    workdir: Path
    branch: str | None = None


@dataclass(frozen=True)
class IssueMetadata:
    """GitHub issue metadata used to prompt the local coder."""

    number: int
    title: str
    body: str
    url: str
    labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class PullRequestMetadata:
    """GitHub pull request metadata emitted by the local coder lane."""

    number: int
    url: str
    head_ref_name: str
    head_ref_oid: str
    action: str = "opened"


@dataclass(frozen=True)
class EpicTask:
    """Atomic task generated from a master-plan issue."""

    title: str
    body: str


@dataclass(frozen=True)
class AiderInvocation:
    """Concrete Aider subprocess invocation."""

    command: list[str]
    env: dict[str, str]
    cwd: Path


@dataclass(frozen=True)
class ManagedRepoStatus:
    """Read-only managed repository status before implementation dispatch."""

    repo: str
    name: str
    workdir: Path
    branch: str | None
    protected: bool
    violating_paths: tuple[str, ...]
    ignored_paths: tuple[str, ...]
    ok: bool
    reason: str


@dataclass(frozen=True)
class ReviewRoutingTag:
    """Structured routing tag emitted by the cloud reviewer."""

    state: str
    next_action: str | None = None
    head_ref_oid: str | None = None
    tier: str | None = None
    suggestions_count: int = 0
    source: str | None = None


@dataclass(frozen=True)
class IssueRun:
    """Persisted queued issue run."""

    id: int
    repo: str
    issue_number: int
    workdir: Path
    branch: str | None
    kanban_task_id: str | None
    kanban_board: str | None
    status: IssueRunStatus
    run_type: IssueRunType
    parent_run_id: int | None
    master_issue_number: int | None
    pr_number: int | None
    pr_url: str | None
    error: str | None
    review_findings_count: int = 0
    attempt_count: int = 0
    next_attempt_at: float = 0.0


@dataclass(frozen=True)
class SubmitResult:
    """Result returned when a run is submitted to the queue."""

    run_id: int
    status: IssueRunStatus
    reused: bool


@dataclass(frozen=True)
class CancellationResult:
    """Result returned when an operator cancels an issue run."""

    run_id: int
    status: IssueRunStatus
    cancelled: bool


_RUN_NOTIFIERS: dict[int, Callable[[str], Awaitable[None]]] = {}
_DEFAULT_NOTIFY: Callable[[str], Awaitable[None]] | None = None
_QUEUE_WORKER_TASK: asyncio.Task | None = None
_QUEUE_GUARD: asyncio.Lock | None = None


async def run_issue_resolution(
    request: IssueResolutionRequest,
    *,
    notify: Callable[[str], Awaitable[None]],
) -> None:
    """Run one issue immediately without enqueueing it."""
    _enforce_issue_repo_allowed(request.repo)
    store = IssueStateStore()
    issue = await _load_issue(request.repo, request.issue_number)
    if is_master_issue(issue):
        run = store.enqueue_run(request, run_type=IssueRunType.MASTER)
        await _execute_master_issue(store, run, issue, notify)
        return
    run = store.enqueue_run(request, run_type=IssueRunType.ISSUE)
    await _execute_single_issue(store, run, issue, notify)


async def submit_issue_resolution(
    request: IssueResolutionRequest,
    *,
    notify: Callable[[str], Awaitable[None]],
) -> SubmitResult:
    """Persist a run and ensure the single-flight worker is active."""
    _enforce_issue_repo_allowed(request.repo)
    issue = await _load_issue(request.repo, request.issue_number)
    run_type = IssueRunType.MASTER if is_master_issue(issue) else IssueRunType.ISSUE
    store = IssueStateStore()
    existing = store.find_incomplete_run(request.repo, request.issue_number)
    run = existing or store.enqueue_run(request, run_type=run_type)
    _RUN_NOTIFIERS[run.id] = notify
    await notify(
        f"Hermes: Queued {'master epic' if run_type is IssueRunType.MASTER else 'issue'} "
        f"#{request.issue_number} as run #{run.id}."
    )
    await ensure_issue_queue_worker(notify=notify)
    return SubmitResult(run_id=run.id, status=run.status, reused=existing is not None)


async def submit_next_issue_resolution(
    request: IssueSelectionRequest,
    *,
    notify: Callable[[str], Awaitable[None]],
) -> SubmitResult:
    """Resolve the oldest open issue in the repo and queue it for coding."""
    _enforce_issue_repo_allowed(request.repo)
    issue = await _load_next_open_issue(request.repo)
    return await submit_issue_resolution(
        IssueResolutionRequest(
            repo=request.repo,
            issue_number=issue.number,
            workdir=request.workdir,
            branch=request.branch,
            kanban_task_id=request.kanban_task_id,
            kanban_board=request.kanban_board,
        ),
        notify=notify,
    )


async def cancel_issue_resolution(
    run_id: int,
    *,
    reason: str,
    notify: Callable[[str], Awaitable[None]],
) -> CancellationResult:
    """Cancel a queued or running issue run and write an audit note when possible."""
    store = IssueStateStore()
    result = store.cancel_run(run_id, reason)
    run = store.get_run(run_id)
    if result.cancelled:
        await notify(f"Hermes: Cancelled issue run #{run_id}: {reason}")
        await _post_issue_audit_comment(
            run.repo,
            run.issue_number,
            f"Hermes audit: run #{run_id} cancelled by operator. Reason: {reason}",
        )
        await _record_kanban_task_audit(
            run,
            "issue_run_cancelled",
            _issue_run_audit_payload(run, reason=reason),
            comment=f"Hermes audit: issue-resolution run #{run_id} cancelled. Reason: {reason}",
        )
    return result


def allowed_issue_repos() -> tuple[str, ...]:
    """Return repositories allowed to run issue automation."""
    configured = os.getenv("HERMES_ISSUE_ALLOWED_REPOS", "").strip()
    if not configured:
        return DEFAULT_ALLOWED_ISSUE_REPOS
    repos = tuple(repo.strip() for repo in configured.split(",") if repo.strip())
    return repos or DEFAULT_ALLOWED_ISSUE_REPOS


def is_issue_repo_allowed(repo: str) -> bool:
    """Return true when the issue lane may execute for this repository."""
    return repo in allowed_issue_repos()


def _enforce_issue_repo_allowed(repo: str) -> None:
    if is_issue_repo_allowed(repo):
        return
    allowed = ", ".join(allowed_issue_repos())
    raise RuntimeError(
        f"Hermes issue automation is not allowed for {repo}; allowed repositories: {allowed}"
    )


async def resume_issue_resolution_queue(
    *,
    notify: Callable[[str], Awaitable[None]] | None = None,
) -> int:
    """Reset interrupted runs and resume the persistent queue after gateway startup."""
    store = IssueStateStore()
    resumed = store.reset_interrupted_runs()
    pending = store.count_pending_runs()
    if pending:
        if notify:
            await notify(f"Hermes: Resuming {pending} queued issue-resolution run(s).")
        await ensure_issue_queue_worker(notify=notify)
    return resumed + pending


async def ensure_issue_queue_worker(
    *,
    notify: Callable[[str], Awaitable[None]] | None = None,
) -> None:
    """Start the FIFO worker if it is not already running."""
    global _DEFAULT_NOTIFY, _QUEUE_GUARD, _QUEUE_WORKER_TASK
    if notify is not None:
        _DEFAULT_NOTIFY = notify
    if _QUEUE_GUARD is None:
        _QUEUE_GUARD = asyncio.Lock()
    async with _QUEUE_GUARD:
        if _QUEUE_WORKER_TASK and not _QUEUE_WORKER_TASK.done():
            return
        _QUEUE_WORKER_TASK = asyncio.create_task(_issue_queue_worker())


async def _issue_queue_worker() -> None:
    """Process queued issue runs in strict FIFO order."""
    store = IssueStateStore()
    while True:
        run = store.claim_next_run()
        if run is None:
            retry_delay = store.next_queued_delay()
            if retry_delay is None:
                return
            await asyncio.sleep(min(retry_delay, ISSUE_RUN_IDLE_POLL_SECONDS))
            continue
        notify = _RUN_NOTIFIERS.get(run.id) or _DEFAULT_NOTIFY or _noop_notify
        try:
            await _execute_run(store, run, notify)
        except (
            ReviewTagParseError,
            ReviewLoopCircuitBreaker,
            ReviewMergeGateError,
        ) as exc:
            store.mark_failed(run.id, str(exc))
            await _record_kanban_task_audit(
                run,
                "issue_run_failed",
                _issue_run_audit_payload(run, error=str(exc), failure_kind="review_safety_gate"),
                comment=f"Hermes audit: issue-resolution run #{run.id} failed review safety gate: {exc}",
            )
            await notify(
                f"Hermes: Issue run #{run.id} failed review safety gate: {exc}"
            )
            logger.exception(
                "issue-resolution run %s failed review safety gate", run.id
            )
        except Exception as exc:
            if store.mark_retry_or_failed(run, str(exc)):
                retrying = store.get_run(run.id)
                delay = max(0, int(retrying.next_attempt_at - _now()))
                await _record_kanban_task_audit(
                    retrying,
                    "issue_run_retry_queued",
                    _issue_run_audit_payload(
                        retrying,
                        error=str(exc),
                        attempt_count=retrying.attempt_count,
                        max_attempts=ISSUE_RUN_MAX_ATTEMPTS,
                        retry_delay_seconds=delay,
                    ),
                )
                await notify(
                    f"Hermes: Issue run #{run.id} failed attempt "
                    f"{retrying.attempt_count}/{ISSUE_RUN_MAX_ATTEMPTS}; "
                    f"retrying in {delay}s: {exc}"
                )
                logger.exception("issue-resolution run %s failed; queued retry", run.id)
            else:
                failed = store.get_run(run.id)
                await _record_kanban_task_audit(
                    failed,
                    "issue_run_failed",
                    _issue_run_audit_payload(
                        failed,
                        error=str(exc),
                        attempt_count=failed.attempt_count,
                        max_attempts=ISSUE_RUN_MAX_ATTEMPTS,
                    ),
                    comment=(
                        f"Hermes audit: issue-resolution run #{run.id} failed after "
                        f"{failed.attempt_count}/{ISSUE_RUN_MAX_ATTEMPTS} attempts: {exc}"
                    ),
                )
                await notify(
                    f"Hermes: Issue run #{run.id} failed after "
                    f"{failed.attempt_count}/{ISSUE_RUN_MAX_ATTEMPTS} attempts: {exc}"
                )
                logger.exception("issue-resolution run %s failed permanently", run.id)
        finally:
            _RUN_NOTIFIERS.pop(run.id, None)
            await _complete_ready_masters(store, notify)


async def _execute_run(
    store: "IssueStateStore",
    run: IssueRun,
    notify: Callable[[str], Awaitable[None]],
) -> None:
    issue = await _load_issue(run.repo, run.issue_number)
    if run.run_type is IssueRunType.MASTER or is_master_issue(issue):
        await _execute_master_issue(store, run, issue, notify)
        return
    await _execute_single_issue(store, run, issue, notify)


async def _execute_master_issue(
    store: "IssueStateStore",
    run: IssueRun,
    issue: IssueMetadata,
    notify: Callable[[str], Awaitable[None]],
) -> None:
    """Expand a master plan issue into queued sub-issues."""
    await notify(
        f"Hermes: Master Epic #{issue.number} detected; decomposing plan locally."
    )
    tasks = await decompose_master_plan(issue)
    if not tasks:
        raise RuntimeError("Guardian returned no atomic tasks for the master plan.")

    created = 0
    for position, task in enumerate(tasks, start=1):
        if store.subissue_exists(run.id, position):
            continue
        sub_issue = await _find_existing_sub_issue(run.repo, issue, task, position)
        if sub_issue is None:
            sub_issue = await _create_sub_issue(run.repo, issue, task, position)
        store.record_subissue(run.id, position, task, sub_issue)
        store.enqueue_run(
            IssueResolutionRequest(
                repo=run.repo,
                issue_number=sub_issue.number,
                workdir=run.workdir,
            ),
            run_type=IssueRunType.SUB_ISSUE,
            parent_run_id=run.id,
            master_issue_number=issue.number,
        )
        created += 1

    store.mark_expanded(run.id)
    await notify(
        f"Hermes: Master Epic #{issue.number} expanded into {created} new sub-issue(s)."
    )


async def _execute_single_issue(
    store: "IssueStateStore",
    run: IssueRun,
    issue: IssueMetadata,
    notify: Callable[[str], Awaitable[None]],
) -> None:
    """Sync the base branch, run local coder, push branch, and trigger review."""
    default_branch = await _load_default_branch(run.repo)
    branch = run.branch or _issue_branch_name(issue, run.repo)
    await _guard_managed_repo_before_issue_dispatch(run, issue, branch)
    await _prepare_issue_branch_from_synced_default(
        run.workdir, default_branch, branch
    )

    await notify(f"Hermes: Starting local coder for Issue #{issue.number}.")
    await _post_issue_audit_comment(
        run.repo,
        issue.number,
        f"Hermes audit: run #{run.id} claimed issue for branch `{branch}`.",
    )
    await _record_kanban_task_audit(
        run,
        "issue_run_claimed",
        _issue_run_audit_payload(
            run,
            issue=issue,
            branch=branch,
            default_branch=default_branch,
        ),
    )
    local_prompt = _local_coder_prompt(run.repo, issue, branch)
    coder_tier = (
        _coder_tier_for_findings(
            findings_cycles=run.review_findings_count,
            complex_findings=False,
        )
        if run.review_findings_count > 0
        else None
    )
    if coder_tier is not None:
        await notify(
            f"Hermes: routing same-branch fix for run #{run.id} to "
            f"{coder_tier.name} coder ({coder_tier.model})."
        )
    local_invocation = build_aider_invocation(
        AiderRole.LOCAL_CODER,
        run.workdir,
        local_prompt,
        coder_tier=coder_tier,
    )
    await _run(
        local_invocation.command, cwd=local_invocation.cwd, env=local_invocation.env
    )
    await _assert_issue_branch_ready_for_pr(
        run.repo, run.workdir, branch, default_branch
    )

    await notify(
        f"Hermes: Local coder finished Issue #{issue.number}; pushing `{branch}`."
    )
    await _push_issue_branch(run.workdir, branch)

    pr = await _create_or_find_pr(run.repo, issue, branch, default_branch, run=run)
    store.record_pr(run.id, pr)
    await notify(f"Hermes: PR #{pr.number} created, triggering reviewer: {pr.url}")
    await _post_issue_audit_comment(
        run.repo,
        issue.number,
        f"Hermes audit: run #{run.id} opened or reused PR #{pr.number}: {pr.url}",
    )
    pr_event_kind = "pr_reused" if pr.action == "reused" else "pr_opened"
    await _record_kanban_task_audit(
        run,
        pr_event_kind,
        _issue_run_audit_payload(run, issue=issue, pr=pr),
        comment=(
            f"Hermes audit: run #{run.id} {'reused' if pr.action == 'reused' else 'opened'} "
            f"PR #{pr.number} for GitHub issue {run.repo}#{issue.number}: {pr.url}"
        ),
    )
    await _post_pr_audit_comment(
        run.repo,
        pr,
        f"Hermes audit: cloud review requested for run #{run.id} at head `{pr.head_ref_oid}`.",
    )
    await _record_kanban_task_audit(
        run,
        "review_requested",
        _issue_run_audit_payload(run, issue=issue, pr=pr),
    )

    review_output, routing_tag = await _run_cloud_reviewer_with_tag(
        run.repo, run.workdir, pr, notify
    )
    await _post_pr_feedback(run.repo, pr, _review_body(review_output))
    if is_review_findings_for_coder(routing_tag):
        findings_count = store.record_review_findings(run.id)
        if findings_count > REVIEW_FINDINGS_MAX_FIX_ATTEMPTS:
            await _post_pr_audit_comment(
                run.repo,
                pr,
                f"Hermes audit: review loop circuit breaker tripped for run #{run.id} "
                f"after {findings_count} reviewer findings cycles.",
            )
            await _record_kanban_task_audit(
                run,
                "review_loop_circuit_breaker",
                _issue_run_audit_payload(
                    run,
                    issue=issue,
                    pr=pr,
                    findings_count=findings_count,
                    max_fix_attempts=REVIEW_FINDINGS_MAX_FIX_ATTEMPTS,
                ),
                comment=(
                    f"Hermes audit: review loop circuit breaker tripped for run #{run.id} "
                    f"after {findings_count} reviewer findings cycles."
                ),
            )
            raise ReviewLoopCircuitBreaker(
                f"review_findings repeated {findings_count} times for PR #{pr.number}; "
                "manual escalation required"
            )
        await notify(
            f"Hermes: Reviewer found fix work on PR #{pr.number}; keeping run #{run.id} "
            f"queued for same-branch coding ({findings_count}/"
            f"{REVIEW_FINDINGS_MAX_FIX_ATTEMPTS})."
        )
        await _post_pr_audit_comment(
            run.repo,
            pr,
            f"Hermes audit: reviewer findings routed run #{run.id} back to same-branch coding "
            f"({findings_count}/{REVIEW_FINDINGS_MAX_FIX_ATTEMPTS}).",
        )
        await _record_kanban_task_audit(
            run,
            "review_fix_routed",
            _issue_run_audit_payload(
                run,
                issue=issue,
                pr=pr,
                findings_count=findings_count,
                max_fix_attempts=REVIEW_FINDINGS_MAX_FIX_ATTEMPTS,
            ),
            comment=(
                f"Hermes audit: reviewer findings routed run #{run.id} back to "
                f"same-branch coding ({findings_count}/{REVIEW_FINDINGS_MAX_FIX_ATTEMPTS})."
            ),
        )
        raise ReviewFindingsRetry("review_findings queued same-branch coding fix")

    if not can_merge_pr(routing_tag, pr):
        routing_state = routing_tag.state if routing_tag is not None else "missing"
        routing_next_action = routing_tag.next_action if routing_tag is not None else "missing"
        await _post_pr_audit_comment(
            run.repo,
            pr,
            f"Hermes audit: automatic merge blocked for run #{run.id}; routing state "
            f"`{routing_state}` next action `{routing_next_action}`.",
        )
        await _record_kanban_task_audit(
            run,
            "merge_blocked",
            _issue_run_audit_payload(
                run,
                issue=issue,
                pr=pr,
                routing_state=routing_state,
                next_action=routing_next_action,
            ),
            comment=(
                f"Hermes audit: automatic merge blocked for run #{run.id}; routing state "
                f"`{routing_state}` next action `{routing_next_action}`."
            ),
        )
        raise ReviewMergeGateError(
            f"PR #{pr.number} did not receive a current-head ready_for_merge tag"
        )

    await _merge_ready_pr(run.repo, issue, pr, run)
    store.mark_completed(run.id)
    await _record_kanban_task_audit(
        run,
        "issue_run_completed",
        _issue_run_audit_payload(run, issue=issue, pr=pr),
        comment=f"Hermes audit: issue-resolution run #{run.id} completed successfully.",
        complete_task=True,
    )
    await _post_pr_audit_comment(
        run.repo,
        pr,
        f"Hermes audit: review completed for run #{run.id}; routing state `{routing_tag.state}`; "
        "PR merged and linked issue closure requested.",
    )
    await notify(f"Hermes: PR #{pr.number} merged and Issue #{issue.number} closed.")


async def _run_cloud_reviewer_with_tag(
    repo: str,
    workdir: Path,
    pr: PullRequestMetadata,
    notify: Callable[[str], Awaitable[None]],
) -> tuple[str, ReviewRoutingTag]:
    """Run tiered reviewers and require a valid routing tag with one retry per tier."""
    last_output = ""
    existing_suggestion_stats = await _load_pr_review_suggestion_stats(repo, pr)
    for reviewer_tier in REVIEWER_TIERS:
        for attempt in range(1, REVIEW_TAG_MAX_PARSE_ATTEMPTS + 1):
            reviewer_prompt = _cloud_reviewer_prompt(
                repo,
                pr,
                retry_tag_required=attempt > 1,
                reviewer_tier=reviewer_tier,
            )
            reviewer_invocation = build_aider_invocation(
                AiderRole.CLOUD_REVIEWER,
                workdir,
                reviewer_prompt,
                reviewer_tier=reviewer_tier,
            )
            reviewer_output = await _run(
                reviewer_invocation.command,
                cwd=reviewer_invocation.cwd,
                env=reviewer_invocation.env,
            )
            last_output = reviewer_output.stdout
            routing_tag = parse_review_routing_tag(last_output)
            if routing_tag is not None:
                suggestion_stats = _combine_review_suggestion_stats(
                    routing_tag,
                    last_output,
                    existing_suggestion_stats,
                )
                decision = plan_pr_manager_next_action(
                    current_reviewer_tier=reviewer_tier.name,
                    suggestion_stats=suggestion_stats,
                    review_state=routing_tag.state,
                )
                if decision.next_action == "coding_subagent":
                    await notify(
                        f"Hermes: {reviewer_tier.name} found "
                        f"{suggestion_stats.total_suggestions_count} actionable suggestion(s) "
                        f"for PR #{pr.number}; routing to {decision.coder_tier.name}."
                    )
                    return last_output, _review_findings_tag_from_signal(
                        routing_tag,
                        reviewer_tier,
                        suggestion_stats,
                    )
                if routing_tag.state == "review_findings":
                    return last_output, routing_tag
                if routing_tag.state == "ready_for_merge":
                    if decision.next_action == "ready_for_merge":
                        await _post_pr_audit_comment(repo, pr, "Ready for merge")
                        return last_output, routing_tag
                    await notify(
                        f"Hermes: {reviewer_tier.name} clean for PR #{pr.number}; "
                        f"escalating to {decision.reviewer_tier.name}."
                    )
                    break
                if routing_tag.state == "review_inconclusive":
                    if decision.reviewer_tier is not None:
                        await notify(
                            f"Hermes: {reviewer_tier.name} inconclusive for PR #{pr.number}; "
                            f"escalating to {decision.reviewer_tier.name}."
                        )
                        break
                return last_output, routing_tag
            if attempt < REVIEW_TAG_MAX_PARSE_ATTEMPTS:
                await notify(
                    f"Hermes: {reviewer_tier.name} output for PR #{pr.number} had no valid kyber-tag; "
                    "retrying once."
                )

    raise ReviewTagParseError(
        f"reviewer output for PR #{pr.number} lacked a valid kyber-tag after "
        f"{REVIEW_TAG_MAX_PARSE_ATTEMPTS} attempt(s); last output: {last_output[:300]}"
    )


async def _complete_ready_masters(
    store: "IssueStateStore",
    notify: Callable[[str], Awaitable[None]],
) -> None:
    for run in store.expanded_masters_ready_to_complete():
        store.mark_completed(run.id)
        await notify(
            f"Hermes: Master Epic #{run.issue_number} completed; all sub-issues are finished."
        )


class IssueStateStore:
    """SQLite persistence for queued issue-resolution runs."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DEFAULT_STATE_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def enqueue_run(
        self,
        request: IssueResolutionRequest,
        *,
        run_type: IssueRunType,
        parent_run_id: int | None = None,
        master_issue_number: int | None = None,
    ) -> IssueRun:
        existing = self.find_incomplete_run(request.repo, request.issue_number)
        if existing:
            return existing
        now = _now()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO issue_runs (
                    repo, issue_number, workdir, branch, kanban_task_id, kanban_board,
                    status, run_type, parent_run_id, master_issue_number, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request.repo,
                    request.issue_number,
                    str(request.workdir),
                    request.branch,
                    request.kanban_task_id,
                    request.kanban_board,
                    IssueRunStatus.QUEUED.value,
                    run_type.value,
                    parent_run_id,
                    master_issue_number,
                    now,
                    now,
                ),
            )
            conn.commit()
            run_id = int(cur.lastrowid)
        return self.get_run(run_id)

    def find_incomplete_run(self, repo: str, issue_number: int) -> IssueRun | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM issue_runs
                WHERE repo = ? AND issue_number = ?
                  AND status IN (?, ?, ?)
                ORDER BY id ASC LIMIT 1
                """,
                (
                    repo,
                    issue_number,
                    IssueRunStatus.QUEUED.value,
                    IssueRunStatus.RUNNING.value,
                    IssueRunStatus.EXPANDED.value,
                ),
            ).fetchone()
        return self._row_to_run(row) if row else None

    def get_run(self, run_id: int) -> IssueRun:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM issue_runs WHERE id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Issue run {run_id} does not exist.")
        return self._row_to_run(row)

    def claim_next_run(self) -> IssueRun | None:
        now = _now()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM issue_runs
                WHERE status = ? AND next_attempt_at <= ?
                ORDER BY next_attempt_at ASC, id ASC LIMIT 1
                """,
                (IssueRunStatus.QUEUED.value, now),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE issue_runs SET status = ?, updated_at = ? WHERE id = ?",
                (IssueRunStatus.RUNNING.value, now, int(row["id"])),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM issue_runs WHERE id = ?", (int(row["id"]),)
            ).fetchone()
        return self._row_to_run(row)

    def next_queued_delay(self) -> float | None:
        """Return seconds until the next delayed queued run is due."""
        now = _now()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT MIN(next_attempt_at) AS next_attempt_at
                FROM issue_runs
                WHERE status = ?
                """,
                (IssueRunStatus.QUEUED.value,),
            ).fetchone()
        if row is None or row["next_attempt_at"] is None:
            return None
        return max(0.0, float(row["next_attempt_at"]) - now)

    def reset_interrupted_runs(self) -> int:
        now = _now()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE issue_runs SET status = ?, updated_at = ? WHERE status = ?",
                (IssueRunStatus.QUEUED.value, now, IssueRunStatus.RUNNING.value),
            )
            conn.commit()
            return int(cur.rowcount or 0)

    def count_pending_runs(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM issue_runs WHERE status = ?",
                (IssueRunStatus.QUEUED.value,),
            ).fetchone()
        return int(row["count"] if row else 0)

    def mark_expanded(self, run_id: int) -> None:
        self._set_status(run_id, IssueRunStatus.EXPANDED)

    def mark_completed(self, run_id: int) -> None:
        self._set_status(run_id, IssueRunStatus.COMPLETED)

    def mark_failed(self, run_id: int, error: str) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE issue_runs SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                (IssueRunStatus.FAILED.value, error[:4000], now, run_id),
            )
            conn.commit()

    def mark_retry_or_failed(self, run: IssueRun, error: str) -> bool:
        """Retry a failed run when attempts remain, otherwise fail it permanently."""
        attempt_count = run.attempt_count + 1
        now = _now()
        if attempt_count >= ISSUE_RUN_MAX_ATTEMPTS:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE issue_runs
                    SET status = ?, error = ?, attempt_count = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        IssueRunStatus.FAILED.value,
                        error[:4000],
                        attempt_count,
                        now,
                        run.id,
                    ),
                )
                conn.commit()
            return False

        delay = _retry_delay_seconds(attempt_count)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE issue_runs
                SET status = ?, error = ?, attempt_count = ?, next_attempt_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    IssueRunStatus.QUEUED.value,
                    error[:4000],
                    attempt_count,
                    now + delay,
                    now,
                    run.id,
                ),
            )
            conn.commit()
        return True

    def record_pr(self, run_id: int, pr: PullRequestMetadata) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE issue_runs
                SET pr_number = ?, pr_url = ?, branch = COALESCE(branch, ?), updated_at = ?
                WHERE id = ?
                """,
                (pr.number, pr.url, pr.head_ref_name, now, run_id),
            )
            conn.commit()

    def cancel_run(self, run_id: int, reason: str) -> CancellationResult:
        now = _now()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM issue_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"issue run #{run_id} not found")
            status = IssueRunStatus(str(row["status"]))
            if status not in {IssueRunStatus.QUEUED, IssueRunStatus.RUNNING}:
                return CancellationResult(run_id=run_id, status=status, cancelled=False)
            conn.execute(
                """
                UPDATE issue_runs
                SET status = ?, error = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    IssueRunStatus.CANCELLED.value,
                    f"cancelled by operator: {reason}"[:4000],
                    now,
                    run_id,
                ),
            )
            conn.commit()
        return CancellationResult(
            run_id=run_id, status=IssueRunStatus.CANCELLED, cancelled=True
        )

    def record_review_findings(self, run_id: int) -> int:
        """Increment and return the review-findings loop count for this run."""
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE issue_runs
                SET review_findings_count = review_findings_count + 1,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, run_id),
            )
            row = conn.execute(
                "SELECT review_findings_count FROM issue_runs WHERE id = ?", (run_id,)
            ).fetchone()
            conn.commit()
        return int(row["review_findings_count"] if row else 0)

    def record_subissue(
        self,
        master_run_id: int,
        position: int,
        task: EpicTask,
        sub_issue: IssueMetadata,
    ) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO master_subissues (
                    master_run_id, position, sub_issue_number, title, body, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (master_run_id, position, sub_issue.number, task.title, task.body, now),
            )
            conn.commit()

    def subissue_exists(self, master_run_id: int, position: int) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM master_subissues
                WHERE master_run_id = ? AND position = ?
                """,
                (master_run_id, position),
            ).fetchone()
        return row is not None

    def list_child_runs(self, master_run_id: int) -> list[IssueRun]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM issue_runs WHERE parent_run_id = ? ORDER BY id ASC",
                (master_run_id,),
            ).fetchall()
        return [self._row_to_run(row) for row in rows]

    def expanded_masters_ready_to_complete(self) -> list[IssueRun]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM issue_runs WHERE status = ? AND run_type = ? ORDER BY id ASC",
                (IssueRunStatus.EXPANDED.value, IssueRunType.MASTER.value),
            ).fetchall()
        ready: list[IssueRun] = []
        for row in rows:
            run = self._row_to_run(row)
            children = self.list_child_runs(run.id)
            if children and all(
                child.status is IssueRunStatus.COMPLETED for child in children
            ):
                ready.append(run)
        return ready

    def _set_status(self, run_id: int, status: IssueRunStatus) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE issue_runs SET status = ?, updated_at = ? WHERE id = ?",
                (status.value, now, run_id),
            )
            conn.commit()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS issue_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    repo TEXT NOT NULL,
                    issue_number INTEGER NOT NULL,
                    workdir TEXT NOT NULL,
                    branch TEXT,
                    kanban_task_id TEXT,
                    kanban_board TEXT,
                    status TEXT NOT NULL,
                    run_type TEXT NOT NULL DEFAULT 'issue',
                    parent_run_id INTEGER,
                    master_issue_number INTEGER,
                    pr_number INTEGER,
                    pr_url TEXT,
                    error TEXT,
                    review_findings_count INTEGER NOT NULL DEFAULT 0,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at REAL NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_issue_runs_status_id
                    ON issue_runs(status, id);
                CREATE INDEX IF NOT EXISTS idx_issue_runs_repo_issue
                    ON issue_runs(repo, issue_number);
                CREATE INDEX IF NOT EXISTS idx_issue_runs_parent
                    ON issue_runs(parent_run_id);

                CREATE TABLE IF NOT EXISTS master_subissues (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    master_run_id INTEGER NOT NULL,
                    position INTEGER NOT NULL,
                    sub_issue_number INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE(master_run_id, position)
                );
                """
            )
            self._ensure_column(
                conn,
                "issue_runs",
                "review_findings_count",
                "INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(
                conn, "issue_runs", "attempt_count", "INTEGER NOT NULL DEFAULT 0"
            )
            self._ensure_column(
                conn, "issue_runs", "next_attempt_at", "REAL NOT NULL DEFAULT 0"
            )
            self._ensure_column(conn, "issue_runs", "kanban_task_id", "TEXT")
            self._ensure_column(conn, "issue_runs", "kanban_board", "TEXT")
            conn.commit()

    @staticmethod
    def _ensure_column(
        conn: sqlite3.Connection, table: str, column: str, declaration: str
    ) -> None:
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _row_to_run(self, row: sqlite3.Row) -> IssueRun:
        return IssueRun(
            id=int(row["id"]),
            repo=str(row["repo"]),
            issue_number=int(row["issue_number"]),
            workdir=Path(str(row["workdir"])),
            branch=str(row["branch"]) if row["branch"] else None,
            kanban_task_id=str(row["kanban_task_id"]) if row["kanban_task_id"] else None,
            kanban_board=str(row["kanban_board"]) if row["kanban_board"] else None,
            status=IssueRunStatus(str(row["status"])),
            run_type=IssueRunType(str(row["run_type"])),
            parent_run_id=int(row["parent_run_id"])
            if row["parent_run_id"] is not None
            else None,
            master_issue_number=int(row["master_issue_number"])
            if row["master_issue_number"] is not None
            else None,
            pr_number=int(row["pr_number"]) if row["pr_number"] is not None else None,
            pr_url=str(row["pr_url"]) if row["pr_url"] else None,
            error=str(row["error"]) if row["error"] else None,
            review_findings_count=int(row["review_findings_count"] or 0),
            attempt_count=int(row["attempt_count"] or 0),
            next_attempt_at=float(row["next_attempt_at"] or 0.0),
        )


def parse_issue_command_args(raw_args: str) -> IssueResolutionRequest:
    """Parse `/issue` arguments from Telegram or another gateway client."""
    tokens = shlex.split(raw_args or "")
    repo = ""
    issue_number: int | None = None
    workdir: Path | None = None
    branch: str | None = None
    kanban_task_id: str | None = None
    kanban_board: str | None = None

    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token in {"--repo", "-r"} and idx + 1 < len(tokens):
            repo = tokens[idx + 1]
            idx += 2
            continue
        if token in {"--issue", "-i"} and idx + 1 < len(tokens):
            issue_number = _parse_issue_number(tokens[idx + 1])
            idx += 2
            continue
        if token in {"--workdir", "-C"} and idx + 1 < len(tokens):
            workdir = Path(tokens[idx + 1]).expanduser()
            idx += 2
            continue
        if token in {"--branch", "-b"} and idx + 1 < len(tokens):
            branch = tokens[idx + 1]
            idx += 2
            continue
        if token == "--kanban-task" and idx + 1 < len(tokens):
            kanban_task_id = tokens[idx + 1]
            idx += 2
            continue
        if token == "--kanban-board" and idx + 1 < len(tokens):
            kanban_board = tokens[idx + 1]
            idx += 2
            continue
        if not repo:
            maybe_repo, maybe_issue = _parse_issue_url(token)
            if maybe_repo and maybe_issue is not None:
                repo = maybe_repo
                issue_number = maybe_issue
            else:
                repo = token
            idx += 1
            continue
        if issue_number is None:
            issue_number = _parse_issue_number(token)
            idx += 1
            continue
        raise ValueError(f"Unexpected argument: {token}")

    if not repo:
        repo = os.getenv("HERMES_ISSUE_REPO", "").strip()
    if issue_number is None:
        raise ValueError("Missing issue number.")
    if not _valid_repo(repo):
        raise ValueError("Repository must look like owner/name.")

    return IssueResolutionRequest(
        repo=repo,
        issue_number=issue_number,
        workdir=workdir or _default_workdir(repo),
        branch=branch,
        kanban_task_id=kanban_task_id,
        kanban_board=kanban_board,
    )


def parse_issue_cancel_command_args(raw_args: str) -> tuple[int, str]:
    """Parse `/issue-cancel` arguments into a run id and audit reason."""
    tokens = shlex.split(raw_args or "")
    if not tokens:
        raise ValueError("missing run id")
    try:
        run_id = int(tokens[0].lstrip("#"))
    except ValueError as exc:
        raise ValueError(f"invalid run id: {tokens[0]}") from exc
    if run_id <= 0:
        raise ValueError("run id must be positive")
    reason = " ".join(tokens[1:]).strip() or "operator requested cancellation"
    return run_id, reason


def parse_issue_next_command_args(raw_args: str) -> IssueSelectionRequest:
    """Parse `/issue-next` arguments from Telegram or another gateway client."""
    tokens = shlex.split(raw_args or "")
    repo = ""
    workdir: Path | None = None
    branch: str | None = None

    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token in {"--repo", "-r"} and idx + 1 < len(tokens):
            repo = tokens[idx + 1]
            idx += 2
            continue
        if token in {"--workdir", "-C"} and idx + 1 < len(tokens):
            workdir = Path(tokens[idx + 1]).expanduser()
            idx += 2
            continue
        if token in {"--branch", "-b"} and idx + 1 < len(tokens):
            branch = tokens[idx + 1]
            idx += 2
            continue
        if not repo:
            repo = token
            idx += 1
            continue
        raise ValueError(f"Unexpected argument: {token}")

    if not repo:
        repo = os.getenv("HERMES_ISSUE_REPO", "").strip()
    if not _valid_repo(repo):
        raise ValueError("Repository must look like owner/name.")

    return IssueSelectionRequest(
        repo=repo, workdir=workdir or _default_workdir(repo), branch=branch
    )


def github_issue_webhook_command(payload: dict[str, Any]) -> str | None:
    """Build a `/issue` command from a GitHub `issues` webhook payload."""
    issue = payload.get("issue") if isinstance(payload, dict) else None
    repository = payload.get("repository") if isinstance(payload, dict) else None
    if not isinstance(issue, dict) or not isinstance(repository, dict):
        return None
    if isinstance(issue.get("pull_request"), dict):
        return None
    repo = str(repository.get("full_name") or "").strip()
    number = issue.get("number")
    if not repo or not isinstance(number, int):
        return None
    return f"/issue --repo {shlex.quote(repo)} --issue {number}"


def is_master_issue(issue: IssueMetadata) -> bool:
    """Return True when an issue should be handled as a master epic."""
    labels = {label.strip().lower() for label in issue.labels}
    if MASTER_PLAN_LABEL in labels:
        return True
    return issue.body.lstrip().startswith(MASTER_PLAN_HEADING)


async def decompose_master_plan(
    issue: IssueMetadata,
    *,
    runtime_env: dict[str, str] | None = None,
) -> list[EpicTask]:
    """Ask Guardian to decompose a master plan into ordered atomic tasks."""
    env = _merged_runtime_env(runtime_env)
    content = await asyncio.to_thread(_guardian_decompose_request, issue, env)
    return parse_decomposition_response(content)


def parse_decomposition_response(content: str) -> list[EpicTask]:
    """Parse Guardian's JSON-only decomposition response."""
    raw = (content or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"```$", "", raw).strip()
    match = re.search(r"(\{.*\}|\[.*\])", raw, flags=re.DOTALL)
    if match:
        raw = match.group(1)
    data = json.loads(raw)
    if isinstance(data, dict):
        items = data.get("tasks", [])
    else:
        items = data
    tasks: list[EpicTask] = []
    for idx, item in enumerate(items, start=1):
        if isinstance(item, str):
            title = item.strip()
            body = title
        elif isinstance(item, dict):
            title = str(item.get("title") or item.get("name") or f"Task {idx}").strip()
            body = str(item.get("body") or item.get("description") or title).strip()
        else:
            continue
        if title:
            tasks.append(EpicTask(title=title[:180], body=body))
    return tasks


def build_aider_invocation(
    role: AiderRole,
    cwd: Path,
    prompt: str,
    *,
    runtime_env: dict[str, str] | None = None,
    reviewer_tier: ModelTier | None = None,
    coder_tier: ModelTier | None = None,
) -> AiderInvocation:
    """Build the Aider subprocess command for a role."""
    env = _merged_runtime_env(runtime_env)
    aider_bin = Path(env.get("AIDER_BIN") or DEFAULT_AIDER_BIN).expanduser()
    if role is AiderRole.LOCAL_CODER:
        if coder_tier is not None:
            model = coder_tier.model
            if coder_tier.provider is ModelProvider.OPENROUTER:
                openrouter_key = _resolve_secret(
                    env, "OPENROUTER_API_KEY", "OPENROUTER_API_KEY_FILE"
                )
                if not openrouter_key:
                    raise RuntimeError(
                        "OPENROUTER_API_KEY or OPENROUTER_API_KEY_FILE is required."
                    )
                env["OPENROUTER_API_KEY"] = openrouter_key
                env["OPENAI_API_KEY"] = openrouter_key
                env.pop("OPENAI_API_BASE", None)
                env.pop("OPENAI_BASE_URL", None)
            else:
                env["OPENAI_API_BASE"] = _normalize_guardian_base(
                    env.get("GUARDIAN_BASE_URL") or DEFAULT_GUARDIAN_BASE_URL
                )
                env["OPENAI_API_KEY"] = _require_secret(env, "AIDER_GUARDIAN_API_KEY")
                env.pop("OPENAI_BASE_URL", None)
        else:
            env["OPENAI_API_BASE"] = _normalize_guardian_base(
                env.get("GUARDIAN_BASE_URL") or DEFAULT_GUARDIAN_BASE_URL
            )
            env["OPENAI_API_KEY"] = _require_secret(env, "AIDER_GUARDIAN_API_KEY")
            model = (
                env.get("AIDER_LOCAL_MODEL")
                or env.get("AIDER_MODEL")
                or (
                    f"openai/{env['DEFAULT_MODEL']}"
                    if env.get("DEFAULT_MODEL")
                    else DEFAULT_LOCAL_MODEL
                )
            )
        command = [
            str(aider_bin),
            "--model",
            model,
            "--yes",
            "--no-gitignore",
            "--message",
            prompt,
        ]
        return AiderInvocation(command=command, env=env, cwd=cwd)

    if role is AiderRole.CLOUD_REVIEWER:
        tier = reviewer_tier or REVIEWER_TIERS[0]
        model = env.get("AIDER_CLOUD_REVIEW_MODEL") or tier.model
        if reviewer_tier is not None:
            model = reviewer_tier.model
        if tier.provider is ModelProvider.OPENROUTER:
            openrouter_key = _resolve_secret(
                env, "OPENROUTER_API_KEY", "OPENROUTER_API_KEY_FILE"
            )
            if not openrouter_key:
                raise RuntimeError(
                    "OPENROUTER_API_KEY or OPENROUTER_API_KEY_FILE is required."
                )
            env["OPENROUTER_API_KEY"] = openrouter_key
            env["OPENAI_API_KEY"] = openrouter_key
            env.pop("OPENAI_API_BASE", None)
            env.pop("OPENAI_BASE_URL", None)
        else:
            env["OPENAI_API_BASE"] = _normalize_guardian_base(
                env.get("GUARDIAN_BASE_URL") or DEFAULT_GUARDIAN_BASE_URL
            )
            env["OPENAI_API_KEY"] = _require_secret(env, "AIDER_GUARDIAN_API_KEY")
            env.pop("OPENAI_BASE_URL", None)
        command = [
            str(aider_bin),
            "--model",
            model,
            "--cache-prompts",
            "--no-auto-commits",
            "--yes",
            "--no-gitignore",
            "--message",
            prompt,
        ]
        return AiderInvocation(command=command, env=env, cwd=cwd)

    raise ValueError(f"Unsupported Aider role: {role}")


def _local_coder_prompt(repo: str, issue: IssueMetadata, branch: str) -> str:
    return (
        f"Resolve Issue #{issue.number} only on the prepared issue branch and leave "
        "the work ready for a GitHub pull request.\n\n"
        "Hard policy:\n"
        "- KyberM0nk-managed downstream projects must never receive direct implementation "
        "changes on their default/protected branch.\n"
        "- All changes outside the KyberM0nk framework scope must be submitted through "
        "a branch -> PR -> review flow.\n"
        "- Do not switch to `main`, `master`, or any unrelated branch.\n"
        "- Do not leave local-only implementation changes outside the PR branch.\n"
        "- Because this work is issue-driven, the PR must mention and link the source "
        f"Issue #{issue.number}; Hermes will create/reuse that PR after your branch work.\n\n"
        f"Repository: {repo}\n"
        f"Branch: {branch}\n"
        f"Issue URL: {issue.url}\n"
        f"Issue title: {issue.title}\n\n"
        f"Issue body:\n{issue.body or '(empty)'}\n"
    )


def _cloud_reviewer_prompt(
    repo: str,
    pr: PullRequestMetadata,
    *,
    retry_tag_required: bool = False,
    reviewer_tier: ModelTier | None = None,
) -> str:
    retry_note = ""
    if retry_tag_required:
        retry_note = (
            "\nYour previous response lacked a valid kyber-tag. Return the review again "
            "with a valid routing tag.\n"
        )
    tier = reviewer_tier or REVIEWER_TIERS[0]
    next_clean_action = "ready_for_merge" if tier.name in {"tier2", "tier3"} else "tier2_review"
    return (
        "Review this new PR and create an inline comment thread directly inside "
        "the GitHub PR with feedback.\n\n"
        f"Repository: {repo}\n"
        f"PR: #{pr.number}\n"
        f"PR URL: {pr.url}\n"
        f"Current PR head SHA: {pr.head_ref_oid}\n"
        f"Reviewer tier: {tier.name} ({tier.purpose}) using {tier.model}\n"
        f"{retry_note}\n"
        "Return concise feedback suitable for GitHub inline review threads. "
        "Do not edit files or create commits. Count both your own findings and existing "
        "Copilot/code-quality inline review comments as suggestions_count when they are actionable.\n\n"
        "End with a routing tag on separate lines using exactly one of these states:\n"
        "kyber-tag.state=review_findings | ready_for_merge | review_inconclusive\n"
        f"next_action=coding_subagent | rerun_reviewer | tier2_review | tier3_review | ready_for_merge\n"
        f"review_tier={tier.name}\n"
        "suggestions_count=<integer actionable findings count>\n"
        "source=hermes-pr-manager\n"
        f"If this tier is clean, use next_action={next_clean_action}.\n"
        f"head_ref_oid={pr.head_ref_oid}"
    )


def _review_body(output: str) -> str:
    body = (output or "").strip()
    if not body:
        body = "Automated cloud reviewer completed without additional feedback."
    if len(body) > 6000:
        body = body[:6000] + "\n\n[truncated]"
    return f"Hermes Cloud Reviewer:\n\n{body}"


def parse_review_routing_tag(output: str) -> ReviewRoutingTag | None:
    """Extract and validate the kyber-tag routing state from reviewer output."""
    state: str | None = None
    next_action: str | None = None
    head_ref_oid: str | None = None
    tier: str | None = None
    suggestions_count = 0
    source: str | None = None
    for raw_line in (output or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("kyber-tag.state="):
            state = line.split("=", 1)[1].strip()
        elif line.startswith("next_action="):
            next_action = line.split("=", 1)[1].strip()
        elif line.startswith("head_ref_oid="):
            head_ref_oid = line.split("=", 1)[1].strip()
        elif line.startswith("review_tier="):
            tier = line.split("=", 1)[1].strip()
        elif line.startswith("suggestions_count="):
            raw_count = line.split("=", 1)[1].strip()
            suggestions_count = int(raw_count) if raw_count.isdigit() else 0
        elif line.startswith("source="):
            source = line.split("=", 1)[1].strip()
    if state not in REVIEW_TAG_STATES:
        return None
    if next_action not in REVIEW_TAG_NEXT_ACTIONS:
        return None
    if not head_ref_oid:
        return None
    return ReviewRoutingTag(
        state=state,
        next_action=next_action,
        head_ref_oid=head_ref_oid,
        tier=tier,
        suggestions_count=suggestions_count,
        source=source,
    )


def is_review_findings_for_coder(tag: ReviewRoutingTag | None) -> bool:
    """Return true when review output should queue same-branch coder fixes."""
    return bool(
        tag and tag.state == "review_findings" and tag.next_action == "coding_subagent"
    )


def can_merge_pr(tag: ReviewRoutingTag | None, pr: PullRequestMetadata) -> bool:
    """Fail closed unless review output says the current PR head is ready."""
    return bool(
        tag
        and tag.state == "ready_for_merge"
        and tag.next_action == "ready_for_merge"
        and tag.head_ref_oid == pr.head_ref_oid
    )


async def _load_pr_review_suggestion_stats(
    repo: str, pr: PullRequestMetadata
) -> ReviewSuggestionStats:
    """Load actionable PR review/comment metadata for suggestion routing."""
    try:
        reviews = await _load_pr_review_records(repo, pr, "reviews")
        comments = await _load_pr_review_records(
            repo,
            pr,
            "comments",
            current_head_only=False,
        )
        commit_shas = await _load_pr_commit_shas(repo, pr)
        comments = await _filter_unresolved_inline_comments(
            repo,
            pr,
            comments,
            commit_shas,
        )
    except Exception as exc:  # pragma: no cover - defensive telemetry fallback.
        logger.warning(
            "Could not load PR review metadata for %s#%s: %s",
            repo,
            pr.number,
            exc,
        )
        return ReviewSuggestionStats(0, 0, False, False)
    return summarize_review_suggestions(reviews, comments)


async def _load_pr_review_records(
    repo: str,
    pr: PullRequestMetadata,
    record_type: str,
    *,
    current_head_only: bool = True,
) -> list[dict[str, Any]]:
    """Return current-head GitHub PR review records or an empty list on API failure."""
    result = await _run(
        ["gh", "api", f"repos/{repo}/pulls/{pr.number}/{record_type}"],
        check=False,
    )
    if result.returncode != 0:
        logger.warning(
            "GitHub PR %s metadata endpoint %s failed: %s",
            pr.number,
            record_type,
            result.stderr.strip(),
        )
        return []
    try:
        records = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        logger.warning(
            "GitHub PR %s metadata endpoint %s returned invalid JSON",
            pr.number,
            record_type,
        )
        return []
    if not isinstance(records, list):
        return []
    return [
        record
        for record in records
        if isinstance(record, dict)
        and (
            not current_head_only
            or _review_record_matches_head(record, pr.head_ref_oid)
        )
    ]


async def _load_pr_commit_shas(repo: str, pr: PullRequestMetadata) -> list[str]:
    """Return the ordered PR commit SHA list or an empty list on API failure."""
    result = await _run(
        ["gh", "api", f"repos/{repo}/pulls/{pr.number}/commits"],
        check=False,
    )
    if result.returncode != 0:
        logger.warning(
            "GitHub PR %s commit list failed: %s",
            pr.number,
            result.stderr.strip(),
        )
        return []
    try:
        records = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        logger.warning(
            "GitHub PR %s commit list returned invalid JSON",
            pr.number,
        )
        return []
    if not isinstance(records, list):
        return []
    return [
        str(record.get("sha") or "").strip()
        for record in records
        if isinstance(record, dict) and str(record.get("sha") or "").strip()
    ]


async def _filter_unresolved_inline_comments(
    repo: str,
    pr: PullRequestMetadata,
    comments: list[dict[str, Any]],
    commit_shas: list[str],
) -> list[dict[str, Any]]:
    """Keep inline comments that still block this PR head.

    A comment stops blocking only when a later commit in the same PR changes the
    exact file/line that the comment targeted.
    """
    unresolved: list[dict[str, Any]] = []
    changed_lines_cache: dict[str, dict[str, set[int]]] = {}

    for comment in comments:
        if not _review_record_is_inline(comment):
            if _review_record_matches_head(comment, pr.head_ref_oid):
                unresolved.append(comment)
            continue
        if await _inline_comment_cleared_by_later_commit(
            repo,
            pr,
            comment,
            commit_shas,
            changed_lines_cache,
        ):
            continue
        unresolved.append(comment)

    return unresolved


async def _load_commit_changed_lines(
    repo: str,
    commit_sha: str,
    changed_lines_cache: dict[str, dict[str, set[int]]],
) -> dict[str, set[int]]:
    cached = changed_lines_cache.get(commit_sha)
    if cached is not None:
        return cached

    result = await _run(["gh", "api", f"repos/{repo}/commits/{commit_sha}"], check=False)
    if result.returncode != 0:
        logger.warning(
            "GitHub commit %s metadata failed: %s",
            commit_sha,
            result.stderr.strip(),
        )
        changed_lines_cache[commit_sha] = {}
        return changed_lines_cache[commit_sha]

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        logger.warning("GitHub commit %s metadata returned invalid JSON", commit_sha)
        changed_lines_cache[commit_sha] = {}
        return changed_lines_cache[commit_sha]

    changed_by_path: dict[str, set[int]] = {}
    files = payload.get("files") if isinstance(payload, dict) else []
    if isinstance(files, list):
        for item in files:
            if not isinstance(item, dict):
                continue
            path = str(item.get("filename") or "").strip()
            patch = str(item.get("patch") or "")
            if not path:
                continue
            changed_by_path[path] = _parse_changed_lines_from_patch(patch)

    changed_lines_cache[commit_sha] = changed_by_path
    return changed_by_path


def _parse_changed_lines_from_patch(patch: str) -> set[int]:
    changed: set[int] = set()
    current_new_line = 0
    for raw in patch.splitlines():
        line = raw.rstrip("\n")
        if line.startswith("@@"):
            match = re.search(r"\+([0-9]+)(?:,([0-9]+))?", line)
            if not match:
                current_new_line = 0
                continue
            current_new_line = int(match.group(1))
            continue
        if current_new_line <= 0:
            continue
        if line.startswith("+++"):
            continue
        if line.startswith("+"):
            changed.add(current_new_line)
            current_new_line += 1
            continue
        if line.startswith("-"):
            continue
        current_new_line += 1
    return changed


def _review_record_line(record: dict[str, Any]) -> int:
    for key in ("original_line", "line", "originalLine", "start_line"):
        value = record.get(key)
        try:
            line = int(value or 0)
        except (TypeError, ValueError):
            line = 0
        if line > 0:
            return line
    return 0


def _review_record_is_inline(record: dict[str, Any]) -> bool:
    return bool(str(record.get("path") or "").strip()) and _review_record_line(record) > 0


async def _inline_comment_cleared_by_later_commit(
    repo: str,
    pr: PullRequestMetadata,
    comment: dict[str, Any],
    commit_shas: list[str],
    changed_lines_cache: dict[str, dict[str, set[int]]],
) -> bool:
    comment_commit = str(
        comment.get("original_commit_id") or comment.get("commit_id") or ""
    ).strip()
    if not comment_commit or comment_commit == pr.head_ref_oid:
        return False

    try:
        start_index = commit_shas.index(comment_commit)
    except ValueError:
        return False

    path = str(comment.get("path") or "").strip()
    line = _review_record_line(comment)
    if not path or line <= 0:
        return False

    for sha in commit_shas[start_index + 1 :]:
        changed_by_path = await _load_commit_changed_lines(repo, sha, changed_lines_cache)
        if line in changed_by_path.get(path, set()):
            return True
    return False


def _review_record_matches_head(record: dict[str, Any], head_ref_oid: str) -> bool:
    """Return true when a GitHub review/comment record applies to the current head."""
    commit_id = str(record.get("commit_id") or record.get("original_commit_id") or "")
    return not commit_id or commit_id == head_ref_oid


def _combine_review_suggestion_stats(
    tag: ReviewRoutingTag,
    review_output: str,
    existing_stats: ReviewSuggestionStats,
) -> ReviewSuggestionStats:
    """Merge reviewer tag counts with existing Copilot/code-quality metadata."""
    internal_count = max(
        existing_stats.internal_suggestions_count,
        tag.suggestions_count - existing_stats.copilot_suggestions_count,
        1 if tag.state == "review_findings" else 0,
    )
    return ReviewSuggestionStats(
        internal_suggestions_count=internal_count,
        copilot_suggestions_count=existing_stats.copilot_suggestions_count,
        copilot_review_detected=existing_stats.copilot_review_detected,
        complex_findings_detected=(
            existing_stats.complex_findings_detected
            or _looks_like_complex_finding(review_output)
        ),
    )


def _review_findings_tag_from_signal(
    source_tag: ReviewRoutingTag,
    reviewer_tier: ModelTier,
    suggestion_stats: ReviewSuggestionStats,
) -> ReviewRoutingTag:
    """Build a blocking routing tag when metadata contains actionable suggestions."""
    return ReviewRoutingTag(
        state="review_findings",
        next_action="coding_subagent",
        head_ref_oid=source_tag.head_ref_oid,
        tier=reviewer_tier.name,
        suggestions_count=suggestion_stats.total_suggestions_count,
        source=source_tag.source or "hermes-pr-manager",
    )


def is_copilot_review_author(login: str) -> bool:
    """Return true for GitHub Copilot/code-quality review bot identities."""
    normalized = (login or "").strip().lower()
    return normalized in {
        "copilot",
        "copilot-pull-request-reviewer[bot]",
        "github-code-quality[bot]",
    } or normalized.startswith("copilot-")


def summarize_review_suggestions(
    reviews: list[dict[str, Any]],
    comments: list[dict[str, Any]],
) -> ReviewSuggestionStats:
    """Count actionable internal, human, and Copilot suggestions from PR review metadata."""
    internal_count = 0
    copilot_count = 0
    copilot_detected = False
    complex_detected = False

    for record in [*reviews, *comments]:
        if not isinstance(record, dict):
            continue
        body = _review_record_body(record)
        login = _review_record_author_login(record)
        is_copilot = is_copilot_review_author(login) or _looks_like_copilot_body(body)
        is_internal = _looks_like_internal_review_body(body)
        suggestion_count = _explicit_suggestion_count(record)
        if suggestion_count <= 0:
            continue
        if is_copilot:
            copilot_detected = True
            copilot_count += suggestion_count
        elif is_internal or _review_record_is_inline(record):
            internal_count += suggestion_count
        if _looks_like_complex_finding(body):
            complex_detected = True

    return ReviewSuggestionStats(
        internal_suggestions_count=internal_count,
        copilot_suggestions_count=copilot_count,
        copilot_review_detected=copilot_detected,
        complex_findings_detected=complex_detected,
    )


def plan_pr_manager_next_action(
    *,
    current_reviewer_tier: str | None,
    suggestion_stats: ReviewSuggestionStats,
    findings_cycles: int = 0,
    review_state: str = "ready_for_merge",
) -> PRManagerDecision:
    """Choose the next PR-manager reviewer/coder step from review signals."""
    if suggestion_stats.total_suggestions_count > 0:
        coder_tier = _coder_tier_for_findings(
            findings_cycles=findings_cycles,
            complex_findings=suggestion_stats.complex_findings_detected,
        )
        return PRManagerDecision(next_action="coding_subagent", coder_tier=coder_tier)

    if review_state == "review_inconclusive":
        next_tier = _next_reviewer_tier(current_reviewer_tier, include_tier3=True)
        if next_tier is not None:
            return PRManagerDecision(
                next_action=f"{next_tier.name}_review",
                reviewer_tier=next_tier,
            )

    next_tier = _next_reviewer_tier(current_reviewer_tier)
    if next_tier is not None:
        return PRManagerDecision(
            next_action=f"{next_tier.name}_review",
            reviewer_tier=next_tier,
        )

    return PRManagerDecision(next_action="ready_for_merge", ready_comment="Ready for merge")


def _review_record_author_login(record: dict[str, Any]) -> str:
    author = record.get("author") if isinstance(record.get("author"), dict) else record.get("user")
    if isinstance(author, dict):
        return str(author.get("login") or "")
    return ""


def _review_record_body(record: dict[str, Any]) -> str:
    return str(record.get("body") or "")


def _looks_like_internal_review_body(body: str) -> bool:
    normalized = body.lower()
    return "[pr-manager]" in normalized or "kyber-tag" in normalized or "aider-reviewer" in normalized


def _looks_like_copilot_body(body: str) -> bool:
    normalized = body.lower()
    return "copilot" in normalized or "github code quality" in normalized


def _explicit_suggestion_count(record: dict[str, Any]) -> int:
    body = _review_record_body(record)
    suggestion_blocks = len(re.findall(r"```suggestion\b", body, flags=re.IGNORECASE))
    posted_match = re.search(r"Inline suggestions posted:\s*(\d+)", body, flags=re.IGNORECASE)
    posted_count = int(posted_match.group(1)) if posted_match else 0
    review_findings_count = 1 if "review_findings" in body else 0
    inline_comment_count = 1 if record.get("path") and (record.get("line") or record.get("position")) else 0
    return max(suggestion_blocks, posted_count, review_findings_count, inline_comment_count)


def _looks_like_complex_finding(body: str) -> bool:
    normalized = body.lower()
    complex_terms = (
        "credential",
        "secret",
        "database",
        "migration",
        "race",
        "data loss",
        "money",
        "trade",
        "execution",
        "risk",
        "security",
        "hard veto",
    )
    return any(term in normalized for term in complex_terms)


def _coder_tier_for_findings(*, findings_cycles: int, complex_findings: bool) -> ModelTier:
    if complex_findings or findings_cycles >= REVIEW_FINDINGS_MAX_FIX_ATTEMPTS:
        return CODER_TIERS[1]
    return CODER_TIERS[0]


def _next_reviewer_tier(current_tier: str | None, *, include_tier3: bool = False) -> ModelTier | None:
    if current_tier is None:
        return REVIEWER_TIERS[0]
    for index, tier in enumerate(REVIEWER_TIERS):
        if tier.name != current_tier:
            continue
        if index + 1 >= len(REVIEWER_TIERS):
            return None
        next_tier = REVIEWER_TIERS[index + 1]
        if next_tier.name == "tier3" and not include_tier3:
            return None
        return next_tier
    return REVIEWER_TIERS[0]


def issue_auto_merge_enabled() -> bool:
    """Return true only when the issue PR-manager may merge PRs."""
    return os.environ.get(ISSUE_AUTO_MERGE_ENV, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


async def _merge_ready_pr(
    repo: str,
    issue: IssueMetadata,
    pr: PullRequestMetadata,
    run: IssueRun,
) -> None:
    """Merge a reviewer-approved PR and explicitly close the linked issue."""
    if not issue_auto_merge_enabled():
        await _post_pr_audit_comment(
            repo,
            pr,
            f"Hermes audit: automatic merge disabled for run #{run.id}; set "
            f"`{ISSUE_AUTO_MERGE_ENV}=1` to re-enable PR-manager merges.",
        )
        await _record_kanban_task_audit(
            run,
            "merge_disabled",
            _issue_run_audit_payload(run, issue=issue, pr=pr),
            comment=(
                f"Hermes audit: automatic merge disabled for run #{run.id}; "
                "operator merge required."
            ),
        )
        raise ReviewMergeGateError(
            f"automatic merge disabled by {ISSUE_AUTO_MERGE_ENV}=0"
        )

    await _assert_pr_merge_ready(repo, pr)
    await _record_kanban_task_audit(
        run,
        "merge_started",
        _issue_run_audit_payload(run, issue=issue, pr=pr),
    )
    await _post_pr_audit_comment(
        repo,
        pr,
        f"Hermes audit: automatic merge started for run #{run.id} at head `{pr.head_ref_oid}`.",
    )
    await _run([
        "gh",
        "pr",
        "merge",
        str(pr.number),
        "--repo",
        repo,
        "--squash",
        "--delete-branch",
    ])
    await _post_pr_audit_comment(
        repo,
        pr,
        f"Hermes audit: PR #{pr.number} merged automatically for run #{run.id}.",
    )
    await _record_kanban_task_audit(
        run,
        "pr_merged",
        _issue_run_audit_payload(run, issue=issue, pr=pr),
        comment=f"Hermes audit: PR #{pr.number} merged automatically for issue-resolution run #{run.id}.",
    )
    await _post_issue_audit_comment(
        repo,
        issue.number,
        f"Hermes audit: PR #{pr.number} merged automatically for run #{run.id}; closing issue.",
    )
    await _run([
        "gh",
        "issue",
        "close",
        str(issue.number),
        "--repo",
        repo,
        "--comment",
        f"Hermes audit: Issue closed after automatic merge of PR #{pr.number} for run #{run.id}.",
    ])
    await _record_kanban_task_audit(
        run,
        "issue_closed",
        _issue_run_audit_payload(run, issue=issue, pr=pr),
        comment=(
            f"Hermes audit: GitHub issue {repo}#{issue.number} closed after "
            f"automatic merge of PR #{pr.number}."
        ),
    )


async def _assert_pr_merge_ready(repo: str, pr: PullRequestMetadata) -> None:
    """Fail closed unless GitHub reports the live PR as clean and current-head."""
    result = await _run([
        "gh",
        "pr",
        "view",
        str(pr.number),
        "--repo",
        repo,
        "--json",
        "state,isDraft,mergeStateStatus,headRefOid",
    ])
    data = json.loads(result.stdout or "{}")
    live_head = str(data.get("headRefOid") or "")
    if live_head != pr.head_ref_oid:
        raise ReviewMergeGateError(
            f"PR #{pr.number} head changed from {pr.head_ref_oid} to {live_head}; rerun review"
        )
    if str(data.get("state") or "").upper() != "OPEN":
        raise ReviewMergeGateError(f"PR #{pr.number} is not open")
    if bool(data.get("isDraft")):
        raise ReviewMergeGateError(f"PR #{pr.number} is draft")
    merge_state = str(data.get("mergeStateStatus") or "").upper()
    if merge_state != "CLEAN":
        raise ReviewMergeGateError(
            f"PR #{pr.number} mergeStateStatus is {merge_state or 'unknown'}; required checks not clean"
        )


async def _load_issue(repo: str, issue_number: int) -> IssueMetadata:
    result = await _run([
        "gh",
        "issue",
        "view",
        str(issue_number),
        "--repo",
        repo,
        "--json",
        "number,title,body,url,labels",
    ])
    data = json.loads(result.stdout)
    labels = data.get("labels") or []
    label_names = tuple(
        str(item.get("name") or "")
        for item in labels
        if isinstance(item, dict) and item.get("name")
    )
    return IssueMetadata(
        number=int(data["number"]),
        title=str(data.get("title") or f"Issue #{issue_number}"),
        body=str(data.get("body") or ""),
        url=str(data.get("url") or ""),
        labels=label_names,
    )


async def _load_default_branch(repo: str) -> str:
    result = await _run(["gh", "repo", "view", repo, "--json", "defaultBranchRef"])
    data = json.loads(result.stdout)
    ref = data.get("defaultBranchRef") if isinstance(data, dict) else {}
    return str(ref.get("name") or "main") if isinstance(ref, dict) else "main"


async def _find_existing_sub_issue(
    repo: str,
    master_issue: IssueMetadata,
    task: EpicTask,
    position: int,
) -> IssueMetadata | None:
    """Return an existing Master Epic sub-issue for this task when a retry finds one."""
    result = await _run([
        "gh",
        "issue",
        "list",
        "--repo",
        repo,
        "--state",
        "all",
        "--search",
        f'"Part of Master Issue #{master_issue.number}" "## Task {position}"',
        "--json",
        "number,title,body,url",
        "--limit",
        "100",
    ])
    rows = json.loads(result.stdout or "[]")
    if not isinstance(rows, list):
        return None
    expected_markers = (
        f"Part of Master Issue #{master_issue.number}",
        f"## Task {position}",
        task.body.strip(),
    )
    for row in rows:
        if not isinstance(row, dict):
            continue
        body = str(row.get("body") or "")
        title = str(row.get("title") or "")
        if title != task.title:
            continue
        if all(marker in body for marker in expected_markers if marker):
            return IssueMetadata(
                number=int(row["number"]),
                title=title,
                body=body,
                url=str(row.get("url") or ""),
            )
    return None


async def _create_sub_issue(
    repo: str,
    master_issue: IssueMetadata,
    task: EpicTask,
    position: int,
) -> IssueMetadata:
    body = (
        f"Part of Master Issue #{master_issue.number}.\n\n"
        f"Master issue: {master_issue.url}\n\n"
        f"## Task {position}\n\n{task.body}"
    )
    result = await _run([
        "gh",
        "issue",
        "create",
        "--repo",
        repo,
        "--title",
        task.title,
        "--body",
        body,
    ])
    url = result.stdout.strip().splitlines()[-1].strip()
    match = re.search(r"/issues/(\d+)", url)
    if not match:
        raise RuntimeError(f"Could not parse issue number from gh output: {url}")
    return IssueMetadata(
        number=int(match.group(1)), title=task.title, body=body, url=url
    )


async def _create_or_find_pr(
    repo: str,
    issue: IssueMetadata,
    branch: str,
    base_branch: str,
    *,
    run: IssueRun | None = None,
) -> PullRequestMetadata:
    title = f"Fix #{issue.number}: {issue.title}"
    body = _pr_body(repo, issue, branch, base_branch, run=run)
    existing = await _find_pr_for_branch(repo, branch)
    if existing is not None:
        return PullRequestMetadata(
            number=existing.number,
            url=existing.url,
            head_ref_name=existing.head_ref_name,
            head_ref_oid=existing.head_ref_oid,
            action="reused",
        )

    create = await _run(
        [
            "gh",
            "pr",
            "create",
            "--repo",
            repo,
            "--head",
            branch,
            "--base",
            base_branch,
            "--title",
            title,
            "--body",
            body,
        ],
        check=False,
    )
    if create.returncode not in {0, 1}:
        create.raise_for_status()

    pr = await _find_pr_for_branch(repo, branch)
    if pr is None:
        raise RuntimeError("PR creation completed but no PR was found for the branch.")
    return PullRequestMetadata(
        number=pr.number,
        url=pr.url,
        head_ref_name=pr.head_ref_name,
        head_ref_oid=pr.head_ref_oid,
        action="opened",
    )


async def _find_pr_for_branch(repo: str, branch: str) -> PullRequestMetadata | None:
    result = await _run([
        "gh",
        "pr",
        "list",
        "--repo",
        repo,
        "--head",
        branch,
        "--json",
        "number,url,headRefName,headRefOid",
        "--limit",
        "1",
    ])
    rows = json.loads(result.stdout or "[]")
    if not rows:
        return None
    row = rows[0]
    return PullRequestMetadata(
        number=int(row["number"]),
        url=str(row["url"]),
        head_ref_name=str(row["headRefName"]),
        head_ref_oid=str(row["headRefOid"]),
        action="found",
    )


async def _load_next_open_issue(repo: str) -> IssueMetadata:
    result = await _run([
        "gh",
        "issue",
        "list",
        "--repo",
        repo,
        "--state",
        "open",
        "--limit",
        "1000",
        "--json",
        "number,title,body,url,labels,createdAt",
    ])
    rows = json.loads(result.stdout or "[]")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"No open issues found in {repo}.")

    def _created_at(row: dict[str, Any]) -> datetime:
        value = str(row.get("createdAt") or "").replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return datetime.max.replace(tzinfo=timezone.utc)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    candidates = [
        row for row in rows if isinstance(row, dict) and row.get("number") is not None
    ]
    if not candidates:
        raise RuntimeError(f"No open issues found in {repo}.")
    chosen = min(candidates, key=_created_at)
    return await _load_issue(repo, int(chosen["number"]))


async def _post_issue_audit_comment(repo: str, issue_number: int, body: str) -> None:
    """Post a compact audit note to the GitHub issue."""
    await _run([
        "gh",
        "issue",
        "comment",
        str(issue_number),
        "--repo",
        repo,
        "--body",
        body,
    ])


def _issue_run_audit_payload(
    run: IssueRun,
    *,
    issue: IssueMetadata | None = None,
    pr: PullRequestMetadata | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "issue_run_id": run.id,
        "repo": run.repo,
        "issue_number": run.issue_number,
        "branch": run.branch,
        "run_type": run.run_type.value,
        "parent_run_id": run.parent_run_id,
        "master_issue_number": run.master_issue_number,
    }
    if issue is not None:
        payload.update({"issue_title": issue.title, "issue_url": issue.url})
    if pr is not None:
        payload.update(
            {
                "pr_number": pr.number,
                "pr_url": pr.url,
                "head_ref_name": pr.head_ref_name,
                "head_ref_oid": pr.head_ref_oid,
                "pr_action": pr.action,
            }
        )
    payload.update(extra)
    return payload


async def _record_kanban_task_audit(
    run: IssueRun,
    kind: str,
    payload: dict[str, Any],
    *,
    comment: str | None = None,
    complete_task: bool = False,
) -> None:
    """Record a lifecycle event/comment on the linked Kanban task, when any."""
    if not run.kanban_task_id:
        return
    await asyncio.to_thread(
        _record_kanban_task_audit_sync,
        run,
        kind,
        payload,
        comment,
        complete_task,
    )


def _record_kanban_task_audit_sync(
    run: IssueRun,
    kind: str,
    payload: dict[str, Any],
    comment: str | None,
    complete_task_flag: bool,
) -> None:
    from hermes_cli import kanban_db

    with kanban_db.connect(board=run.kanban_board) as conn:
        kanban_db.record_task_event(
            conn,
            run.kanban_task_id or "",
            kind,
            payload,
            run_id=run.id,
        )
        if comment:
            kanban_db.add_audit_comment(conn, run.kanban_task_id or "", comment)
        if complete_task_flag:
            kanban_db.complete_task(
                conn,
                run.kanban_task_id or "",
                result=f"GitHub issue {run.repo}#{run.issue_number} merged and closed by Hermes run #{run.id}.",
                summary=f"Hermes run #{run.id} completed issue {run.repo}#{run.issue_number}.",
                metadata=payload,
            )


async def _post_pr_audit_comment(repo: str, pr: PullRequestMetadata, body: str) -> None:
    """Post a compact audit note to the GitHub pull request."""
    await _run([
        "gh",
        "pr",
        "comment",
        str(pr.number),
        "--repo",
        repo,
        "--body",
        body,
    ])


async def _post_pr_feedback(repo: str, pr: PullRequestMetadata, body: str) -> None:
    diff = await _run(["gh", "pr", "diff", str(pr.number), "--repo", repo, "--patch"])
    anchor = _first_added_line(diff.stdout)
    if anchor is not None:
        path, line = anchor
        comment = await _run(
            [
                "gh",
                "api",
                f"repos/{repo}/pulls/{pr.number}/comments",
                "-f",
                f"body={body}",
                "-f",
                f"commit_id={pr.head_ref_oid}",
                "-f",
                f"path={path}",
                "-F",
                f"line={line}",
                "-f",
                "side=RIGHT",
            ],
            check=False,
        )
        if comment.returncode == 0:
            return

    await _run([
        "gh",
        "pr",
        "review",
        str(pr.number),
        "--repo",
        repo,
        "--comment",
        "--body",
        body,
    ])


def _guardian_decompose_request(issue: IssueMetadata, env: dict[str, str]) -> str:
    base_url = _normalize_guardian_base(
        env.get("GUARDIAN_BASE_URL") or DEFAULT_GUARDIAN_BASE_URL
    )
    api_key = (
        env.get("KYBERM0NK_GUARDIAN_API_KEY")
        or env.get("AIDER_GUARDIAN_API_KEY")
        or env.get("GUARDIAN_API_KEY")
        or env.get("OPENAI_API_KEY")
        or ""
    ).strip()
    if not api_key:
        raise RuntimeError(
            "Guardian API key is required for master-plan decomposition."
        )
    model = (
        env.get("HERMES_ISSUE_DECOMPOSE_MODEL")
        or env.get("DEFAULT_MODEL")
        or DEFAULT_DECOMPOSE_MODEL
    )
    payload = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": 2500,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Decompose master project plans into atomic implementation issues. "
                    'Return only JSON shaped as {"tasks":[{"title":"...","body":"..."}]}. '
                    "Each task must be independently implementable and ordered by dependency."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Issue #{issue.number}: {issue.title}\n\n"
                    f"{issue.body}\n\n"
                    "Create the smallest practical ordered list of technical tasks."
                ),
            },
        ],
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Guardian decomposition request failed: {exc}") from exc
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("Guardian decomposition response had no choices.")
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    if not content:
        raise RuntimeError("Guardian decomposition response was empty.")
    return str(content)


def _first_added_line(diff: str) -> tuple[str, int] | None:
    path: str | None = None
    new_line: int | None = None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line.removeprefix("+++ b/")
            continue
        if line.startswith("@@"):
            match = re.search(r"\+(\d+)(?:,\d+)?", line)
            new_line = int(match.group(1)) if match else None
            continue
        if path is None or new_line is None:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            return path, new_line
        if line.startswith("-") and not line.startswith("---"):
            continue
        new_line += 1
    return None


@dataclass
class CompletedProcess:
    """Small subprocess result wrapper with a status helper."""

    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    def raise_for_status(self) -> None:
        if self.returncode != 0:
            rendered = shlex.join(self.command)
            detail = self.stderr.strip() or self.stdout.strip()
            raise RuntimeError(
                f"Command failed ({self.returncode}): {rendered}\n{detail}"
            )


async def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> CompletedProcess:
    proc = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(cwd) if cwd else None,
        env={str(k): str(v) for k, v in (env or os.environ).items()},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    result = CompletedProcess(
        command=command,
        returncode=proc.returncode,
        stdout=stdout.decode(errors="replace"),
        stderr=stderr.decode(errors="replace"),
    )
    if check:
        result.raise_for_status()
    return result


async def _guard_managed_repo_before_issue_dispatch(
    run: IssueRun,
    issue: IssueMetadata,
    branch: str,
) -> ManagedRepoStatus | None:
    """Fail before local coder dispatch when a managed repo has protected-branch drift."""
    status = await _inspect_managed_repo(run.repo, run.workdir)
    if status is None or status.ok:
        return status
    violating = ", ".join(status.violating_paths) or "unknown paths"
    raise RuntimeError(
        f"Managed repo guard blocked {status.name} issue #{issue.number} before branch {branch!r}: "
        f"{status.reason}; violating paths: {violating}. "
        "Recover or clean the protected checkout before starting the autonomous lane."
    )


async def _prepare_issue_branch_from_synced_default(
    workdir: Path,
    default_branch: str,
    branch: str,
) -> None:
    """Refresh the default branch, then branch issue work from that synced base."""
    if branch == default_branch:
        raise RuntimeError(
            f"Hermes PR discipline violation: issue work cannot target the default branch `{default_branch}`."
        )

    await _run(["git", "checkout", default_branch], cwd=workdir)
    await _run(["git", "pull", "--ff-only", "origin", default_branch], cwd=workdir)
    await _run(["git", "checkout", "-B", branch, default_branch], cwd=workdir)


async def _push_issue_branch(workdir: Path, branch: str) -> None:
    """Push only non-protected issue branches to origin."""
    current_branch = (
        await _run(["git", "branch", "--show-current"], cwd=workdir)
    ).stdout.strip()
    if current_branch in STRICT_PROTECTED_PUSH_BRANCHES or branch in STRICT_PROTECTED_PUSH_BRANCHES:
        raise RuntimeError(DIRECT_MASTER_PUSH_ERROR)

    await _run(["git", "push", "-u", "origin", branch], cwd=workdir)


async def _assert_issue_branch_ready_for_pr(
    repo: str,
    workdir: Path,
    expected_branch: str,
    default_branch: str,
) -> None:
    """Fail when coder output is not fully captured by the issue PR branch."""
    if expected_branch == default_branch:
        raise RuntimeError(
            f"Hermes PR discipline violation for {repo}: issue work cannot target "
            f"the default branch `{default_branch}`. Use a dedicated issue branch."
        )

    branch_result = await _run(["git", "branch", "--show-current"], cwd=workdir)
    current_branch = branch_result.stdout.strip()
    if current_branch != expected_branch:
        current = current_branch or "detached HEAD"
        raise RuntimeError(
            f"Hermes PR discipline violation for {repo}: expected issue branch "
            f"`{expected_branch}` before PR creation, but checkout is on `{current}`. "
            "All changes outside the KyberM0nk framework scope must stay on the "
            "issue branch and be submitted through a PR."
        )

    status_result = await _run(["git", "status", "--porcelain"], cwd=workdir)
    allowed_dirty_prefixes = _managed_repo_allowed_dirty_prefixes(repo)
    dirty_paths = tuple(
        _normalize_status_path(line)
        for line in status_result.stdout.splitlines()
        if line.strip()
    )
    violating = tuple(
        path
        for path in dirty_paths
        if not _is_allowed_dirty_path(path, allowed_dirty_prefixes)
    )
    if violating:
        paths = ", ".join(violating)
        raise RuntimeError(
            f"Hermes PR discipline violation for {repo}: uncommitted implementation "
            f"drift remains on `{expected_branch}` before PR creation: {paths}. "
            "Commit it to the issue branch or stop; do not leave local-only changes "
            "outside the PR."
        )


async def _inspect_managed_repo(repo: str, workdir: Path) -> ManagedRepoStatus | None:
    """Inspect a configured managed repo without mutating it."""
    policy = MANAGED_REPO_POLICIES.get(repo.lower())
    if policy is None:
        return None

    name = str(policy["name"])
    protected_branches = tuple(str(value) for value in policy["protected_branches"])
    allowed_dirty_prefixes = _managed_repo_allowed_dirty_prefixes(repo)

    if not workdir.exists():
        return ManagedRepoStatus(
            repo=repo,
            name=name,
            workdir=workdir,
            branch=None,
            protected=False,
            violating_paths=("<missing repository path>",),
            ignored_paths=(),
            ok=False,
            reason=f"managed repository path does not exist: {workdir}",
        )

    branch_result = await _run(["git", "branch", "--show-current"], cwd=workdir)
    branch = branch_result.stdout.strip() or None
    protected = branch in protected_branches if branch else False
    status_result = await _run(["git", "status", "--porcelain"], cwd=workdir)
    dirty_paths = tuple(
        _normalize_status_path(line)
        for line in status_result.stdout.splitlines()
        if line.strip()
    )
    ignored = tuple(
        path
        for path in dirty_paths
        if _is_allowed_dirty_path(path, allowed_dirty_prefixes)
    )
    violating = tuple(
        path
        for path in dirty_paths
        if not _is_allowed_dirty_path(path, allowed_dirty_prefixes)
    )

    if protected and violating:
        ok = False
        reason = f"protected branch {branch!r} has implementation drift"
    else:
        ok = True
        if protected and ignored:
            reason = f"protected branch {branch!r} has only allowed local tool noise"
        elif protected:
            reason = f"protected branch {branch!r} is clean"
        elif violating:
            reason = f"feature branch {branch!r} has local changes"
        else:
            reason = f"branch {branch!r} is clean"

    return ManagedRepoStatus(
        repo=repo,
        name=name,
        workdir=workdir,
        branch=branch,
        protected=protected,
        violating_paths=violating,
        ignored_paths=ignored,
        ok=ok,
        reason=reason,
    )


def _normalize_status_path(line: str) -> str:
    """Extract the path portion from git status --porcelain output."""
    raw = line[3:] if len(line) > 3 else line
    if " -> " in raw:
        raw = raw.split(" -> ", 1)[1]
    return raw.strip()


def _managed_repo_allowed_dirty_prefixes(repo: str) -> tuple[str, ...]:
    """Return local tool-noise prefixes allowed for a managed repository."""
    policy = MANAGED_REPO_POLICIES.get(repo.lower())
    if policy is None:
        return ()
    return tuple(str(value) for value in policy["allowed_dirty_prefixes"])


def _is_allowed_dirty_path(path: str, prefixes: tuple[str, ...]) -> bool:
    """Return true when a dirty path is allowed local tool noise."""
    return any(
        path == prefix.rstrip("/") or path.startswith(prefix) for prefix in prefixes
    )


def _issue_branch_name(issue: IssueMetadata, repo: str | None = None) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", issue.title.lower()).strip("-")[:48]
    repo_slug = ISSUE_BRANCH_REPO_SLUGS.get((repo or "").casefold(), "issue")
    return f"issue/{repo_slug}-{issue.number}-{slug or 'fix'}"


def _pr_body(
    repo: str,
    issue: IssueMetadata,
    branch: str,
    base_branch: str,
    *,
    run: IssueRun | None = None,
) -> str:
    run_line = (
        f"Hermes issue run: #{run.id}"
        if run is not None
        else "Hermes issue run: pending"
    )
    run_type = run.run_type.value if run is not None else "issue"
    parent_line = ""
    if run is not None and run.parent_run_id is not None:
        parent_line = f"\nParent run: #{run.parent_run_id}"
    master_line = ""
    if run is not None and run.master_issue_number is not None:
        master_line = f"\nMaster issue: #{run.master_issue_number}"

    return (
        f"Closes #{issue.number}\n\n"
        "## Hermes execution contract\n\n"
        f"- Repository: `{repo}`\n"
        f"- Issue: #{issue.number} — {issue.url}\n"
        f"- Branch: `{branch}`\n"
        f"- Base branch: `{base_branch}`\n"
        f"- {run_line}\n"
        f"- Run type: `{run_type}`{parent_line}{master_line}\n\n"
        "## Validation evidence\n\n"
        f"- {PR_BODY_VALIDATION_PLACEHOLDER}\n\n"
        "## Risk notes\n\n"
        "- Hermes must preserve the issue-to-branch-to-PR-to-review-to-merge discipline.\n"
        "- All implementation changes outside the KyberM0nk framework scope must be submitted through this PR.\n"
        "- Direct implementation drift on protected CryptoTrader `master`/`main` is forbidden.\n"
        "- Merge remains blocked until review output is clean or an explicit audited override is recorded.\n\n"
        "## Review handoff\n\n"
        "- State: `ready_for_review`\n"
        "- Reviewer lane: `cloud_reviewer`\n"
        "- Next action: post structured review feedback before merge.\n\n"
        "Generated by Hermes automated issue resolution.\n"
    )


def _parse_issue_url(value: str) -> tuple[str | None, int | None]:
    match = re.search(r"github\.com/([^/]+/[^/]+)/issues/(\d+)", value)
    if not match:
        return None, None
    return match.group(1), int(match.group(2))


def _parse_issue_number(value: str) -> int:
    cleaned = value.strip().removeprefix("#")
    if not cleaned.isdigit():
        raise ValueError("Issue number must be numeric.")
    return int(cleaned)


def _valid_repo(repo: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo or ""))


def _default_workdir(repo: str) -> Path:
    configured = os.getenv("HERMES_ISSUE_WORKDIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    path = Path.home() / repo.split("/", 1)[1]
    if path.exists():
        return path
    return Path.cwd()


def _merged_runtime_env(runtime_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if runtime_env is None else runtime_env)
    env_file = Path(env.get("KYBERM0NK_ENV", str(DEFAULT_KYBER_ENV))).expanduser()
    for key, value in _read_env_file(env_file).items():
        env.setdefault(key, value)
    _resolve_file_backed_secret(env, "OPENROUTER_API_KEY", "OPENROUTER_API_KEY_FILE")
    _resolve_file_backed_secret(env, "GITHUB_TOKEN", "GITHUB_TOKEN_FILE")
    if env.get("GITHUB_TOKEN") and not env.get("GH_TOKEN"):
        env["GH_TOKEN"] = env["GITHUB_TOKEN"]
    return env


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = value.strip().strip('"').strip("'")
    return values


def _resolve_file_backed_secret(
    env: dict[str, str], key_name: str, file_key_name: str
) -> None:
    if env.get(key_name):
        return
    value = _read_secret_file(env.get(file_key_name, ""))
    if value:
        env[key_name] = value


def _resolve_secret(env: dict[str, str], key_name: str, file_key_name: str) -> str:
    value = env.get(key_name, "").strip()
    if value:
        return value
    return _read_secret_file(env.get(file_key_name, ""))


def _read_secret_file(path_value: str) -> str:
    if not path_value:
        return ""
    path = Path(path_value).expanduser()
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _require_secret(env: dict[str, str], key_name: str) -> str:
    value = env.get(key_name, "").strip()
    if not value:
        raise RuntimeError(f"{key_name} is required.")
    return value


def _normalize_guardian_base(value: str) -> str:
    return value.replace("host.docker.internal", "127.0.0.1").rstrip("/")


async def _noop_notify(_message: str) -> None:
    return None


def _now() -> float:
    return time.time()


def _retry_delay_seconds(failed_attempt_count: int) -> int:
    index = max(0, failed_attempt_count - 1)
    if index >= len(ISSUE_RUN_RETRY_DELAYS_SECONDS):
        return ISSUE_RUN_RETRY_DELAYS_SECONDS[-1]
    return ISSUE_RUN_RETRY_DELAYS_SECONDS[index]
