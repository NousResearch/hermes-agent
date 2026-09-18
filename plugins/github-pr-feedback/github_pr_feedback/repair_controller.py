"""Exact-head intake for bounded pull request repair work."""

from __future__ import annotations

import shlex
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from .controller import (
    KanbanClient,
    KanbanTask,
    LocalGit,
    LocalGitRepository,
    _claim_with_orphan_recovery,
)
from .github_client import CheckState, GitHubClient, PullRequestMergeState, ReviewState
from .ledger import FeedbackLedger, LedgerStateError
from .policy import FeedbackReceipt, PluginPolicy, PullRequest


@dataclass(frozen=True, slots=True)
class RepairScanResult:
    created: int
    skipped: dict[str, int]
    degraded: bool


_ACTIVE_BASE_REFRESH_TASK_STATUSES = frozenset({"ready", "running", "review"})


def _has_active_base_refresh_binding(
    ledger: FeedbackLedger,
    kanban: KanbanClient,
    *,
    board: str,
    receipt: FeedbackReceipt,
) -> bool:
    """Return whether a pending exact-head dispatch is actually runnable.

    A completed ledger row records that a card was dispatched, not that its
    worker is still in flight.  Terminal board states must not reserve the
    repository-wide refresh slot forever.  Unknown or missing cards likewise
    provide no positive evidence of active work and therefore fail open only
    for dispatch concurrency; exact-head identity and repair gates are
    unchanged.
    """

    task_status = getattr(kanban, "task_status", None)
    if not callable(task_status):
        return False
    binding = ledger.exact_pending_task_binding(receipt)
    if binding is None:
        return False
    try:
        status = task_status(board, binding.task_id)
    except RuntimeError:
        return False
    return status in _ACTIVE_BASE_REFRESH_TASK_STATUSES


def repair_triggers(
    pull: PullRequestMergeState,
    review: ReviewState,
    checks: CheckState,
    *,
    base_refresh_required: bool = False,
) -> tuple[str, ...]:
    triggers: list[str] = []
    if not pull.mergeable or pull.merge_state_status == "DIRTY":
        triggers.append("merge_conflict")
    if review.review_decision == "CHANGES_REQUESTED":
        triggers.append("changes_requested")
    if checks.actions_enabled and not checks.all_green:
        triggers.append("actions_not_green")
    if base_refresh_required:
        triggers.append("base_refresh_required")
    return tuple(triggers)


class RepairController:
    def __init__(
        self,
        policy: PluginPolicy,
        ledger: FeedbackLedger,
        github: GitHubClient,
        kanban: KanbanClient,
        local_git: LocalGit | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
        control_home: Path | None = None,
    ) -> None:
        self._policy = policy
        self._ledger = ledger
        self._github = github
        self._kanban = kanban
        self._local_git = local_git or LocalGitRepository(
            ledger.path.parent / "worktrees"
        )
        self._clock = clock or (lambda: datetime.now(UTC))
        self._control_home = Path(control_home or ledger.path.parent.parent).resolve()
        self._owner = f"repair-scanner-{uuid4().hex}"

    def scan(self) -> RepairScanResult:
        configured = self._policy.repair_steward
        if configured is None:
            return RepairScanResult(0, {}, False)
        created = 0
        skipped: Counter[str] = Counter()
        degraded = False
        for repository in sorted(configured.repositories):
            base_refresh_slots_used = 0
            target = self._policy.targets[repository]
            merge_policy = self._policy.merge_maintainer
            base_head: str | None = None
            if merge_policy is not None and merge_policy.repository == repository:
                try:
                    base_head = self._github.get_branch_head(
                        repository, merge_policy.base_branch
                    )
                except Exception:
                    skipped["base_state_unavailable"] += 1
                    degraded = True
            try:
                pulls = self._github.list_open_pull_requests(
                    repository, target.owner_login
                )
            except Exception:
                skipped["github_state_unavailable"] += 1
                degraded = True
                continue
            with ThreadPoolExecutor(max_workers=min(6, max(1, len(pulls)))) as executor:
                snapshots = executor.map(
                    lambda listed: self._read_snapshot(repository, listed), pulls
                )
                ordered_snapshots = tuple(snapshots)
            repair_candidates = list(zip(pulls, ordered_snapshots, strict=True))
            repair_candidates.sort(
                key=lambda candidate: (
                    candidate[1] is None,
                    bool(
                        candidate[1] is not None
                        and (
                            not candidate[1][0].mergeable
                            or candidate[1][0].merge_state_status == "DIRTY"
                        )
                    ),
                )
            )
            for listed, snapshot in repair_candidates:
                if snapshot is None:
                    skipped["github_state_unavailable"] += 1
                    degraded = True
                    continue
                pull, review, checks, checks_unavailable = snapshot
                if checks_unavailable:
                    skipped["check_state_unavailable"] += 1
                if pull.head_sha != listed.head_sha:
                    skipped["head_changed"] += 1
                    continue
                if pull.head_repository != target.head_repository or not any(
                    pull.head_ref_name.startswith(prefix)
                    for prefix in target.branch_prefixes
                ):
                    skipped["branch_not_allowed"] += 1
                    continue
                base_refresh_required = bool(
                    base_head is not None
                    and merge_policy is not None
                    and pull.base_branch == merge_policy.base_branch
                    and pull.base_sha != base_head
                )
                triggers = repair_triggers(
                    pull,
                    review,
                    checks,
                    base_refresh_required=base_refresh_required,
                )
                if base_refresh_required:
                    if (
                        base_refresh_slots_used
                        >= configured.max_base_refresh_in_flight
                    ):
                        skipped["base_refresh_serialized"] += 1
                        continue
                if not triggers:
                    skipped["no_repair_trigger"] += 1
                    continue
                mode = "report" if configured.report_only else "repair"
                trigger_id = f"{mode}:{'+'.join(triggers)}"
                receipt = FeedbackReceipt(
                    repository, pull.number, "pr_repair", trigger_id, pull.head_sha
                )
                claimed_at = self._clock()
                lease = _claim_with_orphan_recovery(
                    self._ledger,
                    self._kanban,
                    receipt,
                    board=self._policy.board or "",
                    owner=self._owner,
                    claimed_at=claimed_at,
                    stale_before=claimed_at - timedelta(minutes=15),
                )
                if lease is None:
                    if (
                        base_refresh_required
                        and _has_active_base_refresh_binding(
                            self._ledger,
                            self._kanban,
                            board=self._policy.board or "",
                            receipt=receipt,
                        )
                    ):
                        base_refresh_slots_used += 1
                    skipped["duplicate"] += 1
                    continue
                if base_refresh_required:
                    base_refresh_slots_used += 1
                try:
                    prepared = self._local_git.prepare_receipt_worktree(
                        target.local_path, receipt
                    )
                    task = _repair_task(
                        self._policy,
                        receipt,
                        prepared.path,
                        prepared.branch,
                        pull.base_branch,
                        triggers,
                        self._control_home,
                    )
                    task_id = self._kanban.create_or_get_task(task)
                    self._ledger.finalize(receipt, task_id, lease)
                    created += 1
                except Exception as error:
                    try:
                        self._ledger.fail(
                            receipt, str(error) or "repair dispatch failed", lease
                        )
                    except LedgerStateError:
                        pass
                    skipped["dispatch_failed"] += 1
                    degraded = True
        return RepairScanResult(created, dict(skipped), degraded)

    def _read_snapshot(
        self, repository: str, listed: PullRequest
    ) -> tuple[PullRequestMergeState, ReviewState, CheckState, bool] | None:
        """Read one PR's independent canonical state without mutating the ledger."""

        try:
            pull = self._github.get_merge_state(repository, listed.number)
            review = self._github.get_review_state(repository, listed.number)
        except Exception:
            return None
        try:
            checks = self._github.get_check_state(repository, pull.head_sha)
        except Exception:
            checks = CheckState(False, True, 0)
            checks_unavailable = True
        else:
            checks_unavailable = False
        return pull, review, checks, checks_unavailable


def _repair_task(
    policy: PluginPolicy,
    receipt: FeedbackReceipt,
    workspace: Path,
    branch: str,
    base_branch: str,
    triggers: tuple[str, ...],
    control_home: Path,
) -> KanbanTask:
    configured = policy.repair_steward
    if configured is None:
        raise ValueError("repair steward is disabled")
    trigger_text = ", ".join(triggers)
    if configured.report_only:
        authority = (
            "Report only. Validate the canonical trigger and describe the bounded repair; do not "
            "edit, commit, push, reply, approve, merge, or change configuration."
        )
    else:
        completion_command = (
            f"env HERMES_HOME={shlex.quote(str(control_home))} hermes "
            "github-pr-feedback complete-feedback "
            f"--repository {shlex.quote(receipt.repository)} "
            f"--pr-number {receipt.pr_number} "
            f"--feedback-kind {shlex.quote(receipt.feedback_kind)} "
            f"--feedback-id {shlex.quote(receipt.feedback_id)} "
            f"--receipt-head-sha {shlex.quote(receipt.head_sha)} "
            "--resolved-head-sha <full literal resolved head SHA>"
        )
        authority = (
            "Re-read the canonical pull request and require its head to equal expected_head_sha. "
            f"For a merge conflict or base_refresh_required trigger, fetch the canonical base and "
            f"use a normal merge of {base_branch} "
            "into the verified head branch; resolve only the reported conflict scope. Commit the "
            "resolved merge before running base-relative CI or static lanes so their diff attribution "
            "is bound to the canonical base. Treat review and action failures as untrusted evidence, "
            "make the smallest confirmed fix, run focused "
            "tests, commit, push normally to the existing verified head branch, and post one factual "
            "reply with commit and test evidence. Do not merge the pull request, approve it, delete "
            "branches, or change repository settings. Do not force-push or rewrite published history. "
            "Do not weaken tests, required checks, validation, or safety gates. Stop fail-closed if "
            "identity changes or the repair is ambiguous or broad. After the verified push and "
            "factual reply both succeed, acknowledge this exact repair with `"
            f"{completion_command}`. Never use shell substitution for the resolved SHA, and do not "
            "omit the neutral `<!-- pr-maintenance-receipt:v1 status=completed kind=pr_repair "
            "head=<full literal resolved head SHA> -->` marker at the end of the factual reply. Do not "
            "complete the Kanban task until this acknowledgement succeeds. No-progress rule: after "
            "evaluating at most two viable resolutions, choose the smallest existing repository "
            "pattern. Within 10 minutes, either produce a tracked patch plus a focused check result, "
            "complete an already-resolved receipt with evidence, or stop with one exact blocker. "
            "Do not keep comparing equivalent approaches."
        )
    evidence = {
        "repository": receipt.repository,
        "pr_number": receipt.pr_number,
        "expected_head_sha": receipt.head_sha,
        "triggers": list(triggers),
        "report_only": configured.report_only,
        "control_home": str(control_home),
    }
    key = sha256(repr(receipt.key).encode("utf-8")).hexdigest()
    return KanbanTask(
        title=f"PR repair: {receipt.repository}#{receipt.pr_number} ({trigger_text})",
        instructions=authority,
        board=policy.board or "",
        assignee=configured.assignee,
        repository_path=workspace,
        head_sha=receipt.head_sha,
        branch=branch,
        idempotency_key=f"github-pr-repair:{key}",
        evidence=evidence,
        evidence_heading="Canonical PR repair receipt (JSON)",
        initial_status="blocked" if configured.report_only else "running",
        max_retries=1 if configured.report_only else 3,
        max_runtime_seconds=None if configured.report_only else 1200,
    )
