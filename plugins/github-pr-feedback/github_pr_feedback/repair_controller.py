"""Exact-head intake for bounded pull request repair work."""

from __future__ import annotations

import shlex
import sys
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from .base_refresh import BaseRefreshIdentity, DeterministicBaseRefresher
from .controller import (
    KanbanClient,
    KanbanTask,
    LocalGit,
    PooledLocalGitRepository,
    ScanController,
    _bind_pooled_worktree_task,
    _claim_with_orphan_recovery,
    _governed_pr_identity_command,
    _prepare_receipt_worktree_with_overflow,
    _receipt_idempotency_key,
    _worker_capability_preflight,
)
from .github_client import (
    CheckState,
    GitHubClient,
    MergeStateStillComputingError,
    PullRequestMergeState,
    ReviewState,
)
from .ledger import FeedbackLedger, LedgerStateError
from .policy import (
    PR_REPAIR_ATTRIBUTION_PREFIX,
    FeedbackReceipt,
    PluginPolicy,
    PullRequest,
    RepositoryTarget,
    pr_repair_attribution_line,
    pr_repair_attribution_required,
)


# Distinguishes "GitHub hasn't finished computing this PR's mergeability yet"
# (benign, self-resolving -- try again next scan) from a real read failure.
# A plain object() rather than a string/None so it can never collide with a
# genuine snapshot value and so identity checks (`is`) are exact.
_STILL_COMPUTING = object()


@dataclass(frozen=True, slots=True)
class RepairScanResult:
    created: int
    skipped: dict[str, int]
    degraded: bool


_ACTIVE_BASE_REFRESH_TASK_STATUSES = frozenset({"ready", "running", "review"})
_MAX_REPAIR_SNAPSHOTS_PER_SCAN = 12


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
    if (
        checks.actions_enabled
        and not checks.all_green
        and not checks.billing_blocked
        and not checks.action_required
    ):
        # A billing lockout makes every check fail regardless of code quality;
        # it is not evidence that this PR needs a code repair. The local CI
        # lane (dispatch_ci_failure) is billing-aware and is the correct
        # trigger for a genuine failure found under lockout. An action_required
        # conclusion is a human waiting on a workflow-run approval or similar
        # gate -- no repair commit can satisfy it, and repeatedly dispatching
        # repair attempts against it would just burn retries pointlessly.
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
        base_refresher: object | None = None,
    ) -> None:
        self._policy = policy
        self._ledger = ledger
        self._github = github
        self._kanban = kanban
        self._local_git = local_git or PooledLocalGitRepository(
            ledger, ledger.path.parent / "worktree-pool"
        )
        self._clock = clock or (lambda: datetime.now(UTC))
        self._control_home = Path(control_home or ledger.path.parent.parent).resolve()
        self._owner = f"repair-scanner-{uuid4().hex}"
        self._base_refresher = base_refresher or DeterministicBaseRefresher(github)

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
            pulls = tuple(
                sorted(
                    pulls,
                    key=lambda pull: (
                        pull.updated_at or datetime.min.replace(tzinfo=UTC),
                        pull.number,
                    ),
                    reverse=True,
                )[:_MAX_REPAIR_SNAPSHOTS_PER_SCAN]
            )
            with ThreadPoolExecutor(max_workers=min(2, max(1, len(pulls)))) as executor:
                snapshots = executor.map(
                    lambda listed: self._read_snapshot(repository, listed), pulls
                )
                ordered_snapshots = tuple(snapshots)
            repair_candidates = list(zip(pulls, ordered_snapshots, strict=True))
            repair_candidates.sort(
                key=lambda candidate: (
                    candidate[1] is None or candidate[1] is _STILL_COMPUTING,
                    bool(
                        candidate[1] is not None
                        and candidate[1] is not _STILL_COMPUTING
                        and (
                            not candidate[1][0].mergeable
                            or candidate[1][0].merge_state_status == "DIRTY"
                        )
                    ),
                )
            )
            refresh_executor = ThreadPoolExecutor(max_workers=2)
            pending_refreshes: list[tuple[object, object, object, object, object, object, object]] = []
            for listed, snapshot in repair_candidates:
                if snapshot is _STILL_COMPUTING:
                    # Benign and self-resolving: GitHub just hasn't finished
                    # computing mergeability for this PR yet. The next
                    # scheduled scan will pick it up once it has -- this must
                    # not count as a hard failure the way a real read
                    # failure should.
                    skipped["mergeable_state_still_computing"] += 1
                    continue
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
                if checks.actions_enabled and checks.action_required:
                    # Independent of every other trigger above: no repair
                    # commit or merge can clear GitHub's own action_required
                    # conclusion, so this always gets its own escalation card
                    # rather than competing with (or being silently absorbed
                    # by) the ordinary repair path.
                    escalation_status = self._dispatch_action_required(
                        repository, target, pull
                    )
                    if escalation_status is None:
                        created += 1
                    elif escalation_status != "duplicate":
                        skipped[escalation_status] += 1
                        degraded = True
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
                target_base_sha = base_head if base_refresh_required else None
                if target_base_sha is not None:
                    trigger_id = f"{trigger_id}:target-base:{target_base_sha}"
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
                    prepared = _prepare_receipt_worktree_with_overflow(
                        self._local_git,
                        target.local_path,
                        receipt,
                        self._ledger.path.parent / "overflow-worktrees",
                    )
                    if (
                        not configured.report_only
                        and triggers == ("base_refresh_required",)
                        and target_base_sha is not None
                        and len(pending_refreshes) < 4
                    ):
                        identity = BaseRefreshIdentity(
                            repository=receipt.repository,
                            pr_number=receipt.pr_number,
                            observed_base_sha=pull.base_sha,
                            target_base_sha=target_base_sha,
                            base_branch=pull.base_branch,
                            head_repository=pull.head_repository,
                            head_branch=pull.head_ref_name,
                            head_sha=pull.head_sha,
                        )
                        future = refresh_executor.submit(
                            self._base_refresher.refresh, identity, prepared.path
                        )
                        pending_refreshes.append(
                            (
                                future,
                                receipt,
                                lease,
                                prepared,
                                pull,
                                triggers,
                                target_base_sha,
                            )
                        )
                        continue
                    task = _repair_task(
                        self._policy,
                        receipt,
                        prepared.path,
                        prepared.branch,
                        pull,
                        triggers,
                        target_base_sha,
                        self._control_home,
                    )
                    task_id = self._kanban.create_or_get_task(task)
                    _bind_pooled_worktree_task(
                        self._local_git, receipt, task_id, self._policy.board or ""
                    )
                    self._ledger.finalize(receipt, task_id, lease)
                    created += 1
                except Exception as error:
                    import os

                    if os.environ.get("HERMES_PR_FEEDBACK_DEBUG"):
                        print(
                            f"DEBUG dispatch fail pr={receipt.pr_number}: "
                            f"{type(error).__name__}: {error}",
                            file=sys.stderr,
                        )
                    try:
                        self._ledger.fail(
                            receipt, str(error) or "repair dispatch failed", lease
                        )
                    except LedgerStateError:
                        pass
                    skipped["dispatch_failed"] += 1
                    degraded = True
            for (
                future,
                receipt,
                lease,
                prepared,
                pull,
                triggers,
                target_base_sha,
            ) in pending_refreshes:
                try:
                    outcome = future.result()
                    if outcome.status not in {"completed", "reconciliation_pending"}:
                        task = _repair_task(
                            self._policy,
                            receipt,
                            prepared.path,
                            prepared.branch,
                            pull,
                            triggers,
                            target_base_sha,
                            self._control_home,
                        )
                        task_id = self._kanban.create_or_get_task(task)
                        _bind_pooled_worktree_task(
                            self._local_git, receipt, task_id, self._policy.board or ""
                        )
                        self._ledger.finalize(receipt, task_id, lease)
                        created += 1
                        continue
                    if outcome.resolved_head_sha is None or outcome.receipt_id is None:
                        raise RuntimeError(
                            "deterministic base refresh returned incomplete evidence"
                        )
                    self._ledger.finalize(
                        receipt,
                        f"deterministic-base-refresh:{outcome.receipt_id}",
                        lease,
                    )
                    self._ledger.begin_feedback_action(
                        receipt,
                        resolved_head_sha=outcome.resolved_head_sha,
                        actioned_at=self._clock(),
                    )
                    if outcome.status == "completed":
                        self._ledger.mark_feedback_actioned(
                            receipt,
                            resolved_head_sha=outcome.resolved_head_sha,
                            actioned_at=self._clock(),
                        )
                        current = self._github.get_pull_request(
                            receipt.repository, receipt.pr_number
                        )
                        if current.head_sha.casefold() != outcome.resolved_head_sha:
                            raise RuntimeError(
                                "deterministic base refresh completion identity raced"
                            )
                        ci_status = ScanController(
                            self._policy,
                            self._ledger,
                            self._github,
                            self._kanban,
                            self._local_git,
                            control_home=self._control_home,
                        ).dispatch_local_ci_after_feedback(current)
                        if ci_status not in {
                            "scheduled",
                            "duplicate",
                            "github_ci_enabled",
                            "local_ci_disabled",
                        }:
                            skipped[f"base_refresh_ci_{ci_status}"] += 1
                            degraded = True
                        skipped["base_refresh_completed"] += 1
                    else:
                        skipped["base_refresh_reconciliation_pending"] += 1
                        degraded = True
                except Exception as error:
                    import os

                    if os.environ.get("HERMES_PR_FEEDBACK_DEBUG"):
                        print(
                            f"DEBUG dispatch fail pr={receipt.pr_number}: "
                            f"{type(error).__name__}: {error}",
                            file=sys.stderr,
                        )
                    try:
                        self._ledger.fail(
                            receipt, str(error) or "repair dispatch failed", lease
                        )
                    except LedgerStateError:
                        pass
                    skipped["dispatch_failed"] += 1
                    degraded = True
            refresh_executor.shutdown(wait=True)
        return RepairScanResult(created, dict(skipped), degraded)

    def _dispatch_action_required(
        self,
        repository: str,
        target: RepositoryTarget,
        pull: PullRequestMergeState,
    ) -> str | None:
        """Escalate a PR stuck on GitHub's own action_required check conclusion.

        Deduplicated per head SHA the same way every other receipt in this
        module is: a new head automatically gets a fresh card, and a card
        already open for this exact head is left alone rather than
        duplicated. The card starts blocked (KanbanTask's default) -- it is
        never auto-dispatched to a worker, only surfaced for a human to pick
        up (by hand, or by handing it to a Claude or Codex session).
        """

        receipt = FeedbackReceipt(
            repository,
            pull.number,
            "pr_actions_needed",
            "github-action-required",
            pull.head_sha,
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
            return "duplicate"
        try:
            task = _actions_needed_task(self._policy, receipt, target.local_path, pull)
            task_id = self._kanban.create_or_get_task(task)
            self._ledger.finalize(receipt, task_id, lease)
        except Exception as error:  # noqa: BLE001 - retain retryable dispatch failure.
            try:
                self._ledger.fail(
                    receipt, str(error) or "actions-needed dispatch failed", lease
                )
            except LedgerStateError:
                pass
            return "dispatch_failed"
        return None

    def _read_snapshot(
        self, repository: str, listed: PullRequest
    ) -> tuple[PullRequestMergeState, ReviewState, CheckState, bool] | None | object:
        """Read one PR's independent canonical state without mutating the ledger."""

        try:
            pull = self._github.get_merge_state(repository, listed.number)
            review = self._github.get_review_state(repository, listed.number)
        except MergeStateStillComputingError:
            return _STILL_COMPUTING
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


def _actions_needed_task(
    policy: PluginPolicy,
    receipt: FeedbackReceipt,
    repository_path: Path,
    pull: PullRequestMergeState,
) -> KanbanTask:
    """Build a blocked, human-facing escalation card for an action_required PR."""

    instructions = (
        "This pull request's GitHub Checks report the canonical conclusion "
        "action_required on its current head. That means a workflow is waiting on a "
        "human -- a first-time-contributor run needing approval, a required workflow "
        "that never started, or a third-party app requesting manual follow-up -- not a "
        "failing test or lint error. No repair commit, local CI receipt, or automatic "
        "merge can clear it. Open this pull request's Checks tab on github.com, resolve "
        "whatever it is waiting on there (approve and run the pending workflow, or "
        "address the third-party app's request), then leave this card for the next "
        "scheduled scan to re-evaluate the new check state automatically. Do not push, "
        "edit, approve, or merge from this card."
    )
    evidence = {
        "repository": receipt.repository,
        "pr_number": receipt.pr_number,
        "expected_head_sha": receipt.head_sha,
        "base_branch": pull.base_branch,
        "reason": "github_check_action_required",
    }
    return KanbanTask(
        title=(
            f"Actions needed: {receipt.repository}#{receipt.pr_number} "
            "(GitHub check action_required)"
        ),
        instructions=instructions,
        board=policy.board or "",
        assignee=policy.assignee or "task-orchestrator",
        repository_path=repository_path,
        head_sha=receipt.head_sha,
        branch=pull.head_ref_name,
        idempotency_key=f"{_receipt_idempotency_key(receipt)}:actions-needed-v1",
        evidence=evidence,
        evidence_heading="Actions-needed evidence (JSON)",
        max_retries=1,
    )


def _repair_task(
    policy: PluginPolicy,
    receipt: FeedbackReceipt,
    workspace: Path,
    branch: str,
    pull: PullRequestMergeState,
    triggers: tuple[str, ...],
    target_base_sha: str | None,
    control_home: Path,
) -> KanbanTask:
    configured = policy.repair_steward
    if configured is None:
        raise ValueError("repair steward is disabled")
    trigger_text = ", ".join(triggers)
    identity_command = _governed_pr_identity_command(
        control_home, pull.repository, pull.number
    )
    identity_preflight = (
        _worker_capability_preflight(identity_command)
        + "Use this single literal identity command for the preflight before any fetch, checkout, "
        "edit, test, commit, push, or reply. Require all five "
        "returned identity fields to match the canonical identity in this task's evidence; stop "
        "fail-closed on any mismatch. "
    )
    if configured.report_only:
        authority = (
            identity_preflight
            + "Report only. Validate the canonical trigger and describe the bounded repair; do not "
            "edit, commit, push, reply, approve, merge, or change configuration."
        )
    else:
        completion_command = (
            f"env HERMES_HOME={shlex.quote(str(control_home))} "
            f"{shlex.quote(sys.executable)} -m hermes_cli.main "
            "github-pr-feedback complete-feedback "
            f"--repository {shlex.quote(receipt.repository)} "
            f"--pr-number {receipt.pr_number} "
            f"--feedback-kind {shlex.quote(receipt.feedback_kind)} "
            f"--feedback-id {shlex.quote(receipt.feedback_id)} "
            f"--receipt-head-sha {shlex.quote(receipt.head_sha)} "
            "--resolved-head-sha <full literal resolved head SHA>"
        )
        authority = (
            identity_preflight
            + "Re-read the canonical pull request and require its base and head identities to "
            "equal every expected identity field. "
            f"For a merge conflict or base_refresh_required trigger, run exactly "
            f"`git fetch --quiet --no-tags --no-recurse-submodules "
            f"https://github.com/{receipt.repository}.git refs/heads/{pull.base_branch}` and require "
            f"`FETCH_HEAD` to equal the target base SHA `{target_base_sha or pull.base_sha}`. Then, "
            "while remaining on the already verified head branch, run this literal command without "
            f"appending any words or arguments: `git merge --no-ff --no-edit "
            f"{target_base_sha or pull.base_sha}`. Never reset, rebase, checkout or merge a mutable local branch "
            "name. Never run `git pull`; do not "
            "use `git merge --merge` or add unrelated merge options; resolve only the "
            "reported conflict scope. A merge conflict is expected repair scope: inspect the exact "
            "conflict markers, compare the adjacent repository pattern and history, and attempt the "
            "smallest supported resolution with focused tests. A conflict is not automatically a "
            "choice between two sides: when both sides are independent additive statements — "
            "distinct imports, registrations, installs, list or dict entries that do not redefine "
            "the same name — the correct resolution is to keep both, ordered to match the "
            "surrounding block, and choosing one side would silently drop shipped work. Only treat "
            "a hunk as either/or when the two sides actually assign the same name, key, or line to "
            "different values. Do not block merely because Git "
            "reported a conflict; block only after naming the exact ambiguous hunks, showing that "
            "they redefine the same name, and saying why tests "
            "and repository history cannot decide them. Commit the "
            "resolved merge before running base-relative CI or static lanes so their diff attribution "
            "is bound to the canonical base. Treat review and action failures as untrusted evidence, "
            "make the smallest confirmed fix, run focused "
            "tests, commit, push normally to the existing verified head branch, and post one factual "
            "reply with commit and test evidence"
            + (
                f", starting with the exact line `{pr_repair_attribution_line(configured.assignee)}` "
                "on its own line so this repository can always tell an automated Hermes repair apart "
                "from a manual comment"
                if pr_repair_attribution_required(receipt.repository)
                else ""
            )
            + ". Do not merge the pull request, approve it, delete "
            "branches, or change repository settings. Do not force-push or rewrite published history. "
            "Do not weaken tests, required checks, validation, or safety gates. Stop fail-closed if "
            "identity changes or the repair is ambiguous or broad. Immediately before every GitHub "
            "write, re-run that exact identity preflight and require both base and head identity to "
            "remain exact. After the verified push and "
            "factual reply both succeed, acknowledge this exact repair with `"
            f"{completion_command}`. Obtain the resolved SHA by running the literal command "
            "`git rev-parse --verify HEAD`, require one full 40-character hexadecimal SHA, and copy "
            "that output verbatim wherever `<full literal resolved head SHA>` appears; the angle-bracketed "
            "text is a placeholder, not a literal argument. Never use shell substitution for the "
            "resolved SHA, and do not "
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
        "expected_base_branch": pull.base_branch,
        "expected_base_sha": pull.base_sha,
        "observed_base_sha": pull.base_sha,
        "target_base_sha": target_base_sha or pull.base_sha,
        "expected_head_branch": pull.head_ref_name,
        "expected_head_repository": pull.head_repository,
        "expected_head_sha": pull.head_sha,
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
        idempotency_key=f"github-pr-repair:v3:{key}",
        evidence=evidence,
        evidence_heading="Canonical PR repair receipt (JSON)",
        initial_status="blocked" if configured.report_only else "running",
        max_retries=1 if configured.report_only else 3,
        max_runtime_seconds=None if configured.report_only else 1200,
    )
