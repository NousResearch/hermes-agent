"""Fail-closed merge evaluation and exact-head leased execution."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable, Mapping, Protocol

from .ci_runner import CIAuditReceipt
from .github_client import (
    CheckState,
    Feedback,
    GitHubClient,
    GitHubClientError,
    PullRequestMergeState,
    RepositoryMergePolicy,
    ReviewState,
)
from .ledger import FeedbackLedger, MergeLease
from .policy import FeedbackReceipt, MergeMaintainerPolicy, PluginPolicy, PullRequest
from .intent_review import pending_intent_review


@dataclass(frozen=True, slots=True)
class MergeSnapshot:
    repository_private: bool
    pull_request: PullRequestMergeState
    branch_allowed: bool
    repository_merge_policy: RepositoryMergePolicy
    review_state: ReviewState
    check_state: CheckState
    ci_receipt: CIAuditReceipt | None
    manifest_digest: str
    feedback_clear: bool
    base_head_sha: str
    intent_review_pending: bool = False
    codex_review_pending: bool = False


@dataclass(frozen=True, slots=True)
class MergeDecision:
    eligible: bool
    blockers: tuple[str, ...]
    method: str | None
    snapshot_digest: str


@dataclass(frozen=True, slots=True)
class MergeReceipt:
    repository: str
    pr_number: int
    author_login: str
    base_branch: str
    tested_head_sha: str
    ci_receipt_id: str
    snapshot_digest: str
    method: str
    merge_commit_oid: str
    merged_at: datetime
    executor: str

    def to_payload(self) -> dict[str, object]:
        return {
            "repository": self.repository,
            "pr_number": self.pr_number,
            "author_login": self.author_login,
            "base_branch": self.base_branch,
            "tested_head_sha": self.tested_head_sha,
            "ci_receipt_id": self.ci_receipt_id,
            "snapshot_digest": self.snapshot_digest,
            "method": self.method,
            "merge_commit_oid": self.merge_commit_oid,
            "merged_at": self.merged_at.isoformat(),
            "executor": self.executor,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> MergeReceipt:
        return cls(
            repository=str(payload["repository"]),
            pr_number=int(payload["pr_number"]),
            author_login=str(payload["author_login"]),
            base_branch=str(payload["base_branch"]),
            tested_head_sha=str(payload["tested_head_sha"]),
            ci_receipt_id=str(payload["ci_receipt_id"]),
            snapshot_digest=str(payload["snapshot_digest"]),
            method=str(payload["method"]),
            merge_commit_oid=str(payload["merge_commit_oid"]),
            merged_at=datetime.fromisoformat(str(payload["merged_at"])),
            executor=str(payload["executor"]),
        )


@dataclass(frozen=True, slots=True)
class MergeRunResult:
    decision: MergeDecision
    receipt: MergeReceipt | None


class MergeEvidenceSource(Protocol):
    def snapshot(self, number: int) -> MergeSnapshot: ...


# The Codex GitHub App (chatgpt-codex-connector[bot]) edits one running
# summary comment in place rather than posting a new comment per review; each
# row names the short commit SHA it reviewed and whether that review has
# finished. Matching on this is what actually waits for Codex to weigh in on
# the exact current head, instead of merging the instant CI goes green while
# Codex is still queued or only reviewed an earlier push. If Codex found
# something worth flagging, it leaves an ordinary review/issue comment
# alongside this summary -- admitted as feedback like any other reviewer's,
# so the existing feedback_clear gate below is what actually blocks on a
# real Codex finding; this gate only enforces that Codex has weighed in
# at all before merge is even considered.
_CODEX_REVIEW_LOGIN = "chatgpt-codex-connector[bot]"
_CODEX_REVIEW_MARKER = "codex-pull-request-review-summary"
_CODEX_REVIEW_ROW = re.compile(
    r"\|\s*[^|\n]*\|\s*(?P<status>[^<|]*?)\s*<relative-time[^>]*>.*?</relative-time>\s*"
    r"\|\s*`(?P<sha>[0-9a-fA-F]{7,40})`\s*\|",
    re.DOTALL,
)


def _codex_reviewed_head(feedback: tuple[Feedback, ...], head_sha: str) -> bool:
    short_head = head_sha[:7].casefold()
    for item in feedback:
        if (
            item.reviewer.login.casefold() != _CODEX_REVIEW_LOGIN
            or _CODEX_REVIEW_MARKER not in item.body
        ):
            continue
        for match in _CODEX_REVIEW_ROW.finditer(item.body):
            if (
                match.group("sha").casefold() == short_head
                and "completed" in match.group("status").casefold()
            ):
                return True
    return False


def _allowed_merge_method(
    policy: MergeMaintainerPolicy, repository_policy: RepositoryMergePolicy
) -> str | None:
    return next(
        (
            candidate
            for candidate in policy.merge_methods
            if repository_policy.allows(candidate)
        ),
        None,
    )


def evaluate_merge(
    policy: MergeMaintainerPolicy, snapshot: MergeSnapshot, *, now: datetime
) -> MergeDecision:
    """Evaluate a typed snapshot without side effects or model judgment."""

    now = _aware_utc(now)
    pull = snapshot.pull_request
    receipt = snapshot.ci_receipt
    blockers: list[str] = []
    if not snapshot.repository_private:
        blockers.append("repository_not_private")
    if pull.repository != policy.repository:
        blockers.append("repository_mismatch")
    if pull.author_login.casefold() != policy.author_login.casefold():
        blockers.append("author_not_allowed")
    if pull.head_repository != policy.repository:
        blockers.append("head_repository_not_allowed")
    if pull.base_branch != policy.base_branch:
        blockers.append("base_branch_mismatch")
    if pull.base_sha != snapshot.base_head_sha:
        blockers.append("base_head_changed")
    if not snapshot.branch_allowed:
        blockers.append("head_branch_not_allowed")
    if pull.state != "OPEN" or pull.merged:
        blockers.append("pull_request_not_open")
    if pull.is_draft:
        blockers.append("pull_request_draft")
    if not pull.mergeable:
        blockers.append("pull_request_conflicted")
    if pull.merge_state_status != "CLEAN":
        blockers.append("merge_state_not_clean")
    if receipt is None:
        blockers.append("ci_receipt_missing")
    else:
        if receipt.status != "passed":
            blockers.append("ci_receipt_not_passing")
        if (
            receipt.identity.repository != policy.repository
            or receipt.identity.pr_number != pull.number
            or receipt.identity.base_sha != pull.base_sha
            or receipt.identity.head_sha != pull.head_sha
        ):
            blockers.append("ci_identity_mismatch")
        if receipt.manifest_digest != snapshot.manifest_digest:
            blockers.append("ci_manifest_mismatch")
        if receipt.completed_at < now - timedelta(seconds=policy.receipt_max_age_seconds):
            blockers.append("ci_receipt_stale")
    if snapshot.check_state.actions_enabled and snapshot.check_state.action_required:
        # A distinct, more precise code than github_checks_not_green: no CI
        # receipt or repair commit can clear this, only a human approving the
        # gated workflow run (or otherwise resolving it) on GitHub itself.
        blockers.append("action_required")
    elif (
        snapshot.check_state.actions_enabled
        and not snapshot.check_state.all_green
        and not snapshot.check_state.billing_blocked
    ):
        blockers.append("github_checks_not_green")
    if snapshot.review_state.review_decision == "CHANGES_REQUESTED":
        blockers.append("changes_requested")
    if snapshot.review_state.unresolved_thread_count:
        blockers.append("unresolved_review_threads")
    labels = {label.casefold() for label in pull.labels}
    governed_review_required = any(
        label.startswith("sweeper:risk-")
        or label in {"sweeper:blast-broad", "sweeper:blast-massive", "telemetry"}
        for label in labels
    )
    if governed_review_required and "ci-reviewed" not in labels:
        blockers.append("governed_review_missing")
    if not snapshot.feedback_clear:
        blockers.append("feedback_unprocessed")
    if snapshot.intent_review_pending:
        blockers.append("intent_review_required")
    if snapshot.codex_review_pending:
        blockers.append("codex_review_pending")
    method = _allowed_merge_method(policy, snapshot.repository_merge_policy)
    if method is None:
        blockers.append("merge_method_unavailable")
    digest = _snapshot_digest(snapshot, receipt)
    unique_blockers = tuple(dict.fromkeys(blockers))
    return MergeDecision(
        eligible=not unique_blockers,
        blockers=unique_blockers,
        method=method if not unique_blockers else None,
        snapshot_digest=digest,
    )


class MergeController:
    def __init__(
        self,
        policy: MergeMaintainerPolicy,
        source: MergeEvidenceSource,
        github: GitHubClient,
        ledger: FeedbackLedger,
        *,
        owner: str,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._policy = policy
        self._source = source
        self._github = github
        self._ledger = ledger
        self._owner = owner
        self._now = now or (lambda: datetime.now(UTC))

    def run(self, number: int) -> MergeRunResult:
        completed = self._ledger.completed_merge_receipt(self._policy.repository, number)
        if isinstance(completed, MergeReceipt):
            decision = MergeDecision(True, (), completed.method, completed.snapshot_digest)
            return MergeRunResult(decision, completed)
        pending = self._ledger.verification_required_merge_lease(
            self._policy.repository, number
        )
        if pending is not None:
            snapshot = self._source.snapshot(number)
            reconciled = self._reconcile_verified_merge(pending, snapshot)
            if reconciled is not None:
                return reconciled
            blocked = MergeDecision(
                False,
                ("merge_verification_required",),
                None,
                _snapshot_digest(snapshot, snapshot.ci_receipt),
            )
            return MergeRunResult(blocked, None)
        first_snapshot = self._source.snapshot(number)
        first = evaluate_merge(self._policy, first_snapshot, now=self._now())
        if not first.eligible or self._policy.report_only:
            return MergeRunResult(first, None)
        lease = self._ledger.claim_merge_lease(
            self._policy.repository,
            number,
            first_snapshot.pull_request.head_sha,
            owner=self._owner,
            claimed_at=self._now(),
        )
        if lease is None:
            blocked = MergeDecision(
                False, ("merge_lease_unavailable",), None, first.snapshot_digest
            )
            return MergeRunResult(blocked, None)
        second_snapshot = self._source.snapshot(number)
        second = evaluate_merge(self._policy, second_snapshot, now=self._now())
        if not second.eligible or second.snapshot_digest != first.snapshot_digest:
            blockers = second.blockers or ("merge_snapshot_changed",)
            raced = MergeDecision(False, blockers, None, second.snapshot_digest)
            self._ledger.finish_merge_lease(
                lease, status="failed", updated_at=self._now(), error=",".join(blockers)
            )
            return MergeRunResult(raced, None)
        assert second.method is not None
        write_error: Exception | None = None
        try:
            self._github.merge_pull_request(
                self._policy.repository,
                number,
                second_snapshot.pull_request.head_sha,
                method=second.method,
            )
        except GitHubClientError as error:
            write_error = error
        try:
            readback = self._github.get_merge_state(self._policy.repository, number)
        except GitHubClientError as error:
            self._ledger.finish_merge_lease(
                lease,
                status="verification_required",
                updated_at=self._now(),
                error=type(error).__name__,
            )
            blocked = MergeDecision(
                False, ("merge_verification_required",), None, second.snapshot_digest
            )
            return MergeRunResult(blocked, None)
        if (
            not readback.merged
            or readback.repository != self._policy.repository
            or readback.number != number
            or readback.head_sha != second_snapshot.pull_request.head_sha
            or readback.merge_commit_oid is None
        ):
            self._ledger.finish_merge_lease(
                lease,
                status="verification_required" if write_error else "failed",
                updated_at=self._now(),
                error="canonical readback did not confirm the merge",
            )
            blocked = MergeDecision(
                False, ("merge_verification_required",), None, second.snapshot_digest
            )
            return MergeRunResult(blocked, None)
        receipt = MergeReceipt(
            repository=self._policy.repository,
            pr_number=number,
            author_login=second_snapshot.pull_request.author_login,
            base_branch=second_snapshot.pull_request.base_branch,
            tested_head_sha=second_snapshot.pull_request.head_sha,
            ci_receipt_id=second_snapshot.ci_receipt.receipt_id,
            snapshot_digest=second.snapshot_digest,
            method=second.method,
            merge_commit_oid=readback.merge_commit_oid,
            merged_at=_aware_utc(self._now()),
            executor=self._owner,
        )
        self._ledger.finish_merge_lease(
            lease, status="completed", updated_at=self._now(), receipt=receipt
        )
        return MergeRunResult(second, receipt)

    def _reconcile_verified_merge(
        self, lease: MergeLease, snapshot: MergeSnapshot
    ) -> MergeRunResult | None:
        """Complete an ambiguous attempt only from exact canonical merged truth."""

        pull = snapshot.pull_request
        ci_receipt = snapshot.ci_receipt
        method = _allowed_merge_method(self._policy, snapshot.repository_merge_policy)
        if (
            not snapshot.repository_private
            or not snapshot.branch_allowed
            or pull.repository != lease.repository
            or pull.number != lease.pr_number
            or pull.head_sha != lease.head_sha
            or pull.state != "CLOSED"
            or not pull.merged
            or pull.merge_commit_oid is None
            or pull.author_login.casefold() != self._policy.author_login.casefold()
            or pull.head_repository != self._policy.repository
            or pull.base_branch != self._policy.base_branch
            or method is None
            or not isinstance(ci_receipt, CIAuditReceipt)
            or ci_receipt.status != "passed"
            or ci_receipt.identity.repository != lease.repository
            or ci_receipt.identity.pr_number != lease.pr_number
            or ci_receipt.identity.head_sha != lease.head_sha
            or ci_receipt.identity.base_sha != pull.base_sha
            or ci_receipt.manifest_digest != snapshot.manifest_digest
        ):
            return None
        digest = _snapshot_digest(snapshot, ci_receipt)
        receipt = MergeReceipt(
            repository=lease.repository,
            pr_number=lease.pr_number,
            author_login=pull.author_login,
            base_branch=pull.base_branch,
            tested_head_sha=lease.head_sha,
            ci_receipt_id=ci_receipt.receipt_id,
            snapshot_digest=digest,
            method=method,
            merge_commit_oid=pull.merge_commit_oid,
            merged_at=_aware_utc(self._now()),
            executor=self._owner,
        )
        self._ledger.complete_verified_merge(
            lease, updated_at=self._now(), receipt=receipt
        )
        return MergeRunResult(MergeDecision(True, (), method, digest), receipt)


class CanonicalMergeEvidenceSource:
    """Build merge evidence only from canonical GitHub reads and typed ledger state."""

    def __init__(
        self, plugin_policy: PluginPolicy, github: GitHubClient, ledger: FeedbackLedger
    ) -> None:
        if plugin_policy.merge_maintainer is None:
            raise ValueError("merge maintainer is disabled")
        self._plugin_policy = plugin_policy
        self._merge_policy = plugin_policy.merge_maintainer
        self._github = github
        self._ledger = ledger

    def snapshot(self, number: int) -> MergeSnapshot:
        policy = self._merge_policy
        pull = self._github.get_merge_state(policy.repository, number)
        target = self._plugin_policy.targets[policy.repository]
        manifest_path = target.local_path / "tests/manifests/test_lanes.toml"
        if not manifest_path.is_file():
            raise GitHubClientError("CI manifest was unavailable")
        manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        receipt = self._ledger.latest_ci_receipt(
            policy.repository,
            number,
            pull.head_sha,
            manifest_digest=manifest_digest,
            not_before=datetime.min.replace(tzinfo=UTC),
        )
        if receipt is not None and not isinstance(receipt, CIAuditReceipt):
            raise GitHubClientError("CI receipt had an invalid type")
        feedback_clear = self._feedback_clear(pull)
        feedback = self._github.list_feedback(policy.repository, number)
        intent_pending = pending_intent_review(feedback, owner_login=target.owner_login)
        codex_pending = not _codex_reviewed_head(feedback, pull.head_sha)
        return MergeSnapshot(
            repository_private=self._github.repository_is_private(policy.repository),
            pull_request=pull,
            branch_allowed=any(
                pull.head_ref_name.startswith(prefix) for prefix in target.branch_prefixes
            ),
            repository_merge_policy=self._github.get_repository_merge_policy(
                policy.repository
            ),
            review_state=self._github.get_review_state(policy.repository, number),
            check_state=self._github.get_check_state(policy.repository, pull.head_sha),
            ci_receipt=receipt,
            manifest_digest=manifest_digest,
            feedback_clear=feedback_clear,
            base_head_sha=self._github.get_branch_head(
                policy.repository, policy.base_branch
            ),
            intent_review_pending=intent_pending,
            codex_review_pending=codex_pending,
        )

    def _feedback_clear(self, pull: PullRequestMergeState) -> bool:
        from .controller import (
            _ci_receipt_feedback_reason,
            _is_codex_review_summary_tracker,
            _is_non_actionable_review_container,
            _is_self_resolution_receipt,
        )

        policy = self._plugin_policy
        target = policy.targets[pull.repository]
        canonical_pull = PullRequest(
            number=pull.number,
            state=pull.state,
            base_repository=pull.repository,
            head_repository=pull.head_repository,
            author_login=pull.author_login,
            head_ref_name=pull.head_ref_name,
            head_sha=pull.head_sha,
        )
        for feedback in self._github.list_feedback(pull.repository, pull.number):
            if policy.not_before is not None and feedback.created_at < policy.not_before:
                continue
            if (
                _is_non_actionable_review_container(feedback)
                or _is_codex_review_summary_tracker(feedback)
                or _is_self_resolution_receipt(feedback, owner_login=target.owner_login)
            ):
                continue
            receipt = FeedbackReceipt(
                pull.repository,
                pull.number,
                feedback.kind,
                feedback.feedback_id,
                pull.head_sha,
            )
            if _ci_receipt_feedback_reason(self._ledger, receipt, feedback.body) is not None:
                continue
            admission = policy.admit(
                canonical_pull, feedback.reviewer, receipt, is_bot=feedback.is_bot
            )
            if admission.admitted and not self._ledger.was_actioned_on_any_head(receipt):
                return False
        return True


def _snapshot_digest(snapshot: MergeSnapshot, receipt: CIAuditReceipt | None) -> str:
    pull = snapshot.pull_request
    payload = {
        "private": snapshot.repository_private,
        "pull": {
            "repository": pull.repository,
            "number": pull.number,
            "state": pull.state,
            "draft": pull.is_draft,
            "mergeable": pull.mergeable,
            "merge_state": pull.merge_state_status,
            "base": pull.base_branch,
            "base_sha": pull.base_sha,
            "head_repository": pull.head_repository,
            "author": pull.author_login,
            "head_ref": pull.head_ref_name,
            "head_sha": pull.head_sha,
            "merged": pull.merged,
        },
        "branch_allowed": snapshot.branch_allowed,
        "base_head_sha": snapshot.base_head_sha,
        "methods": {
            "squash": snapshot.repository_merge_policy.squash,
            "rebase": snapshot.repository_merge_policy.rebase,
            "merge": snapshot.repository_merge_policy.merge,
        },
        "review": {
            "decision": snapshot.review_state.review_decision,
            "unresolved": snapshot.review_state.unresolved_thread_count,
        },
        "checks": {
            "enabled": snapshot.check_state.actions_enabled,
            "green": snapshot.check_state.all_green,
            "count": snapshot.check_state.check_count,
        },
        "ci_receipt": receipt.receipt_id if receipt else None,
        "manifest": snapshot.manifest_digest,
        "feedback_clear": snapshot.feedback_clear,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _aware_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return value.astimezone(UTC)
