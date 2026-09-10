"""Fail-closed merge evaluation and exact-head leased execution."""

from __future__ import annotations

from .ci_contract import manifest_path as ci_manifest_path

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable, Mapping, Protocol

from .ci_runner import (
    ActionsDisabledLocalCIEvidence,
    CIAuditIdentity,
    CIAuditReceipt,
    CI_MODE_BUDGET_EXHAUSTED_LOCAL_EQUIVALENT,
    actions_disabled_local_ci_evidence,
)
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
    ci_receipt: CIAuditReceipt | CIReceiptComment | None
    manifest_digest: str
    feedback_clear: bool
    base_head_sha: str
    intent_review_pending: bool = False
    codex_review_pending: bool = False
    actions_disabled_local_ci: ActionsDisabledLocalCIEvidence | None = None


@dataclass(frozen=True, slots=True)
class MergeDecision:
    eligible: bool
    blockers: tuple[str, ...]
    method: str | None
    snapshot_digest: str
    evidence: Mapping[str, object] = field(default_factory=dict)


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


@dataclass(frozen=True, slots=True)
class CIReceiptComment:
    """A governed PR comment carrying a worker's exact-head CI identity."""

    receipt_id: str
    identity: CIAuditIdentity
    manifest_digest: str
    status: str
    completed_at: datetime
    ci_mode: str


_CI_RECEIPT_COMMENT = re.compile(
    r"<!--\s*pr-ci-receipt:v2\b"
    r"\s+status=(?P<status>passed|failed)"
    r"\s+id=(?P<id>[0-9a-f]{64})"
    r"\s+head=(?P<head>[0-9a-f]{40})"
    r"\s+base=(?P<base>[0-9a-f]{40})"
    r"\s+manifest=(?P<manifest>[0-9a-f]{64})"
    r"\s+completed=(?P<completed>[^\s>]+)"
    r"\s+ci_mode=(?P<ci_mode>[a-z0-9-]+)\s*-->",
    flags=re.IGNORECASE,
)


def ci_receipt_comment_from_feedback(
    feedback_items: tuple[Feedback, ...],
    pull: PullRequestMergeState,
    *,
    expected_login: str | None,
    manifest_digest: str,
) -> CIReceiptComment | None:
    """Return the newest exact, bot-authored worker receipt comment.

    v1 comments remain display/retirement markers because they do not carry
    enough identity to prove the current manifest. Only v2 comments bridge a
    worker's profile-local receipt to the merge maintainer.
    """

    if not expected_login:
        return None
    candidates: list[CIReceiptComment] = []
    for feedback in feedback_items:
        if (
            feedback.kind not in {"issue_comment", "review_comment"}
            or not feedback.is_bot
            or feedback.reviewer.login.casefold() != expected_login.casefold()
        ):
            continue
        match = _CI_RECEIPT_COMMENT.search(feedback.body)
        if match is None:
            continue
        try:
            completed_at = datetime.fromisoformat(match.group("completed"))
            if completed_at.tzinfo is None or completed_at.utcoffset() is None:
                continue
            identity = CIAuditIdentity(
                pull.repository,
                pull.number,
                match.group("base"),
                match.group("head"),
            )
            candidate = CIReceiptComment(
                receipt_id=match.group("id").casefold(),
                identity=identity,
                manifest_digest=match.group("manifest").casefold(),
                status=match.group("status").casefold(),
                completed_at=completed_at.astimezone(UTC),
                ci_mode=match.group("ci_mode"),
            )
        except (TypeError, ValueError):
            continue
        if (
            candidate.identity.base_sha != pull.base_sha.casefold()
            or candidate.identity.head_sha != pull.head_sha.casefold()
            or candidate.manifest_digest != manifest_digest.casefold()
        ):
            continue
        candidates.append(candidate)
    return max(candidates, key=lambda item: item.completed_at) if candidates else None


def _is_ci_receipt_comment_for_head(
    feedback: Feedback, *, expected_login: str, head_sha: str
) -> bool:
    """Keep the CI transport comment out of ordinary feedback admission."""

    match = _CI_RECEIPT_COMMENT.search(feedback.body)
    return bool(
        match is not None
        and feedback.kind in {"issue_comment", "review_comment"}
        and feedback.is_bot
        and feedback.reviewer.login.casefold() == expected_login.casefold()
        and match.group("head").casefold() == head_sha.casefold()
    )


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


def _is_governed_approval_receipt(
    feedback: Feedback, *, expected_login: str, head_sha: str
) -> bool:
    """Recognize only the configured automation identity's exact-head approval review."""

    return bool(
        feedback.kind == "review"
        and feedback.reviewer.login.casefold() == expected_login.casefold()
        and feedback.review_state == "APPROVED"
        and feedback.reviewed_head_sha == head_sha.casefold()
    )


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
    elif snapshot.check_state.actions_enabled and snapshot.check_state.billing_blocked:
        if (
            not policy.allow_budget_exhausted_local_ci
            or receipt is None
            or not isinstance(receipt, CIAuditReceipt)
            or receipt.status != "passed"
            or receipt.ci_mode != CI_MODE_BUDGET_EXHAUSTED_LOCAL_EQUIVALENT
            or not receipt.actions_state.billing_blocked
        ):
            blockers.append("github_actions_budget_exhausted")
    elif (
        snapshot.check_state.actions_enabled
        and not snapshot.check_state.all_green
        and not snapshot.check_state.billing_blocked
    ):
        blockers.append("github_checks_not_green")
    elif not snapshot.check_state.actions_enabled:
        local_evidence = snapshot.actions_disabled_local_ci
        if (
            not policy.allow_budget_exhausted_local_ci
            or receipt is None
            or local_evidence is None
            or local_evidence.receipt_id != receipt.receipt_id
            or local_evidence.manifest_digest != snapshot.manifest_digest
            or local_evidence.command_count != len(receipt.commands)
            or local_evidence.command_count < local_evidence.required_command_count
        ):
            blockers.append("github_actions_disabled")
    if snapshot.review_state.review_decision == "CHANGES_REQUESTED":
        blockers.append("changes_requested")
    elif snapshot.review_state.review_decision == "REVIEW_REQUIRED":
        blockers.append("review_required")
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
    evidence: dict[str, object] = {
        "repository": pull.repository,
        "pr_number": pull.number,
        "base_branch": pull.base_branch,
        "base_sha": pull.base_sha,
        "base_head_sha": snapshot.base_head_sha,
        "head_sha": pull.head_sha,
        "mergeable": pull.mergeable,
        "merge_state_status": pull.merge_state_status,
        "branch_allowed": snapshot.branch_allowed,
        "review_decision": snapshot.review_state.review_decision,
        "unresolved_review_threads": snapshot.review_state.unresolved_thread_count,
        "feedback_clear": snapshot.feedback_clear,
        "intent_review_pending": snapshot.intent_review_pending,
        "codex_review_pending": snapshot.codex_review_pending,
        "actions_enabled": snapshot.check_state.actions_enabled,
        "checks_all_green": snapshot.check_state.all_green,
        "checks_action_required": snapshot.check_state.action_required,
        "checks_billing_blocked": snapshot.check_state.billing_blocked,
        "ci_receipt_id": receipt.receipt_id if receipt is not None else None,
        "ci_receipt_status": receipt.status if receipt is not None else None,
        "ci_manifest_digest": receipt.manifest_digest if receipt is not None else None,
        "manifest_digest": snapshot.manifest_digest,
    }
    return MergeDecision(
        eligible=not unique_blockers,
        blockers=unique_blockers,
        method=method if not unique_blockers else None,
        snapshot_digest=digest,
        evidence=evidence,
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
            blockers = (
                ("merge_lease_unavailable",)
                if self._ledger.is_merge_enrolled(self._policy.repository, number)
                else ("merge_pr_not_enrolled",)
            )
            blocked = MergeDecision(
                False, blockers, None, first.snapshot_digest
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
        authorized = self._ledger.authorize_merge_write(
            lease, updated_at=self._now()
        )
        if not authorized:
            self._ledger.finish_merge_lease(
                lease,
                status="failed",
                updated_at=self._now(),
                error="merge_pr_not_enrolled",
            )
            blocked = MergeDecision(
                False,
                ("merge_pr_not_enrolled",),
                None,
                second.snapshot_digest,
            )
            return MergeRunResult(blocked, None)
        try:
            self._github.merge_pull_request(
                self._policy.repository,
                number,
                second_snapshot.pull_request.head_sha,
                method=second.method,
            )
        except GitHubClientError:
            # A transport error cannot prove that GitHub rejected the write.
            # Canonical readback below remains the only completion authority.
            pass
        try:
            readback = self._github.get_merge_state(self._policy.repository, number)
        except GitHubClientError as error:
            self._ledger.finish_merge_lease(
                lease,
                status="verification_required",
                updated_at=self._now(),
                error=type(error).__name__,
                expected_status="verification_required",
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
                status="verification_required",
                updated_at=self._now(),
                error="canonical readback did not confirm the merge",
                expected_status="verification_required",
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
            lease,
            status="completed",
            updated_at=self._now(),
            receipt=receipt,
            expected_status="verification_required",
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
        self,
        plugin_policy: PluginPolicy,
        github: GitHubClient,
        ledger: FeedbackLedger,
        merge_policy: MergeMaintainerPolicy | None = None,
    ) -> None:
        selected_policy = merge_policy or plugin_policy.merge_maintainer
        if selected_policy is None:
            raise ValueError("merge maintainer is disabled")
        self._plugin_policy = plugin_policy
        self._merge_policy = selected_policy
        self._github = github
        self._ledger = ledger

    def snapshot(self, number: int) -> MergeSnapshot:
        policy = self._merge_policy
        pull = self._github.get_merge_state(policy.repository, number)
        target = self._plugin_policy.targets[policy.repository]
        manifest_path = ci_manifest_path(target.local_path)
        if not manifest_path.is_file():
            raise GitHubClientError("CI manifest was unavailable")
        manifest_bytes = manifest_path.read_bytes()
        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
        receipt = self._ledger.latest_ci_receipt(
            policy.repository,
            number,
            pull.head_sha,
            manifest_digest=manifest_digest,
            not_before=datetime.min.replace(tzinfo=UTC),
        )
        if receipt is None:
            comment_reader = getattr(type(self._github), "list_ci_receipt_comments", None)
            if callable(comment_reader):
                receipt_comments = comment_reader(self._github, policy.repository, number)
                expected_login = (
                    self._plugin_policy.github_identity.expected_login
                    if self._plugin_policy.github_identity is not None
                    else None
                )
                receipt = ci_receipt_comment_from_feedback(
                    receipt_comments,
                    pull,
                    expected_login=expected_login,
                    manifest_digest=manifest_digest,
                )
        if receipt is not None and not isinstance(receipt, CIAuditReceipt):
            if not isinstance(receipt, CIReceiptComment):
                raise GitHubClientError("CI receipt had an invalid type")
        feedback = self._github.list_feedback(policy.repository, number)
        feedback_clear = self._feedback_clear(pull, feedback)
        intent_pending = pending_intent_review(feedback, owner_login=target.owner_login)
        codex_pending = not _codex_reviewed_head(feedback, pull.head_sha)
        if self._plugin_policy.uses_budget_exhausted_local_ci(policy.repository):
            actions_enabled = self._github.actions_enabled(policy.repository, refresh=True)
            check_state = self._github.get_check_state(
                policy.repository,
                pull.head_sha,
                actions_enabled_hint=actions_enabled,
            )
        else:
            check_state = self._github.get_check_state(policy.repository, pull.head_sha)
        actions_disabled_evidence = (
            actions_disabled_local_ci_evidence(receipt, manifest_bytes)
            if not check_state.actions_enabled
            and self._plugin_policy.uses_budget_exhausted_local_ci(policy.repository)
            and isinstance(receipt, CIAuditReceipt)
            else None
        )
        return MergeSnapshot(
            repository_private=self._github.repository_is_private(policy.repository),
            pull_request=pull,
            branch_allowed=(
                not target.branch_prefixes
                or any(
                    pull.head_ref_name.startswith(prefix)
                    for prefix in target.branch_prefixes
                )
            ),
            repository_merge_policy=self._github.get_repository_merge_policy(
                policy.repository
            ),
            review_state=self._github.get_review_state(policy.repository, number),
            check_state=check_state,
            ci_receipt=receipt,
            manifest_digest=manifest_digest,
            feedback_clear=feedback_clear,
            base_head_sha=self._github.get_branch_head(
                policy.repository, policy.base_branch
            ),
            intent_review_pending=intent_pending,
            codex_review_pending=codex_pending,
            actions_disabled_local_ci=actions_disabled_evidence,
        )

    def _feedback_clear(
        self, pull: PullRequestMergeState, feedback_items: tuple[Feedback, ...]
    ) -> bool:
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
        automation_login = (
            policy.github_identity.expected_login if policy.github_identity is not None else None
        )
        for feedback in feedback_items:
            if policy.not_before is not None and feedback.created_at < policy.not_before:
                continue
            if (
                _is_non_actionable_review_container(feedback)
                or _is_codex_review_summary_tracker(feedback)
                or _is_self_resolution_receipt(feedback, owner_login=target.owner_login)
                or (
                    automation_login is not None
                    and _is_governed_approval_receipt(
                        feedback,
                        expected_login=automation_login,
                        head_sha=pull.head_sha,
                    )
                )
                or (
                    automation_login is not None
                    and _is_ci_receipt_comment_for_head(
                        feedback,
                        expected_login=automation_login,
                        head_sha=pull.head_sha,
                    )
                )
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


def _snapshot_digest(
    snapshot: MergeSnapshot, receipt: CIAuditReceipt | CIReceiptComment | None
) -> str:
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
            "billing_blocked": snapshot.check_state.billing_blocked,
        },
        "ci_receipt": receipt.receipt_id if receipt else None,
        "ci_mode": receipt.ci_mode if receipt else None,
        "manifest": snapshot.manifest_digest,
        "feedback_clear": snapshot.feedback_clear,
        "actions_disabled_local_ci": (
            {
                "receipt_id": snapshot.actions_disabled_local_ci.receipt_id,
                "manifest_digest": snapshot.actions_disabled_local_ci.manifest_digest,
                "command_count": snapshot.actions_disabled_local_ci.command_count,
                "required_command_count": (
                    snapshot.actions_disabled_local_ci.required_command_count
                ),
            }
            if snapshot.actions_disabled_local_ci is not None
            else None
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _aware_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return value.astimezone(UTC)
