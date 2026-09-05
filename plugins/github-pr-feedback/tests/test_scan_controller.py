from __future__ import annotations

import hashlib
import sqlite3
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from github_pr_feedback.controller import (
    LOCAL_CI_FEEDBACK_ID,
    MAX_ADMISSIONS_PER_SCAN,
    MAX_PARALLEL_PR_READS,
    GitCommandResult,
    LocalGitRepository,
    PreparedWorktree,
    ScanController,
    _ci_failure_assignee,
    _ci_receipt_feedback_reason,
    _is_self_resolution_receipt,
    _intent_review_task,
    _task,
)
from github_pr_feedback.ci_runner import CIAuditIdentity, CIAuditReceipt, CommandEvidence
from github_pr_feedback.github_client import MAX_FEEDBACK_BODY_CHARS, CheckState, Feedback
from github_pr_feedback.ledger import FeedbackLedger
from github_pr_feedback.policy import (
    FeedbackReceipt,
    PullRequest,
    Reviewer,
    load_policy,
)


def test_scan_admission_budget_covers_a_large_pr_repair_queue() -> None:
    assert MAX_ADMISSIONS_PER_SCAN >= 128


def test_intent_review_card_uses_valid_zero_retry_encoding(tmp_path: Path) -> None:
    policy = SimpleNamespace(
        merge_maintainer=None,
        assignee="pr-merge-maintainer",
        board="repairs",
    )
    receipt = FeedbackReceipt(
        "acme/widgets",
        17,
        "review_comment",
        "comment-1",
        "a" * 40,
    )

    task = _intent_review_task(policy, receipt, "use the replacement approach", tmp_path)

    assert task.initial_status == "blocked"
    assert task.max_retries == 1
    assert task.idempotency_key.startswith("github-pr-feedback:intent-review:")
    assert "operator intent decision" in task.instructions.casefold()


class FakeGitHub:
    def __init__(
        self,
        pull_request: PullRequest,
        feedback: tuple[Feedback, ...],
        current: PullRequest | None = None,
        pull_requests: tuple[PullRequest, ...] | None = None,
    ) -> None:
        self.pull_request = pull_request
        self.feedback = feedback
        self.current = current or pull_request
        self.pull_requests = pull_requests
        self.current_by_number = {
            item.number: item for item in (pull_requests or (pull_request,))
        }
        self.current_calls: list[tuple[str, int]] = []
        self.feedback_calls: list[tuple[str, int]] = []
        self.branch_calls: list[tuple[str, str]] = []
        self.actions_are_enabled = True
        self.billing_blocked = False
        self.branch_head = self.current.base_sha

    def list_open_pull_requests(
        self, repository: str, owner_login: str
    ) -> tuple[PullRequest, ...]:
        assert repository == self.pull_request.base_repository
        assert owner_login == self.pull_request.author_login
        return self.pull_requests or (self.pull_request,)

    def list_feedback(self, repository: str, number: int) -> tuple[Feedback, ...]:
        assert repository == self.pull_request.base_repository
        assert number in self.current_by_number
        self.feedback_calls.append((repository, number))
        return self.feedback

    def get_pull_request(self, repository: str, number: int) -> PullRequest:
        self.current_calls.append((repository, number))
        if number == self.current.number:
            return self.current
        return self.current_by_number[number]

    def actions_enabled(self, repository: str) -> bool:
        assert repository == self.pull_request.base_repository
        return self.actions_are_enabled

    def get_check_state(self, repository: str, head_sha: str) -> CheckState:
        assert repository == self.pull_request.base_repository
        if not self.actions_are_enabled:
            return CheckState(actions_enabled=False, all_green=True, check_count=0)
        if self.billing_blocked:
            return CheckState(
                actions_enabled=True,
                all_green=False,
                check_count=1,
                billing_blocked=True,
            )
        return CheckState(actions_enabled=True, all_green=True, check_count=0)

    def get_branch_head(self, repository: str, branch: str) -> str:
        self.branch_calls.append((repository, branch))
        assert repository == self.pull_request.base_repository
        assert branch == self.current.base_branch
        assert self.branch_head is not None
        return self.branch_head


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (("python3", "scripts/run_hygiene_lane.py"), "ci-hygiene-fixer"),
        (("python3", "scripts/run_static_lane.py"), "ci-static-fixer"),
        (("python3", "scripts/run_test_lane.py", "--lane", "unit"), "ci-test-fixer"),
        (("npm", "run", "build"), "ci-frontend-fixer"),
        (("python3", "scripts/unknown_ci_check.py"), "ci-general-fixer"),
    ],
)
def test_typed_ci_receipt_routes_to_the_exact_failure_owner(
    argv: tuple[str, ...], expected: str
) -> None:
    receipt = CIAuditReceipt(
        receipt_id="f" * 64,
        identity=CIAuditIdentity("acme/widgets", 17, "b" * 40, "a" * 40),
        manifest_digest="e" * 64,
        status="failed",
        started_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        completed_at=datetime(2026, 8, 25, 12, 1, tzinfo=UTC),
        actions_state=CheckState(False, True, 0),
        commands=(
            CommandEvidence(
                argv=argv,
                cwd="/tmp/worktree",
                returncode=1,
                duration_ms=1,
                timed_out=False,
                stdout_sha256="0" * 64,
                stderr_sha256="0" * 64,
                classification="logic-regression",
            ),
        ),
    )

    assert _ci_failure_assignee(receipt) == expected


def test_environment_blocked_ci_receipt_does_not_dispatch_a_fixer() -> None:
    receipt = CIAuditReceipt(
        receipt_id="f" * 64,
        identity=CIAuditIdentity("acme/widgets", 17, "b" * 40, "a" * 40),
        manifest_digest="e" * 64,
        status="failed",
        started_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        completed_at=datetime(2026, 8, 25, 12, 1, tzinfo=UTC),
        actions_state=CheckState(False, True, 0),
        commands=(
            CommandEvidence(
                argv=("python3", "scripts/run_static_lane.py"),
                cwd=".",
                returncode=127,
                duration_ms=1,
                timed_out=False,
                stdout_sha256="0" * 64,
                stderr_sha256="0" * 64,
                classification="environment-blocked",
            ),
        ),
    )

    assert _ci_failure_assignee(receipt) is None


def test_unchanged_pr_update_watermark_skips_repeated_feedback_reads(
    tmp_path: Path,
) -> None:
    local_path, head_sha = initialized_repository(tmp_path)
    policy = configured_policy(
        local_path,
        not_before="2026-08-24T00:00:00Z",
    )
    pull_request = PullRequest(
        17,
        "OPEN",
        "acme/widgets",
        "acme/widgets",
        "owner",
        "codex/fix",
        head_sha,
        updated_at=datetime(2026, 8, 26, 8, 0, tzinfo=UTC),
    )
    github = FakeGitHub(pull_request, ())
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    controller = ScanController(
        policy,
        ledger,
        github,
        RecordingKanban(),
        RecordingLocalGit(),
    )

    first = controller.scan()
    second = controller.scan()

    assert first.degraded is False
    assert second.degraded is False
    assert github.feedback_calls == [("acme/widgets", 17)]


def test_failed_ci_receipt_waits_for_current_base_before_dispatching_fixer(
    tmp_path: Path,
) -> None:
    local_path, head_sha = initialized_repository(tmp_path)
    old_base = "b" * 40
    policy = configured_policy(
        local_path,
        not_before="2026-08-24T00:00:00Z",
        local_ci_audit=True,
        merge_maintainer=True,
    )
    current = PullRequest(
        17,
        "OPEN",
        "acme/widgets",
        "acme/widgets",
        "owner",
        "codex/fix",
        head_sha,
        base_branch="stable",
        base_sha=old_base,
    )
    github = FakeGitHub(current, ())
    github.branch_head = "c" * 40
    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    receipt = CIAuditReceipt(
        receipt_id="f" * 64,
        identity=CIAuditIdentity("acme/widgets", 17, old_base, head_sha),
        manifest_digest="e" * 64,
        status="failed",
        started_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        completed_at=datetime(2026, 8, 25, 12, 1, tzinfo=UTC),
        actions_state=CheckState(False, True, 0),
        commands=(
            CommandEvidence(
                argv=(".venv/bin/python", "scripts/run_static_lane.py"),
                cwd=".",
                returncode=1,
                duration_ms=1,
                timed_out=False,
                stdout_sha256="0" * 64,
                stderr_sha256="0" * 64,
                classification="logic-regression",
            ),
        ),
    )
    ledger.record_ci_receipt(receipt)

    result = ScanController(policy, ledger, github, kanban, RecordingLocalGit()).dispatch_ci_failure(
        receipt
    )

    assert result == "base_refresh_required"
    assert kanban.tasks == []
    ledger.close()


def test_failed_ci_audit_comment_waits_for_current_base_before_typed_routing(
    tmp_path: Path,
) -> None:
    local_path, head_sha = initialized_repository(tmp_path)
    old_base = "b" * 40
    policy = configured_policy(
        local_path,
        not_before="2026-08-24T00:00:00Z",
        local_ci_audit=True,
        merge_maintainer=True,
    )
    current = PullRequest(
        17,
        "OPEN",
        "acme/widgets",
        "acme/widgets",
        "owner",
        "codex/fix",
        head_sha,
        base_branch="stable",
        base_sha=old_base,
    )
    audit = CIAuditReceipt(
        receipt_id="f" * 64,
        identity=CIAuditIdentity("acme/widgets", 17, old_base, head_sha),
        manifest_digest="e" * 64,
        status="failed",
        started_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        completed_at=datetime(2026, 8, 25, 12, 1, tzinfo=UTC),
        actions_state=CheckState(False, True, 0),
        commands=(
            CommandEvidence(
                argv=(".venv/bin/python", "scripts/run_static_lane.py"),
                cwd=".",
                returncode=1,
                duration_ms=1,
                timed_out=False,
                stdout_sha256="0" * 64,
                stderr_sha256="0" * 64,
                classification="logic-regression",
            ),
        ),
    )
    item = feedback(
        "stale-ci-comment",
        reviewer="owner",
        body=(
            "Local CI audit (hosted GitHub Actions disabled): failed. "
            f"Receipt: `{audit.receipt_id}` (base {old_base}). "
            "scripts/run_static_lane.py failed, logic regression."
        ),
    )
    github = FakeGitHub(current, (item,))
    github.branch_head = "c" * 40
    github.actions_are_enabled = False
    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    ledger.record_ci_receipt(audit)

    result = ScanController(policy, ledger, github, kanban, RecordingLocalGit()).scan()

    assert result.created == 0
    assert result.skipped["base_refresh_required"] == 1
    assert result.skipped.get("feedback_pending", 0) == 0
    assert github.current_calls == [("acme/widgets", 17)]
    assert github.branch_calls == [("acme/widgets", "stable")]
    assert kanban.tasks == []
    ledger.close()


def test_ordinary_feedback_referencing_ci_is_not_bound_to_a_failed_receipt(
    tmp_path: Path,
) -> None:
    local_path, head_sha = initialized_repository(tmp_path)
    old_base = "b" * 40
    policy = configured_policy(
        local_path,
        not_before="2026-08-24T00:00:00Z",
        local_ci_audit=True,
        merge_maintainer=True,
    )
    current = PullRequest(
        17,
        "OPEN",
        "acme/widgets",
        "acme/widgets",
        "owner",
        "codex/fix",
        head_sha,
        base_branch="stable",
        base_sha=old_base,
    )
    audit = CIAuditReceipt(
        receipt_id="f" * 64,
        identity=CIAuditIdentity("acme/widgets", 17, old_base, head_sha),
        manifest_digest="e" * 64,
        status="failed",
        started_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        completed_at=datetime(2026, 8, 25, 12, 1, tzinfo=UTC),
        actions_state=CheckState(False, True, 0),
        commands=(),
    )
    item = feedback(
        "ordinary-ci-reference",
        reviewer="owner",
        body="Please fix the PR regression described by the local CI audit; it still fails.",
    )
    github = FakeGitHub(current, (item,))
    github.branch_head = "c" * 40
    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    ledger.record_ci_receipt(audit)

    result = ScanController(policy, ledger, github, kanban, RecordingLocalGit()).scan()

    assert result.created == 1
    assert result.skipped.get("base_refresh_required", 0) == 0
    assert github.branch_calls == []
    assert [task.evidence["feedback_id"] for task in kanban.tasks] == [
        "ordinary-ci-reference"
    ]
    ledger.close()


def test_failed_ci_comment_is_suppressed_after_base_refresh_changes_head(
    tmp_path: Path,
) -> None:
    local_path, stale_head = initialized_repository(tmp_path)
    old_base = "b" * 40
    refreshed_head = "d" * 40
    policy = configured_policy(
        local_path,
        not_before="2026-08-24T00:00:00Z",
        local_ci_audit=True,
        merge_maintainer=True,
    )
    current = PullRequest(
        17,
        "OPEN",
        "acme/widgets",
        "acme/widgets",
        "owner",
        "codex/fix",
        refreshed_head,
        base_branch="stable",
        base_sha="c" * 40,
    )
    audit = CIAuditReceipt(
        receipt_id="f" * 64,
        identity=CIAuditIdentity("acme/widgets", 17, old_base, stale_head),
        manifest_digest="e" * 64,
        status="failed",
        started_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        completed_at=datetime(2026, 8, 25, 12, 1, tzinfo=UTC),
        actions_state=CheckState(False, True, 0),
        commands=(),
    )
    item = feedback(
        "stale-ci-comment",
        reviewer="owner",
        body=(
            "Local CI audit failed. "
            f"Authoritative receipt: `{audit.receipt_id}`."
        ),
    )
    github = FakeGitHub(current, (item,))
    github.actions_are_enabled = True
    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    ledger.record_ci_receipt(audit)

    result = ScanController(policy, ledger, github, kanban, RecordingLocalGit()).scan()

    assert result.created == 0
    assert result.skipped["superseded_ci_receipt"] == 1
    assert result.skipped.get("feedback_pending", 0) == 0
    assert github.branch_calls == []
    assert kanban.tasks == []
    ledger.close()


def test_duplicate_failed_ci_comments_share_one_base_head_lookup(tmp_path: Path) -> None:
    local_path, head_sha = initialized_repository(tmp_path)
    old_base = "b" * 40
    policy = configured_policy(
        local_path,
        not_before="2026-08-24T00:00:00Z",
        local_ci_audit=True,
        merge_maintainer=True,
    )
    current = PullRequest(
        17,
        "OPEN",
        "acme/widgets",
        "acme/widgets",
        "owner",
        "codex/fix",
        head_sha,
        base_branch="stable",
        base_sha=old_base,
    )
    audit = CIAuditReceipt(
        receipt_id="f" * 64,
        identity=CIAuditIdentity("acme/widgets", 17, old_base, head_sha),
        manifest_digest="e" * 64,
        status="failed",
        started_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        completed_at=datetime(2026, 8, 25, 12, 1, tzinfo=UTC),
        actions_state=CheckState(False, True, 0),
        commands=(),
    )
    body = f"Local CI audit failed. Receipt: `{audit.receipt_id}`."
    github = FakeGitHub(
        current,
        (
            feedback("ci-comment-a", reviewer="owner", body=body),
            feedback("ci-comment-b", reviewer="owner", body=body),
        ),
    )
    github.branch_head = "c" * 40
    github.actions_are_enabled = False
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    ledger.record_ci_receipt(audit)

    result = ScanController(
        policy, ledger, github, RecordingKanban(), RecordingLocalGit()
    ).scan()

    assert result.created == 0
    assert result.skipped["base_refresh_required"] == 2
    assert github.branch_calls == [("acme/widgets", "stable")]
    ledger.close()


def test_failed_ci_comment_base_lookup_failure_is_degraded(tmp_path: Path) -> None:
    local_path, head_sha = initialized_repository(tmp_path)
    base_sha = "b" * 40
    policy = configured_policy(
        local_path,
        not_before="2026-08-24T00:00:00Z",
        local_ci_audit=True,
        merge_maintainer=True,
    )
    current = PullRequest(
        17,
        "OPEN",
        "acme/widgets",
        "acme/widgets",
        "owner",
        "codex/fix",
        head_sha,
        base_branch="stable",
        base_sha=base_sha,
    )
    audit = CIAuditReceipt(
        receipt_id="f" * 64,
        identity=CIAuditIdentity("acme/widgets", 17, base_sha, head_sha),
        manifest_digest="e" * 64,
        status="failed",
        started_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        completed_at=datetime(2026, 8, 25, 12, 1, tzinfo=UTC),
        actions_state=CheckState(False, True, 0),
        commands=(),
    )
    item = feedback(
        "ci-comment",
        reviewer="owner",
        body=f"Local CI audit failed. Receipt: `{audit.receipt_id}`.",
    )

    class UnavailableBaseGitHub(FakeGitHub):
        def get_branch_head(self, repository: str, branch: str) -> str:
            self.branch_calls.append((repository, branch))
            raise RuntimeError("base unavailable")

    github = UnavailableBaseGitHub(current, (item,))
    github.actions_are_enabled = False
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    ledger.record_ci_receipt(audit)

    result = ScanController(
        policy, ledger, github, RecordingKanban(), RecordingLocalGit()
    ).scan()

    assert result.created == 0
    assert result.skipped["base_state_unavailable"] == 1
    assert result.degraded is True
    ledger.close()


def test_retry_admission_precedes_stale_ci_comment_base_lookup(tmp_path: Path) -> None:
    local_path, head_sha = initialized_repository(tmp_path)
    old_base = "b" * 40
    policy = configured_policy(
        local_path,
        not_before="2026-08-24T00:00:00Z",
        local_ci_audit=True,
        merge_maintainer=True,
    )
    closed = PullRequest(
        17,
        "CLOSED",
        "acme/widgets",
        "acme/widgets",
        "owner",
        "codex/fix",
        head_sha,
        base_branch="stable",
        base_sha=old_base,
    )
    receipt_id = "f" * 64
    item = feedback(
        "stale-ci-comment",
        reviewer="owner",
        body=(
            "Local CI audit failed. "
            f"Authoritative receipt: `{receipt_id}`."
        ),
    )
    github = FakeGitHub(closed, (item,))
    github.branch_head = "c" * 40
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    ledger.record_ci_receipt(
        CIAuditReceipt(
            receipt_id=receipt_id,
            identity=CIAuditIdentity("acme/widgets", 17, old_base, head_sha),
            manifest_digest="e" * 64,
            status="failed",
            started_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
            completed_at=datetime(2026, 8, 25, 12, 1, tzinfo=UTC),
            actions_state=CheckState(False, True, 0),
            commands=(),
        )
    )
    skipped: Counter[str] = Counter()
    controller = ScanController(
        policy, ledger, github, RecordingKanban(), RecordingLocalGit()
    )
    receipt = FeedbackReceipt(
        "acme/widgets", 17, "issue_comment", "stale-ci-comment", head_sha
    )

    assert controller._revalidate(receipt, skipped) is None
    assert skipped == {"pull_request_not_open": 1}
    assert github.branch_calls == []
    ledger.close()


def test_failed_exact_head_static_receipt_immediately_dispatches_one_typed_fixer(
    tmp_path: Path,
) -> None:
    local_path, head_sha = initialized_repository(tmp_path)
    base_sha = "b" * 40
    raw = {
        "enabled": True,
        "repositories": [
            {
                "base_repository": "acme/widgets",
                "head_repository": "acme/widgets",
                "local_path": str(local_path),
                "owner_login": "owner",
                "branch_prefixes": ["codex/"],
            }
        ],
        "reviewer_logins": ["reviewer"],
        "reviewer_associations": [],
        "not_before": "2026-08-24T00:00:00Z",
        "assignee": "task-orchestrator",
        "routing_rules": [
            {
                "assignee": "ci-static-fixer",
                "precedence": 150,
                "match_any": ["static lane"],
                "match_labels_any": ["ci/static"],
                "tags": ["type/ci", "ci/static"],
                "priority": "P2",
                "blast_radius": "contained",
                "risks": [],
                "requires_review": False,
            }
        ],
        "auto_dispatch": True,
        "board": "repairs",
        "local_ci_audit": {
            "enabled": True,
            "assignee": "pr-local-ci-auditor",
            "post_results": True,
        },
    }
    current = PullRequest(
        17,
        "OPEN",
        "acme/widgets",
        "acme/widgets",
        "owner",
        "codex/fix",
        head_sha,
        base_branch="main",
        base_sha=base_sha,
    )
    github = FakeGitHub(current, ())
    kanban = RecordingKanban()
    local_git = RecordingLocalGit()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    receipt = CIAuditReceipt(
        receipt_id="f" * 64,
        identity=CIAuditIdentity("acme/widgets", 17, base_sha, head_sha),
        manifest_digest="e" * 64,
        status="failed",
        started_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        completed_at=datetime(2026, 8, 25, 12, 1, tzinfo=UTC),
        actions_state=CheckState(False, True, 0),
        commands=(
            CommandEvidence(
                argv=(".venv/bin/python", "scripts/run_static_lane.py"),
                cwd=".",
                returncode=1,
                duration_ms=1,
                timed_out=False,
                stdout_sha256="0" * 64,
                stderr_sha256="0" * 64,
                classification="logic-regression",
            ),
        ),
    )
    ledger.record_ci_receipt(receipt)
    controller = ScanController(
        load_policy(raw), ledger, github, kanban, local_git, control_home=tmp_path
    )

    assert controller.dispatch_ci_failure(receipt) == "scheduled"
    assert controller.dispatch_ci_failure(receipt) == "duplicate"
    assert len(kanban.tasks) == 1
    task = kanban.tasks[0]
    assert task.assignee == "ci-static-fixer"
    assert task.head_sha == head_sha
    assert task.initial_status == "running"
    assert task.max_runtime_seconds == 60 * 60
    assert task.max_retries == 2
    assert task.idempotency_key.endswith(":typed-fixer-v3")
    assert "first 90 seconds" in task.instructions
    assert "do not repeat completed work" in task.instructions
    assert task.evidence["ci_receipt_id"] == "f" * 64
    assert task.evidence["failed_command"]["classification"] == "logic-regression"
    assert "pr-maintenance-receipt:v1" in task.instructions
    assert local_git.calls[0][1].feedback_kind == "pr_repair"
    ledger.close()


@pytest.mark.parametrize(
    ("stored_task_status", "expected_second_result", "expected_task_count"),
    [("archived", "scheduled", 2), (None, "duplicate", 1)],
)
def test_only_archived_exact_head_fixer_is_recreated(
    tmp_path: Path,
    stored_task_status: str | None,
    expected_second_result: str,
    expected_task_count: int,
) -> None:
    local_path, head_sha = initialized_repository(tmp_path)
    base_sha = "b" * 40
    configured = configured_policy(
        local_path,
        not_before="2026-08-24T00:00:00Z",
        local_ci_audit=True,
    )
    current = PullRequest(
        17,
        "OPEN",
        "acme/widgets",
        "acme/widgets",
        "owner",
        "codex/fix",
        head_sha,
        base_branch="main",
        base_sha=base_sha,
    )
    audit = CIAuditReceipt(
        receipt_id="f" * 64,
        identity=CIAuditIdentity("acme/widgets", 17, base_sha, head_sha),
        manifest_digest="e" * 64,
        status="failed",
        started_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        completed_at=datetime(2026, 8, 25, 12, 1, tzinfo=UTC),
        actions_state=CheckState(False, True, 0),
        commands=(
            CommandEvidence(
                argv=(".venv/bin/python", "scripts/run_static_lane.py"),
                cwd=".",
                returncode=1,
                duration_ms=1,
                timed_out=False,
                stdout_sha256="0" * 64,
                stderr_sha256="0" * 64,
                classification="logic-regression",
            ),
        ),
    )

    class ArchivedKanban(RecordingKanban):
        def task_status(self, board: str, task_id: str) -> str | None:
            assert board == "repairs"
            assert task_id == "kanban-1"
            return stored_task_status

    kanban = ArchivedKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    ledger.record_ci_receipt(audit)
    controller = ScanController(
        configured,
        ledger,
        FakeGitHub(current, ()),
        kanban,
        RecordingLocalGit(),
        control_home=tmp_path,
    )

    assert controller.dispatch_ci_failure(audit) == "scheduled"
    assert controller.dispatch_ci_failure(audit) == expected_second_result
    assert len(kanban.tasks) == expected_task_count
    ledger.close()


@pytest.mark.parametrize(
    ("stored_task_status", "expected_second_result", "expected_task_count"),
    [("archived", "scheduled", 2), (None, "duplicate", 1)],
)
def test_only_archived_exact_head_local_ci_audit_is_recreated(
    tmp_path: Path,
    stored_task_status: str | None,
    expected_second_result: str,
    expected_task_count: int,
) -> None:
    local_path, head_sha = initialized_repository(tmp_path)
    base_sha = "b" * 40
    configured = configured_policy(
        local_path,
        not_before="2026-08-24T00:00:00Z",
        local_ci_audit=True,
    )
    current = PullRequest(
        17,
        "OPEN",
        "acme/widgets",
        "acme/widgets",
        "owner",
        "codex/fix",
        head_sha,
        base_branch="main",
        base_sha=base_sha,
    )

    class ArchivedKanban(RecordingKanban):
        def task_status(self, board: str, task_id: str) -> str | None:
            assert board == "repairs"
            assert task_id == "kanban-1"
            return stored_task_status

    github = FakeGitHub(current, ())
    github.actions_are_enabled = False
    kanban = ArchivedKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    controller = ScanController(
        configured,
        ledger,
        github,
        kanban,
        RecordingLocalGit(),
        control_home=tmp_path,
    )

    assert controller.dispatch_local_ci_after_feedback(current) == "scheduled"
    assert (
        controller.dispatch_local_ci_after_feedback(current)
        == expected_second_result
    )
    assert len(kanban.tasks) == expected_task_count
    ledger.close()


@pytest.mark.parametrize(
    "receipt_text",
    (
        "Authoritative receipt: `{receipt_id}`.",
        "Receipt `{receipt_id}`, manifest digest present.",
        "receipt_id `{receipt_id}`, status failed.",
        "Receipt id: {receipt_id} — status failed.",
        "Status failed for {receipt_id}; this is the authoritative local-CI receipt.",
    ),
)
def test_superseded_ci_receipt_comment_does_not_create_duplicate_repair(
    tmp_path: Path, receipt_text: str
) -> None:
    identity = CIAuditIdentity("acme/widgets", 17, "b" * 40, "a" * 40)
    older = CIAuditReceipt(
        receipt_id="1" * 64,
        identity=identity,
        manifest_digest="e" * 64,
        status="failed",
        started_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        completed_at=datetime(2026, 8, 25, 12, 1, tzinfo=UTC),
        actions_state=CheckState(False, True, 0),
        commands=(),
    )
    latest = CIAuditReceipt(
        receipt_id="2" * 64,
        identity=identity,
        manifest_digest="e" * 64,
        status="failed",
        started_at=datetime(2026, 8, 25, 12, 2, tzinfo=UTC),
        completed_at=datetime(2026, 8, 25, 12, 3, tzinfo=UTC),
        actions_state=CheckState(False, True, 0),
        commands=(),
    )
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    ledger.record_ci_receipt(older)
    ledger.record_ci_receipt(latest)
    feedback_receipt = FeedbackReceipt(
        "acme/widgets", 17, "issue_comment", "comment-1", "a" * 40
    )

    reason = _ci_receipt_feedback_reason(
        ledger,
        feedback_receipt,
        "Local CI audit completed. "
        + receipt_text.format(receipt_id="1" * 64),
    )

    assert reason == "superseded_ci_receipt"
    ledger.close()


def test_ci_receipt_for_an_older_tested_head_is_superseded_after_repair(
    tmp_path: Path,
) -> None:
    old_head = "a" * 40
    current_head = "c" * 40
    audit = CIAuditReceipt(
        receipt_id="1" * 64,
        identity=CIAuditIdentity("acme/widgets", 17, "b" * 40, old_head),
        manifest_digest="e" * 64,
        status="failed",
        started_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        completed_at=datetime(2026, 8, 25, 12, 1, tzinfo=UTC),
        actions_state=CheckState(False, True, 0),
        commands=(),
    )
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    ledger.record_ci_receipt(audit)
    receipt = FeedbackReceipt(
        "acme/widgets", 17, "issue_comment", "comment-1", current_head
    )

    reason = _ci_receipt_feedback_reason(
        ledger,
        receipt,
        "Local CI audit (read-only, exact-head): tested head "
        f"{old_head}. Receipt {'1' * 64}: FAILED. Recommend a separate repair card.",
    )

    assert reason == "superseded_ci_receipt"
    ledger.close()


def test_self_resolution_accepts_exact_fixed_commit_with_historic_failures() -> None:
    item = feedback(
        "fixed-with-historic-failures",
        reviewer="owner",
        body=(
            f"Fixed in {'a' * 40}. Verification: 28 focused tests passed. "
            "Three pre-existing failures reproduce at the pristine receipt SHA and "
            "are unrelated to this change."
        ),
    )

    assert _is_self_resolution_receipt(item, owner_login="owner") is True


def test_self_resolution_accepts_fixed_import_with_reproduced_old_failures() -> None:
    item = feedback(
        "fixed-import-with-old-failures",
        reviewer="owner",
        body=(
            f"Fixed in {'d' * 40}.\n\n"
            f"Confirmed at {'9' * 40}: importing the research module fails under "
            "python3 -S before any scenario executes.\n\n"
            "Fix: the owner module is now loaded directly from its file path.\n\n"
            "Verification:\n- python3 -S import now succeeds.\n"
            "- Focused gate: 28 passed.\n"
            "- Note: 3 pre-existing failures reproduced identically at the pristine "
            "receipt SHA, so they are unrelated to this change."
        ),
    )

    assert _is_self_resolution_receipt(item, owner_login="owner") is True


@pytest.mark.parametrize(
    "body",
    (
        (
            f"Reproduced the static-lane failure at head {'a' * 40}. "
            f"Fix (commit {'b' * 40}): added the scoped header. "
            "Verification after the fix: static lane status: pass (rc=0), 47 passed. "
            "No safety gate was relaxed; merge remains gated."
        ),
        (
            "Static-lane repair for the local-CI receipt, pushed at the verified "
            f"head {'a' * 40}. Commit: {'b' * 40}. Failure reproduced before the fix. "
            "After the fix the same lane returns status: pass, rc=0; 50 passed. "
            "No required check or safety gate was relaxed; merge remains gated."
        ),
    ),
)
def test_self_resolution_understands_semantic_static_repair_receipts(
    body: str,
) -> None:
    item = feedback("semantic-static-repair", reviewer="owner", body=body)

    assert _is_self_resolution_receipt(item, owner_login="owner") is True


def test_self_resolution_accepts_a_bounded_long_static_repair_receipt() -> None:
    body = (
        "Static-lane repair for the local-CI receipt, pushed at verified "
        f"head {'a' * 40}. Commit: {'b' * 40}. "
        "Verification after the fix: status: pass, rc=0; 50 passed. "
        "No required check or safety gate was relaxed; merge remains gated. "
        + ("Bounded command evidence and file attribution. " * 70)
    )
    item = feedback("long-static-repair", reviewer="owner", body=body)

    assert 2_000 < len(item.body) < 16_384
    assert _is_self_resolution_receipt(item, owner_login="owner") is True


@pytest.mark.parametrize(
    "body",
    (
        (
            f"Base refresh: normal-merged origin/stable ({'b' * 40}) into the "
            f"verified PR head {'a' * 40} (clean ort merge, zero conflicts). "
            f"Focused check: 20 passed. Canonical head is now {'c' * 40}."
        ),
        (
            f"Base refresh for this PR: canonical head was re-verified at {'a' * 40}. "
            f"Canonical stable ({'b' * 40}) was merged via a normal ort merge with "
            f"zero conflicts. Merge commit {'c' * 40} was pushed as a plain "
            "fast-forward. Focused tests: 50 passed."
        ),
        (
            f"Base refresh repair (base_refresh_required): verified head {'a' * 40}; "
            f"normal-merged stable ({'b' * 40}) with zero conflicts. Merge commit "
            f"{'c' * 40} pushed as a normal fast-forward. Focused tests: 34 passed. "
            "No tests or gates were modified."
        ),
    ),
)
def test_self_resolution_accepts_completed_base_refresh_receipts(body: str) -> None:
    item = feedback("base-refresh-receipt", reviewer="owner", body=body)

    assert _is_self_resolution_receipt(item, owner_login="owner") is True


def test_self_resolution_understands_reordered_base_refresh_evidence() -> None:
    item = feedback(
        "reordered-base-refresh-receipt",
        reviewer="owner",
        body=(
            "Verification finished with 20 focused tests passed. The canonical head is "
            f"now {'c' * 40} after a zero-conflict normal merge of stable {'b' * 40}; "
            "the base has therefore been refreshed and the commit was pushed normally."
        ),
    )

    assert _is_self_resolution_receipt(item, owner_login="owner") is True


def test_self_resolution_understands_neutral_machine_receipt_marker() -> None:
    item = feedback(
        "structured-repair-receipt",
        reviewer="owner",
        body=(
            "Repair completed and focused checks passed.\n"
            "<!-- pr-maintenance-receipt:v1 status=completed kind=base_refresh "
            f"head={'c' * 40} -->"
        ),
    )

    assert _is_self_resolution_receipt(item, owner_login="owner") is True


class MixedPullRequestGitHub(FakeGitHub):
    def __init__(self, admitted: PullRequest, foreign: PullRequest, feedback: tuple[Feedback, ...]) -> None:
        super().__init__(admitted, feedback)
        self.foreign = foreign
        self.feedback_calls: list[int] = []

    def list_open_pull_requests(
        self, repository: str, owner_login: str
    ) -> tuple[PullRequest, ...]:
        assert repository == self.pull_request.base_repository
        assert owner_login == self.pull_request.author_login
        return (self.foreign, self.pull_request)

    def list_feedback(self, repository: str, number: int) -> tuple[Feedback, ...]:
        self.feedback_calls.append(number)
        if number != self.pull_request.number:
            raise AssertionError("feedback for an out-of-policy PR must not be fetched")
        return self.feedback


class ConcurrentScanGitHub(FakeGitHub):
    def __init__(self, pulls: tuple[PullRequest, ...]) -> None:
        super().__init__(pulls[0], ())
        self.pulls = pulls
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def list_open_pull_requests(
        self, repository: str, owner_login: str
    ) -> tuple[PullRequest, ...]:
        return self.pulls

    def list_feedback(self, repository: str, number: int) -> tuple[Feedback, ...]:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self.lock:
            self.active -= 1
        return ()


class RecordingLocalGit:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, object]] = []

    def prepare_receipt_worktree(self, path: Path, receipt: FeedbackReceipt) -> PreparedWorktree:
        self.calls.append((path, receipt))
        return PreparedWorktree(
            path.parent / "prepared-worktree",
            "hermes/github-pr-feedback/receipt-branch",
            receipt.head_sha,
        )


class RecordingKanban:
    def __init__(self) -> None:
        self.tasks: list[object] = []

    def create_task(self, task: object) -> str:
        self.tasks.append(task)
        return f"kanban-{len(self.tasks)}"

    def create_or_get_task(self, task: object) -> str:
        self.tasks.append(task)
        return f"kanban-{len(self.tasks)}"


class FailingKanban:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def create_or_get_task(self, task: object) -> str:
        self.calls.append(task)
        raise RuntimeError("Kanban unavailable")


class LostResponseKanban:
    def __init__(self) -> None:
        self.calls: list[object] = []
        self.created_by_key: dict[str, str] = {}

    def create_or_get_task(self, task: object) -> str:
        self.calls.append(task)
        key = task.idempotency_key
        if key not in self.created_by_key:
            self.created_by_key[key] = "kanban-recovered"
            raise RuntimeError("Kanban response lost")
        return self.created_by_key[key]


class RecordingGitRunner:
    def __init__(self, results: list[GitCommandResult]) -> None:
        self.results = iter(results)
        self.calls: list[list[str]] = []

    def run(self, argv: list[str]) -> GitCommandResult:
        self.calls.append(argv)
        return next(self.results)


def test_scan_creates_one_bounded_untrusted_task_and_deduplicates_with_sqlite(tmp_path: Path) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(local_path, not_before="2026-08-24T00:00:00Z")
    pull_request = admitted_pull_request(sha)
    github = FakeGitHub(
        pull_request,
        (
            feedback("old", created_at="2026-08-23T23:59:59Z"),
            feedback("self", reviewer="owner"),
            feedback("bot", is_bot=True),
            feedback("allowed", body="x" * (MAX_FEEDBACK_BODY_CHARS + 1_000)),
        ),
    )
    kanban = RecordingKanban()
    local_git = RecordingLocalGit()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    scanner = ScanController(policy, ledger, github, kanban, local_git)

    scans = []
    for feedback_id, kind in (("self", "issue_comment"), ("bot", "issue_comment")):
        scans.append(scanner.scan())
        ledger.mark_feedback_actioned(
            FeedbackReceipt("acme/widgets", 17, kind, feedback_id, sha),
            resolved_head_sha=sha,
            actioned_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        )
    scans.append(scanner.scan())
    duplicate_scan = scanner.scan()

    assert [result.created for result in scans] == [1, 1, 1]
    assert duplicate_scan.created == 0
    assert len(kanban.tasks) == 3
    task = next(task for task in kanban.tasks if task.evidence["feedback_id"] == "allowed")
    assert task.repository_path == local_path.parent / "prepared-worktree"
    assert task.head_sha == sha
    assert task.branch == "hermes/github-pr-feedback/receipt-branch"
    assert task.board == "repairs"
    assert task.assignee == "repair-agent"
    assert len(task.evidence["body"]) == MAX_FEEDBACK_BODY_CHARS
    assert task.evidence["untrusted"] is True
    assert "push/reply/merge require operator approval" in task.instructions
    assert local_git.calls[0][0] == local_path
    persisted = ledger._connection.execute(
        "SELECT workspace_path, expected_sha FROM feedback_receipts WHERE feedback_id = 'allowed'"
    ).fetchone()
    assert persisted == (str(local_path.parent / "prepared-worktree"), sha)
    assert {task.evidence["feedback_id"] for task in kanban.tasks} == {"self", "bot", "allowed"}
    ledger.close()


def test_scan_serializes_distinct_feedback_for_the_same_pr_head(tmp_path: Path) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(local_path, not_before="2026-08-24T00:00:00Z")
    github = FakeGitHub(
        admitted_pull_request(sha),
        (
            feedback("first", body="[P1] Fix the first regression."),
            feedback("second", body="[P1] Fix the second regression."),
        ),
    )
    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    scanner = ScanController(policy, ledger, github, kanban, RecordingLocalGit())

    first_scan = scanner.scan()
    second_scan = scanner.scan()

    assert first_scan.created == 1
    assert second_scan.created == 0
    assert [task.evidence["feedback_id"] for task in kanban.tasks] == ["first"]
    first_receipt = FeedbackReceipt("acme/widgets", 17, "issue_comment", "first", sha)
    ledger.mark_feedback_actioned(
        first_receipt,
        resolved_head_sha=sha,
        actioned_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
    )

    third_scan = scanner.scan()

    assert third_scan.created == 1
    assert [task.evidence["feedback_id"] for task in kanban.tasks] == ["first", "second"]
    ledger.close()


def test_scan_routes_actionable_feedback_to_the_best_configured_specialist(
    tmp_path: Path,
) -> None:
    local_path, sha = initialized_repository(tmp_path)
    raw = {
        "enabled": True,
        "repositories": [
            {
                "base_repository": "acme/widgets",
                "head_repository": "acme/widgets",
                "local_path": str(local_path),
                "owner_login": "owner",
                "branch_prefixes": ["codex/"],
            }
        ],
        "reviewer_logins": ["reviewer"],
        "reviewer_associations": [],
        "not_before": "2026-08-24T00:00:00Z",
        "assignee": "task-orchestrator",
        "assignee_rules": [
            {"assignee": "performance-specialist", "match_any": ["latency", "throughput"]},
            {"assignee": "market-data-specialist", "match_any": ["market data", "quote"]},
        ],
        "board": "repairs",
    }
    policy = load_policy(raw)
    github = FakeGitHub(
        admitted_pull_request(sha),
        (feedback("slow", body="[P1] Reduce latency and improve throughput in this hot path."),),
    )
    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    result = ScanController(policy, ledger, github, kanban, RecordingLocalGit()).scan()

    assert result.created == 1
    assert kanban.tasks[0].assignee == "performance-specialist"
    ledger.close()


def test_scan_records_label_driven_routing_and_review_gate_on_the_task(
    tmp_path: Path,
) -> None:
    local_path, sha = initialized_repository(tmp_path)
    raw = {
        "enabled": True,
        "repositories": [
            {
                "base_repository": "acme/widgets",
                "head_repository": "acme/widgets",
                "local_path": str(local_path),
                "owner_login": "owner",
                "branch_prefixes": ["codex/"],
            }
        ],
        "reviewer_logins": ["reviewer"],
        "reviewer_associations": [],
        "not_before": "2026-08-24T00:00:00Z",
        "assignee": "task-orchestrator",
        "routing_rules": [
            {
                "assignee": "session-state-steward",
                "precedence": 100,
                "match_any": ["resume"],
                "match_labels_any": ["sweeper:risk-session-state"],
                "tags": ["type/bug", "area/sessions"],
                "priority": "P1",
                "blast_radius": "broad",
                "risks": ["session-state"],
                "requires_review": True,
            }
        ],
        "board": "repairs",
    }
    pull = PullRequest(
        17,
        "OPEN",
        "acme/widgets",
        "acme/widgets",
        "owner",
        "codex/fix",
        sha,
        labels=("sweeper:risk-session-state",),
    )
    github = FakeGitHub(
        pull,
        (feedback("resume", body="Resume can open the wrong session."),),
    )
    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    result = ScanController(
        load_policy(raw), ledger, github, kanban, RecordingLocalGit()
    ).scan()

    assert result.created == 1
    task = kanban.tasks[0]
    assert task.assignee == "session-state-steward"
    assert task.evidence["routing"] == {
        "tags": ["type/bug", "area/sessions"],
        "priority": "P1",
        "blast_radius": "broad",
        "risks": ["session-state"],
        "requires_review": True,
        "ambiguous": False,
    }
    assert "independent safety review" in task.instructions
    ledger.close()


def test_typed_ci_owner_override_preserves_risk_metadata_and_review_gate(
    tmp_path: Path,
) -> None:
    local_path, sha = initialized_repository(tmp_path)
    raw = {
        "enabled": True,
        "repositories": [
            {
                "base_repository": "acme/widgets",
                "head_repository": "acme/widgets",
                "local_path": str(local_path),
                "owner_login": "owner",
                "branch_prefixes": ["codex/"],
            }
        ],
        "reviewer_logins": ["reviewer"],
        "reviewer_associations": [],
        "not_before": "2026-08-24T00:00:00Z",
        "assignee": "task-orchestrator",
        "routing_rules": [
            {
                "assignee": "session-state-steward",
                "precedence": 100,
                "match_any": [],
                "match_labels_any": ["sweeper:risk-session-state"],
                "tags": ["area/sessions"],
                "priority": "P1",
                "blast_radius": "broad",
                "risks": ["session-state"],
                "requires_review": True,
            }
        ],
        "board": "repairs",
    }
    receipt = FeedbackReceipt(
        "acme/widgets", 17, "issue_comment", "ci-failure", sha
    )

    task = _task(
        load_policy(raw),
        receipt,
        PreparedWorktree(tmp_path / "worktree", "codex/fix", sha),
        "Local CI audit reports a static failure.",
        control_home=tmp_path,
        assignee_override="ci-static-fixer",
        labels=("sweeper:risk-session-state",),
    )

    assert task.assignee == "ci-static-fixer"
    assert task.evidence["routing"]["risks"] == ["session-state"]
    assert task.evidence["routing"]["requires_review"] is True
    assert "independent safety review" in task.instructions


def test_auto_dispatch_starts_an_admitted_exact_head_repair_ready_with_push_and_reply_scope(
    tmp_path: Path,
) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(
        local_path,
        not_before="2026-08-24T00:00:00Z",
        auto_dispatch=True,
    )
    github = FakeGitHub(
        admitted_pull_request(sha),
        (feedback("actionable", body="[P1] Fix the confirmed runtime regression."),),
    )
    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    control_home = tmp_path / "control home"
    result = ScanController(
        policy,
        ledger,
        github,
        kanban,
        RecordingLocalGit(),
        control_home=control_home,
    ).scan()

    assert result.created == 1
    task = kanban.tasks[0]
    assert getattr(task, "initial_status", None) == "running"
    assert getattr(task, "max_retries", None) == 2
    assert task.max_runtime_seconds == 1200
    assert "first 90 seconds" in task.instructions
    assert "do not retry a tool-blocked command" in task.instructions.casefold()
    assert "Do not keep re-evaluating equivalent approaches" in task.instructions
    assert "commit and push" in task.instructions
    assert "post a factual PR reply" in task.instructions
    assert "Do not merge" in task.instructions
    assert "still equals the expected receipt SHA" in task.instructions
    assert "complete-feedback" in task.instructions
    assert (
        f"env HERMES_HOME='{control_home}' {sys.executable} -m hermes_cli.main "
        "github-pr-feedback complete-feedback"
    ) in (
        task.instructions
    )
    assert "full literal resolved head SHA" in task.instructions
    assert (
        "<!-- pr-maintenance-receipt:v1 status=completed kind=issue_comment "
        "head=<full literal resolved head SHA> -->" in task.instructions
    )
    assert "Hermes automated repair (" in task.instructions
    ledger.close()


def test_scan_dispatches_one_read_only_exact_head_ci_audit_when_actions_are_disabled(
    tmp_path: Path,
) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(
        local_path,
        not_before="2026-08-24T00:00:00Z",
        local_ci_audit=True,
    )
    github = FakeGitHub(admitted_pull_request(sha), ())
    github.actions_are_enabled = False
    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    control_home = tmp_path / "control home"
    first = ScanController(
        policy,
        ledger,
        github,
        kanban,
        RecordingLocalGit(),
        control_home=control_home,
    ).scan()
    second = ScanController(
        policy,
        ledger,
        github,
        kanban,
        RecordingLocalGit(),
        control_home=control_home,
    ).scan()

    assert first.created == 1
    assert second.created == 0
    assert len(kanban.tasks) == 1
    task = kanban.tasks[0]
    assert task.title == "Local PR CI audit: acme/widgets#17"
    assert task.assignee == "pr-local-ci-auditor"
    assert task.initial_status == "running"
    assert task.max_retries == 3
    assert task.max_runtime_seconds == 8 * 60 * 60
    assert task.idempotency_key.endswith(":supervised-v4")
    assert task.evidence_heading == "Canonical PR audit receipt (JSON)"
    assert task.evidence == {
        "repository": "acme/widgets",
        "pr_number": 17,
        "expected_head_sha": sha,
        "github_actions_required_disabled": True,
        "post_results": True,
    }
    assert "Do not edit source files" in task.instructions
    assert "Do not publish, approve, or merge" in task.instructions
    assert "sole GitHub comment publisher" in task.instructions
    assert "Do not post a second manual summary" in task.instructions
    assert "scripts/run_hygiene_lane.py" in task.instructions
    assert "scripts/run_static_lane.py" in task.instructions
    assert "scripts/run_test_lane.py" in task.instructions
    assert "-m hermes_cli.main github-pr-feedback audit-pr" in task.instructions
    assert "background terminal process" in task.instructions
    assert "process poll or wait" in task.instructions
    assert "do not run the audit command again" in task.instructions.casefold()
    assert (
        f"env HERMES_HOME='{control_home}' {sys.executable} -m hermes_cli.main "
        "github-pr-feedback audit-pr"
    ) in (
        task.instructions
    )
    assert f"--head-sha {sha}" in task.instructions
    ledger.close()


def test_scan_reconciles_existing_failed_exact_head_receipt_to_typed_fixer(
    tmp_path: Path,
) -> None:
    local_path, head_sha = initialized_repository(tmp_path)
    base_sha = "b" * 40
    policy = configured_policy(
        local_path,
        not_before="2026-08-24T00:00:00Z",
        local_ci_audit=True,
    )
    pull_request = PullRequest(
        17,
        "OPEN",
        "acme/widgets",
        "acme/widgets",
        "owner",
        "codex/fix",
        head_sha,
        base_branch="stable",
        base_sha=base_sha,
    )
    github = FakeGitHub(pull_request, ())
    github.actions_are_enabled = False
    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    ledger.record_ci_receipt(
        CIAuditReceipt(
            receipt_id="f" * 64,
            identity=CIAuditIdentity("acme/widgets", 17, base_sha, head_sha),
            manifest_digest="e" * 64,
            status="failed",
            started_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
            completed_at=datetime(2026, 8, 25, 12, 1, tzinfo=UTC),
            actions_state=CheckState(False, True, 0),
            commands=(
                CommandEvidence(
                    argv=(".venv/bin/python", "scripts/run_static_lane.py"),
                    cwd=".",
                    returncode=1,
                    duration_ms=1,
                    timed_out=False,
                    stdout_sha256="0" * 64,
                    stderr_sha256="0" * 64,
                    classification="logic-regression",
                ),
            ),
        )
    )
    scanner = ScanController(policy, ledger, github, kanban, RecordingLocalGit())

    first = scanner.scan()
    second = scanner.scan()

    assert first.created == 1
    assert second.created == 0
    assert len(kanban.tasks) == 1
    assert kanban.tasks[0].assignee == "ci-static-fixer"
    assert kanban.tasks[0].evidence["ci_receipt_id"] == "f" * 64
    assert "background terminal process" in kanban.tasks[0].instructions
    assert "process wait" in kanban.tasks[0].instructions
    assert "in the foreground" not in kanban.tasks[0].instructions
    ledger.close()


def test_scan_does_not_dispatch_local_ci_when_github_actions_are_enabled(
    tmp_path: Path,
) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(
        local_path,
        not_before="2026-08-24T00:00:00Z",
        local_ci_audit=True,
    )
    github = FakeGitHub(admitted_pull_request(sha), ())
    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    result = ScanController(policy, ledger, github, kanban, RecordingLocalGit()).scan()

    assert result.created == 0
    assert result.skipped["github_ci_enabled"] == 1
    assert kanban.tasks == []
    ledger.close()


def test_scan_dispatches_local_ci_when_policy_requires_it_despite_hosted_actions(
    tmp_path: Path,
) -> None:
    local_path, sha = initialized_repository(tmp_path)
    raw = {
        "enabled": True,
        "repositories": [
            {
                "base_repository": "acme/widgets",
                "head_repository": "acme/widgets",
                "local_path": str(local_path),
                "owner_login": "owner",
                "branch_prefixes": ["codex/"],
            }
        ],
        "reviewer_logins": ["owner"],
        "reviewer_associations": [],
        "not_before": "2026-08-24T00:00:00Z",
        "assignee": "repair-agent",
        "board": "repairs",
        "local_ci_audit": {
            "enabled": True,
            "assignee": "pr-local-ci-auditor",
            "post_results": True,
            "required_for_open_prs": True,
        },
    }
    policy = load_policy(raw)
    github = FakeGitHub(admitted_pull_request(sha), ())
    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    result = ScanController(policy, ledger, github, kanban, RecordingLocalGit()).scan()

    assert result.created == 1
    assert [task.title for task in kanban.tasks] == ["Local PR CI audit: acme/widgets#17"]
    assert "GitHub Actions remain disabled" not in kanban.tasks[0].instructions
    assert "may remain enabled" in kanban.tasks[0].instructions
    assert kanban.tasks[0].evidence["github_actions_required_disabled"] is False
    ledger.close()


def test_scan_dispatches_local_ci_oldest_backlog_first(tmp_path: Path) -> None:
    local_path, sha = initialized_repository(tmp_path)
    older = PullRequest(
        17, "OPEN", "acme/widgets", "acme/widgets", "owner", "codex/older", sha
    )
    newer = PullRequest(
        18, "OPEN", "acme/widgets", "acme/widgets", "owner", "codex/newer", sha
    )
    policy = configured_policy(
        local_path,
        not_before="2026-08-24T00:00:00Z",
        local_ci_audit=True,
    )
    github = FakeGitHub(newer, (), pull_requests=(newer, older))
    github.actions_are_enabled = False
    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    result = ScanController(policy, ledger, github, kanban, RecordingLocalGit()).scan()

    assert result.created == 1
    assert result.skipped["local_ci_dispatch_cap"] == 1
    assert [task.title for task in kanban.tasks] == [
        "Local PR CI audit: acme/widgets#17",
    ]
    ledger.close()


def test_scan_bounds_newest_first_per_pr_reads_for_local_ci(tmp_path: Path) -> None:
    local_path, sha = initialized_repository(tmp_path)
    older = PullRequest(
        17, "OPEN", "acme/widgets", "acme/widgets", "owner", "codex/older", sha
    )
    newer = PullRequest(
        18, "OPEN", "acme/widgets", "acme/widgets", "owner", "codex/newer", sha
    )
    raw = {
        "enabled": True,
        "repositories": [
            {
                "base_repository": "acme/widgets",
                "head_repository": "acme/widgets",
                "local_path": str(local_path),
                "owner_login": "owner",
                "branch_prefixes": ["codex/"],
            }
        ],
        "reviewer_logins": ["reviewer"],
        "reviewer_associations": [],
        "not_before": "2026-08-24T00:00:00Z",
        "assignee": "repair-agent",
        "board": "repairs",
        "local_ci_audit": {
            "enabled": True,
            "assignee": "pr-local-ci-auditor",
            "post_results": True,
            "required_for_open_prs": True,
            "max_open_prs_per_scan": 1,
        },
    }
    github = FakeGitHub(newer, (), pull_requests=(newer, older))
    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    result = ScanController(
        load_policy(raw), ledger, github, kanban, RecordingLocalGit()
    ).scan()

    assert result.created == 1
    assert result.skipped["local_ci_open_pr_scan_cap"] == 1
    assert github.feedback_calls == [("acme/widgets", 17), ("acme/widgets", 18)]
    assert github.current_calls == [("acme/widgets", 17)]
    ledger.close()


def test_local_ci_catalogue_selects_oldest_missing_head_beyond_freshness_window(
    tmp_path: Path,
) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(
        local_path,
        not_before="2026-08-24T00:00:00Z",
        local_ci_audit=True,
        max_open_prs_per_scan=12,
    )
    pulls = tuple(
        PullRequest(
            number,
            "OPEN",
            "acme/widgets",
            "acme/widgets",
            "owner",
            f"codex/pr-{number}",
            sha,
            updated_at=datetime(2026, 8, 25, 12, number - 700, tzinfo=UTC),
        )
        for number in range(700, 714)
    )
    # #715 is deliberately outside the ordinary newest-PR read window.
    oldest = PullRequest(
        715,
        "OPEN",
        "acme/widgets",
        "acme/widgets",
        "owner",
        "codex/old-backlog",
        "b" * 40,
        updated_at=datetime(2026, 8, 24, 0, 0, tzinfo=UTC),
    )
    github = FakeGitHub(pulls[0], (), pull_requests=(*pulls, oldest))
    github.actions_are_enabled = False
    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    result = ScanController(policy, ledger, github, kanban, RecordingLocalGit()).scan()

    assert result.created == 1
    assert [task.evidence["pr_number"] for task in kanban.tasks] == [715]
    assert result.skipped["local_ci_open_pr_scan_cap"] == 3
    ledger.close()


def test_failed_local_ci_dispatch_due_is_reclaimed_without_duplicate(tmp_path: Path) -> None:
    local_path, sha = initialized_repository(tmp_path)
    now = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)
    policy = configured_policy(
        local_path, not_before="2026-08-24T00:00:00Z", local_ci_audit=True
    )
    pull = admitted_pull_request(sha)
    github = FakeGitHub(pull, ())
    github.actions_are_enabled = False
    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    receipt = FeedbackReceipt(
        "acme/widgets", 17, "pr_local_ci", LOCAL_CI_FEEDBACK_ID, sha
    )
    lease = ledger.claim(
        receipt,
        owner="previous-dispatch",
        claimed_at=now - timedelta(minutes=10),
        stale_before=now - timedelta(minutes=15),
    )
    assert lease is not None
    ledger.fail(receipt, "worktree pool was temporarily full", lease)

    result = ScanController(
        policy, ledger, github, kanban, RecordingLocalGit(), clock=lambda: now
    ).scan()

    assert result.created == 1
    assert result.skipped.get("duplicate", 0) == 0
    assert len(kanban.tasks) == 1
    assert ledger.exact_receipt_status(receipt) == "completed"
    ledger.close()


def test_failed_local_ci_dispatch_backoff_is_visible_and_does_not_consume_slot(
    tmp_path: Path,
) -> None:
    local_path, sha = initialized_repository(tmp_path)
    now = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)
    policy = configured_policy(
        local_path, not_before="2026-08-24T00:00:00Z", local_ci_audit=True
    )
    pull = admitted_pull_request(sha)
    github = FakeGitHub(pull, ())
    github.actions_are_enabled = False
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    receipt = FeedbackReceipt(
        "acme/widgets", 17, "pr_local_ci", LOCAL_CI_FEEDBACK_ID, sha
    )
    lease = ledger.claim(
        receipt,
        owner="previous-dispatch",
        claimed_at=now,
        stale_before=now - timedelta(minutes=5),
    )
    assert lease is not None
    ledger.fail(receipt, "transient dispatch failure", lease)

    result = ScanController(
        policy, ledger, github, RecordingKanban(), RecordingLocalGit(), clock=lambda: now
    ).scan()

    assert result.created == 0
    assert result.skipped["retry_backoff"] == 1
    assert result.skipped.get("local_ci_dispatch_cap", 0) == 0
    assert ledger.exact_receipt_status(receipt) == "failed"
    ledger.close()


def test_required_local_ci_backlog_signal_counts_missing_receipts_below_read_cap(
    tmp_path: Path,
) -> None:
    local_path, sha = initialized_repository(tmp_path)
    manifest = local_path / "tests" / "manifests" / "test_lanes.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("version = 1\n", encoding="utf-8")
    base_sha = "b" * 40
    older = PullRequest(
        17,
        "OPEN",
        "acme/widgets",
        "acme/widgets",
        "owner",
        "codex/older",
        sha,
        base_branch="stable",
        base_sha=base_sha,
    )
    newer = PullRequest(
        18,
        "OPEN",
        "acme/widgets",
        "acme/widgets",
        "owner",
        "codex/newer",
        sha,
        base_branch="stable",
        base_sha=base_sha,
    )
    raw = {
        "enabled": True,
        "repositories": [
            {
                "base_repository": "acme/widgets",
                "head_repository": "acme/widgets",
                "local_path": str(local_path),
                "owner_login": "owner",
                "branch_prefixes": ["codex/"],
            }
        ],
        "reviewer_logins": ["reviewer"],
        "reviewer_associations": [],
        "not_before": "2026-08-24T00:00:00Z",
        "assignee": "repair-agent",
        "board": "repairs",
        "local_ci_audit": {
            "enabled": True,
            "assignee": "pr-local-ci-auditor",
            "post_results": True,
            "required_for_open_prs": True,
            "max_dispatches_per_scan": 1,
            "max_open_prs_per_scan": 300,
        },
    }
    github = FakeGitHub(newer, (), pull_requests=(newer, older))
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    result = ScanController(
        load_policy(raw), ledger, github, RecordingKanban(), RecordingLocalGit()
    ).scan()

    assert getattr(result, "required_local_ci_backlog", 0) == 2
    assert result.skipped.get("local_ci_open_pr_scan_cap", 0) == 0
    assert github.feedback_calls == [("acme/widgets", 17), ("acme/widgets", 18)]
    assert github.current_calls == [("acme/widgets", 17)]
    ledger.close()


def test_required_local_ci_backlog_signal_ignores_read_cap_when_receipts_are_current(
    tmp_path: Path,
) -> None:
    local_path, sha = initialized_repository(tmp_path)
    manifest = local_path / "tests" / "manifests" / "test_lanes.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("version = 1\n", encoding="utf-8")
    manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    base_sha = "b" * 40
    pulls = tuple(
        PullRequest(
            number,
            "OPEN",
            "acme/widgets",
            "acme/widgets",
            "owner",
            f"codex/pr-{number}",
            sha,
            base_branch="stable",
            base_sha=base_sha,
        )
        for number in (17, 18)
    )
    raw = {
        "enabled": True,
        "repositories": [
            {
                "base_repository": "acme/widgets",
                "head_repository": "acme/widgets",
                "local_path": str(local_path),
                "owner_login": "owner",
                "branch_prefixes": ["codex/"],
            }
        ],
        "reviewer_logins": ["reviewer"],
        "reviewer_associations": [],
        "not_before": "2026-08-24T00:00:00Z",
        "assignee": "repair-agent",
        "board": "repairs",
        "local_ci_audit": {
            "enabled": True,
            "assignee": "pr-local-ci-auditor",
            "post_results": True,
            "required_for_open_prs": True,
            "max_open_prs_per_scan": 1,
        },
    }
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    recorded_at = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)
    for pull in pulls:
        ledger.record_ci_receipt(
            CIAuditReceipt(
                receipt_id=f"{pull.number - 16}" * 64,
                identity=CIAuditIdentity(
                    "acme/widgets", pull.number, base_sha, pull.head_sha
                ),
                manifest_digest=manifest_digest,
                status="passed",
                started_at=recorded_at,
                completed_at=recorded_at + timedelta(minutes=1),
                actions_state=CheckState(True, True, 1),
                commands=(),
            )
        )
        feedback_receipt = FeedbackReceipt(
            "acme/widgets",
            pull.number,
            "pr_local_ci",
            f"{LOCAL_CI_FEEDBACK_ID}:{base_sha}",
            pull.head_sha,
        )
        lease = ledger.claim(
            feedback_receipt,
            owner="seed",
            claimed_at=recorded_at,
            stale_before=recorded_at - timedelta(minutes=5),
        )
        assert lease is not None
        ledger.finalize(feedback_receipt, f"audit-{pull.number}", lease)
    github = FakeGitHub(pulls[0], (), pull_requests=pulls)

    result = ScanController(
        load_policy(raw), ledger, github, RecordingKanban(), RecordingLocalGit()
    ).scan()

    assert getattr(result, "required_local_ci_backlog", 0) == 0
    assert result.skipped.get("local_ci_open_pr_scan_cap", 0) == 0
    assert github.feedback_calls == [("acme/widgets", 18)]
    assert github.current_calls == []
    ledger.close()


def test_scan_dispatches_local_ci_when_actions_are_enabled_but_billing_blocked(
    tmp_path: Path,
) -> None:
    """The repo-level scan() gate reads actions_enabled() (a repo setting)

    separately from the per-PR dispatch paths -- it needs its own billing
    check, sampled from one open PR's check state, or it silently believes
    GitHub CI is fine for an entire repository stuck in a billing lockout.
    """

    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(
        local_path,
        not_before="2026-08-24T00:00:00Z",
        local_ci_audit=True,
    )
    github = FakeGitHub(admitted_pull_request(sha), ())
    github.actions_are_enabled = True
    github.billing_blocked = True
    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    result = ScanController(policy, ledger, github, kanban, RecordingLocalGit()).scan()

    assert result.skipped.get("github_ci_enabled", 0) == 0
    assert result.created == 1
    assert [task.title for task in kanban.tasks] == ["Local PR CI audit: acme/widgets#17"]
    ledger.close()


def test_scan_fails_closed_and_reports_degraded_when_actions_state_is_unavailable(
    tmp_path: Path,
) -> None:
    class UnavailableActionsGitHub(FakeGitHub):
        def actions_enabled(self, repository: str) -> bool:
            raise RuntimeError(f"cannot read {repository}")

    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(
        local_path,
        not_before="2026-08-24T00:00:00Z",
        local_ci_audit=True,
    )
    github = UnavailableActionsGitHub(admitted_pull_request(sha), ())
    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    result = ScanController(policy, ledger, github, kanban, RecordingLocalGit()).scan()

    assert result.created == 0
    assert result.skipped["github_ci_state_unavailable"] == 1
    assert result.degraded is True
    assert kanban.tasks == []
    ledger.close()


def test_scan_dispatches_a_new_local_ci_audit_when_the_pr_head_changes(
    tmp_path: Path,
) -> None:
    local_path, first_sha = initialized_repository(tmp_path)
    policy = configured_policy(
        local_path,
        not_before="2026-08-24T00:00:00Z",
        local_ci_audit=True,
    )
    github = FakeGitHub(admitted_pull_request(first_sha), ())
    github.actions_are_enabled = False
    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    scanner = ScanController(policy, ledger, github, kanban, RecordingLocalGit())

    first = scanner.scan()
    second_sha = "b" * 40
    github.pull_request = admitted_pull_request(second_sha)
    github.current = github.pull_request
    second = scanner.scan()

    assert first.created == 1
    assert second.created == 1
    assert [task.head_sha for task in kanban.tasks] == [first_sha, second_sha]
    assert kanban.tasks[0].idempotency_key != kanban.tasks[1].idempotency_key
    ledger.close()


def test_scan_dispatches_a_new_local_ci_audit_when_only_the_base_head_changes(
    tmp_path: Path,
) -> None:
    local_path, head_sha = initialized_repository(tmp_path)
    policy = configured_policy(
        local_path,
        not_before="2026-08-24T00:00:00Z",
        local_ci_audit=True,
    )
    first = PullRequest(
        17,
        "OPEN",
        "acme/widgets",
        "acme/widgets",
        "owner",
        "codex/fix",
        head_sha,
        base_branch="stable",
        base_sha="1" * 40,
    )
    github = FakeGitHub(first, ())
    github.actions_are_enabled = False
    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    scanner = ScanController(policy, ledger, github, kanban, RecordingLocalGit())

    first_scan = scanner.scan()
    second = PullRequest(
        17,
        "OPEN",
        "acme/widgets",
        "acme/widgets",
        "owner",
        "codex/fix",
        head_sha,
        base_branch="stable",
        base_sha="2" * 40,
    )
    github.pull_request = second
    github.current = second
    second_scan = scanner.scan()
    duplicate_scan = scanner.scan()

    assert first_scan.created == 1
    assert second_scan.created == 1
    assert duplicate_scan.created == 0
    assert [task.head_sha for task in kanban.tasks] == [head_sha, head_sha]
    assert kanban.tasks[0].idempotency_key != kanban.tasks[1].idempotency_key
    ledger.close()


def test_local_ci_waits_while_admitted_feedback_is_unactioned(tmp_path: Path) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(
        local_path,
        not_before="2026-08-24T00:00:00Z",
        local_ci_audit=True,
    )
    github = FakeGitHub(admitted_pull_request(sha), (feedback("needs-fix"),))
    github.actions_are_enabled = False
    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    result = ScanController(policy, ledger, github, kanban, RecordingLocalGit()).scan()

    assert result.created == 1
    assert result.skipped["feedback_pending"] == 1
    assert [task.title for task in kanban.tasks] == ["GitHub PR feedback: acme/widgets#17"]
    ledger.close()


def test_local_ci_runs_after_all_admitted_feedback_is_actioned(tmp_path: Path) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(
        local_path,
        not_before="2026-08-24T00:00:00Z",
        local_ci_audit=True,
    )
    item = feedback("fixed")
    github = FakeGitHub(admitted_pull_request(sha), (item,))
    github.actions_are_enabled = False
    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    receipt = FeedbackReceipt("acme/widgets", 17, item.kind, item.feedback_id, sha)
    now = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)
    lease = ledger.claim(
        receipt,
        owner="seed",
        claimed_at=now,
        stale_before=now - timedelta(minutes=5),
    )
    assert lease is not None
    ledger.finalize(receipt, "feedback-task", lease)
    ledger.mark_feedback_actioned(receipt, resolved_head_sha=sha, actioned_at=now)

    result = ScanController(policy, ledger, github, kanban, RecordingLocalGit()).scan()

    assert result.created == 1
    assert result.skipped["already_actioned"] == 1
    assert result.skipped.get("feedback_pending", 0) == 0
    assert [task.title for task in kanban.tasks] == ["Local PR CI audit: acme/widgets#17"]
    ledger.close()


def test_completed_feedback_immediately_schedules_exact_head_local_ci(tmp_path: Path) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(
        local_path,
        not_before="2026-08-24T00:00:00Z",
        auto_dispatch=True,
        local_ci_audit=True,
    )
    item = feedback("fixed")
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    receipt = FeedbackReceipt("acme/widgets", 17, item.kind, item.feedback_id, sha)
    lease = ledger.claim(
        receipt,
        owner="feedback-worker",
        claimed_at=datetime(2026, 8, 24, 1, 0, tzinfo=UTC),
        stale_before=datetime(2026, 8, 24, 0, 55, tzinfo=UTC),
    )
    assert lease is not None
    ledger.finalize(receipt, "feedback-task", lease)
    ledger.mark_feedback_actioned(
        receipt,
        resolved_head_sha=sha,
        actioned_at=datetime(2026, 8, 24, 2, 0, tzinfo=UTC),
    )
    kanban = RecordingKanban()
    github = FakeGitHub(admitted_pull_request(sha), (item,))
    github.actions_are_enabled = False
    controller = ScanController(
        policy,
        ledger,
        github,
        kanban,
        RecordingLocalGit(),
    )

    status = controller.dispatch_local_ci_after_feedback(admitted_pull_request(sha))

    assert status == "scheduled"
    assert [task.title for task in kanban.tasks] == ["Local PR CI audit: acme/widgets#17"]
    ledger.close()


def test_completed_feedback_schedules_local_ci_when_actions_are_billing_blocked(
    tmp_path: Path,
) -> None:
    """Actions enabled as a repo setting but every check billing-blocked must not

    read as "GitHub CI is fine" — local CI must still be dispatched.
    """

    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(
        local_path,
        not_before="2026-08-24T00:00:00Z",
        auto_dispatch=True,
        local_ci_audit=True,
    )
    item = feedback("fixed")
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    receipt = FeedbackReceipt("acme/widgets", 17, item.kind, item.feedback_id, sha)
    lease = ledger.claim(
        receipt,
        owner="feedback-worker",
        claimed_at=datetime(2026, 8, 24, 1, 0, tzinfo=UTC),
        stale_before=datetime(2026, 8, 24, 0, 55, tzinfo=UTC),
    )
    assert lease is not None
    ledger.finalize(receipt, "feedback-task", lease)
    ledger.mark_feedback_actioned(
        receipt,
        resolved_head_sha=sha,
        actioned_at=datetime(2026, 8, 24, 2, 0, tzinfo=UTC),
    )
    kanban = RecordingKanban()
    github = FakeGitHub(admitted_pull_request(sha), (item,))
    github.actions_are_enabled = True
    github.billing_blocked = True
    controller = ScanController(
        policy,
        ledger,
        github,
        kanban,
        RecordingLocalGit(),
    )

    status = controller.dispatch_local_ci_after_feedback(admitted_pull_request(sha))

    assert status == "scheduled"
    assert [task.title for task in kanban.tasks] == ["Local PR CI audit: acme/widgets#17"]
    ledger.close()


def test_duplicate_local_ci_receipts_do_not_starve_a_new_head_after_comment_fixes(
    tmp_path: Path,
) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(
        local_path,
        not_before="2026-08-24T00:00:00Z",
        local_ci_audit=True,
    )
    stale_pulls = tuple(
        PullRequest(number, "OPEN", "acme/widgets", "acme/widgets", "owner", "codex/fix", sha)
        for number in range(1, MAX_ADMISSIONS_PER_SCAN + 1)
    )
    repaired = PullRequest(
        MAX_ADMISSIONS_PER_SCAN + 1,
        "OPEN",
        "acme/widgets",
        "acme/widgets",
        "owner",
        "codex/comment-fix",
        "b" * 40,
    )

    class ManyPullsGitHub(FakeGitHub):
        def list_open_pull_requests(self, repository: str, owner_login: str):
            return (*stale_pulls, repaired)

        def get_pull_request(self, repository: str, number: int):
            return next(pull for pull in (*stale_pulls, repaired) if pull.number == number)

        def list_feedback(self, repository: str, number: int):
            return ()

    github = ManyPullsGitHub(stale_pulls[0], ())
    github.actions_are_enabled = False
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    claimed_at = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)
    for pull in stale_pulls:
        receipt = FeedbackReceipt(
            "acme/widgets", pull.number, "pr_local_ci", LOCAL_CI_FEEDBACK_ID, pull.head_sha
        )
        lease = ledger.claim(
            receipt,
            owner="seed",
            claimed_at=claimed_at,
            stale_before=claimed_at - timedelta(minutes=5),
        )
        assert lease is not None
        ledger.finalize(receipt, f"done-{pull.number}", lease)
    kanban = RecordingKanban()

    result = ScanController(policy, ledger, github, kanban, RecordingLocalGit()).scan()

    assert result.created == 1
    assert result.skipped["local_ci_exact_head_deferred"] == MAX_ADMISSIONS_PER_SCAN
    assert result.skipped.get("admission_cap", 0) == 0
    assert [task.evidence["pr_number"] for task in kanban.tasks] == [repaired.number]
    ledger.close()


def test_scan_refetches_head_before_dispatching_a_local_ci_audit(
    tmp_path: Path,
) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(
        local_path,
        not_before="2026-08-24T00:00:00Z",
        local_ci_audit=True,
    )
    github = FakeGitHub(
        admitted_pull_request(sha),
        (),
        current=admitted_pull_request("b" * 40),
    )
    github.actions_are_enabled = False
    kanban = RecordingKanban()
    local_git = RecordingLocalGit()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    result = ScanController(policy, ledger, github, kanban, local_git).scan()

    assert result.created == 0
    assert result.skipped["head_changed"] == 1
    assert local_git.calls == []
    assert kanban.tasks == []
    ledger.close()


def test_scan_suppresses_high_confidence_self_resolution_receipts(tmp_path: Path) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(local_path, not_before="2026-08-24T00:00:00Z")
    github = FakeGitHub(
        admitted_pull_request(sha),
        (
            feedback(
                "addressed",
                reviewer="owner",
                body="Addressed the review at exact head abc123. Verification passed.",
            ),
            feedback(
                "superseded",
                reviewer="owner",
                body="Confirmed the gap. This narrower PR is superseded by #22.",
            ),
            feedback(
                "worker-completion-receipt",
                reviewer="owner",
                body=(
                    "Resolved the PR-introduced static failures in abc123. "
                    "Verification: 8 focused tests passed. No merge performed."
                ),
            ),
            feedback(
                "static-worker-completion-receipt",
                reviewer="owner",
                body=(
                    "## Static lane fix — verified\n\nFixed the reported static failures.\n\n"
                    "**Commit:** `abc123`\n\n**Focused verification:** flake8 passed.\n\n"
                    "**Files changed:** 1 file."
                ),
            ),
            feedback(
                "still-actionable",
                reviewer="owner",
                body="Fixed the first case, but one blocker remains and still needs work.",
            ),
        ),
    )
    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    result = ScanController(policy, ledger, github, kanban, RecordingLocalGit()).scan()

    assert result.created == 1
    assert result.skipped["self_resolution_receipt"] == 3
    assert result.skipped["duplicate"] == 1
    assert [task.evidence["feedback_id"] for task in kanban.tasks] == ["superseded"]
    ledger.close()


def test_scan_suppresses_owner_ci_receipt_marker_without_profile_local_ledger(
    tmp_path: Path,
) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(local_path, not_before="2026-08-24T00:00:00Z")
    item = feedback(
        "ci-result",
        reviewer="owner",
        body=(
            "Addressed local CI audit. Authoritative receipt: "
            f"`{'d' * 64}`.\n\n"
            f"<!-- pr-ci-receipt:v1 status=passed id={'d' * 64} head={sha} -->"
        ),
    )
    ledger = FeedbackLedger(tmp_path / "scanner-ledger.sqlite3")
    kanban = RecordingKanban()

    result = ScanController(
        policy,
        ledger,
        FakeGitHub(admitted_pull_request(sha), (item,)),
        kanban,
        RecordingLocalGit(),
    ).scan()

    assert result.created == 0
    assert result.skipped["self_ci_receipt"] == 1
    assert kanban.tasks == []
    ledger.close()


@pytest.mark.parametrize(
    "body",
    [
        (
            "Thanks for the careful review — points 1 and 3 are addressed in "
            f"{'a' * 40}. Focused verification: 3 tests pass."
        ),
        (
            f"Pushed {'b' * 40} addressing both review points. Verification: "
            "40 tests passed, 0 failed."
        ),
        (
            "Thanks for the careful review — both points were checked against the tree "
            f"at {'c' * 40}; neither required a code change."
        ),
        (
            "Updated the existing branch against current main with the complete exact-head "
            "repair/CI/merge continuation series. Fresh verification: 335 plugin tests passed; "
            "Ruff and git diff --check passed."
        ),
        (
            "Integrated the latest approved follow-ups in two commits. Fresh verification on "
            "the updated head: 356 focused tests passed, Ruff passed, and git diff --check passed."
        ),
        (
            "Exact-head verification of merge head "
            f"{'d' * 40} in a clean worktree confirms the re-review findings. "
            "A runtime import check passes. No code changes were made at this head."
        ),
    ],
)
def test_self_resolution_suppresses_semantic_owner_completion_receipts(body: str) -> None:
    item = feedback("semantic-owner-completion", reviewer="owner", body=body)

    assert _is_self_resolution_receipt(item, owner_login="owner") is True


def test_self_resolution_does_not_trust_unverified_supersession_claim() -> None:
    item = feedback(
        "unverified-supersession",
        reviewer="owner",
        body="Confirmed the gap. This narrower PR is superseded by #94495.",
    )

    assert _is_self_resolution_receipt(item, owner_login="owner") is False


@pytest.mark.parametrize(
    ("kind", "body"),
    [
        (
            "issue_comment",
            "Local-CI static-lane repair for this PR is in place at commit {sha} "
            "(current PR head). Re-validated today on the exact receipt tree: "
            "run_static_lane.py status: pass, rc=0; zero findings. No merge was "
            "performed; merge remains controlled by the deterministic safety gates.",
        ),
        (
            "review_comment",
            "Local CI repair for receipt e7cd950e landed at head {sha} "
            "(fast-forward push to this branch; no gate, cap, baseline, or manifest "
            "relaxed). Evidence: reproduced at the pristine receipt head; with the fix, "
            "run_hygiene_lane.py rc=0, all 42 checks passed; run_static_lane.py rc=0, "
            "status=pass.",
        ),
        (
            "issue_comment",
            "Repaired the local-CI static-lane failure reported for exact head "
            "`6d5e11b8d62429e1c3042086f9269b539475c3d4`. Root cause and fix "
            "(commit `50d0baa40effd44085194438fc9fab73381efa94`, 2 files changed). "
            "Verification, all at commit `50d0baa40effd44085194438fc9fab73381efa94`; "
            "no CI configuration, required checks, or safety gates were modified: "
            "run_static_lane.py -> status: pass, errors: []; focused pytest -> 14 passed.",
        ),
        (
            "issue_comment",
            "Verification at head {sha} (current canonical PR head): merge parents "
            "confirmed with no history rewrite. Focused verification: 14 tests passed.",
        ),
        (
            "issue_comment",
            "Validated against exact head {sha}. The reported static failure no longer "
            "reproduces. Focused verification: run_static_lane.py status=pass. No source "
            "files were changed.",
        ),
        (
            "issue_comment",
            "Static-lane repair pushed at {sha} (head was re-read before any write). "
            "Changes (3 files): formatting and scoped lint provenance only. Verification: "
            "run_static_lane.py -> pass; run_hygiene_lane.py -> pass; runtime tests passed. "
            "Merge remains gated; no CI configuration, required checks, or safety gates were "
            "modified.",
        ),
        (
            "issue_comment",
            "Re: local PR CI audit receipt 4743945a -- already resolved at this PR's "
            "current head. Validated at {sha} (canonical PR head): "
            "scripts/run_hygiene_lane.py exits 0 -- all 42 admission checks pass. "
            "The failures predate the shared stable-base repair and are fixed by lineage "
            "with no edit to this branch. No gates or caps weakened; no merge action taken.",
        ),
    ],
)
def test_scan_suppresses_factual_owner_ci_completion_comments(
    tmp_path: Path,
    kind: str,
    body: str,
) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(local_path, not_before="2026-08-24T00:00:00Z")
    owner = Reviewer("owner", "MEMBER")
    github = FakeGitHub(
        admitted_pull_request(sha),
        (
            Feedback(
                kind,
                "factual-owner-completion",
                owner,
                body.format(sha=sha),
                datetime.fromisoformat("2026-08-24T00:00:00+00:00"),
                False,
            ),
        ),
    )
    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    result = ScanController(policy, ledger, github, kanban, RecordingLocalGit()).scan()

    assert result.created == 0
    assert result.skipped["self_resolution_receipt"] == 1
    assert kanban.tasks == []
    ledger.close()


@pytest.mark.parametrize(
    ("kind", "reviewer", "is_bot", "body"),
    [
        (
            "issue_comment",
            "owner",
            False,
            "Please perform a local CI repair for this PR at current head {sha}. "
            "The static lane currently fails. Do not merge.",
        ),
        (
            "review_comment",
            "ci-review-bot",
            True,
            "Local CI repair for this PR is in place at commit {sha}. Static lane "
            "verification passed. No merge was performed.",
        ),
        (
            "issue_comment",
            "owner",
            False,
            "Local-CI static-lane repair for this PR is in place at commit {sha}. "
            "Static lane verification passed and no merge was performed, but one "
            "blocker remains.",
        ),
        (
            "issue_comment",
            "owner",
            False,
            "Static-lane repair pushed at {sha}. Changes (1 file). Verification: "
            "unrelated-test -> pass.",
        ),
    ],
)
def test_scan_keeps_owner_ci_repair_requests_and_non_owner_bot_comments(
    tmp_path: Path,
    kind: str,
    reviewer: str,
    is_bot: bool,
    body: str,
) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(local_path, not_before="2026-08-24T00:00:00Z")
    github = FakeGitHub(
        admitted_pull_request(sha),
        (
            Feedback(
                kind,
                "genuine-feedback",
                Reviewer(reviewer, "MEMBER"),
                body.format(sha=sha),
                datetime.fromisoformat("2026-08-24T00:00:00+00:00"),
                is_bot,
            ),
        ),
    )
    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    result = ScanController(policy, ledger, github, kanban, RecordingLocalGit()).scan()

    assert result.created == 1
    assert result.skipped.get("self_resolution_receipt", 0) == 0
    assert [task.evidence["feedback_id"] for task in kanban.tasks] == ["genuine-feedback"]
    ledger.close()


@pytest.mark.parametrize(
    "action_language",
    [
        "The current static lane is still failing.",
        "This needs fixing.",
        "This needs repair.",
        "One remaining failure requires action.",
        "Two remaining failures require action.",
        "The current static lane fails.",
        "The current static lane still fails.",
    ],
)
def test_self_resolution_fail_opens_for_bounded_unresolved_action_language(
    action_language: str,
) -> None:
    item = feedback(
        "unresolved-completion-shape",
        reviewer="owner",
        body=(
            "Local-CI static-lane repair for this PR is in place at commit "
            f"{'a' * 40} (current PR head). Re-validated on the exact receipt tree: "
            "run_static_lane.py status: pass, rc=0. No merge was performed. "
            f"{action_language}"
        ),
    )

    assert _is_self_resolution_receipt(item, owner_login="owner") is False


def test_self_resolution_fail_opens_when_the_github_body_reaches_the_intake_cap() -> None:
    prefix = (
        "Local-CI static-lane repair for this PR is in place at commit "
        f"{'a' * 40} (current PR head). Re-validated on the exact receipt tree: "
        "run_static_lane.py status: pass, rc=0. No merge was performed. "
    )
    body = prefix + "x" * (MAX_FEEDBACK_BODY_CHARS - len(prefix))
    assert len(body) == MAX_FEEDBACK_BODY_CHARS
    item = feedback("capped-completion-shape", reviewer="owner", body=body)

    assert _is_self_resolution_receipt(item, owner_login="owner") is False


def test_self_resolution_keeps_factual_historic_failure_reproduction() -> None:
    item = feedback(
        "historic-reproduction",
        reviewer="owner",
        body=(
            "Local CI repair for receipt e7cd950e landed at head "
            f"{'a' * 40}. Evidence: reproduced at the pristine receipt head where "
            "run_static_lane.py fails; with the fix, run_static_lane.py status=pass, "
            "all checks passed. No merge was performed."
        ),
    )

    assert _is_self_resolution_receipt(item, owner_login="owner") is True


def test_self_resolution_accepts_same_lane_status_equals_pass_evidence() -> None:
    item = feedback(
        "historic-status-equals-pass",
        reviewer="owner",
        body=(
            "Local CI repair for receipt e7cd950e landed at head "
            f"{'a' * 40}. Evidence: reproduced at the pristine receipt where "
            "run_static_lane.py fails; with the fix, run_static_lane.py status=pass. "
            "No merge was performed."
        ),
    )

    assert _is_self_resolution_receipt(item, owner_login="owner") is True


@pytest.mark.parametrize(
    "evidence",
    [
        (
            "reproduced at the pristine receipt where run_static_lane.py fails and "
            "still fails after the fix; with the fix, unrelated focused checks passed."
        ),
        (
            "reproduced at the pristine receipt where run_static_lane.py fails; "
            "with the fix, unrelated focused checks passed."
        ),
    ],
)
def test_self_resolution_does_not_treat_unrelated_passes_as_same_lane_resolution(
    evidence: str,
) -> None:
    item = feedback(
        "ambiguous-historic-reproduction",
        reviewer="owner",
        body=(
            "Local CI repair for receipt e7cd950e landed at head "
            f"{'a' * 40}. Evidence: {evidence} No merge was performed."
        ),
    )

    assert _is_self_resolution_receipt(item, owner_login="owner") is False


@pytest.mark.parametrize(
    ("kind", "body"),
    [
        (
            "issue_comment",
            "Local-CI static-lane repair for this PR is in place at commit "
            f"{'a' * 40} (current PR head). Re-validated on the receipt tree: "
            "run_static_lane.py status: pass. No merge was performed.",
        ),
        (
            "review_comment",
            "Local CI repair for receipt e7cd950e landed at head "
            f"{'a' * 40}. Evidence: run_hygiene_lane.py rc=0, all checks passed; "
            "no gate was relaxed.",
        ),
        (
            "issue_comment",
            "Repaired the local-CI static-lane failure reported for exact head "
            f"`{'b' * 40}`. Root cause and fix (commit `{'a' * 40}`, 2 files changed). "
            f"Verification, all at commit `{'a' * 40}`; no CI configuration, required "
            "checks, or safety gates were modified: run_static_lane.py status: pass; "
            "focused pytest -> 14 passed.",
        ),
    ],
)
def test_exact_live_completion_shapes_remain_admitted_when_authored_by_a_non_owner_bot(
    kind: str,
    body: str,
) -> None:
    item = Feedback(
        kind,
        "non-owner-live-shape",
        Reviewer("ci-repair-bot", "MEMBER"),
        body,
        datetime.fromisoformat("2026-08-24T00:00:00+00:00"),
        True,
    )

    assert _is_self_resolution_receipt(item, owner_login="owner") is False


@pytest.mark.parametrize(
    ("action_language", "expected_title", "expected_pending"),
    [
        ("", "Local PR CI audit: acme/widgets#17", 0),
        (" The current static lane still fails.", "GitHub PR feedback: acme/widgets#17", 1),
    ],
)
def test_local_ci_ordering_ignores_only_completed_owner_receipts(
    tmp_path: Path,
    action_language: str,
    expected_title: str,
    expected_pending: int,
) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(
        local_path,
        not_before="2026-08-24T00:00:00Z",
        local_ci_audit=True,
    )
    item = feedback(
        "owner-completion",
        reviewer="owner",
        body=(
            "Local-CI static-lane repair for this PR is in place at commit "
            f"{sha} (current PR head). Re-validated on the exact receipt tree: "
            "run_static_lane.py status: pass, rc=0. No merge was performed."
            f"{action_language}"
        ),
    )
    github = FakeGitHub(admitted_pull_request(sha), (item,))
    github.actions_are_enabled = False
    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    result = ScanController(policy, ledger, github, kanban, RecordingLocalGit()).scan()

    assert result.created == 1
    assert result.skipped.get("feedback_pending", 0) == expected_pending
    assert [task.title for task in kanban.tasks] == [expected_title]
    ledger.close()


def test_scan_suppresses_base_inherited_self_audits_but_keeps_pr_regressions(
    tmp_path: Path,
) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(local_path, not_before="2026-08-24T00:00:00Z")
    github = FakeGitHub(
        admitted_pull_request(sha),
        (
            feedback(
                "base-inherited",
                reviewer="owner",
                body=(
                    "Local PR CI audit: the hygiene failures are pre-existing and "
                    "reproduce on the stable base. Recommendation: use a separate "
                    "repair card; no PR-head repair was performed."
                ),
            ),
            feedback(
                "already-fixed",
                reviewer="owner",
                body=(
                    "Validated at the receipt head. No additional changes required; "
                    "focused verification passed."
                ),
            ),
            feedback(
                "confirmed-no-code-change",
                reviewer="owner",
                body=(
                    "Confirmed resolved at the exact PR head. No further code change "
                    "required; no new commit pushed."
                ),
            ),
            feedback(
                "confirmed-no-source-change",
                reviewer="owner",
                body=(
                    "Verified independently at the receipt worktree. Confirmed accurate — "
                    "no further source change required. No merge performed."
                ),
            ),
            feedback(
                "reverified-no-change-needed",
                reviewer="owner",
                body=(
                    "Re-verified at the exact head: the governed transition is enforced and "
                    "the focused suite passes. No further change needed."
                ),
            ),
            feedback(
                "stable-tip-separate-card",
                reviewer="owner",
                body=(
                    "Re: local PR CI audit receipt 4743945a — validated and reproduced "
                    "at the exact tested SHA. Assessment: pre-existing governance drift "
                    "on the branch lineage, not introduced by this PR; consistent with "
                    "failures reproducing on stable tip. A separate repair card has been "
                    "opened rather than patching inside this refactor."
                ),
            ),
            feedback(
                "shared-base-independent-validation",
                reviewer="owner",
                body=(
                    "Independent validation of this audit at the exact receipt head. "
                    "Reproduction confirmed: the failures are pre-existing shared-base "
                    "failures, not regressions of this PR."
                ),
            ),
            feedback(
                "pr-regression",
                reviewer="owner",
                body=(
                    "Local CI audit: static lane failed with a PR-introduced logic "
                    "regression. The changed extraction source still needs work."
                ),
            ),
        ),
    )
    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    result = ScanController(policy, ledger, github, kanban, RecordingLocalGit()).scan()

    assert result.created == 1
    assert result.skipped["self_resolution_receipt"] == 7
    assert [task.evidence["feedback_id"] for task in kanban.tasks] == ["pr-regression"]
    ledger.close()


def test_scan_suppresses_non_actionable_review_containers_but_keeps_inline_findings(
    tmp_path: Path,
) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(local_path, not_before="2026-08-24T00:00:00Z")
    created_at = datetime.fromisoformat("2026-08-24T00:00:00+00:00")
    reviewer = Reviewer("codex-review-bot", "MEMBER")
    boilerplate = (
        "### 💡 Codex Review\n"
        "Here are some automated review suggestions for this pull request.\n"
        "**Reviewed commit:** `abc123`"
    )
    github = FakeGitHub(
        admitted_pull_request(sha),
        (
            Feedback("review", "empty-review", reviewer, "", created_at, True),
            Feedback("review", "review-envelope", reviewer, boilerplate, created_at, True),
            Feedback(
                "review_comment",
                "inline-finding",
                reviewer,
                "[P1] Preserve the fallback when the provider omits strikes.",
                created_at,
                True,
            ),
        ),
    )
    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    result = ScanController(policy, ledger, github, kanban, RecordingLocalGit()).scan()

    assert result.created == 1
    assert result.skipped["non_actionable_review_container"] == 2
    assert [task.evidence["feedback_id"] for task in kanban.tasks] == ["inline-finding"]
    ledger.close()


def test_scan_suppresses_codexs_own_review_summary_tracker_comment(tmp_path: Path) -> None:
    """Codex's running review-status comment (an issue comment, not a GitHub

    review, so the container check above never sees it) only ever reports
    which review ran and when -- never itself a finding. Admitting it as
    actionable feedback wastes a repair dispatch, and its unicode-heavy table
    has tripped a fallback provider's egress secret-scanner as a false
    positive.
    """

    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(local_path, not_before="2026-08-24T00:00:00Z")
    created_at = datetime.fromisoformat("2026-08-24T00:00:00+00:00")
    tracker = (
        "<!-- codex-pull-request-review-summary -->\n\n## Codex Review Summary\n\n"
        "| Review | Status | Commit | Review trigger |\n| --- | --- | --- | --- |\n"
        "| Code Review | Completed | `abc1234` | PR opened |"
    )
    github = FakeGitHub(
        admitted_pull_request(sha),
        (
            Feedback(
                "issue_comment",
                "codex-tracker",
                Reviewer("chatgpt-codex-connector[bot]", None),
                tracker,
                created_at,
                True,
            ),
        ),
    )
    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    result = ScanController(policy, ledger, github, kanban, RecordingLocalGit()).scan()

    assert result.created == 0
    assert result.skipped["codex_review_summary_tracker"] == 1
    assert kanban.tasks == []
    ledger.close()


def test_scan_suppresses_self_review_comment_receipt_only_when_no_action_remains(
    tmp_path: Path,
) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(local_path, not_before="2026-08-24T00:00:00Z")
    created_at = datetime.fromisoformat("2026-08-24T00:00:00+00:00")
    owner = Reviewer("owner", "MEMBER")
    github = FakeGitHub(
        admitted_pull_request(sha),
        (
            Feedback(
                "review_comment",
                "fixed-reply",
                owner,
                "Fixed at 026649c79. Focused verification passed.",
                created_at,
                False,
            ),
            Feedback(
                "review_comment",
                "remaining-reply",
                owner,
                "Fixed the first case, but one blocker remains and still needs work.",
                created_at,
                False,
            ),
        ),
    )
    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    result = ScanController(policy, ledger, github, kanban, RecordingLocalGit()).scan()

    assert result.created == 1
    assert result.skipped["self_resolution_receipt"] == 1
    assert [task.evidence["feedback_id"] for task in kanban.tasks] == ["remaining-reply"]
    ledger.close()


def test_scan_refetches_the_head_and_does_not_claim_or_dispatch_a_race(tmp_path: Path) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(local_path, not_before="2026-08-24T00:00:00Z")
    listed = admitted_pull_request(sha)
    current = admitted_pull_request("b" * 40)
    github = FakeGitHub(listed, (feedback("race"),), current)
    kanban = RecordingKanban()
    local_git = RecordingLocalGit()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    result = ScanController(policy, ledger, github, kanban, local_git).scan()

    assert result.created == 0
    assert result.skipped["head_changed"] == 1
    assert kanban.tasks == []
    assert local_git.calls == []
    assert sqlite3.connect(tmp_path / "ledger.sqlite3").execute(
        "SELECT COUNT(*) FROM feedback_receipts"
    ).fetchone() == (0,)
    ledger.close()


def test_scan_filters_out_of_policy_pull_requests_before_fetching_their_feedback(
    tmp_path: Path,
) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(local_path, not_before="2026-08-24T00:00:00Z")
    admitted = admitted_pull_request(sha)
    foreign = PullRequest(
        99,
        "OPEN",
        "acme/widgets",
        "acme/widgets",
        "someone-else",
        "codex/unrelated",
        "b" * 40,
    )
    github = MixedPullRequestGitHub(admitted, foreign, (feedback("allowed"),))
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    result = ScanController(policy, ledger, github, RecordingKanban(), RecordingLocalGit()).scan()

    assert result.created == 1
    assert result.skipped["author_not_allowed"] == 1
    assert result.skipped.get("github_error", 0) == 0
    assert github.feedback_calls == [17]
    ledger.close()


def test_scan_reads_independent_pull_feedback_concurrently(tmp_path: Path) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(local_path, not_before="2026-08-24T00:00:00Z")
    pulls = tuple(
        PullRequest(
            number,
            "OPEN",
            "acme/widgets",
            "acme/widgets",
            "owner",
            f"codex/fix-{number}",
            sha,
        )
        for number in range(17, 17 + MAX_PARALLEL_PR_READS + 1)
    )
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    github = ConcurrentScanGitHub(pulls)
    result = ScanController(
        policy,
        ledger,
        github,
        RecordingKanban(),
        RecordingLocalGit(),
    ).scan()

    assert result.created == 0
    assert result.skipped == {}
    assert result.degraded is False
    assert github.max_active == MAX_PARALLEL_PR_READS
    ledger.close()


def test_scan_requeues_unactioned_feedback_after_the_pr_head_changes(
    tmp_path: Path,
) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(local_path, not_before="2026-08-24T00:00:00Z")
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    prior = FeedbackReceipt("acme/widgets", 17, "issue_comment", "already-seen", "b" * 40)
    lease = ledger.claim(
        prior,
        owner="prior-scanner",
        claimed_at=datetime(2026, 8, 24, 1, 0, tzinfo=UTC),
        stale_before=datetime(2026, 8, 24, 0, 55, tzinfo=UTC),
    )
    assert lease is not None
    ledger.finalize(prior, "kanban-prior", lease)
    kanban = RecordingKanban()

    result = ScanController(
        policy,
        ledger,
        FakeGitHub(admitted_pull_request(sha), (feedback("already-seen"),)),
        kanban,
        RecordingLocalGit(),
    ).scan()

    assert result.created == 1
    assert [task.evidence["feedback_id"] for task in kanban.tasks] == ["already-seen"]
    assert kanban.tasks[0].head_sha == sha
    ledger.close()


def test_scan_does_not_requeue_feedback_actioned_on_an_older_head(
    tmp_path: Path,
) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(local_path, not_before="2026-08-24T00:00:00Z")
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    prior = FeedbackReceipt("acme/widgets", 17, "issue_comment", "already-fixed", "b" * 40)
    lease = ledger.claim(
        prior,
        owner="prior-scanner",
        claimed_at=datetime(2026, 8, 24, 1, 0, tzinfo=UTC),
        stale_before=datetime(2026, 8, 24, 0, 55, tzinfo=UTC),
    )
    assert lease is not None
    ledger.finalize(prior, "kanban-prior", lease)
    ledger.mark_feedback_actioned(
        prior,
        resolved_head_sha="c" * 40,
        actioned_at=datetime(2026, 8, 24, 2, 0, tzinfo=UTC),
    )
    kanban = RecordingKanban()

    result = ScanController(
        policy,
        ledger,
        FakeGitHub(admitted_pull_request(sha), (feedback("already-fixed"),)),
        kanban,
        RecordingLocalGit(),
    ).scan()

    assert result.created == 0
    assert result.skipped["already_actioned"] == 1
    assert kanban.tasks == []
    ledger.close()


def test_scan_has_a_fixed_per_run_admission_cap(tmp_path: Path) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(local_path, not_before="2026-08-24T00:00:00Z")
    pull_requests = tuple(
        PullRequest(
            number,
            "OPEN",
            "acme/widgets",
            "acme/widgets",
            "owner",
            f"codex/fix-{number}",
            sha,
        )
        for number in range(1, MAX_ADMISSIONS_PER_SCAN + 3)
    )

    class ManyPullRequestGitHub(FakeGitHub):
        def list_open_pull_requests(
            self, repository: str, owner_login: str
        ) -> tuple[PullRequest, ...]:
            assert repository == "acme/widgets"
            assert owner_login == "owner"
            return pull_requests

        def list_feedback(self, repository: str, number: int) -> tuple[Feedback, ...]:
            return (feedback(str(number)),)

        def get_pull_request(self, repository: str, number: int) -> PullRequest:
            return pull_requests[number - 1]

    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    result = ScanController(
        policy,
        ledger,
        ManyPullRequestGitHub(pull_requests[0], ()),
        kanban,
        RecordingLocalGit(),
    ).scan()

    assert result.created == MAX_ADMISSIONS_PER_SCAN
    assert result.skipped["admission_cap"] == 2
    assert result.degraded is True
    ledger.close()


def test_scan_cap_counts_failed_dispatch_attempts(tmp_path: Path) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(local_path, not_before="2026-08-24T00:00:00Z")
    feedback_items = tuple(feedback(str(number)) for number in range(MAX_ADMISSIONS_PER_SCAN + 2))
    kanban = FailingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    result = ScanController(
        policy, ledger, FakeGitHub(admitted_pull_request(sha), feedback_items), kanban, RecordingLocalGit()
    ).scan()

    assert result.created == 0
    assert len(kanban.calls) == MAX_ADMISSIONS_PER_SCAN
    assert result.skipped["admission_cap"] == 2
    assert result.degraded is True
    ledger.close()


def test_explicit_retry_revalidates_canonical_head_before_claiming_failed_receipt(tmp_path: Path) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(local_path, not_before="2026-08-24T00:00:00Z")
    receipt = FeedbackReceipt("acme/widgets", 17, "issue_comment", "failed", sha)
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    lease = ledger.claim(
        receipt,
        owner="test-scanner",
        claimed_at=datetime(2026, 8, 24, 12, 0, tzinfo=UTC),
        stale_before=datetime(2026, 8, 24, 11, 55, tzinfo=UTC),
    )
    assert lease is not None
    ledger.fail(receipt, "Kanban unavailable", lease)
    github = FakeGitHub(admitted_pull_request(sha), (feedback("failed"),), admitted_pull_request("b" * 40))

    result = ScanController(policy, ledger, github, RecordingKanban(), RecordingLocalGit()).retry_failed(receipt)

    assert result.created == 0
    assert result.skipped["head_changed"] == 1
    assert github.current_calls == [("acme/widgets", 17)]
    assert receipt_status(ledger, receipt) == "failed"
    ledger.close()


def test_explicit_retry_recovers_a_lost_kanban_response_with_the_same_receipt_key(tmp_path: Path) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(local_path, not_before="2026-08-24T00:00:00Z")
    github = FakeGitHub(admitted_pull_request(sha), (feedback("lost-response"),))
    kanban = LostResponseKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    scanner = ScanController(policy, ledger, github, kanban, RecordingLocalGit())

    first = scanner.scan()
    receipt = FeedbackReceipt("acme/widgets", 17, "issue_comment", "lost-response", sha)
    retry = scanner.retry_failed(receipt)

    assert first.created == 0
    assert retry.created == 1
    assert len(kanban.created_by_key) == 1
    assert [task.idempotency_key for task in kanban.calls] == [
        kanban.calls[0].idempotency_key,
        kanban.calls[0].idempotency_key,
    ]
    assert receipt_status(ledger, receipt) == "completed"
    assert receipt_task_id(ledger, receipt) == "kanban-recovered"
    ledger.close()


def test_scan_recovers_a_stale_receipt_after_process_death_immediately_after_claim(
    tmp_path: Path,
) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(local_path, not_before="2026-08-24T00:00:00Z")
    github = FakeGitHub(admitted_pull_request(sha), (feedback("death-after-claim"),))
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    claimed_at = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)

    class CrashAfterClaim:
        def prepare_receipt_worktree(self, _path: Path, _receipt: FeedbackReceipt) -> PreparedWorktree:
            raise SystemExit("simulated process death after claim")

    with pytest.raises(SystemExit, match="after claim"):
        ScanController(
            policy,
            ledger,
            github,
            RecordingKanban(),
            CrashAfterClaim(),
            claim_owner="scanner-a",
            clock=lambda: claimed_at,
        ).scan()
    receipt = FeedbackReceipt("acme/widgets", 17, "issue_comment", "death-after-claim", sha)
    assert receipt_expected_sha(ledger, receipt) == sha

    kanban = RecordingKanban()
    recovered = ScanController(
        policy,
        ledger,
        github,
        kanban,
        RecordingLocalGit(),
        claim_owner="scanner-b",
        clock=lambda: claimed_at + timedelta(minutes=6),
    ).scan()

    assert recovered.created == 1
    assert receipt_status(ledger, receipt) == "completed"
    assert receipt_lease_version(ledger, receipt) == 2
    ledger.close()


def test_scan_recovers_a_stale_receipt_after_process_death_after_worktree_creation(
    tmp_path: Path,
) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(local_path, not_before="2026-08-24T00:00:00Z")
    github = FakeGitHub(admitted_pull_request(sha), (feedback("death-after-worktree"),))
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    claimed_at = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)
    local_git = LocalGitRepository(tmp_path / "profile" / "worktrees")

    class CrashAfterWorktree:
        def __init__(self) -> None:
            self.prepared: PreparedWorktree | None = None

        def prepare_receipt_worktree(self, path: Path, receipt: FeedbackReceipt) -> PreparedWorktree:
            self.prepared = local_git.prepare_receipt_worktree(path, receipt)
            raise SystemExit("simulated process death after worktree")

    crashing_git = CrashAfterWorktree()
    with pytest.raises(SystemExit, match="after worktree"):
        ScanController(
            policy,
            ledger,
            github,
            RecordingKanban(),
            crashing_git,
            claim_owner="scanner-a",
            clock=lambda: claimed_at,
        ).scan()
    assert crashing_git.prepared is not None

    kanban = RecordingKanban()
    recovered = ScanController(
        policy,
        ledger,
        github,
        kanban,
        local_git,
        claim_owner="scanner-b",
        clock=lambda: claimed_at + timedelta(minutes=6),
    ).scan()

    assert recovered.created == 1
    assert kanban.tasks[0].repository_path == crashing_git.prepared.path
    assert git_output(crashing_git.prepared.path, "rev-parse", "HEAD") == sha
    ledger.close()


def test_scan_recovers_a_stale_receipt_after_task_creation_response_is_lost_to_process_death(
    tmp_path: Path,
) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(local_path, not_before="2026-08-24T00:00:00Z")
    github = FakeGitHub(admitted_pull_request(sha), (feedback("death-after-task"),))
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    claimed_at = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)

    class CrashAfterTaskCreation:
        def __init__(self) -> None:
            self.calls: list[object] = []
            self.created_by_key: dict[str, str] = {}

        def create_or_get_task(self, task: object) -> str:
            self.calls.append(task)
            key = task.idempotency_key
            if key not in self.created_by_key:
                self.created_by_key[key] = "kanban-existing"
                raise SystemExit("simulated process death after task creation")
            return self.created_by_key[key]

    kanban = CrashAfterTaskCreation()
    with pytest.raises(SystemExit, match="after task creation"):
        ScanController(
            policy,
            ledger,
            github,
            kanban,
            RecordingLocalGit(),
            claim_owner="scanner-a",
            clock=lambda: claimed_at,
        ).scan()

    recovered = ScanController(
        policy,
        ledger,
        github,
        kanban,
        RecordingLocalGit(),
        claim_owner="scanner-b",
        clock=lambda: claimed_at + timedelta(minutes=6),
    ).scan()
    receipt = FeedbackReceipt("acme/widgets", 17, "issue_comment", "death-after-task", sha)

    assert recovered.created == 1
    assert len(kanban.created_by_key) == 1
    assert [task.idempotency_key for task in kanban.calls] == [
        kanban.calls[0].idempotency_key,
        kanban.calls[0].idempotency_key,
    ]
    assert receipt_task_id(ledger, receipt) == "kanban-existing"
    assert receipt_lease_version(ledger, receipt) == 2
    ledger.close()


def test_explicit_retry_revalidates_the_canonical_reviewer_before_retrying(tmp_path: Path) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(local_path, not_before="2026-08-24T00:00:00Z")
    receipt = FeedbackReceipt("acme/widgets", 17, "issue_comment", "untrusted", sha)
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    lease = ledger.claim(
        receipt,
        owner="test-scanner",
        claimed_at=datetime(2026, 8, 24, 12, 0, tzinfo=UTC),
        stale_before=datetime(2026, 8, 24, 11, 55, tzinfo=UTC),
    )
    assert lease is not None
    ledger.fail(receipt, "Kanban unavailable", lease)
    github = FakeGitHub(admitted_pull_request(sha), (feedback("untrusted", reviewer="stranger"),))

    result = ScanController(policy, ledger, github, RecordingKanban(), RecordingLocalGit()).retry_failed(receipt)

    assert result.created == 0
    assert result.skipped["reviewer_not_allowed"] == 1
    assert github.current_calls == [("acme/widgets", 17)]
    assert receipt_status(ledger, receipt) == "failed"
    ledger.close()


def test_scan_rejects_a_feedback_timestamp_without_a_timezone(tmp_path: Path) -> None:
    local_path, sha = initialized_repository(tmp_path)
    policy = configured_policy(local_path, not_before="2026-08-24T00:00:00Z")
    item = feedback("naive")
    object.__setattr__(item, "created_at", datetime.fromisoformat("2026-08-24T00:00:00"))
    kanban = RecordingKanban()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    result = ScanController(
        policy, ledger, FakeGitHub(admitted_pull_request(sha), (item,)), kanban, RecordingLocalGit()
    ).scan()

    assert result.created == 0
    assert result.skipped["invalid_feedback_timestamp"] == 1
    assert kanban.tasks == []
    ledger.close()


def test_scan_marks_incomplete_canonical_github_coverage_as_degraded(tmp_path: Path) -> None:
    local_path, _sha = initialized_repository(tmp_path)
    policy = configured_policy(local_path, not_before="2026-08-24T00:00:00Z")
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    class UnavailableGitHub:
        def list_open_pull_requests(
            self, _repository: str, _owner_login: str
        ) -> tuple[PullRequest, ...]:
            raise RuntimeError("canonical read unavailable")

    result = ScanController(
        policy,
        ledger,
        UnavailableGitHub(),
        RecordingKanban(),
        RecordingLocalGit(),
    ).scan()

    assert result.created == 0
    assert result.skipped == {"github_error": 1}
    assert result.degraded is True
    ledger.close()


def test_scan_marks_missing_local_exact_head_as_degraded_with_a_precise_reason(
    tmp_path: Path,
) -> None:
    local_path, local_sha = initialized_repository(tmp_path)
    unavailable_sha = "b" * 40
    assert local_sha != unavailable_sha
    policy = configured_policy(local_path, not_before="2026-08-24T00:00:00Z")
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    result = ScanController(
        policy,
        ledger,
        FakeGitHub(admitted_pull_request(unavailable_sha), (feedback("missing-head"),)),
        RecordingKanban(),
        LocalGitRepository(tmp_path / "profile" / "worktrees"),
    ).scan()

    assert result.created == 0
    assert result.skipped == {"exact_head_unavailable": 1}
    assert result.degraded is True
    receipt = FeedbackReceipt("acme/widgets", 17, "issue_comment", "missing-head", unavailable_sha)
    assert receipt_status(ledger, receipt) == "failed"
    ledger.close()


def test_local_git_creates_a_deterministic_receipt_branch_at_the_verified_head_with_fixed_argv() -> None:
    sha = "a" * 40
    runner = RecordingGitRunner(
        [
            GitCommandResult(0, ""),
            GitCommandResult(1, ""),
            GitCommandResult(0, ""),
            GitCommandResult(0, f"{sha}\n"),
        ]
    )
    repository = LocalGitRepository(runner)
    receipt = FeedbackReceipt("acme/widgets", 17, "issue_comment", "feedback-1", sha)

    branch = repository.prepare_receipt_branch(Path("/repositories/widgets"), receipt)

    assert branch.startswith("hermes/github-pr-feedback/")
    assert runner.calls == [
        ["git", "-C", "/repositories/widgets", "cat-file", "-e", f"{sha}^{{commit}}"],
        [
            "git",
            "-C",
            "/repositories/widgets",
            "rev-parse",
            "--verify",
            "--quiet",
            f"refs/heads/{branch}^{{commit}}",
        ],
        ["git", "-C", "/repositories/widgets", "branch", branch, sha],
        [
            "git",
            "-C",
            "/repositories/widgets",
            "rev-parse",
            "--verify",
            "--quiet",
            f"refs/heads/{branch}^{{commit}}",
        ],
    ]


def test_local_git_rejects_an_existing_deterministic_branch_at_a_different_head() -> None:
    sha = "a" * 40
    runner = RecordingGitRunner([GitCommandResult(0, ""), GitCommandResult(0, f"{'b' * 40}\n")])
    repository = LocalGitRepository(runner)
    receipt = FeedbackReceipt("acme/widgets", 17, "issue_comment", "feedback-1", sha)

    with pytest.raises(RuntimeError, match="collides"):
        repository.prepare_receipt_branch(Path("/repositories/widgets"), receipt)


def test_local_git_materializes_and_reuses_a_deterministic_linked_worktree_at_exact_head(
    tmp_path: Path,
) -> None:
    source, original_sha = initialized_repository(tmp_path)
    receipt = FeedbackReceipt("acme/widgets", 17, "issue_comment", "exact-head", original_sha)
    repository = LocalGitRepository(tmp_path / "profile" / "worktrees")

    first = repository.prepare_receipt_worktree(source, receipt)
    (source / "README.md").write_text("branch moved\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "README.md"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Test User",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "move source branch",
        ],
        check=True,
    )
    moved_sha = git_output(source, "rev-parse", "HEAD")
    second = repository.prepare_receipt_worktree(source, receipt)

    assert moved_sha != original_sha
    assert second == first
    assert first.path != source
    assert git_output(first.path, "rev-parse", "HEAD") == original_sha
    assert first.expected_sha == original_sha
    assert first.branch.startswith("hermes/github-pr-feedback/")


def test_local_git_hard_fails_if_a_materialized_receipt_worktree_head_changes(
    tmp_path: Path,
) -> None:
    source, original_sha = initialized_repository(tmp_path)
    receipt = FeedbackReceipt("acme/widgets", 17, "issue_comment", "tampered", original_sha)
    repository = LocalGitRepository(tmp_path / "profile" / "worktrees")
    prepared = repository.prepare_receipt_worktree(source, receipt)
    (source / "second.txt").write_text("second\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "second.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Test User",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "second",
        ],
        check=True,
    )
    moved_sha = git_output(source, "rev-parse", "HEAD")
    subprocess.run(["git", "-C", str(prepared.path), "checkout", "--quiet", "--detach", moved_sha], check=True)

    with pytest.raises(RuntimeError, match="worktree HEAD does not match expected SHA"):
        repository.prepare_receipt_worktree(source, receipt)


def test_local_git_hard_fails_if_a_materialized_receipt_worktree_is_dirty(
    tmp_path: Path,
) -> None:
    source, original_sha = initialized_repository(tmp_path)
    receipt = FeedbackReceipt("acme/widgets", 17, "issue_comment", "dirty", original_sha)
    repository = LocalGitRepository(tmp_path / "profile" / "worktrees")
    prepared = repository.prepare_receipt_worktree(source, receipt)
    (prepared.path / "untrusted.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="receipt worktree is not clean"):
        repository.prepare_receipt_worktree(source, receipt)


def test_local_git_reports_an_unavailable_exact_head_without_falling_back_to_local_head(
    tmp_path: Path,
) -> None:
    source, local_sha = initialized_repository(tmp_path)
    unavailable_sha = "b" * 40
    assert unavailable_sha != local_sha
    receipt = FeedbackReceipt("acme/widgets", 17, "issue_comment", "unavailable", unavailable_sha)
    repository = LocalGitRepository(tmp_path / "profile" / "worktrees")

    with pytest.raises(RuntimeError, match="exact head is unavailable in configured repository"):
        repository.prepare_receipt_worktree(source, receipt)

    assert list((tmp_path / "profile" / "worktrees").glob("*")) == []


def initialized_repository(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "repository"
    subprocess.run(["git", "init", "--quiet", str(path)], check=True)
    (path / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "README.md"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Test User",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        check=True,
    )
    sha = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return path, sha


def git_output(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def configured_policy(
    local_path: Path,
    *,
    not_before: str,
    auto_dispatch: bool = False,
    local_ci_audit: bool = False,
    merge_maintainer: bool = False,
    max_open_prs_per_scan: int | None = None,
):
    raw = {
            "enabled": True,
            "repositories": [
                {
                    "base_repository": "acme/widgets",
                    "head_repository": "acme/widgets",
                    "local_path": str(local_path),
                    "owner_login": "owner",
                    "branch_prefixes": ["codex/"],
                }
            ],
            "reviewer_logins": ["reviewer"],
            "reviewer_associations": [],
            "include_self_feedback": True,
            "include_bot_feedback": True,
            "auto_dispatch": auto_dispatch,
            "not_before": not_before,
            "assignee": "repair-agent",
            "board": "repairs",
        }
    if local_ci_audit:
        raw["local_ci_audit"] = {
            "enabled": True,
            "assignee": "pr-local-ci-auditor",
            "post_results": True,
        }
        if max_open_prs_per_scan is not None:
            raw["local_ci_audit"]["max_open_prs_per_scan"] = max_open_prs_per_scan
        raw["routing_rules"] = [
            {
                "assignee": "ci-static-fixer",
                "precedence": 150,
                "match_any": ["static lane"],
                "match_labels_any": ["ci/static"],
                "tags": ["type/ci", "ci/static"],
                "priority": "P2",
                "blast_radius": "contained",
                "risks": [],
                "requires_review": False,
            }
        ]
    if merge_maintainer:
        raw["merge_maintainer"] = {
            "enabled": True,
            "assignee": "pr-merge-maintainer",
            "repository": "acme/widgets",
            "author_login": "owner",
            "base_branch": "stable",
            "merge_methods": ["squash"],
            "receipt_max_age_seconds": 3600,
            "report_only": False,
            "post_merge": {"enabled": False},
        }
    return load_policy(raw)


def admitted_pull_request(sha: str) -> PullRequest:
    return PullRequest(17, "OPEN", "acme/widgets", "acme/widgets", "owner", "codex/fix", sha)


def feedback(
    feedback_id: str,
    *,
    body: str = "please change this",
    reviewer: str = "reviewer",
    is_bot: bool = False,
    created_at: str = "2026-08-24T00:00:00Z",
) -> Feedback:
    return Feedback(
        "issue_comment",
        feedback_id,
        Reviewer(reviewer, "MEMBER"),
        body,
        datetime.fromisoformat(created_at),
        is_bot,
    )


def receipt_status(ledger: FeedbackLedger, receipt: FeedbackReceipt) -> str:
    row = ledger._connection.execute(
        "SELECT status FROM feedback_receipts WHERE repository = ? AND pr_number = ? "
        "AND feedback_kind = ? AND feedback_id = ? AND head_sha = ?",
        receipt.key,
    ).fetchone()
    assert row is not None
    return row[0]


def receipt_task_id(ledger: FeedbackLedger, receipt: FeedbackReceipt) -> str | None:
    row = ledger._connection.execute(
        "SELECT task_id FROM feedback_receipts WHERE repository = ? AND pr_number = ? "
        "AND feedback_kind = ? AND feedback_id = ? AND head_sha = ?",
        receipt.key,
    ).fetchone()
    assert row is not None
    return row[0]


def receipt_lease_version(ledger: FeedbackLedger, receipt: FeedbackReceipt) -> int:
    row = ledger._connection.execute(
        "SELECT lease_version FROM feedback_receipts WHERE repository = ? AND pr_number = ? "
        "AND feedback_kind = ? AND feedback_id = ? AND head_sha = ?",
        receipt.key,
    ).fetchone()
    assert row is not None
    return int(row[0])


def receipt_expected_sha(ledger: FeedbackLedger, receipt: FeedbackReceipt) -> str | None:
    row = ledger._connection.execute(
        "SELECT expected_sha FROM feedback_receipts WHERE repository = ? AND pr_number = ? "
        "AND feedback_kind = ? AND feedback_id = ? AND head_sha = ?",
        receipt.key,
    ).fetchone()
    assert row is not None
    return row[0]
