from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
import subprocess
import threading

from github_pr_feedback.controller import FeedbackReceipt, PreparedWorktree
from github_pr_feedback.github_client import (
    CheckState,
    PullRequestMergeState,
    ReviewState,
)
from github_pr_feedback.ledger import FeedbackLedger
from github_pr_feedback.policy import load_policy
from github_pr_feedback.repair_controller import RepairController, repair_triggers


SHA = "a" * 40


def merge_state(
    *, number: int = 17, mergeable: bool = True, status: str = "CLEAN"
) -> PullRequestMergeState:
    return PullRequestMergeState(
        repository="acme/widgets",
        number=number,
        state="OPEN",
        is_draft=False,
        mergeable=mergeable,
        merge_state_status=status,
        base_branch="main",
        base_sha="b" * 40,
        head_repository="acme/widgets",
        author_login="owner",
        head_ref_name="codex/fix",
        head_sha=SHA,
        merged=False,
        merge_commit_oid=None,
    )


def policy(
    tmp_path: Path,
    *,
    report_only: bool = False,
    merge_maintainer: bool = False,
    max_base_refresh_in_flight: int | None = None,
):
    repository = tmp_path / "repo"
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    raw = {
        "enabled": True,
        "repositories": [
            {
                "base_repository": "acme/widgets",
                "head_repository": "acme/widgets",
                "local_path": str(repository),
                "owner_login": "owner",
                "branch_prefixes": ["codex/"],
            }
        ],
        "reviewer_logins": ["reviewer"],
        "reviewer_associations": [],
        "not_before": "2026-08-25T00:00:00Z",
        "assignee": "fallback",
        "board": "repairs",
        "repair_steward": {
            "enabled": True,
            "assignee": "pr-repair-steward",
            "repositories": ["acme/widgets"],
            "report_only": report_only,
        },
    }
    if max_base_refresh_in_flight is not None:
        raw["repair_steward"]["max_base_refresh_in_flight"] = (
            max_base_refresh_in_flight
        )
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


def test_repair_triggers_cover_conflicts_changes_requested_and_non_green_actions() -> (
    None
):
    assert repair_triggers(
        merge_state(mergeable=False, status="DIRTY"),
        ReviewState("CHANGES_REQUESTED", 1),
        CheckState(True, False, 2),
    ) == ("merge_conflict", "changes_requested", "actions_not_green")


def test_repair_triggers_does_not_treat_a_billing_lockout_as_a_repair_trigger() -> None:
    """A GitHub Actions billing lockout fails every check regardless of code quality.

    It is not evidence this PR needs a repair; the local-CI lane is the
    billing-aware trigger for a genuine failure found under lockout.
    """

    assert repair_triggers(
        merge_state(),
        ReviewState(None, 0),
        CheckState(True, False, 2, True),
    ) == ()


def test_repair_triggers_does_not_treat_action_required_as_a_repair_trigger() -> None:
    """A check waiting on human workflow-run approval is not a code defect.

    No repair commit can satisfy GitHub's own action_required conclusion; only
    a human approving the gated run (or otherwise resolving it) can.
    """

    assert repair_triggers(
        merge_state(),
        ReviewState(None, 0),
        CheckState(True, False, 2, False, True),
    ) == ()


class GitHub:
    def list_open_pull_requests(self, repository: str, owner: str):
        from github_pr_feedback.policy import PullRequest

        return (
            PullRequest(17, "OPEN", repository, repository, owner, "codex/fix", SHA),
        )

    def get_merge_state(self, repository: str, number: int):
        return merge_state(mergeable=False, status="DIRTY")

    def get_review_state(self, repository: str, number: int):
        return ReviewState(None, 0)

    def get_check_state(self, repository: str, head_sha: str):
        return CheckState(False, True, 0)


class GitHubWithoutChecks(GitHub):
    def get_check_state(self, repository: str, head_sha: str):
        raise RuntimeError("check state unavailable")


class ActionRequiredGitHub(GitHub):
    """A PR with no conflict/review trigger, only a GitHub action_required check."""

    def get_merge_state(self, repository: str, number: int):
        return merge_state()

    def get_check_state(self, repository: str, head_sha: str):
        return CheckState(True, False, 1, False, True)


class BehindBaseGitHub(GitHub):
    def get_merge_state(self, repository: str, number: int):
        return replace(merge_state(), base_branch="stable")

    def get_branch_head(self, repository: str, branch: str):
        assert (repository, branch) == ("acme/widgets", "stable")
        return "c" * 40


class ManyBehindBaseGitHub(BehindBaseGitHub):
    def list_open_pull_requests(self, repository: str, owner: str):
        from github_pr_feedback.policy import PullRequest

        return tuple(
            PullRequest(
                number,
                "OPEN",
                repository,
                repository,
                owner,
                "codex/fix",
                str(number % 10) * 40,
            )
            for number in (17, 18)
        )

    def get_merge_state(self, repository: str, number: int):
        return replace(
            merge_state(number=number),
            base_branch="stable",
            head_sha=str(number % 10) * 40,
        )


class RefreshInFlightGitHub(ManyBehindBaseGitHub):
    def get_merge_state(self, repository: str, number: int):
        state = super().get_merge_state(repository, number)
        if number == 17:
            return replace(state, base_sha="c" * 40)
        return state


class MixedConflictBaseGitHub(ManyBehindBaseGitHub):
    def get_merge_state(self, repository: str, number: int):
        state = super().get_merge_state(repository, number)
        if number == 17:
            return replace(state, mergeable=False, merge_state_status="DIRTY")
        return state


class ConcurrentReadGitHub(GitHub):
    def __init__(self) -> None:
        self.barrier = threading.Barrier(2)

    def list_open_pull_requests(self, repository: str, owner: str):
        from github_pr_feedback.policy import PullRequest

        return tuple(
            PullRequest(number, "OPEN", repository, repository, owner, "codex/fix", SHA)
            for number in (17, 18)
        )

    def get_merge_state(self, repository: str, number: int):
        self.barrier.wait(timeout=1)
        return merge_state(number=number)


class LocalGit:
    def prepare_receipt_worktree(self, path: Path, receipt):
        return PreparedWorktree(path / "exact", "hermes/repair", receipt.head_sha)


class Kanban:
    def __init__(self):
        self.tasks = []

    def create_or_get_task(self, task):
        self.tasks.append(task)
        return "repair-task"


class StatusKanban(Kanban):
    def __init__(self, statuses: dict[str, str | None]):
        super().__init__()
        self.statuses = statuses

    def task_status(self, board: str, task_id: str) -> str | None:
        assert board == "repairs"
        return self.statuses.get(task_id)


def test_repair_controller_dedupes_exact_head_and_preserves_merge_authority(
    tmp_path: Path,
) -> None:
    configured = policy(tmp_path)
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    kanban = Kanban()
    controller = RepairController(
        configured,
        ledger,
        GitHub(),
        kanban,
        LocalGit(),
        clock=lambda: datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
    )

    first = controller.scan()
    second = controller.scan()

    assert first.created == 1
    assert second.created == 0
    task = kanban.tasks[0]
    assert task.assignee == "pr-repair-steward"
    assert task.initial_status == "running"
    assert task.max_runtime_seconds == 1200
    assert "git merge --no-ff --no-edit" in task.instructions
    assert "Commit the resolved merge before running base-relative" in task.instructions
    assert "Do not merge the pull request" in task.instructions
    assert "Do not force-push" in task.instructions
    assert "Do not weaken" in task.instructions
    assert task.evidence["expected_base_branch"] == "main"
    assert task.evidence["expected_base_sha"] == "b" * 40
    assert task.evidence["expected_head_branch"] == "codex/fix"
    assert task.evidence["expected_head_repository"] == "acme/widgets"
    assert task.evidence["expected_head_sha"] == SHA
    identity_command = (
        "github-pr-feedback inspect-pr --repository acme/widgets --pr-number 17"
    )
    assert task.instructions.count(identity_command) == 1
    assert "gh pr view" not in task.instructions
    assert "before any fetch, checkout, edit, test, commit, push, or reply" in (
        task.instructions
    )
    assert "require all five returned identity fields" in task.instructions.casefold()
    assert task.idempotency_key.startswith("github-pr-repair:v3:")
    ledger.close()


def test_repair_controller_escalates_an_action_required_pr_instead_of_repairing_it(
    tmp_path: Path,
) -> None:
    """A PR with no conflict/review trigger, only action_required, still gets a card.

    It must be a blocked, human-facing escalation -- distinct from (and never
    absorbed into) the ordinary repair path, since no repair commit can clear
    GitHub's own action_required conclusion.
    """

    configured = policy(tmp_path)
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    kanban = Kanban()
    controller = RepairController(
        configured,
        ledger,
        ActionRequiredGitHub(),
        kanban,
        LocalGit(),
        clock=lambda: datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
    )

    first = controller.scan()
    second = controller.scan()

    assert first.created == 1
    assert first.skipped.get("no_repair_trigger") == 1
    assert second.created == 0
    assert len(kanban.tasks) == 1
    task = kanban.tasks[0]
    assert task.assignee == "fallback"
    assert task.initial_status == "blocked"
    assert task.title == "Actions needed: acme/widgets#17 (GitHub check action_required)"
    assert "action_required" in task.instructions
    assert "Do not push, edit, approve, or merge" in task.instructions
    assert task.evidence["reason"] == "github_check_action_required"
    ledger.close()


def test_repair_controller_routes_a_stale_pr_base_into_the_refresh_lane(
    tmp_path: Path,
) -> None:
    configured = policy(tmp_path, merge_maintainer=True)
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    kanban = Kanban()

    result = RepairController(
        configured,
        ledger,
        BehindBaseGitHub(),
        kanban,
        LocalGit(),
    ).scan()

    assert result.created == 1
    assert kanban.tasks[0].evidence["triggers"] == ["base_refresh_required"]
    assert "git merge --no-ff --no-edit" in kanban.tasks[0].instructions
    assert "base_refresh_required" in kanban.tasks[0].instructions
    assert "github-pr-feedback complete-feedback" in kanban.tasks[0].instructions
    assert "--feedback-kind pr_repair" in kanban.tasks[0].instructions
    assert "--feedback-id repair:base_refresh_required" in kanban.tasks[0].instructions
    assert "--resolved-head-sha <full literal resolved head SHA>" in (
        kanban.tasks[0].instructions
    )
    assert "evaluating at most two viable resolutions" in kanban.tasks[0].instructions
    assert "Within 10 minutes" in kanban.tasks[0].instructions
    assert "pr-maintenance-receipt:v1" in kanban.tasks[0].instructions
    ledger.close()


def test_repair_controller_dispatches_base_refreshes_up_to_configured_limit(
    tmp_path: Path,
) -> None:
    configured = policy(
        tmp_path, merge_maintainer=True, max_base_refresh_in_flight=2
    )
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    kanban = Kanban()

    result = RepairController(
        configured,
        ledger,
        ManyBehindBaseGitHub(),
        kanban,
        LocalGit(),
    ).scan()

    assert result.created == 2
    assert result.skipped.get("base_refresh_serialized", 0) == 0
    assert len(kanban.tasks) == 2
    ledger.close()


def test_repair_controller_defaults_to_one_base_refresh_in_flight(
    tmp_path: Path,
) -> None:
    configured = policy(tmp_path, merge_maintainer=True)
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    kanban = Kanban()

    result = RepairController(
        configured,
        ledger,
        ManyBehindBaseGitHub(),
        kanban,
        LocalGit(),
    ).scan()

    assert result.created == 1
    assert result.skipped["base_refresh_serialized"] == 1
    assert len(kanban.tasks) == 1
    ledger.close()


def test_terminal_refresh_binding_does_not_hold_slot_before_archived_recovery(
    tmp_path: Path,
) -> None:
    configured = policy(tmp_path, merge_maintainer=True)
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    now = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)
    terminal = FeedbackReceipt(
        "acme/widgets", 17, "pr_repair", "repair:base_refresh_required", "7" * 40
    )
    archived = FeedbackReceipt(
        "acme/widgets", 18, "pr_repair", "repair:base_refresh_required", "8" * 40
    )
    for receipt, task_id in (
        (terminal, "triage-refresh-task"),
        (archived, "archived-refresh-task"),
    ):
        lease = ledger.claim(
            receipt,
            owner="repair-controller",
            claimed_at=now,
            stale_before=now,
        )
        assert lease is not None
        ledger.finalize(receipt, task_id, lease)
    kanban = StatusKanban(
        {
            "triage-refresh-task": "triage",
            "archived-refresh-task": "archived",
        }
    )

    result = RepairController(
        configured,
        ledger,
        ManyBehindBaseGitHub(),
        kanban,
        LocalGit(),
        clock=lambda: now,
    ).scan()

    assert result.created == 1
    assert result.skipped == {"base_refresh_serialized": 1}
    assert [task.evidence["pr_number"] for task in kanban.tasks] == [18]
    assert any(
        binding.task_id == "repair-task"
        for binding in ledger.pending_task_bindings_for_head(archived)
    )
    ledger.close()


def test_old_head_terminal_binding_does_not_hold_current_refresh_slot(
    tmp_path: Path,
) -> None:
    configured = policy(tmp_path, merge_maintainer=True)
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    now = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)
    old_head = FeedbackReceipt(
        "acme/widgets", 17, "pr_repair", "repair:base_refresh_required", "6" * 40
    )
    lease = ledger.claim(
        old_head,
        owner="repair-controller",
        claimed_at=now,
        stale_before=now,
    )
    assert lease is not None
    ledger.finalize(old_head, "stale-triage-task", lease)
    kanban = StatusKanban({"stale-triage-task": "triage"})

    result = RepairController(
        configured,
        ledger,
        ManyBehindBaseGitHub(),
        kanban,
        LocalGit(),
        clock=lambda: now,
    ).scan()

    assert result.created == 1
    assert result.skipped["base_refresh_serialized"] == 1
    assert [task.evidence["pr_number"] for task in kanban.tasks] == [18]
    ledger.close()


def test_base_refresh_prefers_clean_pr_before_conflicted_pr(tmp_path: Path) -> None:
    configured = policy(tmp_path, merge_maintainer=True)
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    kanban = Kanban()

    result = RepairController(
        configured,
        ledger,
        MixedConflictBaseGitHub(),
        kanban,
        LocalGit(),
    ).scan()

    assert result.created == 1
    assert kanban.tasks[0].evidence["pr_number"] == 18
    assert kanban.tasks[0].evidence["triggers"] == ["base_refresh_required"]
    ledger.close()


def test_current_base_pr_does_not_impersonate_an_in_flight_refresh_task(
    tmp_path: Path,
) -> None:
    configured = policy(tmp_path, merge_maintainer=True)
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    kanban = Kanban()

    result = RepairController(
        configured,
        ledger,
        RefreshInFlightGitHub(),
        kanban,
        LocalGit(),
    ).scan()

    assert result.created == 1
    assert result.skipped.get("base_refresh_in_flight", 0) == 0
    assert kanban.tasks[0].evidence["pr_number"] == 18
    ledger.close()


def test_unrelated_pending_feedback_does_not_consume_the_base_refresh_slot(
    tmp_path: Path,
) -> None:
    configured = policy(tmp_path, merge_maintainer=True)
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    now = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)
    unrelated = FeedbackReceipt(
        "acme/widgets", 17, "review_comment", "review-1", "7" * 40
    )
    lease = ledger.claim(
        unrelated,
        owner="feedback-controller",
        claimed_at=now,
        stale_before=now,
    )
    assert lease is not None
    ledger.finalize(unrelated, "feedback-task", lease)
    kanban = StatusKanban({"feedback-task": "running"})

    result = RepairController(
        configured,
        ledger,
        ManyBehindBaseGitHub(),
        kanban,
        LocalGit(),
        clock=lambda: now,
    ).scan()

    assert result.created == 1
    assert result.skipped["base_refresh_serialized"] == 1
    assert kanban.tasks[0].evidence["pr_number"] == 18
    ledger.close()


def test_archived_pending_feedback_is_superseded_by_current_base_refresh(
    tmp_path: Path,
) -> None:
    configured = policy(tmp_path, merge_maintainer=True)
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    now = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)
    unrelated = FeedbackReceipt(
        "acme/widgets", 17, "review_comment", "review-1", "7" * 40
    )
    lease = ledger.claim(
        unrelated,
        owner="feedback-controller",
        claimed_at=now,
        stale_before=now,
    )
    assert lease is not None
    ledger.finalize(unrelated, "archived-feedback-task", lease)
    report = FeedbackReceipt(
        "acme/widgets", 17, "pr_repair", "report:observation", "7" * 40
    )
    report_lease = ledger.claim(
        report,
        owner="report-controller",
        claimed_at=now,
        stale_before=now,
    )
    assert report_lease is not None
    ledger.finalize(report, "active-report-task", report_lease)

    class ArchivedKanban(Kanban):
        def task_status(self, board: str, task_id: str) -> str | None:
            assert board == "repairs"
            assert task_id == "archived-feedback-task"
            return "archived"

    class NewerSeventeenGitHub(ManyBehindBaseGitHub):
        def list_open_pull_requests(self, repository: str, owner: str):
            pulls = super().list_open_pull_requests(repository, owner)
            timestamp = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)
            return (
                replace(pulls[0], updated_at=timestamp + timedelta(minutes=1)),
                replace(pulls[1], updated_at=timestamp),
            )

    kanban = ArchivedKanban()
    result = RepairController(
        configured,
        ledger,
        NewerSeventeenGitHub(),
        kanban,
        LocalGit(),
        clock=lambda: now,
    ).scan()

    assert result.created == 1
    assert kanban.tasks[0].evidence["pr_number"] == 17
    action_rows = dict(
        ledger._connection.execute(
            "SELECT feedback_id, action_status FROM feedback_receipts "
            "WHERE repository = 'acme/widgets' AND pr_number = 17"
        )
    )
    assert action_rows["review-1"] == "superseded"
    assert action_rows["report:observation"] == "pending"
    ledger.close()


def test_report_only_repair_scan_creates_a_blocked_observation(tmp_path: Path) -> None:
    configured = policy(tmp_path, report_only=True)
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    kanban = Kanban()

    result = RepairController(configured, ledger, GitHub(), kanban, LocalGit()).scan()

    assert result.created == 1
    assert kanban.tasks[0].initial_status == "blocked"
    assert "Report only" in kanban.tasks[0].instructions
    ledger.close()


def test_unavailable_checks_do_not_hide_a_confirmed_merge_conflict(
    tmp_path: Path,
) -> None:
    configured = policy(tmp_path)
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    kanban = Kanban()

    result = RepairController(
        configured,
        ledger,
        GitHubWithoutChecks(),
        kanban,
        LocalGit(),
    ).scan()

    assert result.created == 1
    assert result.skipped["check_state_unavailable"] == 1
    assert result.degraded is False
    assert "merge_conflict" in kanban.tasks[0].evidence["triggers"]
    ledger.close()


def test_repair_scan_reads_independent_pull_states_concurrently(tmp_path: Path) -> None:
    configured = policy(tmp_path)
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    result = RepairController(
        configured,
        ledger,
        ConcurrentReadGitHub(),
        Kanban(),
        LocalGit(),
    ).scan()

    assert result.created == 0
    assert result.skipped == {"no_repair_trigger": 2}
    assert result.degraded is False
    ledger.close()


def test_report_only_receipt_does_not_block_later_active_repair(tmp_path: Path) -> None:
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    kanban = Kanban()
    kwargs = {
        "ledger": ledger,
        "github": GitHub(),
        "kanban": kanban,
        "local_git": LocalGit(),
        "clock": lambda: datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
    }

    report = RepairController(policy(tmp_path, report_only=True), **kwargs).scan()
    active = RepairController(policy(tmp_path, report_only=False), **kwargs).scan()

    assert report.created == 1
    assert active.created == 1
    assert [task.initial_status for task in kanban.tasks] == ["blocked", "running"]
    ledger.close()
