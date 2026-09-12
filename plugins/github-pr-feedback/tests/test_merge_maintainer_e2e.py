from __future__ import annotations

import hashlib
import subprocess
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from github_pr_feedback.ci_runner import CIAuditIdentity, CIAuditReceipt
from github_pr_feedback.cli import _run_merge_scan
from github_pr_feedback.github_client import (
    CheckState,
    Feedback,
    PullRequestMergeState,
    RepositoryMergePolicy,
    ReviewState,
)
from github_pr_feedback.ledger import FeedbackLedger
from github_pr_feedback.policy import PullRequest, Reviewer, load_policy


BASE_SHA = "b" * 40
HEAD_SHA = "a" * 40
MERGE_SHA = "c" * 40


class CanonicalFakeGitHub:
    def __init__(
        self,
        states: list[PullRequestMergeState],
        feedback: tuple[Feedback, ...] = (),
    ) -> None:
        self.states = states
        self.feedback = feedback
        self.merge_calls: list[tuple[str, int, str, str]] = []

    def list_open_pull_requests(self, repository: str, owner: str) -> tuple[PullRequest, ...]:
        return (
            PullRequest(
                17,
                "OPEN",
                repository,
                repository,
                owner,
                "codex/fix; $(printenv GH_TOKEN)",
                HEAD_SHA,
            ),
        )

    def get_merge_state(self, repository: str, number: int) -> PullRequestMergeState:
        return self.states.pop(0)

    def get_branch_head(self, repository: str, branch: str) -> str:
        return BASE_SHA

    def repository_is_private(self, repository: str) -> bool:
        return True

    def get_repository_merge_policy(self, repository: str) -> RepositoryMergePolicy:
        return RepositoryMergePolicy(squash=False, rebase=True, merge=True)

    def get_review_state(self, repository: str, number: int) -> ReviewState:
        return ReviewState("APPROVED", 0)

    def get_check_state(self, repository: str, head_sha: str) -> CheckState:
        return CheckState(False, True, 0)

    def list_feedback(self, repository: str, number: int) -> tuple[Feedback, ...]:
        return self.feedback

    def merge_pull_request(
        self, repository: str, number: int, head_sha: str, *, method: str
    ) -> None:
        self.merge_calls.append((repository, number, head_sha, method))


class RecordingKanban:
    def __init__(self) -> None:
        self.tasks = []

    def create_or_get_task(self, task) -> str:
        self.tasks.append(task)
        return "task-17"


def open_state() -> PullRequestMergeState:
    return PullRequestMergeState(
        repository="acme/widgets",
        number=17,
        state="OPEN",
        is_draft=False,
        mergeable=True,
        merge_state_status="CLEAN",
        base_branch="stable",
        base_sha=BASE_SHA,
        head_repository="acme/widgets",
        author_login="owner",
        head_ref_name="codex/fix; $(printenv GH_TOKEN)",
        head_sha=HEAD_SHA,
        merged=False,
        merge_commit_oid=None,
    )


def configured_policy(repository: Path, *, report_only: bool = False):
    return load_policy(
        {
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
            "include_self_feedback": True,
            "not_before": "2026-08-24T00:00:00Z",
            "assignee": "repair-agent",
            "board": "repairs",
            "merge_maintainer": {
                "enabled": True,
                "assignee": "pr-merge-maintainer",
                "repository": "acme/widgets",
                "author_login": "owner",
                "base_branch": "stable",
                "merge_methods": ["squash", "rebase", "merge"],
                "receipt_max_age_seconds": 21600,
                "report_only": report_only,
                "post_merge": {"enabled": False},
            },
        }
    )


def prepare_receipt(repository: Path, ledger: FeedbackLedger) -> None:
    manifest = repository / "tests/manifests/test_lanes.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('[lanes.unit]\nci_status = "required"\n', encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    now = datetime.now(UTC)
    ledger.record_ci_receipt(
        CIAuditReceipt(
            receipt_id="d" * 64,
            identity=CIAuditIdentity("acme/widgets", 17, BASE_SHA, HEAD_SHA),
            manifest_digest=digest,
            status="passed",
            started_at=now - timedelta(minutes=2),
            completed_at=now - timedelta(minutes=1),
            actions_state=CheckState(False, True, 0),
            commands=(),
        )
    )


def test_end_to_end_exact_head_receipt_selects_enabled_method_and_merges_once(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    prepare_receipt(repository, ledger)
    merged = replace(
        open_state(), state="CLOSED", merged=True, merge_commit_oid=MERGE_SHA
    )
    github = CanonicalFakeGitHub([open_state(), open_state(), merged])
    kanban = RecordingKanban()

    payload = _run_merge_scan(
        configured_policy(repository), ledger, github=github, kanban=kanban
    )

    assert payload["status"] == "ok"
    assert payload["merged"] == [
        {
            "pr_number": 17,
            "head_sha": HEAD_SHA,
            "method": "rebase",
            "merge_commit_oid": MERGE_SHA,
        }
    ]
    assert github.merge_calls == [("acme/widgets", 17, HEAD_SHA, "rebase")]
    assert kanban.tasks == []
    assert all("GH_TOKEN" not in str(field) for field in github.merge_calls[0])
    ledger.close()


def test_end_to_end_report_only_creates_readiness_task_without_a_write(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    prepare_receipt(repository, ledger)
    github = CanonicalFakeGitHub([open_state()])
    kanban = RecordingKanban()

    payload = _run_merge_scan(
        configured_policy(repository, report_only=True),
        ledger,
        github=github,
        kanban=kanban,
    )

    assert payload["report_only"] is True
    assert payload["merged"] == []
    assert github.merge_calls == []
    assert len(kanban.tasks) == 1
    assert kanban.tasks[0].evidence["eligible"] is True
    ledger.close()


def test_superseded_owner_ci_status_comment_does_not_block_merge(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    prepare_receipt(repository, ledger)
    old_head = "e" * 40
    stale_comment = Feedback(
        "issue_comment",
        "5416219124",
        Reviewer("owner", "OWNER"),
        (
            f"Local CI audit completed for exact head `{old_head}` (base `{BASE_SHA}`). "
            f"Authoritative receipt: `{'f' * 64}`. The failed receipt remains "
            "merge-blocking; later fail-fast lanes may be absent."
        ),
        datetime(2026, 8, 25, 20, 18, tzinfo=UTC),
        False,
    )
    merged = replace(
        open_state(), state="CLOSED", merged=True, merge_commit_oid=MERGE_SHA
    )
    github = CanonicalFakeGitHub(
        [open_state(), open_state(), merged], feedback=(stale_comment,)
    )

    payload = _run_merge_scan(
        configured_policy(repository), ledger, github=github, kanban=RecordingKanban()
    )

    assert payload["merged"][0]["pr_number"] == 17
    assert github.merge_calls == [("acme/widgets", 17, HEAD_SHA, "rebase")]
    ledger.close()


def test_current_actionable_review_feedback_remains_merge_blocking(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    prepare_receipt(repository, ledger)
    actionable = Feedback(
        "review_comment",
        "3857347391",
        Reviewer("reviewer", "MEMBER"),
        "Align daily windows to the completed market close.",
        datetime(2026, 8, 25, 20, 59, tzinfo=UTC),
        False,
    )
    github = CanonicalFakeGitHub([open_state()], feedback=(actionable,))

    payload = _run_merge_scan(
        configured_policy(repository), ledger, github=github, kanban=RecordingKanban()
    )

    assert payload["merged"] == []
    assert payload["blocked"] == {"17": ["feedback_unprocessed"]}
    assert github.merge_calls == []
    ledger.close()
