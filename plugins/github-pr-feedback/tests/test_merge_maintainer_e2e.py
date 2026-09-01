from __future__ import annotations

import hashlib
import subprocess
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from github_pr_feedback.ci_runner import (
    CIAuditIdentity,
    CIAuditReceipt,
    CommandEvidence,
    _receipt_id,
)
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


def codex_review_comment(head_sha: str = HEAD_SHA) -> Feedback:
    """A completed chatgpt-codex-connector[bot] review summary for this head.

    Every e2e fixture includes this by default so these tests exercise the
    *other* merge gates -- a test specifically about the codex_review_pending
    gate itself lives in test_merge_controller.py.
    """

    return Feedback(
        "issue_comment",
        "codex-summary-1",
        Reviewer("chatgpt-codex-connector[bot]", None),
        (
            "<!-- codex-pull-request-review-summary -->\n\n## Codex Review Summary\n\n"
            "| Review | Status | Commit | Review trigger |\n"
            "| --- | --- | --- | --- |\n"
            "| Code Review | Completed "
            '<relative-time datetime="2026-08-25T20:00:00Z"></relative-time> | '
            f"`{head_sha[:7]}` | PR opened |"
        ),
        datetime(2026, 8, 25, 20, 0, tzinfo=UTC),
        True,
    )


class CanonicalFakeGitHub:
    def __init__(
        self,
        states: list[PullRequestMergeState],
        feedback: tuple[Feedback, ...] = (),
    ) -> None:
        self.states = states
        self.feedback = (codex_review_comment(), *feedback)
        self.merge_calls: list[tuple[str, int, str, str]] = []
        self.comments: list[tuple[str, int, str]] = []

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

    def post_issue_comment(self, repository: str, number: int, body: str) -> None:
        self.comments.append((repository, number, body))
        self.feedback = (
            *self.feedback,
            Feedback(
                "issue_comment",
                str(len(self.feedback) + 1),
                Reviewer("github-pr-feedback-bot", None),
                body,
                datetime(2026, 8, 25, 21, 0, tzinfo=UTC),
                True,
            ),
        )


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
    started_at = now - timedelta(minutes=2)
    completed_at = now - timedelta(minutes=1)
    identity = CIAuditIdentity("acme/widgets", 17, BASE_SHA, HEAD_SHA)
    commands = (
        CommandEvidence(
            argv=("python3", "scripts/run_test_lane.py"),
            cwd=str(repository),
            returncode=0,
            duration_ms=1,
            timed_out=False,
            stdout_sha256="0" * 64,
            stderr_sha256="0" * 64,
            classification="passed",
        ),
    )
    ledger.record_ci_receipt(
        CIAuditReceipt(
            receipt_id=_receipt_id(identity, digest, "passed", completed_at, commands),
            identity=identity,
            manifest_digest=digest,
            status="passed",
            started_at=started_at,
            completed_at=completed_at,
            actions_state=CheckState(False, True, 0),
            commands=commands,
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
    assert len(github.comments) == 1
    posted_repository, posted_number, posted_body = github.comments[0]
    assert (posted_repository, posted_number) == ("acme/widgets", 17)
    assert "Ready to merge" in posted_body
    assert f"<!-- pr-ready-to-merge-receipt:v1 head={HEAD_SHA} -->" in posted_body
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


def test_codexs_own_review_summary_tracker_does_not_block_merge(tmp_path: Path) -> None:
    """Codex's running review-status comment never itself blocks feedback_unprocessed.

    It only ever reports which review ran and when -- never a finding -- so
    admitting it as pending feedback would block every otherwise-eligible PR
    on a comment that never asked for anything.
    """

    repository = tmp_path / "repository"
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    prepare_receipt(repository, ledger)
    tracker = Feedback(
        "issue_comment",
        "codex-tracker",
        Reviewer("chatgpt-codex-connector[bot]", None),
        "<!-- codex-pull-request-review-summary -->\n\n## Codex Review Summary",
        datetime(2026, 8, 25, 20, 18, tzinfo=UTC),
        True,
    )
    merged = replace(open_state(), state="CLOSED", merged=True, merge_commit_oid=MERGE_SHA)
    github = CanonicalFakeGitHub([open_state(), open_state(), merged], feedback=(tracker,))

    payload = _run_merge_scan(
        configured_policy(repository), ledger, github=github, kanban=RecordingKanban()
    )

    assert payload["merged"][0]["pr_number"] == 17
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
