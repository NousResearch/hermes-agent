from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from github_pr_feedback.github_client import Feedback, ReviewThread
from github_pr_feedback.ledger import FeedbackLedger
from github_pr_feedback.policy import (
    FeedbackReceipt,
    GitHubIdentityPolicy,
    MergeMaintainerPolicy,
    PluginPolicy,
    PullRequest,
    RepositoryTarget,
    Reviewer,
)
from github_pr_feedback.superseded_feedback import (
    SupersededFeedbackController,
    SupersededFeedbackError,
)

HEAD = "a" * 40
FIX = "c" * 40


def policy(repository_path: Path) -> PluginPolicy:
    repository = "acme/widgets"
    return PluginPolicy(
        enabled=True,
        targets={
            repository: RepositoryTarget(
                repository,
                repository,
                repository_path,
                "owner",
                ("codex/", "claude/", "hermes/"),
            )
        },
        reviewer_logins=frozenset(),
        reviewer_associations=frozenset(),
        include_self_feedback=False,
        include_bot_feedback=False,
        auto_dispatch=False,
        not_before=None,
        assignee="fallback",
        board="Pull Request Maintenance",
        merge_maintainer=MergeMaintainerPolicy(
            "merge-maintainer",
            repository,
            "owner",
            "stable",
            ("squash",),
            3600,
            True,
            None,
        ),
        github_identity=GitHubIdentityPolicy("hermes-bot", "HERMES_BOT_TOKEN"),
    )


def pull(
    *, head: str = HEAD, base: str = "codex/parent", state: str = "OPEN"
) -> PullRequest:
    return PullRequest(
        17,
        state,
        "acme/widgets",
        "acme/widgets",
        "owner",
        "codex/child",
        head,
        base_branch=base,
        base_sha="b" * 40,
    )


def feedback(
    kind: str, feedback_id: str, body: str, *, reviewer_login: str = "reviewer"
) -> Feedback:
    return Feedback(
        kind,
        feedback_id,
        Reviewer(reviewer_login, "MEMBER"),
        body,
        datetime(2026, 1, 1, tzinfo=UTC),
        False,
    )


class FakeGitHub:
    def __init__(self, pulls: list[PullRequest], threads: list[ReviewThread]) -> None:
        self.pulls = iter(pulls)
        self.threads = iter(threads)
        self.comments = [
            feedback("review_comment", "42", "Please fix the stale edge case")
        ]
        self.posts: list[str] = []
        self.resolves: list[tuple[str, int, str, str]] = []

    def get_pull_request(self, _repository: str, _number: int) -> PullRequest:
        return next(self.pulls)

    def list_feedback(self, _repository: str, _number: int) -> tuple[Feedback, ...]:
        return tuple(self.comments)

    def get_review_thread_for_comment(
        self,
        _repository: str,
        _number: int,
        _comment_id: str,
        *,
        expected_head_sha: str,
    ) -> ReviewThread:
        assert expected_head_sha == HEAD
        return next(self.threads)

    def post_issue_comment(self, _repository: str, _number: int, body: str) -> None:
        self.posts.append(body)
        self.comments.append(
            feedback("issue_comment", "9001", body, reviewer_login="hermes-bot")
        )

    def resolve_review_thread_for_comment(
        self, repository: str, number: int, comment_id: str, *, expected_head_sha: str
    ) -> bool:
        self.resolves.append((repository, number, comment_id, expected_head_sha))
        return True


def thread(*, resolved: bool = False) -> ReviewThread:
    return ReviewThread("PRRT_exact", "42", HEAD, resolved)


def git_success(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []

    def fake_run(argv, **_kwargs):
        calls.append(tuple(argv))
        stdout = "git@github.com:acme/widgets.git\n" if "get-url" in argv else ""
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr(
        "github_pr_feedback.superseded_feedback.subprocess.run", fake_run
    )
    return calls


def test_resolve_superseded_feedback_proves_identity_ancestry_and_post_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    git_calls = git_success(monkeypatch)
    github = FakeGitHub(
        [pull(), pull(), pull()],
        [thread(), thread(), thread(resolved=True)],
    )

    result = SupersededFeedbackController(
        policy(tmp_path),
        github=github,  # type: ignore[arg-type]
    ).resolve(
        "acme/widgets",
        17,
        HEAD,
        comment_id="42",
        fix_sha=FIX,
        repository_path=tmp_path,
        test_evidence="scripts/run_tests.sh tests/test_regression.py -q: 8 passed",
        git_environment={},
    )

    assert result.thread_resolved is True
    assert result.base_branch == "stable"
    assert github.resolves == [("acme/widgets", 17, "42", HEAD)]
    assert len(github.posts) == 1
    body = github.posts[0]
    assert FIX in body and HEAD in body
    assert "scripts/run_tests.sh tests/test_regression.py -q: 8 passed" in body
    assert f"comment_id=42 fix={FIX}" in body
    assert git_calls == [
        ("git", "-C", str(tmp_path), "remote", "get-url", "origin"),
        (
            "git",
            "-C",
            str(tmp_path),
            "fetch",
            "origin",
            "refs/heads/stable:refs/remotes/origin/stable",
        ),
        ("git", "-C", str(tmp_path), "merge-base", "--is-ancestor", HEAD, FIX),
        (
            "git",
            "-C",
            str(tmp_path),
            "merge-base",
            "--is-ancestor",
            FIX,
            "refs/remotes/origin/stable",
        ),
        (
            "git",
            "-C",
            str(tmp_path),
            "merge-base",
            "--is-ancestor",
            HEAD,
            "refs/remotes/origin/stable",
        ),
    ]


@pytest.mark.parametrize(
    ("pulls", "threads", "match"),
    (
        ([pull(), pull(head="d" * 40)], [thread()], "head changed"),
        ([pull(), pull(base="codex/other")], [thread()], "base changed"),
        ([pull()], [ReviewThread("PRRT_exact", "99", HEAD, False)], "comment identity"),
    ),
)
def test_resolve_superseded_feedback_fails_before_writes_on_identity_drift(
    pulls: list[PullRequest],
    threads: list[ReviewThread],
    match: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    git_success(monkeypatch)
    github = FakeGitHub(pulls, threads)

    with pytest.raises(SupersededFeedbackError, match=match):
        SupersededFeedbackController(
            policy(tmp_path),
            github=github,  # type: ignore[arg-type]
        ).resolve(
            "acme/widgets",
            17,
            HEAD,
            comment_id="42",
            fix_sha=FIX,
            repository_path=tmp_path,
            test_evidence="focused regression: passed",
            git_environment={},
        )

    assert github.posts == []
    assert github.resolves == []


def test_resolve_superseded_feedback_rejects_fix_not_between_head_and_stable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    results = iter((0, 0, 0, 1))

    def fake_run(argv, **_kwargs):
        stdout = "git@github.com:acme/widgets.git\n" if "get-url" in argv else ""
        return subprocess.CompletedProcess(argv, next(results), stdout, "")

    monkeypatch.setattr(
        "github_pr_feedback.superseded_feedback.subprocess.run", fake_run
    )
    github = FakeGitHub([pull()], [thread()])

    with pytest.raises(SupersededFeedbackError, match="fix commit is not an ancestor"):
        SupersededFeedbackController(
            policy(tmp_path),
            github=github,  # type: ignore[arg-type]
        ).resolve(
            "acme/widgets",
            17,
            HEAD,
            comment_id="42",
            fix_sha=FIX,
            repository_path=tmp_path,
            test_evidence="focused regression: passed",
            git_environment={},
        )

    assert github.posts == []
    assert github.resolves == []


def test_resolve_superseded_feedback_rejects_thread_drift_immediately_before_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    git_success(monkeypatch)
    github = FakeGitHub(
        [pull(), pull()],
        [thread(), ReviewThread("PRRT_changed", "42", HEAD, False)],
    )

    with pytest.raises(SupersededFeedbackError, match="thread identity changed"):
        SupersededFeedbackController(
            policy(tmp_path),
            github=github,  # type: ignore[arg-type]
        ).resolve(
            "acme/widgets",
            17,
            HEAD,
            comment_id="42",
            fix_sha=FIX,
            repository_path=tmp_path,
            test_evidence="focused regression: passed",
            git_environment={},
        )

    assert github.posts == []
    assert github.resolves == []


def test_resolve_superseded_feedback_fails_closed_when_thread_post_state_is_unresolved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    git_success(monkeypatch)
    github = FakeGitHub([pull(), pull()], [thread(), thread(), thread()])

    with pytest.raises(SupersededFeedbackError, match="resolution was not confirmed"):
        SupersededFeedbackController(
            policy(tmp_path),
            github=github,  # type: ignore[arg-type]
        ).resolve(
            "acme/widgets",
            17,
            HEAD,
            comment_id="42",
            fix_sha=FIX,
            repository_path=tmp_path,
            test_evidence="focused regression: passed",
            git_environment={},
        )

    assert len(github.posts) == 1
    assert len(github.resolves) == 1


def test_resolve_superseded_feedback_fails_closed_on_pr_post_state_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    git_success(monkeypatch)
    github = FakeGitHub(
        [pull(), pull(), pull(head="d" * 40)],
        [thread(), thread(), thread(resolved=True)],
    )

    with pytest.raises(SupersededFeedbackError, match="head changed"):
        SupersededFeedbackController(
            policy(tmp_path),
            github=github,  # type: ignore[arg-type]
        ).resolve(
            "acme/widgets",
            17,
            HEAD,
            comment_id="42",
            fix_sha=FIX,
            repository_path=tmp_path,
            test_evidence="focused regression: passed",
            git_environment={},
        )

    assert len(github.posts) == 1
    assert len(github.resolves) == 1


@pytest.mark.parametrize("seed_failed", (False, True))
def test_ledger_reconciles_only_exact_superseded_review_comment_idempotently(
    seed_failed: bool, tmp_path: Path
) -> None:
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    receipt = FeedbackReceipt("acme/widgets", 17, "review_comment", "42", HEAD)
    if seed_failed:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        lease = ledger.claim(
            receipt,
            owner="scanner",
            claimed_at=now,
            stale_before=now,
        )
        assert lease is not None
        ledger.fail(receipt, "dispatch failed", lease)

    actioned_at = datetime(2026, 1, 2, tzinfo=UTC)
    ledger.reconcile_superseded_feedback_action(
        receipt,
        stable_fix_sha=FIX,
        actioned_at=actioned_at,
    )
    ledger.reconcile_superseded_feedback_action(
        receipt,
        stable_fix_sha=FIX,
        actioned_at=actioned_at,
    )

    row = ledger._connection.execute(
        "SELECT status, action_status, actioned_head_sha, last_error FROM feedback_receipts "
        "WHERE repository = ? AND pr_number = ? AND feedback_kind = ? "
        "AND feedback_id = ? AND head_sha = ?",
        receipt.key,
    ).fetchone()
    assert row == ("completed", "completed", FIX, None)
    assert ledger.was_actioned_on_any_head(receipt)
    ledger.close()


def test_cli_wires_literal_resolve_superseded_feedback_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from github_pr_feedback import cli

    seen: list[tuple[object, ...]] = []

    class Controller:
        def __init__(self, configured_policy, *, github) -> None:
            assert configured_policy == "policy"
            assert github == "bot-client"

        def resolve(self, repository, pr_number, head_sha, **kwargs):
            seen.append((repository, pr_number, head_sha, kwargs))
            return SimpleNamespace(
                repository=repository,
                pr_number=pr_number,
                head_sha=head_sha,
                fix_sha=kwargs["fix_sha"],
                comment_id=kwargs["comment_id"],
                base_branch="stable",
                thread_resolved=True,
            )

    monkeypatch.setattr(cli, "_load_policy_from_context", lambda _ctx: "policy")
    monkeypatch.setattr(cli, "_github_client", lambda _policy: "bot-client")
    monkeypatch.setattr(cli, "SupersededFeedbackController", Controller)

    class Ledger:
        def __init__(self) -> None:
            self.reconciled: list[tuple[FeedbackReceipt, str]] = []

        def reconcile_superseded_feedback_action(
            self, receipt: FeedbackReceipt, *, stable_fix_sha: str, actioned_at
        ) -> None:
            assert actioned_at.tzinfo is not None
            self.reconciled.append((receipt, stable_fix_sha))

        def close(self) -> None:
            pass

    ledger = Ledger()
    monkeypatch.setattr(cli.FeedbackLedger, "for_current_profile", lambda: ledger)
    parser = argparse.ArgumentParser()
    cli.setup_cli(object(), parser)
    args = parser.parse_args([
        "resolve-superseded-feedback",
        "--repository",
        "acme/widgets",
        "--pr-number",
        "17",
        "--head-sha",
        HEAD,
        "--comment-id",
        "42",
        "--fix-sha",
        FIX,
        "--repository-path",
        str(tmp_path),
        "--test-evidence",
        "focused regression: passed",
    ])

    assert cli.handle_cli_with_context(object(), args) == 0
    assert seen[0][:3] == ("acme/widgets", 17, HEAD)
    assert seen[0][3] == {
        "comment_id": "42",
        "fix_sha": FIX,
        "repository_path": tmp_path,
        "test_evidence": "focused regression: passed",
    }
    assert (
        json.loads(capsys.readouterr().out)["status"] == "resolved_superseded_feedback"
    )
    assert ledger.reconciled == [
        (FeedbackReceipt("acme/widgets", 17, "review_comment", "42", HEAD), FIX)
    ]
