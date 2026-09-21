from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from github_pr_feedback.github_client import GitHubClient
from github_pr_feedback.policy import (
    MergeMaintainerPolicy,
    PluginPolicy,
    PullRequest,
    RepositoryTarget,
)
from github_pr_feedback.superseded import (
    ExactAncestorRunner,
    SupersededCloseError,
    SupersededPullRequestController,
)

SHA = "a" * 40


class RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def run(self, argv: list[str]) -> str:
        self.calls.append(tuple(argv))
        return json.dumps({})


class FakeGitHub:
    def __init__(self, pulls: list[PullRequest]) -> None:
        self.pulls = iter(pulls)
        self.reads = 0
        self.closes: list[tuple[str, int, str, str]] = []

    def get_pull_request(self, repository: str, number: int) -> PullRequest:
        self.reads += 1
        return next(self.pulls)

    def close_pull_request_with_comment(
        self,
        repository: str,
        number: int,
        *,
        head_sha: str,
        comment: str,
    ) -> None:
        self.closes.append((repository, number, head_sha, comment))


def policy(repository_path: Path) -> PluginPolicy:
    repository = "acme/widgets"
    target = RepositoryTarget(
        repository,
        repository,
        repository_path,
        "owner",
        ("codex/", "claude/", "hermes/"),
    )
    merge = MergeMaintainerPolicy(
        "merge-maintainer",
        repository,
        "owner",
        "stable",
        ("squash",),
        3600,
        True,
        None,
    )
    return PluginPolicy(
        enabled=True,
        targets={repository: target},
        reviewer_logins=frozenset(),
        reviewer_associations=frozenset(),
        include_self_feedback=False,
        include_bot_feedback=False,
        auto_dispatch=False,
        not_before=None,
        assignee="fallback",
        board="Pull Request Maintenance",
        merge_maintainer=merge,
    )


def pull(*, state: str = "OPEN", head_sha: str = SHA, base: str = "stable") -> PullRequest:
    return PullRequest(
        17,
        state,
        "acme/widgets",
        "acme/widgets",
        "owner",
        "codex/already-landed",
        head_sha,
        base_branch=base,
        base_sha="b" * 40,
    )


def test_exact_ancestor_runner_fetches_base_then_checks_exact_head(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(argv, **_kwargs):
        calls.append(tuple(argv))
        stdout = "git@github.com:acme/widgets.git\n" if "get-url" in argv else ""
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr("github_pr_feedback.superseded.subprocess.run", fake_run)

    assert ExactAncestorRunner(tmp_path).exact_head_is_on_remote_base(
        SHA, "stable", expected_repository="acme/widgets"
    )
    assert calls == [
        ("git", "-C", str(tmp_path), "remote", "get-url", "origin"),
        ("git", "-C", str(tmp_path), "fetch", "origin", "stable"),
        (
            "git",
            "-C",
            str(tmp_path),
            "merge-base",
            "--is-ancestor",
            SHA,
            "refs/remotes/origin/stable",
        ),
    ]
    assert not any(
        option in calls[2]
        for option in ("merge", "rebase", "push", "--force", "--delete")
    )


def test_close_superseded_rereads_exact_identity_and_closes_without_branch_delete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    git_calls: list[tuple[str, ...]] = []

    def fake_run(argv, **_kwargs):
        git_calls.append(tuple(argv))
        stdout = "https://github.com/acme/widgets.git\n" if "get-url" in argv else ""
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr("github_pr_feedback.superseded.subprocess.run", fake_run)
    github = FakeGitHub([pull(), pull(), pull(state="CLOSED")])

    result = SupersededPullRequestController(
        policy(tmp_path), github=github  # type: ignore[arg-type]
    ).close(
        "acme/widgets", 17, SHA, repository_path=tmp_path, git_environment={}
    )

    assert result.state == "CLOSED"
    assert github.reads == 3
    assert len(github.closes) == 1
    repository, number, head_sha, comment = github.closes[0]
    assert (repository, number, head_sha) == ("acme/widgets", 17, SHA)
    assert "Hermes automated superseded PR close (merge-maintainer)" in comment
    assert f"exact head `{SHA}`" in comment
    assert "origin/stable" in comment
    assert "No branch was deleted" in comment
    assert "hermes-superseded-close:v1" in comment
    assert all("push" not in call and "rebase" not in call for call in git_calls)


def test_close_superseded_allows_stacked_base_when_head_is_on_merge_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    git_calls: list[tuple[str, ...]] = []

    def fake_run(argv, **_kwargs):
        git_calls.append(tuple(argv))
        stdout = "git@github.com:acme/widgets.git\n" if "get-url" in argv else ""
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr("github_pr_feedback.superseded.subprocess.run", fake_run)
    github = FakeGitHub(
        [
            pull(base="codex/stack-parent"),
            pull(base="codex/stack-parent"),
            pull(state="CLOSED", base="codex/stack-parent"),
        ]
    )

    result = SupersededPullRequestController(
        policy(tmp_path), github=github  # type: ignore[arg-type]
    ).close(
        "acme/widgets", 17, SHA, repository_path=tmp_path, git_environment={}
    )

    assert result.state == "CLOSED"
    assert result.base_branch == "stable"
    assert github.reads == 3
    assert len(github.closes) == 1
    assert (
        "git",
        "-C",
        str(tmp_path),
        "merge-base",
        "--is-ancestor",
        SHA,
        "refs/remotes/origin/stable",
    ) in git_calls


def test_close_superseded_rejects_nonancestor_without_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    results = iter((0, 0, 1))

    def fake_run(argv, **_kwargs):
        stdout = "git@github.com:acme/widgets.git\n" if "get-url" in argv else ""
        return subprocess.CompletedProcess(argv, next(results), stdout, "")

    monkeypatch.setattr("github_pr_feedback.superseded.subprocess.run", fake_run)
    github = FakeGitHub([pull()])

    with pytest.raises(SupersededCloseError, match="not an ancestor"):
        SupersededPullRequestController(
            policy(tmp_path), github=github  # type: ignore[arg-type]
        ).close(
            "acme/widgets", 17, SHA, repository_path=tmp_path, git_environment={}
        )

    assert github.reads == 1
    assert github.closes == []


def test_close_superseded_rejects_a_local_checkout_for_another_repository(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(argv, **_kwargs):
        calls.append(tuple(argv))
        return subprocess.CompletedProcess(
            argv, 0, "git@github.com:other/project.git\n", ""
        )

    monkeypatch.setattr("github_pr_feedback.superseded.subprocess.run", fake_run)
    github = FakeGitHub([pull()])

    with pytest.raises(SupersededCloseError, match="local origin does not match"):
        SupersededPullRequestController(
            policy(tmp_path), github=github  # type: ignore[arg-type]
        ).close(
            "acme/widgets", 17, SHA, repository_path=tmp_path, git_environment={}
        )

    assert calls == [("git", "-C", str(tmp_path), "remote", "get-url", "origin")]
    assert github.closes == []


def test_close_superseded_rejects_head_drift_immediately_before_close(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "github_pr_feedback.superseded.subprocess.run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(
            argv,
            0,
            "git@github.com:acme/widgets.git\n" if "get-url" in argv else "",
            "",
        ),
    )
    github = FakeGitHub(
        [
            pull(base="codex/stack-parent"),
            pull(head_sha="c" * 40, base="codex/stack-parent"),
        ]
    )

    with pytest.raises(SupersededCloseError, match="head changed"):
        SupersededPullRequestController(
            policy(tmp_path), github=github  # type: ignore[arg-type]
        ).close(
            "acme/widgets", 17, SHA, repository_path=tmp_path, git_environment={}
        )

    assert github.reads == 2
    assert github.closes == []


def test_close_superseded_rejects_actual_base_drift_immediately_before_close(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    git_calls: list[tuple[str, ...]] = []

    def fake_run(argv, **_kwargs):
        git_calls.append(tuple(argv))
        stdout = "git@github.com:acme/widgets.git\n" if "get-url" in argv else ""
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr("github_pr_feedback.superseded.subprocess.run", fake_run)
    github = FakeGitHub(
        [
            pull(base="codex/stack-parent"),
            pull(base="codex/different-parent"),
        ]
    )

    with pytest.raises(SupersededCloseError, match="base changed"):
        SupersededPullRequestController(
            policy(tmp_path), github=github  # type: ignore[arg-type]
        ).close(
            "acme/widgets", 17, SHA, repository_path=tmp_path, git_environment={}
        )

    assert github.reads == 2
    assert github.closes == []
    assert any("merge-base" in call for call in git_calls)


@pytest.mark.parametrize(
    "changed",
    (
        replace(pull(), state="CLOSED"),
        replace(pull(), head_repository="fork/widgets"),
        replace(pull(), author_login="intruder"),
        replace(pull(), head_ref_name="feature/unowned"),
    ),
)
def test_close_superseded_rejects_nonexact_open_policy_identity(
    changed: PullRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        "github_pr_feedback.superseded.subprocess.run",
        lambda argv, **_kwargs: calls.append(tuple(argv)),
    )
    github = FakeGitHub([changed])

    with pytest.raises(SupersededCloseError):
        SupersededPullRequestController(
            policy(tmp_path), github=github  # type: ignore[arg-type]
        ).close(
            "acme/widgets", 17, SHA, repository_path=tmp_path, git_environment={}
        )

    assert calls == []
    assert github.closes == []


def test_github_close_uses_fixed_argv_without_branch_deletion() -> None:
    runner = RecordingRunner()
    comment = "Hermes automated superseded PR close (merge-maintainer)"

    GitHubClient(runner).close_pull_request_with_comment(
        "acme/widgets", 17, head_sha=SHA, comment=comment
    )

    assert runner.calls == [
        (
            "gh",
            "pr",
            "close",
            "17",
            "--repo",
            "acme/widgets",
            "--comment",
            comment,
        )
    ]
    assert "--delete-branch" not in runner.calls[0]


def test_cli_wires_explicit_exact_head_close_superseded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from github_pr_feedback import cli

    seen: list[tuple[str, int, str, Path]] = []

    class Controller:
        def __init__(self, configured_policy, *, github) -> None:
            assert configured_policy == "policy"
            assert github == "identity-bound-client"

        def close(
            self,
            repository: str,
            pr_number: int,
            head_sha: str,
            *,
            repository_path: Path,
        ):
            seen.append((repository, pr_number, head_sha, repository_path))
            return SimpleNamespace(
                repository=repository,
                pr_number=pr_number,
                head_sha=head_sha,
                base_branch="stable",
            )

    monkeypatch.setattr(cli, "_load_policy_from_context", lambda _ctx: "policy")
    monkeypatch.setattr(cli, "_github_client", lambda _policy: "identity-bound-client")
    monkeypatch.setattr(cli, "SupersededPullRequestController", Controller)
    parser = argparse.ArgumentParser()
    cli.setup_cli(object(), parser)
    args = parser.parse_args(
        [
            "close-superseded",
            "--repository",
            "acme/widgets",
            "--pr-number",
            "17",
            "--head-sha",
            SHA,
            "--repository-path",
            str(tmp_path),
        ]
    )

    assert cli.handle_cli_with_context(object(), args) == 0
    assert seen == [("acme/widgets", 17, SHA, tmp_path)]
    assert json.loads(capsys.readouterr().out) == {
        "status": "closed_superseded",
        "repository": "acme/widgets",
        "pr_number": 17,
        "head_sha": SHA,
        "base_branch": "stable",
    }
