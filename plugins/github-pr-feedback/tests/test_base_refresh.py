from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from github_pr_feedback.base_refresh import (
    BaseRefreshIdentity,
    DeterministicBaseRefresher,
)
from github_pr_feedback.ci_runner import CompletedCommand
from github_pr_feedback.github_client import PullRequestMergeState


OLD_BASE = "b" * 40
TARGET_BASE = "c" * 40
HEAD = "a" * 40
RESOLVED = "d" * 40


def merge_state(**overrides: object) -> PullRequestMergeState:
    values: dict[str, object] = {
        "repository": "acme/widgets",
        "number": 17,
        "state": "OPEN",
        "is_draft": False,
        "mergeable": True,
        "merge_state_status": "CLEAN",
        "base_branch": "stable",
        "base_sha": OLD_BASE,
        "head_repository": "acme/widgets",
        "author_login": "owner",
        "head_ref_name": "codex/fix",
        "head_sha": HEAD,
        "merged": False,
        "merge_commit_oid": None,
    }
    values.update(overrides)
    return PullRequestMergeState(**values)


class FakeGitHub:
    def __init__(self, states: list[PullRequestMergeState]) -> None:
        self.states = states
        self.comments: list[str] = []

    def get_merge_state(self, repository: str, number: int) -> PullRequestMergeState:
        assert (repository, number) == ("acme/widgets", 17)
        return self.states.pop(0)

    def post_issue_comment(self, repository: str, number: int, body: str) -> None:
        assert (repository, number) == ("acme/widgets", 17)
        self.comments.append(body)


class RecordingRunner:
    def __init__(self, *, fail: str | None = None) -> None:
        self.fail = fail
        self.calls: list[tuple[tuple[str, ...], Path, dict[str, str]]] = []
        self.head = HEAD

    def run(self, argv, *, cwd, env, timeout):
        argv = tuple(argv)
        self.calls.append((argv, cwd, dict(env)))
        key = " ".join(argv)
        if argv[:3] == ("git", "rev-parse", "--verify"):
            if argv[-1] == "HEAD":
                output = self.head + "\n"
            elif argv[-1] == "FETCH_HEAD^{commit}":
                output = TARGET_BASE + "\n"
            else:
                output = ""
        elif argv[:2] == ("git", "merge"):
            if self.fail == "merge":
                return CompletedCommand(1, "", "conflict", 4, False)
            self.head = RESOLVED
            output = "merged\n"
        elif argv[:2] == ("git", "status"):
            output = ""
        elif "run_static_lane.py" in key:
            if self.fail == "static":
                return CompletedCommand(1, "", "static failed", 4, False)
            output = '{"status":"pass"}\n'
        elif argv[:2] == ("git", "push"):
            if self.fail == "push":
                return CompletedCommand(1, "", "non-fast-forward", 4, False)
            output = "pushed\n"
        else:
            output = ""
        return CompletedCommand(0, output, "", 4, False)


def identity() -> BaseRefreshIdentity:
    return BaseRefreshIdentity(
        repository="acme/widgets",
        pr_number=17,
        observed_base_sha=OLD_BASE,
        target_base_sha=TARGET_BASE,
        base_branch="stable",
        head_repository="acme/widgets",
        head_branch="codex/fix",
        head_sha=HEAD,
    )


def prepare_worktree(path: Path) -> None:
    script = path / "scripts/run_static_lane.py"
    script.parent.mkdir(parents=True)
    script.write_text("# fixture\n", encoding="utf-8")
    executable = path / ".venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n", encoding="utf-8")


def test_conflict_free_refresh_uses_only_literal_safe_git_and_posts_receipt(
    tmp_path: Path,
) -> None:
    prepare_worktree(tmp_path)
    github = FakeGitHub(
        [
            merge_state(),
            merge_state(),
            merge_state(head_sha=RESOLVED, base_sha=TARGET_BASE),
            merge_state(head_sha=RESOLVED, base_sha=TARGET_BASE),
        ]
    )
    commands = RecordingRunner()

    result = DeterministicBaseRefresher(github, command_runner=commands).refresh(
        identity(), tmp_path
    )

    assert result.status == "completed"
    assert result.resolved_head_sha == RESOLVED
    assert result.receipt_id is not None
    argv = [call[0] for call in commands.calls]
    assert (
        "git",
        "fetch",
        "--quiet",
        "--no-tags",
        "--no-recurse-submodules",
        "https://github.com/acme/widgets.git",
        "refs/heads/stable",
    ) in argv
    assert ("git", "merge", "--no-ff", "--no-edit", TARGET_BASE) in argv
    assert (
        "git",
        "push",
        "https://github.com/acme/widgets.git",
        f"HEAD:refs/heads/codex/fix",
    ) in argv
    assert not any(
        word in {"checkout", "reset", "rebase", "pull", "--force", "-f"}
        for call in argv
        for word in call
    )
    static_call = next(
        call
        for call in commands.calls
        if "scripts/run_static_lane.py" in call[0]
    )
    assert static_call[2]["STATIC_BASE_REF"] == TARGET_BASE
    assert github.comments and result.receipt_id in github.comments[0]
    assert "No pull request merge was performed" in github.comments[0]


def test_static_failure_hands_off_without_push_or_comment(tmp_path: Path) -> None:
    prepare_worktree(tmp_path)
    github = FakeGitHub([merge_state()])
    commands = RecordingRunner(fail="static")

    result = DeterministicBaseRefresher(github, command_runner=commands).refresh(
        identity(), tmp_path
    )

    assert result.status == "handoff"
    assert result.reason == "static_failed"
    assert not any(call[0][:2] == ("git", "push") for call in commands.calls)
    assert github.comments == []


def test_identity_race_hands_off_before_fetch_or_merge(tmp_path: Path) -> None:
    prepare_worktree(tmp_path)
    github = FakeGitHub([replace(merge_state(), head_sha="e" * 40)])
    commands = RecordingRunner()

    result = DeterministicBaseRefresher(github, command_runner=commands).refresh(
        identity(), tmp_path
    )

    assert result.status == "handoff"
    assert result.reason == "identity_race"
    assert not any(call[0][:2] in {("git", "fetch"), ("git", "merge")} for call in commands.calls)


class ConflictLeavesUnmergedRunner(RecordingRunner):
    """Model the real post-conflict worktree: unmerged entries until unwound."""

    def __init__(self, *, abort_works: bool = True) -> None:
        super().__init__(fail="merge")
        self.abort_works = abort_works
        self.unmerged = False

    def run(self, argv, *, cwd, env, timeout):
        argv = tuple(argv)
        if argv[:2] == ("git", "merge") and argv[2:3] != ("--abort",):
            self.unmerged = True
            self.calls.append((argv, cwd, dict(env)))
            return CompletedCommand(1, "", "conflict", 4, False)
        if argv[:3] == ("git", "merge", "--abort"):
            self.calls.append((argv, cwd, dict(env)))
            if not self.abort_works:
                return CompletedCommand(128, "", "no MERGE_HEAD", 4, False)
            self.unmerged = False
            return CompletedCommand(0, "", "", 4, False)
        if argv[:3] == ("git", "reset", "--hard"):
            self.calls.append((argv, cwd, dict(env)))
            self.unmerged = False
            return CompletedCommand(0, "", "", 4, False)
        if argv[:2] == ("git", "status") and self.unmerged:
            self.calls.append((argv, cwd, dict(env)))
            return CompletedCommand(0, "UU live_runner.py\n", "", 4, False)
        return super().run(argv, cwd=cwd, env=env, timeout=timeout)


def test_merge_conflict_leaves_worktree_clean_at_exact_head(tmp_path: Path) -> None:
    # A conflicted merge handed off unresolved used to keep its unmerged index
    # entries, so the next deterministic attempt failed workspace_not_clean and
    # the handoff worker's own merge failed with "Merging is not possible
    # because you have unmerged files."
    prepare_worktree(tmp_path)
    commands = ConflictLeavesUnmergedRunner()

    result = DeterministicBaseRefresher(
        FakeGitHub([merge_state()]), command_runner=commands
    ).refresh(identity(), tmp_path)

    assert result.status == "handoff"
    assert result.reason == "merge_conflict"
    assert commands.unmerged is False, "failed merge must not be left unmerged"
    assert any(call[0][:3] == ("git", "merge", "--abort") for call in commands.calls)
    assert not any(call[0][:2] == ("git", "push") for call in commands.calls)


def test_merge_conflict_hard_resets_when_abort_refuses(tmp_path: Path) -> None:
    # MERGE_HEAD can already be gone while the index still holds unmerged
    # entries; `git merge --abort` refuses there, so the reset is the fallback.
    prepare_worktree(tmp_path)
    commands = ConflictLeavesUnmergedRunner(abort_works=False)

    result = DeterministicBaseRefresher(
        FakeGitHub([merge_state()]), command_runner=commands
    ).refresh(identity(), tmp_path)

    assert result.reason == "merge_conflict"
    assert commands.unmerged is False
    assert any(call[0][:3] == ("git", "reset", "--hard") for call in commands.calls)
    assert any(call[0][:2] == ("git", "clean") for call in commands.calls)


def test_merge_conflict_hands_off_without_push_or_comment(tmp_path: Path) -> None:
    prepare_worktree(tmp_path)
    github = FakeGitHub([merge_state()])
    commands = RecordingRunner(fail="merge")

    result = DeterministicBaseRefresher(github, command_runner=commands).refresh(
        identity(), tmp_path
    )

    assert result.status == "handoff"
    assert result.reason == "merge_conflict"
    assert not any(call[0][:2] == ("git", "push") for call in commands.calls)
    assert github.comments == []
