"""Governed resolution of review feedback superseded by a fix already on stable."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from hermes_cli.github_identity import GitHubAutomationIdentity, GitHubIdentityError

from .github_client import GitHubClient, ReviewThread
from .policy import PluginPolicy, PullRequest, hermes_attribution_line
from .stack import _branch

_GITHUB_REMOTE = re.compile(
    r"^(?:https://github\.com/|ssh://git@github\.com/|git@github\.com:)"
    r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?/?$"
)


class SupersededFeedbackError(RuntimeError):
    """The requested feedback resolution could not be proven safe."""


@dataclass(frozen=True, slots=True)
class SupersededFeedbackResult:
    repository: str
    pr_number: int
    head_sha: str
    comment_id: str
    fix_sha: str
    base_branch: str
    thread_resolved: bool


class ExactFixAncestryRunner:
    """Prove an exact PR head and fix are both ordered on fetched stable."""

    def __init__(
        self, repository: Path, *, environment: Mapping[str, str] | None = None
    ) -> None:
        self.repository = Path(repository)
        self._environment = None if environment is None else dict(environment)

    def prove(
        self,
        head_sha: str,
        fix_sha: str,
        base_branch: str,
        *,
        expected_repository: str,
    ) -> None:
        head_sha = _full_sha(head_sha, "head_sha")
        fix_sha = _full_sha(fix_sha, "fix_sha")
        if head_sha == fix_sha:
            raise SupersededFeedbackError("fix_sha must be a descendant commit")
        base_branch = _branch(base_branch, "base_branch")
        remote_url = self._run("remote", "get-url", "origin").stdout.strip()
        match = _GITHUB_REMOTE.fullmatch(remote_url)
        if match is None or match.group(1).casefold() != expected_repository.casefold():
            raise SupersededFeedbackError(
                "local origin does not match the configured GitHub repository"
            )
        self._run(
            "fetch",
            "origin",
            f"refs/heads/{base_branch}:refs/remotes/origin/{base_branch}",
        )
        remote_base = f"refs/remotes/origin/{base_branch}"
        if not self._is_ancestor(head_sha, fix_sha):
            raise SupersededFeedbackError(
                "fix commit is not a descendant of the exact pull request head"
            )
        if not self._is_ancestor(fix_sha, remote_base):
            raise SupersededFeedbackError(
                "fix commit is not an ancestor of the fetched configured base"
            )
        if not self._is_ancestor(head_sha, remote_base):
            raise SupersededFeedbackError(
                "exact pull request head is not an ancestor of the fetched configured base"
            )

    def _is_ancestor(self, ancestor: str, descendant: str) -> bool:
        try:
            result = subprocess.run(
                (
                    "git",
                    "-C",
                    str(self.repository),
                    "merge-base",
                    "--is-ancestor",
                    ancestor,
                    descendant,
                ),
                check=False,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                env=self._environment,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise SupersededFeedbackError(
                "git ancestry verification was unavailable"
            ) from error
        if result.returncode == 0:
            return True
        if result.returncode == 1:
            return False
        raise SupersededFeedbackError(
            result.stderr.strip() or "git ancestry verification failed"
        )

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                ("git", "-C", str(self.repository), *args),
                check=False,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                env=self._environment,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise SupersededFeedbackError("git verification was unavailable") from error
        if result.returncode:
            raise SupersededFeedbackError(result.stderr.strip() or "git fetch failed")
        return result


class SupersededFeedbackController:
    """Resolve one exact review thread only when its fix is already on stable."""

    def __init__(self, policy: PluginPolicy, *, github: GitHubClient) -> None:
        self.policy = policy
        self.github = github

    def resolve(
        self,
        repository: str,
        pr_number: int,
        head_sha: str,
        *,
        comment_id: str,
        fix_sha: str,
        repository_path: Path,
        test_evidence: str,
        git_environment: Mapping[str, str] | None = None,
    ) -> SupersededFeedbackResult:
        head_sha = _full_sha(head_sha, "head_sha")
        fix_sha = _full_sha(fix_sha, "fix_sha")
        comment_id = _comment_id(comment_id)
        test_evidence = _bounded_evidence(test_evidence)
        if (
            not isinstance(pr_number, int)
            or isinstance(pr_number, bool)
            or pr_number < 1
        ):
            raise ValueError("pr_number must be a positive integer")
        merge_policy = self.policy.merge_policy_for(repository)
        if merge_policy is None:
            raise SupersededFeedbackError(
                "repository is not configured for merge maintenance"
            )
        identity = self.policy.github_identity
        if identity is None:
            raise SupersededFeedbackError(
                "GitHub automation identity is not configured"
            )

        initial = self.github.get_pull_request(repository, pr_number)
        actual_base = self._require_exact_open_pull(
            initial,
            repository=repository,
            pr_number=pr_number,
            head_sha=head_sha,
        )
        self._require_exact_review_comment(repository, pr_number, comment_id)
        initial_thread = self.github.get_review_thread_for_comment(
            repository, pr_number, comment_id, expected_head_sha=head_sha
        )
        self._require_thread_identity(initial_thread, comment_id, head_sha)

        if git_environment is None:
            try:
                git_environment = GitHubAutomationIdentity(
                    identity.expected_login, identity.token_env
                ).git_command_environment()
            except GitHubIdentityError as error:
                raise SupersededFeedbackError(
                    "GitHub automation credential is unavailable"
                ) from error
        ExactFixAncestryRunner(repository_path, environment=git_environment).prove(
            head_sha,
            fix_sha,
            merge_policy.base_branch,
            expected_repository=repository,
        )

        marker = _receipt_marker(
            head_sha=head_sha,
            comment_id=comment_id,
            fix_sha=fix_sha,
            base_branch=merge_policy.base_branch,
        )
        marker_present = self._reply_marker_present(repository, pr_number, marker)

        # Last canonical PR read before any write. The actual stacked base is
        # identity too, even though containment is proved against stable.
        current = self.github.get_pull_request(repository, pr_number)
        current_base = self._require_exact_open_pull(
            current,
            repository=repository,
            pr_number=pr_number,
            head_sha=head_sha,
        )
        if current_base != actual_base:
            raise SupersededFeedbackError("pull request base changed")
        current_thread = self.github.get_review_thread_for_comment(
            repository, pr_number, comment_id, expected_head_sha=head_sha
        )
        self._require_thread_identity(
            current_thread,
            comment_id,
            head_sha,
            expected_thread_id=initial_thread.thread_id,
        )

        if current_thread.is_resolved and not marker_present:
            raise SupersededFeedbackError(
                "review thread was already resolved without the exact factual reply"
            )
        if not marker_present:
            body = (
                f"{hermes_attribution_line(merge_policy.assignee, action='superseded feedback resolution')}\n\n"
                f"Resolving review comment `{comment_id}` on PR #{pr_number}: exact PR head "
                f"`{head_sha}` and fix commit `{fix_sha}` are both reachable from freshly "
                f"fetched `origin/{merge_policy.base_branch}`, with the fix descending from "
                "that PR head.\n\n"
                f"Verified test evidence: {test_evidence}\n\n{marker}"
            )
            self.github.post_issue_comment(repository, pr_number, body)
            if not self._reply_marker_present(repository, pr_number, marker):
                raise SupersededFeedbackError("factual reply post was not confirmed")

        if not current_thread.is_resolved:
            self.github.resolve_review_thread_for_comment(
                repository,
                pr_number,
                comment_id,
                expected_head_sha=head_sha,
            )

        closed_thread = self.github.get_review_thread_for_comment(
            repository, pr_number, comment_id, expected_head_sha=head_sha
        )
        self._require_thread_identity(
            closed_thread,
            comment_id,
            head_sha,
            expected_thread_id=initial_thread.thread_id,
        )
        if not closed_thread.is_resolved:
            raise SupersededFeedbackError("review thread resolution was not confirmed")
        final = self.github.get_pull_request(repository, pr_number)
        final_base = self._require_exact_open_pull(
            final,
            repository=repository,
            pr_number=pr_number,
            head_sha=head_sha,
        )
        if final_base != actual_base:
            raise SupersededFeedbackError("pull request base changed after resolution")
        if not self._reply_marker_present(repository, pr_number, marker):
            raise SupersededFeedbackError("factual reply post-state was not confirmed")
        return SupersededFeedbackResult(
            repository,
            pr_number,
            head_sha,
            comment_id,
            fix_sha,
            merge_policy.base_branch,
            True,
        )

    def _require_exact_open_pull(
        self,
        pull: PullRequest,
        *,
        repository: str,
        pr_number: int,
        head_sha: str,
    ) -> str:
        if pull.base_repository != repository or pull.number != pr_number:
            raise SupersededFeedbackError("pull request repository or number changed")
        if pull.head_sha != head_sha:
            raise SupersededFeedbackError("pull request head changed")
        try:
            actual_base = _branch(pull.base_branch or "", "pull request base")
        except ValueError as error:
            raise SupersededFeedbackError("pull request base is invalid") from error
        admission = self.policy.admit_pull_request(pull)
        if not admission.admitted:
            raise SupersededFeedbackError(
                f"pull request is outside configured ownership: {admission.reason}"
            )
        if pull.state != "OPEN":
            raise SupersededFeedbackError("pull request is not open")
        return actual_base

    def _require_exact_review_comment(
        self, repository: str, pr_number: int, comment_id: str
    ) -> None:
        matches = [
            item
            for item in self.github.list_feedback(repository, pr_number)
            if item.kind == "review_comment" and item.feedback_id == comment_id
        ]
        if len(matches) != 1:
            raise SupersededFeedbackError(
                "review comment identity did not match exactly one review comment"
            )

    @staticmethod
    def _require_thread_identity(
        thread: ReviewThread,
        comment_id: str,
        head_sha: str,
        *,
        expected_thread_id: str | None = None,
    ) -> None:
        if thread.comment_id != comment_id or thread.head_sha != head_sha:
            raise SupersededFeedbackError("review thread comment identity changed")
        if expected_thread_id is not None and thread.thread_id != expected_thread_id:
            raise SupersededFeedbackError("review thread identity changed")

    def _reply_marker_present(
        self, repository: str, pr_number: int, marker: str
    ) -> bool:
        identity = self.policy.github_identity
        if identity is None:
            raise SupersededFeedbackError(
                "GitHub automation identity is not configured"
            )
        return any(
            item.kind == "issue_comment"
            and item.reviewer.login.casefold() == identity.expected_login.casefold()
            and marker in item.body
            for item in self.github.list_feedback(repository, pr_number)
        )


def _receipt_marker(
    *, head_sha: str, comment_id: str, fix_sha: str, base_branch: str
) -> str:
    return (
        f"<!-- hermes-superseded-feedback:v1 head={head_sha} "
        f"comment_id={comment_id} fix={fix_sha} base={base_branch} -->"
    )


def _full_sha(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(char not in "0123456789abcdefABCDEF" for char in value)
    ):
        raise ValueError(f"{field} must be a full hexadecimal SHA")
    return value.casefold()


def _comment_id(value: str) -> str:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("comment_id must be a positive integer") from error
    if parsed < 1 or str(parsed) != value:
        raise ValueError("comment_id must be a canonical positive integer")
    return value


def _bounded_evidence(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 1000:
        raise ValueError("test_evidence must contain 1 to 1000 characters")
    return value.strip()
