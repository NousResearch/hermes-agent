"""Governed closing of open PRs whose exact head is already on the base."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from hermes_cli.github_identity import GitHubAutomationIdentity, GitHubIdentityError

from .github_client import GitHubClient, GitHubClientError
from .policy import PluginPolicy, PullRequest, hermes_attribution_line
from .stack import _branch

_GITHUB_REMOTE = re.compile(
    r"^(?:https://github\.com/|ssh://git@github\.com/|git@github\.com:)"
    r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?/?$"
)


class SupersededCloseError(RuntimeError):
    """The requested close could not be proven safe."""


class ExactAncestorRunner:
    """Fetch and prove exact Git ancestry without changing local branches."""

    def __init__(
        self, repository: Path, *, environment: Mapping[str, str] | None = None
    ) -> None:
        self.repository = Path(repository)
        self._environment = None if environment is None else dict(environment)

    def exact_head_is_on_remote_base(
        self, head_sha: str, base_branch: str, *, expected_repository: str
    ) -> bool:
        head_sha = _full_sha(head_sha)
        base_branch = _branch(base_branch, "base_branch")
        remote_url = self._run("remote", "get-url", "origin").stdout.strip()
        match = _GITHUB_REMOTE.fullmatch(remote_url)
        if match is None or match.group(1).casefold() != expected_repository.casefold():
            raise SupersededCloseError(
                "local origin does not match the configured GitHub repository"
            )
        self._run("fetch", "origin", base_branch)
        try:
            result = subprocess.run(
                (
                    "git",
                    "-C",
                    str(self.repository),
                    "merge-base",
                    "--is-ancestor",
                    head_sha,
                    f"refs/remotes/origin/{base_branch}",
                ),
                check=False,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                env=self._environment,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise SupersededCloseError("git ancestry verification was unavailable") from error
        if result.returncode == 0:
            return True
        if result.returncode == 1:
            return False
        raise SupersededCloseError(
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
            raise SupersededCloseError("git verification was unavailable") from error
        if result.returncode:
            raise SupersededCloseError(result.stderr.strip() or "git fetch failed")
        return result


@dataclass(frozen=True, slots=True)
class SupersededCloseResult:
    repository: str
    pr_number: int
    head_sha: str
    base_branch: str
    state: str


class SupersededPullRequestController:
    """Close only a policy-owned exact head proven reachable from its base."""

    def __init__(self, policy: PluginPolicy, *, github: GitHubClient) -> None:
        self.policy = policy
        self.github = github

    def close(
        self,
        repository: str,
        pr_number: int,
        head_sha: str,
        *,
        repository_path: Path,
        git_environment: Mapping[str, str] | None = None,
    ) -> SupersededCloseResult:
        head_sha = _full_sha(head_sha)
        if not isinstance(pr_number, int) or isinstance(pr_number, bool) or pr_number < 1:
            raise ValueError("pr_number must be a positive integer")
        merge_policy = self.policy.merge_policy_for(repository)
        if merge_policy is None:
            raise SupersededCloseError(
                "repository is not configured for merge maintenance"
            )
        initial = self.github.get_pull_request(repository, pr_number)
        try:
            actual_base_branch = _branch(initial.base_branch, "pull request base")
        except ValueError as error:
            raise SupersededCloseError("pull request base is invalid") from error
        self._require_exact_open_pull(
            initial,
            repository=repository,
            pr_number=pr_number,
            head_sha=head_sha,
            base_branch=actual_base_branch,
        )
        if git_environment is None:
            identity = self.policy.github_identity
            if identity is None:
                raise SupersededCloseError(
                    "GitHub automation identity is not configured"
                )
            try:
                git_environment = GitHubAutomationIdentity(
                    identity.expected_login, identity.token_env
                ).git_command_environment()
            except GitHubIdentityError as error:
                raise SupersededCloseError(
                    "GitHub automation credential is unavailable"
                ) from error
        runner = ExactAncestorRunner(repository_path, environment=git_environment)
        if not runner.exact_head_is_on_remote_base(
            head_sha,
            merge_policy.base_branch,
            expected_repository=repository,
        ):
            raise SupersededCloseError(
                "exact pull request head is not an ancestor of the fetched base"
            )

        # This is deliberately the last read before the one bounded close call.
        # Any head, state, identity, or base drift aborts without mutation.
        current = self.github.get_pull_request(repository, pr_number)
        self._require_exact_open_pull(
            current,
            repository=repository,
            pr_number=pr_number,
            head_sha=head_sha,
            base_branch=actual_base_branch,
        )
        comment = (
            f"{hermes_attribution_line(merge_policy.assignee, action='superseded PR close')}\n\n"
            f"Closing PR #{pr_number} as superseded: exact head `{head_sha}` is already "
            f"reachable from freshly fetched `origin/{merge_policy.base_branch}`. "
            "No branch was deleted and no history was rewritten.\n\n"
            f"<!-- hermes-superseded-close:v1 head={head_sha} "
            f"base={merge_policy.base_branch} -->"
        )
        self.github.close_pull_request_with_comment(
            repository, pr_number, head_sha=head_sha, comment=comment
        )
        closed = self.github.get_pull_request(repository, pr_number)
        self._require_exact_identity(
            closed,
            repository=repository,
            pr_number=pr_number,
            head_sha=head_sha,
            base_branch=actual_base_branch,
        )
        if closed.state != "CLOSED":
            raise GitHubClientError("pull request did not close")
        return SupersededCloseResult(
            repository, pr_number, head_sha, merge_policy.base_branch, closed.state
        )

    def _require_exact_open_pull(
        self,
        pull: PullRequest,
        *,
        repository: str,
        pr_number: int,
        head_sha: str,
        base_branch: str,
    ) -> None:
        self._require_exact_identity(
            pull,
            repository=repository,
            pr_number=pr_number,
            head_sha=head_sha,
            base_branch=base_branch,
        )
        admission = self.policy.admit_pull_request(pull)
        if not admission.admitted:
            raise SupersededCloseError(
                f"pull request is outside configured ownership: {admission.reason}"
            )
        if pull.state != "OPEN":
            raise SupersededCloseError("pull request is not open")

    @staticmethod
    def _require_exact_identity(
        pull: PullRequest,
        *,
        repository: str,
        pr_number: int,
        head_sha: str,
        base_branch: str,
    ) -> None:
        if pull.base_repository != repository or pull.number != pr_number:
            raise SupersededCloseError("pull request repository or number changed")
        if pull.head_sha != head_sha:
            raise SupersededCloseError("pull request head changed")
        if pull.base_branch != base_branch:
            raise SupersededCloseError("pull request base changed")


def _full_sha(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(char not in "0123456789abcdefABCDEF" for char in value)
    ):
        raise ValueError("head_sha must be a full hexadecimal SHA")
    return value.casefold()
