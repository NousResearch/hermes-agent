"""Deterministic, fail-closed refresh of one exact pull-request head."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from .ci_runner import CompletedCommand
from .github_client import PullRequestMergeState
from .policy import CODEX_REVIEW_TRIGGER


_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,240}$")
_SHA = re.compile(r"^[0-9a-fA-F]{40}$")


class BaseRefreshGitHub(Protocol):
    def get_merge_state(self, repository: str, number: int) -> PullRequestMergeState: ...

    def post_issue_comment(self, repository: str, number: int, body: str) -> None: ...


class BaseRefreshCommandRunner(Protocol):
    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout: int,
    ) -> CompletedCommand: ...


class SubprocessBaseRefreshCommandRunner:
    """Run only literal argv; the refresher never invokes a shell."""

    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout: int,
    ) -> CompletedCommand:
        try:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                env=dict(env),
                check=False,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return CompletedCommand(
                returncode=1,
                stdout="",
                stderr=type(error).__name__,
                duration_ms=0,
                timed_out=isinstance(error, subprocess.TimeoutExpired),
            )
        return CompletedCommand(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_ms=0,
            timed_out=False,
        )


@dataclass(frozen=True, slots=True)
class BaseRefreshIdentity:
    repository: str
    pr_number: int
    observed_base_sha: str
    target_base_sha: str
    base_branch: str
    head_repository: str
    head_branch: str
    head_sha: str

    def __post_init__(self) -> None:
        if not _REPOSITORY.fullmatch(self.repository) or not _REPOSITORY.fullmatch(
            self.head_repository
        ):
            raise ValueError("base refresh repository identity is invalid")
        if not isinstance(self.pr_number, int) or isinstance(self.pr_number, bool) or self.pr_number < 1:
            raise ValueError("base refresh pull request number is invalid")
        if not _REF.fullmatch(self.base_branch) or not _REF.fullmatch(self.head_branch):
            raise ValueError("base refresh branch identity is invalid")
        for field in ("observed_base_sha", "target_base_sha", "head_sha"):
            value = getattr(self, field)
            if not isinstance(value, str) or not _SHA.fullmatch(value):
                raise ValueError(f"{field} must be a full hexadecimal SHA")
            object.__setattr__(self, field, value.casefold())


@dataclass(frozen=True, slots=True)
class BaseRefreshResult:
    status: str
    reason: str | None = None
    resolved_head_sha: str | None = None
    receipt_id: str | None = None


class DeterministicBaseRefresher:
    """Refresh a clean exact head without delegating Git choices to a model."""

    def __init__(
        self,
        github: BaseRefreshGitHub,
        *,
        command_runner: BaseRefreshCommandRunner | None = None,
    ) -> None:
        self._github = github
        self._commands = command_runner or SubprocessBaseRefreshCommandRunner()

    def refresh(self, identity: BaseRefreshIdentity, worktree: Path) -> BaseRefreshResult:
        worktree = Path(worktree).resolve()
        if not worktree.is_dir() or not (worktree / "scripts/run_static_lane.py").is_file():
            return BaseRefreshResult("handoff", "workspace_unavailable")
        try:
            initial = self._github.get_merge_state(identity.repository, identity.pr_number)
        except Exception:  # noqa: BLE001 - uncertain canonical identity fails closed.
            return BaseRefreshResult("handoff", "identity_unavailable")
        if not _matches_initial(identity, initial):
            return BaseRefreshResult("handoff", "identity_race")
        if not self._clean_exact_head(worktree, identity.head_sha):
            return BaseRefreshResult("handoff", "workspace_not_clean")

        base_url = f"https://github.com/{identity.repository}.git"
        if not self._ok(
            (
                "git",
                "fetch",
                "--quiet",
                "--no-tags",
                "--no-recurse-submodules",
                base_url,
                f"refs/heads/{identity.base_branch}",
            ),
            worktree,
            timeout=60,
        ):
            return BaseRefreshResult("handoff", "base_fetch_failed")
        fetched = self._stdout(
            ("git", "rev-parse", "--verify", "FETCH_HEAD^{commit}"), worktree
        )
        if fetched.casefold() != identity.target_base_sha:
            return BaseRefreshResult("handoff", "target_base_mismatch")
        if not self._ok(
            ("git", "merge", "--no-ff", "--no-edit", identity.target_base_sha),
            worktree,
            timeout=120,
        ):
            self._restore_exact_head(worktree, identity.head_sha)
            return BaseRefreshResult("handoff", "merge_conflict")
        resolved = self._stdout(("git", "rev-parse", "--verify", "HEAD"), worktree)
        if not _SHA.fullmatch(resolved) or resolved.casefold() == identity.head_sha:
            return BaseRefreshResult("handoff", "merge_result_invalid")
        resolved = resolved.casefold()
        if not self._is_clean(worktree):
            return BaseRefreshResult("handoff", "merge_result_not_clean")

        environment = dict(os.environ)
        environment["STATIC_BASE_REF"] = identity.target_base_sha
        static = self._commands.run(
            (".venv/bin/python", "scripts/run_static_lane.py"),
            cwd=worktree,
            env=environment,
            timeout=3600,
        )
        if static.returncode != 0 or static.timed_out:
            return BaseRefreshResult("handoff", "static_failed")
        if not self._clean_exact_head(worktree, resolved):
            return BaseRefreshResult("handoff", "static_mutated_worktree")
        try:
            before_push = self._github.get_merge_state(
                identity.repository, identity.pr_number
            )
        except Exception:  # noqa: BLE001 - uncertain canonical identity fails closed.
            return BaseRefreshResult("handoff", "identity_unavailable")
        if not _matches_initial(identity, before_push):
            return BaseRefreshResult("handoff", "identity_race")

        if not self._ok(
            (
                "git",
                "push",
                f"https://github.com/{identity.head_repository}.git",
                f"HEAD:refs/heads/{identity.head_branch}",
            ),
            worktree,
            timeout=120,
        ):
            return BaseRefreshResult("handoff", "push_failed")
        receipt_id = _receipt_id(identity, resolved, static)
        try:
            after_push = self._github.get_merge_state(
                identity.repository, identity.pr_number
            )
        except Exception:  # noqa: BLE001 - retain pushed state for reconciliation.
            return BaseRefreshResult(
                "reconciliation_pending",
                "identity_unavailable_after_push",
                resolved,
                receipt_id,
            )
        if not _matches_resolved(identity, after_push, resolved):
            return BaseRefreshResult(
                "reconciliation_pending", "identity_race_after_push", resolved, receipt_id
            )
        try:
            before_comment = self._github.get_merge_state(
                identity.repository, identity.pr_number
            )
            if not _matches_resolved(identity, before_comment, resolved):
                raise ValueError("identity changed before comment")
            self._github.post_issue_comment(
                identity.repository,
                identity.pr_number,
                _receipt_comment(identity, resolved, receipt_id),
            )
        except Exception:  # noqa: BLE001 - retain pushed state for reconciliation.
            return BaseRefreshResult(
                "reconciliation_pending", "comment_unavailable", resolved, receipt_id
            )
        return BaseRefreshResult("completed", None, resolved, receipt_id)

    def _restore_exact_head(self, worktree: Path, head_sha: str) -> None:
        """Leave a failed merge behind as a clean worktree at the exact head.

        A conflicted merge that is handed off unresolved keeps its unmerged
        index entries, which poisons every later attempt: the deterministic
        path then fails ``workspace_not_clean`` and the handoff worker's own
        merge fails "Merging is not possible because you have unmerged files."
        ``--abort`` is tried first so an ordinary conflict is unwound normally;
        the hard reset is the fallback for a half-cleaned state where
        ``MERGE_HEAD`` is already gone and ``--abort`` therefore refuses.
        """

        if self._clean_exact_head(worktree, head_sha):
            return
        self._ok(("git", "merge", "--abort"), worktree, timeout=60)
        if self._clean_exact_head(worktree, head_sha):
            return
        self._ok(("git", "reset", "--hard", head_sha), worktree, timeout=60)
        self._ok(("git", "clean", "-fd"), worktree, timeout=60)

    def _clean_exact_head(self, worktree: Path, expected: str) -> bool:
        head = self._stdout(("git", "rev-parse", "--verify", "HEAD"), worktree)
        return head.casefold() == expected.casefold() and self._is_clean(worktree)

    def _is_clean(self, worktree: Path) -> bool:
        result = self._run(
            ("git", "status", "--porcelain=v1", "--untracked-files=all"),
            worktree,
            timeout=30,
        )
        return result.returncode == 0 and not result.stdout.strip()

    def _stdout(self, argv: tuple[str, ...], worktree: Path) -> str:
        result = self._run(argv, worktree, timeout=30)
        return result.stdout.strip() if result.returncode == 0 else ""

    def _ok(self, argv: tuple[str, ...], worktree: Path, *, timeout: int) -> bool:
        result = self._run(argv, worktree, timeout=timeout)
        return result.returncode == 0 and not result.timed_out

    def _run(
        self, argv: tuple[str, ...], worktree: Path, *, timeout: int
    ) -> CompletedCommand:
        return self._commands.run(argv, cwd=worktree, env=dict(os.environ), timeout=timeout)


def _matches_initial(identity: BaseRefreshIdentity, state: PullRequestMergeState) -> bool:
    return (
        state.repository == identity.repository
        and state.number == identity.pr_number
        and state.state == "OPEN"
        and not state.merged
        and state.base_branch == identity.base_branch
        and state.base_sha.casefold() == identity.observed_base_sha
        and state.head_repository == identity.head_repository
        and state.head_ref_name == identity.head_branch
        and state.head_sha.casefold() == identity.head_sha
    )


def _matches_resolved(
    identity: BaseRefreshIdentity, state: PullRequestMergeState, resolved: str
) -> bool:
    return (
        state.repository == identity.repository
        and state.number == identity.pr_number
        and state.state == "OPEN"
        and not state.merged
        and state.base_branch == identity.base_branch
        and state.base_sha.casefold() == identity.target_base_sha
        and state.head_repository == identity.head_repository
        and state.head_ref_name == identity.head_branch
        and state.head_sha.casefold() == resolved
    )


def _receipt_id(
    identity: BaseRefreshIdentity, resolved: str, static: CompletedCommand
) -> str:
    payload = "\0".join(
        (
            identity.repository,
            str(identity.pr_number),
            identity.head_sha,
            identity.target_base_sha,
            resolved,
            hashlib.sha256(static.stdout.encode("utf-8")).hexdigest(),
            hashlib.sha256(static.stderr.encode("utf-8")).hexdigest(),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _receipt_comment(
    identity: BaseRefreshIdentity, resolved: str, receipt_id: str
) -> str:
    return (
        f"Base refresh completed from exact head `{identity.head_sha}` onto exact base "
        f"`{identity.target_base_sha}` at `{resolved}`. Exact-base static lane passed. "
        f"Deterministic receipt: `{receipt_id}`. No pull request merge was performed; "
        "normal merge gates remain authoritative. "
        f"<!-- pr-maintenance-receipt:v1 status=completed kind=pr_repair head={resolved} -->\n\n"
        # This head has moved past whatever Codex last reviewed (the merge just
        # forwarded it onto a new base); Codex never re-reviews on its own after
        # a push, only on this explicit mention.
        f"{CODEX_REVIEW_TRIGGER}"
    )
