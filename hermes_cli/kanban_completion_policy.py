"""Apply completion contracts before durable Kanban transitions."""

from __future__ import annotations

import os
import re
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse


class CompletionPolicyError(ValueError):
    """A registered completion contract could not be satisfied."""


_FULL_SHA_RE = re.compile(r"[0-9a-fA-F]{40}")
_REPOSITORY_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
GitRunner = Callable[..., str]


def _git(workspace: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(workspace), *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CompletionPolicyError(f"could not inspect repository completion evidence: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "git command failed").strip()
        raise CompletionPolicyError(f"could not inspect repository completion evidence: {detail}")
    return result.stdout.strip()


def _remote_repository(url: str) -> str | None:
    """Return ``OWNER/REPO`` for a GitHub remote URL."""
    value = (url or "").strip()
    if value.startswith("git@github.com:"):
        path = value.removeprefix("git@github.com:")
    else:
        parsed = urlparse(value)
        if parsed.hostname != "github.com":
            return None
        path = parsed.path.lstrip("/")
    path = path.removesuffix(".git").strip("/")
    return path if _REPOSITORY_RE.fullmatch(path) else None


def _require_clean_workspace(workspace: Path, git_runner: GitRunner) -> None:
    if git_runner(workspace, "status", "--porcelain=v1", "--untracked-files=all"):
        raise CompletionPolicyError(
            "repository completion receipt rejected: worktree has uncommitted changes"
        )


def _assigned_base(task) -> tuple[str, str]:
    """Return immutable base evidence persisted by the control-plane dispatcher."""
    base_ref = (task.workspace_base_ref or "").strip()
    base_sha = (task.workspace_base_sha or "").strip().lower()
    if not base_ref or not _FULL_SHA_RE.fullmatch(base_sha):
        raise CompletionPolicyError(
            "repository completion receipt rejected: assigned base evidence is unavailable"
        )
    return base_ref, base_sha


def _load_bundled_github_package() -> None:
    """Load the bundled PR feedback package from the trusted source tree."""
    import sys

    module = sys.modules.get("github_pr_feedback")
    if module is not None:
        return
    import importlib.util

    from hermes_cli._startup_fast import project_root_str

    plugin_dir = Path(project_root_str()) / "plugins" / "github-pr-feedback" / "github_pr_feedback"
    spec = importlib.util.spec_from_file_location(
        "github_pr_feedback", plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load bundled github_pr_feedback from {plugin_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["github_pr_feedback"] = module
    spec.loader.exec_module(module)


def _authorized_github_client():
    """Create the canonical client after verifying the dedicated Hermes identity."""
    import importlib

    from hermes_cli.github_identity import GitHubAutomationIdentity

    _load_bundled_github_package()
    client_type = importlib.import_module("github_pr_feedback.github_client").GitHubClient
    identity = GitHubAutomationIdentity.from_environment()
    return client_type.for_automation_identity(
        expected_login=identity.expected_login,
        token_env=identity.token_env,
    )


def enforce_repository_handoff(
    *, task, metadata, github_client=None, git_runner: GitRunner | None = None,
) -> None:
    """Validate the terminal receipt for a dispatcher-managed Git worktree.

    Scratch work and ordinary directories retain their existing completion
    behavior. A managed worktree must state whether it changed the repository.
    Changed work is bound to the live clean checkout, remote-tracking refs, and
    an exact GitHub PR URL before the task can leave the in-flight state.
    """
    if task is None or task.workspace_kind != "worktree":
        return
    if not task.workspace_path:
        raise CompletionPolicyError(
            "repository completion receipt rejected: worktree path is unresolved"
        )
    receipt = metadata if isinstance(metadata, dict) else {}
    changed = receipt.get("repository_changes")
    if not isinstance(changed, bool):
        raise CompletionPolicyError(
            "repository completion receipt must set metadata.repository_changes to true or false"
        )
    workspace = Path(task.workspace_path).expanduser()
    if not workspace.is_dir():
        raise CompletionPolicyError(
            "repository completion receipt rejected: assigned worktree no longer exists"
        )
    run_git = git_runner or _git
    run_git(workspace, "rev-parse", "--show-toplevel")
    _require_clean_workspace(workspace, run_git)
    head = run_git(workspace, "rev-parse", "HEAD").lower()
    branch = run_git(workspace, "branch", "--show-current")
    if task.branch_name and task.branch_name != branch:
        raise CompletionPolicyError(
            f"repository completion receipt rejected: worktree branch does not match assigned branch {task.branch_name}"
        )
    if changed is False:
        base_ref, base_sha = _assigned_base(task)
        if head != base_sha:
            raise CompletionPolicyError(
                "repository completion receipt rejected: worktree HEAD differs from assigned "
                f"base {base_ref} at {base_sha}; declare repository_changes=true and provide a PR receipt"
            )
        return

    required = ("commit_sha", "pushed_branch", "repository", "base_branch", "pr_url")
    missing = [name for name in required if not isinstance(receipt.get(name), str) or not receipt[name].strip()]
    if missing:
        raise CompletionPolicyError(
            "repository completion receipt is missing: " + ", ".join(missing)
        )
    commit_sha = receipt["commit_sha"].strip().lower()
    pushed_branch = receipt["pushed_branch"].strip()
    repository = receipt["repository"].strip()
    base_branch = receipt["base_branch"].strip()
    pr_url = receipt["pr_url"].strip()
    if not _FULL_SHA_RE.fullmatch(commit_sha):
        raise CompletionPolicyError("repository completion receipt commit_sha must be a full 40-character SHA")
    if not _REPOSITORY_RE.fullmatch(repository):
        raise CompletionPolicyError("repository completion receipt repository must be OWNER/REPO")
    expected_pr_prefix = f"https://github.com/{repository}/pull/"
    if not pr_url.startswith(expected_pr_prefix) or not pr_url.removeprefix(expected_pr_prefix).isdigit():
        raise CompletionPolicyError(
            "repository completion receipt pr_url must be an exact GitHub PR URL for repository"
        )
    if head != commit_sha:
        raise CompletionPolicyError(
            f"repository completion receipt commit_sha does not match worktree HEAD {head}"
        )
    if branch != pushed_branch or (task.branch_name and task.branch_name != pushed_branch):
        raise CompletionPolicyError(
            f"repository completion receipt pushed_branch does not match assigned branch {task.branch_name or branch}"
        )
    if base_branch == pushed_branch:
        raise CompletionPolicyError("repository completion receipt base_branch must differ from pushed_branch")
    tracking_refs = run_git(
        workspace,
        "for-each-ref",
        "--format=%(refname:short)",
        "--points-at",
        "HEAD",
        "refs/remotes",
    ).splitlines()
    remote_names = [
        ref.split("/", 1)[0]
        for ref in tracking_refs
        if ref.endswith(f"/{pushed_branch}") and "/" in ref
    ]
    if not remote_names:
        raise CompletionPolicyError(
            "repository completion receipt has no remote-tracking ref proving the branch was pushed at commit_sha"
        )
    matched_remote = None
    for remote_name in remote_names:
        if (_remote_repository(run_git(workspace, "remote", "get-url", remote_name)) or "").lower() == repository.lower():
            matched_remote = remote_name
            break
    if matched_remote is None:
        raise CompletionPolicyError(
            "repository completion receipt repository does not match the pushed branch remote"
        )
    base_ref = f"refs/remotes/{matched_remote}/{base_branch}"
    run_git(workspace, "rev-parse", "--verify", "--end-of-options", base_ref + "^{commit}")
    try:
        run_git(workspace, "merge-base", "--is-ancestor", base_ref, "HEAD")
    except CompletionPolicyError as exc:
        raise CompletionPolicyError(
            "repository completion receipt base_branch is not an ancestor of commit_sha"
        ) from exc
    pr_number = int(pr_url.removeprefix(expected_pr_prefix))
    if pr_number < 1:
        raise CompletionPolicyError(
            "repository completion receipt pr_url must identify a positive pull request number"
        )
    try:
        pull_request = (github_client or _authorized_github_client()).get_pull_request(
            repository, pr_number,
        )
    except (ImportError, RuntimeError) as exc:
        raise CompletionPolicyError(
            "repository completion receipt could not verify pr_url through the authorized GitHub identity"
        ) from exc
    if (
        pull_request.number != pr_number
        or pull_request.base_repository.casefold() != repository.casefold()
    ):
        raise CompletionPolicyError(
            "repository completion receipt pr_url does not resolve to the declared repository and PR"
        )
    if pull_request.state != "OPEN":
        raise CompletionPolicyError(
            "repository completion receipt pr_url must resolve to an open pull request"
        )
    if pull_request.head_ref_name != pushed_branch:
        raise CompletionPolicyError(
            "repository completion receipt pushed branch does not match the pull request head branch"
        )
    if pull_request.head_sha.casefold() != commit_sha:
        raise CompletionPolicyError(
            "repository completion receipt commit_sha does not match the pull request head SHA"
        )
    if pull_request.base_branch != base_branch:
        raise CompletionPolicyError(
            "repository completion receipt base branch does not match the pull request base branch"
        )


def _load_bundled_github_pr_feedback_guard():
    """Import ``github_pr_feedback.repair_completion_policy`` from the bundled
    plugin source, not the worker's own (possibly plugin-disabled) sys.path.

    A dispatched worker profile that doesn't enable ``github-pr-feedback`` never
    puts it on sys.path, so a bare ``import github_pr_feedback...`` here raises
    ModuleNotFoundError even though its control-plane receipt still needs this
    guard enforced. Load it directly from the bundled plugin directory (part of
    this same trusted Hermes source tree, unlike a profile-local override
    manifest) via its real package name so its relative imports resolve.
    """
    _load_bundled_github_package()
    import importlib

    return importlib.import_module("github_pr_feedback.repair_completion_policy").guard_repair_completion


def enforce_completion_policies(*, task_id, board, assignee, summary):
    from hermes_cli.plugins import invoke_hook

    results = list(invoke_hook(
        "pre_kanban_complete", task_id=task_id, board=board,
        assignee=assignee, summary=summary,
    ))
    results.extend(_control_plane_github_feedback_results(task_id=task_id))
    _raise_on_policy_results(results)


def enforce_review_policies(*, task_id, board, assignee, summary):
    """Apply feedback contracts before handing a task to human review."""
    from hermes_cli.plugins import invoke_hook

    results = list(invoke_hook(
        "pre_kanban_review", task_id=task_id, board=board,
        assignee=assignee, summary=summary,
    ))
    results.extend(_control_plane_github_feedback_results(task_id=task_id))
    _raise_on_policy_results(results)


def _raise_on_policy_results(results):
    for result in results:
        if result is None:
            continue
        if not isinstance(result, dict) or result.get("action") not in {"block", "allow", "approve"}:
            raise CompletionPolicyError("Kanban completion policy returned an invalid decision")
        if result["action"] == "block":
            raise CompletionPolicyError(result.get("message") or "Kanban completion policy rejected this transition")


def _control_plane_github_feedback_results(*, task_id):
    """Run control-plane PR-feedback policy for dispatched workers with a task binding.

    Repair workers may run under a profile that intentionally does not enable the
    optional github-pr-feedback plugin.  Their durable dispatch receipt still
    lives in ``HERMES_CONTROL_HOME``; when that receipt binds this task, enforce
    the control-plane acknowledgement contract before allowing completion.
    """
    control_home = os.environ.get("HERMES_CONTROL_HOME", "").strip()
    worker_task = os.environ.get("HERMES_KANBAN_TASK", "").strip()
    if not control_home or not worker_task:
        return []
    ledger = Path(control_home) / "github-pr-feedback" / "ledger.sqlite3"
    if not ledger.exists():
        return []
    try:
        with closing(sqlite3.connect(ledger.resolve().as_uri() + "?mode=ro", uri=True, timeout=1)) as connection:
            table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='feedback_receipts'"
            ).fetchone()
            if not table:
                return []
            bound = connection.execute(
                "SELECT 1 FROM feedback_receipts WHERE task_id = ? AND feedback_kind IN "
                "('review_comment', 'issue_comment', 'review', 'pr_repair') "
                "AND NOT (feedback_kind = 'pr_repair' AND feedback_id LIKE 'report:%') LIMIT 1",
                (task_id,),
            ).fetchone()
            if not bound:
                return []
    except (OSError, sqlite3.Error, ValueError):
        return [{"action": "block", "message": "Kanban completion policy could not verify the control-plane GitHub PR feedback binding"}]
    try:
        guard_repair_completion = _load_bundled_github_pr_feedback_guard()
    except ImportError:
        return [{"action": "block", "message": "Kanban completion policy could not load the control-plane GitHub PR feedback guard"}]

    ctx = type("ControlPlaneFeedbackContext", (), {"get_config": staticmethod(lambda key, default=None: True if key == "enabled" else default)})()
    return [guard_repair_completion(ctx, task_id=task_id)]
