#!/usr/bin/env python3
"""Preflight GitHub PR push/create strategy and emit a reviewable receipt.

This script intentionally stays small and dependency-free. It answers the
questions that should be known before an agent tries to push or open a PR:

- Which GitHub repository does the current remote point to?
- Is the working tree clean?
- Is GitHub CLI authenticated?
- Does the active account have write-level permission on upstream?
- Should the branch push directly to origin or to a fork remote?
- Is a fork PR blocked on maintainer workflow approval rather than CI failure?

It prints JSON so callers can route, summarize, or write PR bodies without
scraping terminal prose.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

WRITE_LEVEL_PERMISSIONS = {"ADMIN", "MAINTAIN", "WRITE"}
FORK_LEVEL_PERMISSIONS = {"READ", "TRIAGE", "NONE"}
REPO_SLUG_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
PR_NUMBER_RE = re.compile(r"^[1-9][0-9]*$")

CommandRunner = Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]]


def parse_github_remote_url(url: str) -> str | None:
    """Return owner/repo for common GitHub remote URL forms."""

    cleaned = url.strip()
    patterns = (
        r"^https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$",
        r"^git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$",
        r"^ssh://git@github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$",
    )
    for pattern in patterns:
        match = re.match(pattern, cleaned)
        if match:
            candidate = f"{match.group('owner')}/{match.group('repo')}"
            try:
                return validate_repo_slug(candidate)
            except ValueError:
                return None
    return None


def validate_repo_slug(value: str) -> str:
    """Validate OWNER/REPO before passing it to gh as a positional value."""

    candidate = value.strip()
    if not REPO_SLUG_RE.fullmatch(candidate):
        raise ValueError("Repository must be in OWNER/REPO form.")
    if any(part.startswith("-") for part in candidate.split("/")):
        raise ValueError("Repository owner/name must not start with '-'.")
    return candidate


def parse_pr_identifier(value: str) -> str:
    """Return a numeric PR id from a number or GitHub pull-request URL."""

    candidate = value.strip()
    if PR_NUMBER_RE.fullmatch(candidate):
        return candidate
    parsed = urlparse(candidate)
    if parsed.scheme in {"http", "https"} and parsed.netloc.lower() == "github.com":
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 4 and parts[-2] == "pull" and PR_NUMBER_RE.fullmatch(parts[-1]):
            return parts[-1]
    raise ValueError("PR must be a positive integer or a GitHub /pull/<number> URL.")


def recommend_push_strategy(
    *, viewer_permission: str | None, working_tree_clean: bool, gh_authenticated: bool
) -> str:
    """Return the safest push strategy for a GitHub PR workflow."""

    if not gh_authenticated:
        return "blocked_auth"
    if not working_tree_clean:
        return "blocked_dirty_tree"

    normalized = (viewer_permission or "").upper()
    if normalized in WRITE_LEVEL_PERMISSIONS:
        return "direct_origin"
    if normalized in FORK_LEVEL_PERMISSIONS:
        return "fork_remote"
    return "unknown_permission"


def classify_ci_state(
    *, checks: Sequence[dict[str, Any]] | None, workflow_runs: Sequence[dict[str, Any]] | None
) -> dict[str, Any]:
    """Classify PR checks/runs into a compact receipt state.

    GitHub marks fork pull-request workflows as ``action_required`` before an
    upstream maintainer approves them. That state is not a test failure and
    should be surfaced as a waiting-for-maintainer gate.
    """

    checks = list(checks or [])
    workflow_runs = list(workflow_runs or [])

    failed_checks = [
        check
        for check in checks
        if str(check.get("conclusion") or check.get("state") or "").lower()
        in {"failure", "failed", "error", "cancelled", "timed_out"}
    ]
    failed_runs = [
        run
        for run in workflow_runs
        if str(run.get("conclusion") or "").lower()
        in {"failure", "failed", "error", "cancelled", "timed_out"}
    ]
    if failed_checks or failed_runs:
        return {
            "state": "failure",
            "blocked_by": "ci_failure",
            "is_failure": True,
            "next_action": "Inspect failing checks or workflow logs and push a fix.",
            "checks": failed_checks,
            "runs": failed_runs,
        }

    action_required_runs = [
        run
        for run in workflow_runs
        if str(run.get("conclusion") or "").lower() == "action_required"
    ]
    if action_required_runs:
        return {
            "state": "action_required",
            "blocked_by": "maintainer_workflow_approval",
            "is_failure": False,
            "next_action": "Ask an upstream maintainer to approve fork workflow runs.",
            "runs": action_required_runs,
        }

    pending_checks = [
        check
        for check in checks
        if str(check.get("status") or check.get("state") or "").lower()
        in {"queued", "pending", "in_progress", "requested", "waiting", "expected"}
    ]
    pending_runs = [
        run
        for run in workflow_runs
        if str(run.get("status") or "").lower() in {"queued", "pending", "in_progress", "waiting"}
    ]
    if pending_checks or pending_runs:
        return {
            "state": "pending",
            "blocked_by": "ci_pending",
            "is_failure": False,
            "next_action": "Wait for CI to finish.",
            "checks": pending_checks,
            "runs": pending_runs,
        }

    if checks or workflow_runs:
        return {
            "state": "success",
            "blocked_by": None,
            "is_failure": False,
            "next_action": "Review or merge when ready.",
            "checks": checks,
            "runs": workflow_runs,
        }

    return {
        "state": "no_checks",
        "blocked_by": "checks_not_reported",
        "is_failure": False,
        "next_action": "Wait briefly, then inspect repository workflow configuration if checks remain absent.",
        "checks": [],
        "runs": [],
    }


def build_pr_receipt(
    *,
    preflight: dict[str, Any],
    pr: dict[str, Any] | None,
    ci: dict[str, Any] | None,
    local_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a stable JSON receipt for PR delivery/status updates."""

    pr = pr or {}
    ci = ci or classify_ci_state(checks=[], workflow_runs=[])
    return {
        "repo": preflight.get("repo"),
        "branch": preflight.get("branch"),
        "strategy": preflight.get("strategy"),
        "viewer_permission": preflight.get("viewer_permission"),
        "working_tree_clean": preflight.get("working_tree_clean"),
        "pr_number": pr.get("number"),
        "pr_url": pr.get("url"),
        "pr_state": pr.get("state"),
        "draft": pr.get("isDraft"),
        "mergeable": pr.get("mergeable"),
        "base": pr.get("baseRefName"),
        "head": pr.get("headRefName"),
        "ci_state": ci.get("state"),
        "blocked_by": ci.get("blocked_by"),
        "ci_failure": ci.get("is_failure"),
        "next_action": ci.get("next_action"),
        "local_validation": local_validation or {},
    }


def _default_runner(args: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - args are static lists built by this script.
        list(args),
        cwd=str(cwd),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _run_json(runner: CommandRunner, args: Sequence[str], cwd: Path) -> dict[str, Any] | None:
    result = runner(args, cwd)
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return None


def _run_json_list(runner: CommandRunner, args: Sequence[str], cwd: Path) -> list[dict[str, Any]]:
    result = runner(args, cwd)
    if result.returncode != 0:
        return []
    try:
        data = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _run_text(runner: CommandRunner, args: Sequence[str], cwd: Path) -> tuple[int, str, str]:
    result = runner(args, cwd)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def collect_preflight(
    *, cwd: Path, remote: str = "origin", runner: CommandRunner = _default_runner
) -> dict[str, Any]:
    """Collect Git/GitHub preflight data for the current branch."""

    code, branch, branch_err = _run_text(runner, ["git", "branch", "--show-current"], cwd)
    if code != 0:
        branch = ""

    code, status, _ = _run_text(runner, ["git", "status", "--porcelain"], cwd)
    working_tree_clean = code == 0 and status == ""

    code, remote_url, remote_err = _run_text(runner, ["git", "remote", "get-url", remote], cwd)
    repo = parse_github_remote_url(remote_url) if code == 0 else None

    gh_status = runner(["gh", "auth", "status"], cwd)
    gh_authenticated = gh_status.returncode == 0

    repo_view = (
        _run_json(
            runner,
            ["gh", "repo", "view", repo or "", "--json", "nameWithOwner,viewerPermission,defaultBranchRef"],
            cwd,
        )
        if repo and gh_authenticated
        else None
    )
    viewer_permission = (repo_view or {}).get("viewerPermission")
    default_branch_ref = (repo_view or {}).get("defaultBranchRef") or {}
    default_branch = default_branch_ref.get("name")

    existing_prs = (
        _run_json_list(
            runner,
            [
                "gh",
                "pr",
                "list",
                "--repo",
                repo,
                "--head",
                branch,
                "--json",
                "number,url,state,title,isDraft",
            ],
            cwd,
        )
        if repo and branch and gh_authenticated
        else []
    )

    strategy = recommend_push_strategy(
        viewer_permission=viewer_permission,
        working_tree_clean=working_tree_clean,
        gh_authenticated=gh_authenticated,
    )

    return {
        "repo": repo,
        "remote": remote,
        "remote_url": remote_url if code == 0 else None,
        "branch": branch,
        "default_branch": default_branch,
        "viewer_permission": viewer_permission,
        "strategy": strategy,
        "working_tree_clean": working_tree_clean,
        "gh_authenticated": gh_authenticated,
        "existing_pr": existing_prs[0] if existing_prs else None,
        "errors": [err for err in (branch_err, remote_err, gh_status.stderr.strip()) if err and not gh_authenticated],
    }


def collect_pr_status(
    *,
    repo: str,
    pr_number: str,
    branch: str | None = None,
    cwd: Path,
    runner: CommandRunner = _default_runner,
) -> dict[str, Any]:
    """Collect PR metadata and classify its check/workflow state."""

    pr = _run_json(
        runner,
        [
            "gh",
            "pr",
            "view",
            str(pr_number),
            "--repo",
            repo,
            "--json",
            "number,url,title,state,isDraft,headRefName,headRefOid,baseRefName,mergeable,statusCheckRollup",
        ],
        cwd,
    ) or {}
    branch = branch or pr.get("headRefName") or ""
    head_sha = pr.get("headRefOid")

    run_args = [
        "gh",
        "run",
        "list",
        "--repo",
        repo,
    ]
    if head_sha:
        run_args.extend(["--commit", str(head_sha)])
    else:
        run_args.extend(["--branch", branch])
    run_args.extend(
        [
            "--limit",
            "20",
            "--json",
            "databaseId,name,status,conclusion,event,headBranch,headSha,displayTitle,url,createdAt,updatedAt",
        ]
    )

    runs = _run_json_list(runner, run_args, cwd)
    ci = classify_ci_state(checks=pr.get("statusCheckRollup") or [], workflow_runs=runs)
    return {"pr": pr, "ci": ci, "runs": runs}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", default=".", help="Repository working directory")
    parser.add_argument("--remote", default="origin", help="Git remote to inspect")
    parser.add_argument("--pr", help="PR number or URL to include status receipt")
    parser.add_argument("--repo", help="Owner/repo override for PR status")
    parser.add_argument("--local-validation-json", help="JSON object with local validation evidence")
    args = parser.parse_args(argv)

    cwd = Path(args.cwd).resolve()
    preflight = collect_preflight(cwd=cwd, remote=args.remote)

    local_validation: dict[str, Any] = {}
    if args.local_validation_json:
        try:
            parsed = json.loads(args.local_validation_json)
            if isinstance(parsed, dict):
                local_validation = parsed
        except json.JSONDecodeError as exc:
            print(f"Invalid --local-validation-json: {exc}", file=sys.stderr)
            return 2

    output: dict[str, Any] = {"preflight": preflight}
    if args.pr:
        repo = args.repo or preflight.get("repo")
        if not repo:
            print("Cannot collect PR status without a GitHub repo.", file=sys.stderr)
            return 2
        try:
            repo = validate_repo_slug(str(repo))
            pr_number = parse_pr_identifier(args.pr)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        status = collect_pr_status(
            repo=repo,
            pr_number=pr_number,
            # Let collect_pr_status use the PR head branch. The current local branch
            # may be a separate follow-up branch while checking an existing PR.
            branch=None,
            cwd=cwd,
        )
        output["pr"] = status["pr"]
        output["ci"] = status["ci"]
        output["receipt"] = build_pr_receipt(
            preflight=preflight,
            pr=status["pr"],
            ci=status["ci"],
            local_validation=local_validation,
        )

    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
