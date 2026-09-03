#!/usr/bin/env python3
"""Execute guarded git operations for an autoresearch workspace."""

from __future__ import annotations

import argparse
import json
import re
import secrets
import subprocess
from functools import wraps
from pathlib import Path
from typing import Any

from _util import exclusive_lock

MARKER = ".autoresearch-workspace"


class WorkspaceError(RuntimeError):
    """Raised when a workspace invariant is not satisfied."""


def locked_workspace(function):
    @wraps(function)
    def wrapper(workspace_dir, *args, **kwargs):
        workspace = Path(workspace_dir).expanduser().resolve()
        with exclusive_lock(workspace / ".git" / "autoresearch-operation.lock"):
            return function(workspace_dir, *args, **kwargs)

    return wrapper


def safe_branch_name(experiment_id: int, description: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", description.lower())
    safe = re.sub(r"_+", "_", safe).strip("_")[:40]
    return f"exp_{experiment_id}_{safe or 'experiment'}"


def run_git(workspace: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=workspace,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown git error"
        raise WorkspaceError(f"git {' '.join(args)} failed: {detail}")
    return result


def validate_workspace(workspace_dir: str) -> Path:
    workspace = Path(workspace_dir).expanduser().resolve()
    if not (workspace / ".git").is_dir() or not (workspace / MARKER).is_file():
        raise WorkspaceError(f"not an initialized autoresearch workspace: {workspace}")
    top_level = Path(run_git(workspace, "rev-parse", "--show-toplevel").stdout.strip()).resolve()
    if top_level != workspace:
        raise WorkspaceError("workspace path is not the git repository root")
    marker_lines = (workspace / MARKER).read_text(encoding="utf-8").splitlines()
    marker_values = dict(
        line.split("=", 1) for line in marker_lines if "=" in line
    )
    workspace_id = marker_values.get("id", "")
    configured_id = run_git(
        workspace, "config", "--local", "--get", "autoresearch.workspace-id", check=False
    ).stdout.strip()
    if not workspace_id or workspace_id != configured_id:
        raise WorkspaceError("workspace marker is not bound to this repository")
    root_commits = run_git(workspace, "rev-list", "--max-parents=0", "HEAD").stdout.split()
    if len(root_commits) != 1:
        raise WorkspaceError("workspace must have exactly one root commit")
    root_files = run_git(
        workspace, "ls-tree", "-r", "--name-only", root_commits[0]
    ).stdout.splitlines()
    if root_files != [MARKER]:
        raise WorkspaceError("workspace root commit does not match the autoresearch contract")
    return workspace


def current_branch(workspace: Path) -> str:
    return run_git(workspace, "branch", "--show-current").stdout.strip()


def ensure_clean(workspace: Path) -> None:
    status = run_git(workspace, "status", "--porcelain").stdout.strip()
    if status:
        raise WorkspaceError("workspace has uncommitted changes")


def initialize(workspace_dir: str) -> dict[str, Any]:
    workspace = Path(workspace_dir).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    if (workspace / ".git").exists():
        if (workspace / MARKER).is_file():
            validate_workspace(str(workspace))
            return {"status": "already_initialized", "workspace": str(workspace)}
        raise WorkspaceError("refusing to adopt an existing git repository")
    if any(workspace.iterdir()):
        raise WorkspaceError("workspace must be empty before initialization")

    initialized = subprocess.run(
        ["git", "init", "--initial-branch=main"],
        cwd=workspace,
        check=False,
        capture_output=True,
        text=True,
    )
    if initialized.returncode != 0:
        run_git(workspace, "init")
        run_git(workspace, "checkout", "-b", "main")
    run_git(workspace, "config", "user.email", "autoresearch@hermes.local")
    run_git(workspace, "config", "user.name", "Hermes Autoresearch")
    workspace_id = secrets.token_hex(16)
    run_git(workspace, "config", "--local", "autoresearch.workspace-id", workspace_id)
    (workspace / MARKER).write_text(
        f"version=1\nid={workspace_id}\n", encoding="utf-8"
    )
    run_git(workspace, "add", MARKER)
    run_git(workspace, "commit", "-m", "initialize autoresearch workspace")
    return {"status": "initialized", "workspace": str(workspace), "branch": "main"}


@locked_workspace
def create_branch(workspace_dir: str, experiment_id: int, description: str) -> dict[str, Any]:
    workspace = validate_workspace(workspace_dir)
    ensure_clean(workspace)
    if current_branch(workspace) != "main":
        raise WorkspaceError("new experiments must start from main")
    branch = safe_branch_name(experiment_id, description)
    exists = run_git(workspace, "show-ref", "--verify", f"refs/heads/{branch}", check=False)
    if exists.returncode == 0:
        raise WorkspaceError(f"experiment branch already exists: {branch}")
    run_git(workspace, "checkout", "-b", branch)
    return {"status": "branched", "workspace": str(workspace), "branch": branch}


@locked_workspace
def merge_branch(
    workspace_dir: str,
    experiment_id: int,
    description: str,
    commit_message: str,
) -> dict[str, Any]:
    workspace = validate_workspace(workspace_dir)
    branch = safe_branch_name(experiment_id, description)
    if current_branch(workspace) != branch:
        raise WorkspaceError(f"expected active branch {branch}")

    run_git(workspace, "add", "-A")
    unchanged = run_git(workspace, "diff", "--cached", "--quiet", check=False)
    if unchanged.returncode == 0:
        raise WorkspaceError("experiment has no changes to merge")
    if unchanged.returncode not in {0, 1}:
        raise WorkspaceError("unable to inspect staged experiment changes")
    run_git(workspace, "commit", "-m", commit_message)
    run_git(workspace, "checkout", "main")
    run_git(workspace, "merge", "--ff-only", branch)
    run_git(workspace, "branch", "-d", branch)
    ensure_clean(workspace)
    return {"status": "merged", "workspace": str(workspace), "branch": branch}


@locked_workspace
def revert_branch(workspace_dir: str, experiment_id: int, description: str) -> dict[str, Any]:
    workspace = validate_workspace(workspace_dir)
    branch = safe_branch_name(experiment_id, description)
    if current_branch(workspace) != branch:
        raise WorkspaceError(f"expected active branch {branch}")
    run_git(workspace, "reset", "--hard")
    run_git(workspace, "checkout", "-f", "main")
    run_git(workspace, "clean", "-fdx")
    run_git(workspace, "branch", "-D", branch)
    ensure_clean(workspace)
    return {"status": "reverted", "workspace": str(workspace), "branch": branch}


def inspect_diff(workspace_dir: str) -> dict[str, Any]:
    workspace = validate_workspace(workspace_dir)
    return {
        "workspace": str(workspace),
        "branch": current_branch(workspace),
        "diff": run_git(workspace, "diff", "main").stdout,
    }


def inspect_log(workspace_dir: str, oneline: bool = False) -> dict[str, Any]:
    workspace = validate_workspace(workspace_dir)
    args = ["log", "-20"]
    if oneline:
        args.append("--oneline")
    return {"workspace": str(workspace), "log": run_git(workspace, *args).stdout}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    init_parser = commands.add_parser("init")
    init_parser.add_argument("workspace")

    branch_parser = commands.add_parser("branch")
    branch_parser.add_argument("workspace")
    branch_parser.add_argument("experiment_id", type=int)
    branch_parser.add_argument("description")

    name_parser = commands.add_parser("branch-name")
    name_parser.add_argument("experiment_id", type=int)
    name_parser.add_argument("description")

    diff_parser = commands.add_parser("diff")
    diff_parser.add_argument("workspace")

    merge_parser = commands.add_parser("merge")
    merge_parser.add_argument("workspace")
    merge_parser.add_argument("experiment_id", type=int)
    merge_parser.add_argument("description")
    merge_parser.add_argument("commit_message")

    revert_parser = commands.add_parser("revert")
    revert_parser.add_argument("workspace")
    revert_parser.add_argument("experiment_id", type=int)
    revert_parser.add_argument("description")

    log_parser = commands.add_parser("log")
    log_parser.add_argument("workspace")
    log_parser.add_argument("--oneline", action="store_true")

    current_parser = commands.add_parser("current-branch")
    current_parser.add_argument("workspace")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result: Any = None
        if args.command == "init":
            result = initialize(args.workspace)
        elif args.command == "branch":
            result = create_branch(args.workspace, args.experiment_id, args.description)
        elif args.command == "branch-name":
            result = {"branch": safe_branch_name(args.experiment_id, args.description)}
        elif args.command == "diff":
            result = inspect_diff(args.workspace)
        elif args.command == "merge":
            result = merge_branch(
                args.workspace,
                args.experiment_id,
                args.description,
                args.commit_message,
            )
        elif args.command == "revert":
            result = revert_branch(args.workspace, args.experiment_id, args.description)
        elif args.command == "log":
            result = inspect_log(args.workspace, args.oneline)
        elif args.command == "current-branch":
            workspace = validate_workspace(args.workspace)
            result = {"workspace": str(workspace), "branch": current_branch(workspace)}
        print(json.dumps(result, indent=2))
    except WorkspaceError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
