"""Fail-closed dispatcher attestation for the restricted implementer profile.

This validates the dispatcher-owned worker process before agent startup.  Tool
children deliberately scrub task identity, so this module must never infer its
result from a terminal or code-execution descendant environment.
"""
from __future__ import annotations

import os
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Protocol


class ImplementerWorkspaceError(RuntimeError):
    """The implementer process is not the dispatcher-owned worker it claims to be."""


class _Task(Protocol):
    id: str
    workspace_path: Optional[str]
    current_run_id: Optional[int]
    claim_lock: Optional[str]
    branch_name: Optional[str]


@dataclass(frozen=True)
class WorkspaceAttestation:
    task_id: str
    workspace: Path
    run_id: int
    claim_lock: str
    branch: str


def is_implementer_profile(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Whether this process has selected the restricted implementer profile."""
    env = os.environ if environ is None else environ
    profile = (env.get("HERMES_PROFILE") or env.get("HERMES_PROFILE_NAME") or "").strip()
    if profile:
        return profile == "implementer"
    home = (env.get("HERMES_HOME") or "").rstrip(os.sep)
    return Path(home).name == "implementer" and Path(home).parent.name == "profiles"


def _required(env: Mapping[str, str], name: str) -> str:
    value = (env.get(name) or "").strip()
    if not value:
        raise ImplementerWorkspaceError(f"implementer dispatcher attestation missing {name}")
    return value


def _real_dir(value: str, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_dir():
        raise ImplementerWorkspaceError(f"implementer {label} must be an existing absolute directory")
    return path.resolve()


def _git(args: list[str], cwd: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ImplementerWorkspaceError(f"implementer workspace git attestation failed: {exc}") from exc


def _load_task(task_id: str) -> _Task | None:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect() as conn:
        return kb.get_task(conn, task_id)


def require_attested_workspace(
    *,
    task_loader: Callable[[str], _Task | None] = _load_task,
    cwd_getter: Callable[[], str] = os.getcwd,
    git_runner: Callable[[list[str], str], str] = _git,
    environ: Optional[Mapping[str, str]] = None,
) -> WorkspaceAttestation | None:
    """Validate a dispatcher-owned implementer worker or fail before tool setup."""
    env = os.environ if environ is None else environ
    if not is_implementer_profile(env):
        return None

    task_id = _required(env, "HERMES_KANBAN_TASK")
    workspace = _real_dir(_required(env, "HERMES_KANBAN_WORKSPACE"), label="workspace")
    claim_lock = _required(env, "HERMES_KANBAN_CLAIM_LOCK")
    _required(env, "HERMES_KANBAN_DISPATCH_GRANT")
    branch = _required(env, "HERMES_KANBAN_BRANCH")
    raw_run_id = _required(env, "HERMES_KANBAN_RUN_ID")
    try:
        run_id = int(raw_run_id)
    except ValueError as exc:
        raise ImplementerWorkspaceError("implementer HERMES_KANBAN_RUN_ID must be an integer") from exc

    task = task_loader(task_id)
    if task is None:
        raise ImplementerWorkspaceError(f"implementer task {task_id} was not found")
    if not task.workspace_path or _real_dir(task.workspace_path, label="task workspace") != workspace:
        raise ImplementerWorkspaceError("implementer workspace does not match the claimed task")
    if task.current_run_id != run_id:
        raise ImplementerWorkspaceError("implementer run id does not match the claimed task")
    if task.claim_lock != claim_lock:
        raise ImplementerWorkspaceError("implementer claim lock does not match the claimed task")
    if task.branch_name != branch:
        raise ImplementerWorkspaceError("implementer branch does not match the claimed task")

    cwd = _real_dir(cwd_getter(), label="current working directory")
    if cwd != workspace:
        raise ImplementerWorkspaceError("implementer current working directory does not match workspace")
    top_level = _real_dir(git_runner(["rev-parse", "--show-toplevel"], str(workspace)), label="git top-level")
    if top_level != workspace:
        raise ImplementerWorkspaceError("implementer git top-level does not match workspace")
    if git_runner(["branch", "--show-current"], str(workspace)) != branch:
        raise ImplementerWorkspaceError("implementer git branch does not match the claimed task")
    return WorkspaceAttestation(task_id, workspace, run_id, claim_lock, branch)


def verify_original_baseline(
    *, environ: Optional[Mapping[str, str]] = None,
    git_runner: Callable[[list[str], str], str] = _git,
) -> None:
    """Reject implementer completion when the dispatcher's original checkout changed."""
    env = os.environ if environ is None else environ
    if not is_implementer_profile(env):
        return
    raw = _required(env, "HERMES_KANBAN_ORIGINAL_BASELINE")
    try:
        baseline = json.loads(raw)
        original = _real_dir(str(baseline["path"]), label="original repository")
        expected = {key: str(baseline[key]) for key in ("head", "branch", "status")}
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ImplementerWorkspaceError("implementer original-repository baseline is invalid") from exc
    actual = {
        "head": git_runner(["rev-parse", "HEAD"], str(original)),
        "branch": git_runner(["branch", "--show-current"], str(original)),
        "status": git_runner(["status", "--porcelain", "--untracked-files=all"], str(original)),
    }
    if actual != expected:
        raise ImplementerWorkspaceError("implementer original repository differs from dispatcher baseline")


def capture_original_baseline(workspace: str) -> str:
    """Return dispatcher-captured JSON for the primary checkout of a worktree."""
    listing = _git(["worktree", "list", "--porcelain"], workspace)
    first = next((line.removeprefix("worktree ") for line in listing.splitlines()
                  if line.startswith("worktree ")), "")
    original = _real_dir(first, label="original repository")
    return json.dumps({
        "path": str(original),
        "head": _git(["rev-parse", "HEAD"], str(original)),
        "branch": _git(["branch", "--show-current"], str(original)),
        "status": _git(["status", "--porcelain", "--untracked-files=all"], str(original)),
    }, sort_keys=True)
