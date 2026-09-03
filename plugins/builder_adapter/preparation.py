"""Create reviewable, non-activating governed builder job proposals."""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .canonical import canonical_sha256
from .errors import AdapterError


_IDENTIFIER = re.compile(r"^[A-Z][A-Z0-9_]{2,127}$")
_CONTRACT_ID = re.compile(r"^[A-Z][A-Z0-9_-]{2,63}$")
_REPOSITORY_ID = re.compile(r"^[a-z][a-z0-9_-]{2,63}$")


@dataclass(frozen=True)
class RepositoryFacts:
    root: str
    remote: str
    head: str
    branch: str
    clean: bool


def _git(repository: Path, *args: str) -> str:
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "/bin/false",
        "SSH_ASKPASS": "/bin/false",
    }
    try:
        result = subprocess.run(
            ["/usr/bin/git", "--no-pager", "-C", str(repository), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AdapterError("WORKTREE_MISMATCH", "repository inspection failed") from exc
    return result.stdout.strip()


def inspect_repository(repository: str | Path) -> RepositoryFacts:
    requested = Path(repository).expanduser()
    try:
        root = requested.resolve(strict=True)
    except OSError as exc:
        raise AdapterError("WORKTREE_MISMATCH", "repository does not exist") from exc
    if root.is_symlink() or not root.is_dir():
        raise AdapterError("WORKTREE_MISMATCH", "repository path is unsafe")
    top = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    if top != root:
        raise AdapterError("WORKTREE_MISMATCH", "--repo must name the repository root")
    remote = _git(root, "config", "--local", "--get", "remote.origin.url")
    head = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "branch", "--show-current")
    if not remote or not re.fullmatch(r"[0-9a-f]{40}([0-9a-f]{24})?", head):
        raise AdapterError("REPOSITORY_MISMATCH", "repository identity is incomplete")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=normal")
    return RepositoryFacts(str(root), remote, head, branch, clean=not status)


def _allowed_path(value: str) -> str:
    if not value or value.startswith(("/", "\\")) or "\\" in value or "\x00" in value:
        raise AdapterError("MANIFEST_MISMATCH", "allowed paths must be repository-relative")
    path = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise AdapterError("MANIFEST_MISMATCH", "allowed path traversal is forbidden")
    if value in {"*", "**", "**/*"}:
        raise AdapterError("MANIFEST_MISMATCH", "repository-wide write access is forbidden")
    return path.as_posix()


def prepare_bundle(
    *,
    cycle_id: str,
    contract_id: str,
    repository_id: str,
    repository: RepositoryFacts,
    goal: str,
    acceptance_criteria: list[str],
    allowed_paths: list[str],
    planned_branch: str,
    planned_worktree: str,
    validation_profile_id: str,
    max_runtime_seconds: int,
    heartbeat_timeout_seconds: int,
    registered_remote: str | None,
) -> dict:
    if not _IDENTIFIER.fullmatch(cycle_id):
        raise AdapterError("INVALID_REQUEST", "cycle ID must use uppercase letters, digits, and underscores")
    if not _CONTRACT_ID.fullmatch(contract_id):
        raise AdapterError("INVALID_REQUEST", "contract ID is invalid")
    if not _REPOSITORY_ID.fullmatch(repository_id):
        raise AdapterError("INVALID_REQUEST", "repository ID is invalid")
    if not repository.clean:
        raise AdapterError("WORKTREE_MISMATCH", "source repository must be clean")
    if not goal.strip() or len(goal) > 4000:
        raise AdapterError("INVALID_REQUEST", "goal must contain 1 to 4000 characters")
    criteria = [item.strip() for item in acceptance_criteria if item.strip()]
    if not criteria or any(len(item) > 1000 for item in criteria):
        raise AdapterError("INVALID_REQUEST", "at least one bounded acceptance criterion is required")
    paths = sorted(set(_allowed_path(item.strip()) for item in allowed_paths if item.strip()))
    if not paths:
        raise AdapterError("MANIFEST_MISMATCH", "at least one allowed path is required")
    worktree = Path(planned_worktree).expanduser()
    if not worktree.is_absolute() or worktree.exists() or worktree.is_symlink():
        raise AdapterError("WORKTREE_MISMATCH", "planned worktree must be a new absolute path")
    if not planned_branch or planned_branch.startswith("-") or planned_branch == "HEAD":
        raise AdapterError("BRANCH_MISMATCH", "planned branch is invalid")
    if not 60 <= max_runtime_seconds <= 43200:
        raise AdapterError("INVALID_REQUEST", "runtime must be between 60 and 43200 seconds")
    if not 30 <= heartbeat_timeout_seconds <= 3600:
        raise AdapterError("INVALID_REQUEST", "heartbeat timeout must be between 30 and 3600 seconds")

    activation_state = (
        "READY_FOR_GOVERNANCE_REVIEW"
        if registered_remote == repository.remote
        else "NEEDS_REPOSITORY_ALLOWLIST"
    )
    bundle = {
        "schema_version": "1.0.0",
        "bundle_kind": "hermes.builder_job_proposal",
        "cycle_id": cycle_id,
        "contract_id": contract_id,
        "cycle_revision": 1,
        "activation_state": activation_state,
        "repository": {
            "repository_id": repository_id,
            "source_root": repository.root,
            "canonical_remote": repository.remote,
            "base_sha": repository.head,
            "source_branch": repository.branch,
            "planned_branch": planned_branch,
            "planned_worktree": str(worktree),
        },
        "objective": {
            "summary": goal.strip(),
            "success_criteria": criteria,
        },
        "allowed_path_manifest": {
            "default_access": "forbidden",
            "symlinks": "reject",
            "submodules": "reject",
            "read_policy": {
                "source": "git_tracked_regular_files",
                "snapshot": "base_sha",
                "deny_patterns": [
                    ".env*",
                    "**/.env*",
                    "**/*credential*",
                    "**/*secret*",
                    "**/*token*",
                    "**/*.pem",
                    "**/*.key",
                ],
            },
            "rules": [{"pattern": path, "access": "read_write"} for path in paths],
        },
        "validation_profile_id": validation_profile_id,
        "timeout_policy": {
            "max_runtime_seconds": max_runtime_seconds,
            "heartbeat_timeout_seconds": heartbeat_timeout_seconds,
        },
        "retry_policy": {"max_attempts": 1, "retryable_terminal_states": []},
        "prepared_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    bundle["bundle_sha256"] = canonical_sha256(bundle)
    return bundle


def write_bundle(path: str | Path, bundle: dict) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.exists() or destination.is_symlink():
        raise AdapterError("INVALID_REQUEST", "proposal output already exists")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o600)
    try:
        raw = (json.dumps(bundle, indent=2, sort_keys=True) + "\n").encode()
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    info = destination.stat()
    if stat.S_IMODE(info.st_mode) != 0o600:
        raise AdapterError("INTERNAL_ERROR", "proposal permissions are not owner-only")
    return destination

