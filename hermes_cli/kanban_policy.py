"""Config-driven safety policy for Hermes Kanban boards.

Board policies keep project-specific workflow constraints out of core Kanban
logic: canonical project roots, allowed worktree roots, denied scratch roots,
and worker guidance are loaded from declarative per-board JSON files.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


_BOARD_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9\-_]{0,63}$")


def normalize_board_slug(board: str) -> str:
    slug = str(board).strip().lower()
    if not slug or not _BOARD_SLUG_RE.match(slug):
        raise ValueError(
            f"invalid board slug {board!r}: must be 1-64 chars, lowercase alphanumerics / hyphens / underscores"
        )
    return slug


@dataclass(frozen=True)
class BoardPolicy:
    board: str
    project_root: Path | None = None
    base_branch: str = "main"
    worktree_root: Path | None = None
    denied_workspace_roots: list[Path] = field(default_factory=list)
    shared_project_root_writable: bool = False
    scratch_repo_operations_allowed: bool = False
    max_active_issue_pipelines: int | None = None
    cleanup_after_merge: bool = True
    forbid_dirty_completion: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_root", _norm_optional_path(self.project_root))
        object.__setattr__(self, "worktree_root", _norm_optional_path(self.worktree_root))
        object.__setattr__(
            self,
            "denied_workspace_roots",
            [_norm_path(path) for path in self.denied_workspace_roots],
        )
        if self.max_active_issue_pipelines is not None:
            try:
                value = int(self.max_active_issue_pipelines)
            except (TypeError, ValueError) as exc:
                raise ValueError("max_active_issue_pipelines must be an integer") from exc
            if value < 1:
                raise ValueError("max_active_issue_pipelines must be >= 1")
            object.__setattr__(self, "max_active_issue_pipelines", value)

    @property
    def configured(self) -> bool:
        return any(
            [
                self.project_root,
                self.worktree_root,
                self.denied_workspace_roots,
                self.max_active_issue_pipelines,
                self.shared_project_root_writable,
                self.scratch_repo_operations_allowed,
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("project_root", "worktree_root"):
            if data[key] is not None:
                data[key] = str(data[key])
        data["denied_workspace_roots"] = [str(path) for path in self.denied_workspace_roots]
        return data


@dataclass(frozen=True)
class WorkspacePolicyFinding:
    severity: str
    code: str
    message: str
    detail: str = ""


def _norm_optional_path(value: str | Path | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return _norm_path(value)


def _norm_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _kanban_home() -> Path:
    try:
        from hermes_cli import kanban_db as kb

        return kb.kanban_home().resolve(strict=False)
    except Exception:
        override = os.environ.get("HERMES_KANBAN_HOME", "").strip()
        if override:
            return Path(override).expanduser().resolve(strict=False)
        return Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser().resolve(strict=False)


def policies_root(*, home: Path | None = None) -> Path:
    return (home.resolve(strict=False) if home else _kanban_home()) / "kanban" / "policies"


def policy_path_for_board(board: str, *, home: Path | None = None) -> Path:
    return policies_root(home=home) / f"{normalize_board_slug(board)}.json"


def default_policy(board: str) -> BoardPolicy:
    return BoardPolicy(board=board)


def load_policy(board: str, *, home: Path | None = None) -> BoardPolicy:
    path = policy_path_for_board(board, home=home)
    if not path.exists():
        return default_policy(board)
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"policy file {path} must contain a JSON object")
    return BoardPolicy(
        board=board,
        project_root=data.get("project_root"),
        base_branch=data.get("base_branch", "main"),
        worktree_root=data.get("worktree_root"),
        denied_workspace_roots=data.get("denied_workspace_roots", []),
        shared_project_root_writable=bool(data.get("shared_project_root_writable", False)),
        scratch_repo_operations_allowed=bool(data.get("scratch_repo_operations_allowed", False)),
        max_active_issue_pipelines=data.get("max_active_issue_pipelines"),
        cleanup_after_merge=bool(data.get("cleanup_after_merge", True)),
        forbid_dirty_completion=bool(data.get("forbid_dirty_completion", True)),
    )


def policy_report(board: str, *, home: Path | None = None) -> dict[str, Any]:
    path = policy_path_for_board(board, home=home)
    errors: list[str] = []
    policy = default_policy(board)
    if path.exists():
        try:
            policy = load_policy(board, home=home)
        except Exception as exc:
            errors.append(str(exc))
    return {
        "board": board,
        "configured": path.exists(),
        "path": str(path),
        "policy": policy.to_dict(),
        "errors": errors,
    }


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_workspace(
    policy: BoardPolicy,
    *,
    workspace_kind: str | None,
    workspace_path: str | None,
    repo_touching: bool,
) -> list[WorkspacePolicyFinding]:
    findings: list[WorkspacePolicyFinding] = []
    if not repo_touching:
        return findings
    if not policy.project_root and not policy.worktree_root and not policy.denied_workspace_roots:
        return findings

    kind = workspace_kind or "scratch"
    if kind != "worktree":
        if policy.worktree_root:
            findings.append(WorkspacePolicyFinding(
                "error",
                "repo_touching_requires_worktree",
                "Repo-touching card must use workspace_kind=worktree under this board policy.",
                f"workspace_kind={kind!r}; allowed worktree root is {policy.worktree_root}",
            ))
        return findings

    if not workspace_path:
        findings.append(WorkspacePolicyFinding(
            "error",
            "missing_explicit_project_worktree",
            "Repo-touching worktree card lacks an explicit policy-compliant workspace_path.",
            f"Use an absolute path under {policy.worktree_root}." if policy.worktree_root else "Set workspace_path explicitly.",
        ))
        return findings

    path = _norm_path(workspace_path)
    for denied in policy.denied_workspace_roots:
        if _is_relative_to(path, denied):
            findings.append(WorkspacePolicyFinding(
                "error",
                "workspace_under_denied_root",
                f"Workspace is under denied root {denied}.",
                str(path),
            ))

    if policy.project_root and path == policy.project_root and not policy.shared_project_root_writable:
        findings.append(WorkspacePolicyFinding(
            "error",
            "shared_project_root_workspace",
            "Repo-touching card targets the shared project root, which is read-only by policy.",
            f"Use a task-specific worktree under {policy.worktree_root}." if policy.worktree_root else str(path),
        ))

    if policy.worktree_root and not _is_relative_to(path, policy.worktree_root):
        findings.append(WorkspacePolicyFinding(
            "error",
            "worktree_outside_policy_root",
            f"Worktree must be under {policy.worktree_root}.",
            f"workspace_path={workspace_path!r}",
        ))
    return findings


def format_policy_guidance(policy: BoardPolicy) -> str:
    if not policy.configured:
        return ""
    lines = ["## Board policy", ""]
    lines.append(f"Board: {policy.board}")
    if policy.project_root:
        access = "writable" if policy.shared_project_root_writable else "read-only inspection space"
        lines.append(f"Canonical repo: {policy.project_root} ({access})")
    lines.append(f"Base branch: {policy.base_branch}")
    if policy.worktree_root:
        lines.append(f"Allowed task worktree root: {policy.worktree_root}")
    if policy.denied_workspace_roots:
        lines.append("Denied workspace roots: " + ", ".join(str(p) for p in policy.denied_workspace_roots))
    scratch = "allowed" if policy.scratch_repo_operations_allowed else "forbidden for repo clone/build/install operations"
    lines.append(f"Scratch/control-plane repo operations: {scratch}")
    if policy.max_active_issue_pipelines:
        lines.append(f"Max active issue pipelines: {policy.max_active_issue_pipelines}")
    if policy.forbid_dirty_completion:
        lines.append("Completion must report clean git status or block with dirty-file evidence.")
    return "\n".join(lines).rstrip() + "\n"
