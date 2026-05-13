"""Workflow policy loading and validation.

Policy lives at ``<workspace>/.hermes/workflow.yaml``. Missing policy is not an
error: Hermes returns a conservative default policy plus a warning so read-only
status/API surfaces can still function before a project opts in.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import WorkflowValidationError, WorkflowValidationIssue

SUPPORTED_POLICY_VERSION = 1
CANONICAL_ROLES = (
    "planner",
    "architect",
    "reviewer",
    "publisher",
    "decomposer",
    "engineer",
    "integrator",
    "retro",
    "historian",
)
VALID_SCALES = {"small", "medium", "large", "xl"}
VALID_REVIEW_SETTINGS = {"none", "optional", "light", "required", "human", "llm_auditable", "human_at_major_breakpoints"}

DEFAULT_POLICY: dict[str, Any] = {
    "version": SUPPORTED_POLICY_VERSION,
    "project": {"name": "", "board": "default"},
    "roles": {
        "planner": "planner",
        "architect": "architect",
        "reviewer": "reviewer",
        "publisher": "publisher",
        "decomposer": "decomposer",
        "engineer": "engineer",
        "integrator": "integrator",
        "retro": "retro",
        "historian": "historian",
    },
    "review": {
        "default": "llm_auditable",
        "gates": {
            "small": {"prd": "optional", "spec": "optional", "dag": "none", "review": "llm_auditable"},
            "medium": {"prd": "light", "spec": "required", "dag": "optional", "review": "llm_auditable"},
            "large": {"prd": "full", "spec": "required", "dag": "human", "review": "human_at_major_breakpoints"},
            "xl": {"prd": "full", "spec": "required", "dag": "human", "review": "human_at_major_breakpoints"},
        },
    },
    "dag": {
        "format": "yaml",
        "require_integrator_for_large": True,
        "require_definition_of_done": True,
        "max_parallel_engineers": 4,
    },
    "worktrees": {"enabled": True, "root": ".worktrees", "branch_prefix": "workflow"},
}

# The PRD deliberately allows ``full`` for PRD review depth even though it is
# not a gate verdict. Keep it scoped to review.gates.<scale>.prd so other keys
# stay tight.
_VALID_PRD_SETTINGS = VALID_REVIEW_SETTINGS | {"full"}


@dataclass(frozen=True)
class WorkflowPolicy:
    """A normalized workflow policy and its source path."""

    data: dict[str, Any]
    path: Path | None = None
    using_default: bool = False

    @property
    def board(self) -> str:
        return str(self.data.get("project", {}).get("board") or "default")

    @property
    def roles(self) -> dict[str, str | None]:
        roles = self.data.get("roles", {})
        return {role: roles.get(role) for role in CANONICAL_ROLES}


@dataclass(frozen=True)
class PolicyLoadResult:
    """Result of loading workflow policy with structured issues."""

    policy: WorkflowPolicy
    issues: list[WorkflowValidationIssue]

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def warnings(self) -> list[WorkflowValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def errors(self) -> list[WorkflowValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "policy": self.policy.data,
            "path": str(self.policy.path) if self.policy.path else None,
            "using_default": self.policy.using_default,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def policy_path_for_workspace(workspace_path: str | Path) -> Path:
    """Return the canonical workflow policy path for a workspace."""

    return Path(workspace_path).expanduser() / ".hermes" / "workflow.yaml"


def load_policy(workspace_path: str | Path, *, strict: bool = False) -> PolicyLoadResult:
    """Load and validate ``.hermes/workflow.yaml`` for ``workspace_path``.

    Missing policy returns the default policy with a warning. Invalid YAML or
    validation errors return the default/merged data plus structured errors;
    callers that need exception semantics can pass ``strict=True``.
    """

    path = policy_path_for_workspace(workspace_path)
    issues: list[WorkflowValidationIssue] = []

    if not path.exists():
        policy = WorkflowPolicy(_default_policy_for_workspace(workspace_path), path=path, using_default=True)
        issues.append(
            WorkflowValidationIssue(
                code="policy_missing",
                message="No .hermes/workflow.yaml found; using safe default workflow policy.",
                path=str(path),
                severity="warning",
            )
        )
        return PolicyLoadResult(policy=policy, issues=issues)

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        issues.append(
            WorkflowValidationIssue(
                code="policy_invalid_yaml",
                message=f"Invalid workflow policy YAML: {exc}",
                path=str(path),
            )
        )
        result = PolicyLoadResult(
            policy=WorkflowPolicy(_default_policy_for_workspace(workspace_path), path=path, using_default=True),
            issues=issues,
        )
        if strict:
            raise WorkflowValidationError(result.errors)
        return result

    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        issues.append(
            WorkflowValidationIssue(
                code="policy_not_mapping",
                message="Workflow policy must be a YAML mapping/object.",
                path=str(path),
            )
        )
        loaded = {}

    merged = _deep_merge(_default_policy_for_workspace(workspace_path), loaded)
    issues.extend(validate_policy(merged, policy_path=path, workspace_path=Path(workspace_path).expanduser()))
    result = PolicyLoadResult(policy=WorkflowPolicy(merged, path=path, using_default=False), issues=issues)
    if strict and not result.ok:
        raise WorkflowValidationError(result.errors)
    return result


def validate_policy(policy: dict[str, Any], *, policy_path: Path | None = None, workspace_path: Path | None = None) -> list[WorkflowValidationIssue]:
    """Return structured validation issues for a normalized policy mapping."""

    issues: list[WorkflowValidationIssue] = []
    src = str(policy_path) if policy_path else ""

    version = policy.get("version")
    if version != SUPPORTED_POLICY_VERSION:
        issues.append(_issue("unsupported_version", f"Unsupported workflow policy version: {version!r}.", src, "version"))

    project = policy.get("project")
    if not isinstance(project, dict):
        issues.append(_issue("project_not_mapping", "project must be a mapping.", src, "project"))
    else:
        board = project.get("board")
        if not isinstance(board, str) or not board.strip():
            issues.append(_issue("invalid_board", "project.board must be a non-empty string.", src, "project.board"))

    roles = policy.get("roles")
    if not isinstance(roles, dict):
        issues.append(_issue("roles_not_mapping", "roles must be a mapping.", src, "roles"))
    else:
        for role in CANONICAL_ROLES:
            value = roles.get(role)
            if value is not None and not isinstance(value, str):
                issues.append(_issue("invalid_role_mapping", f"roles.{role} must be a profile name string or null.", src, f"roles.{role}"))

    review = policy.get("review")
    if not isinstance(review, dict):
        issues.append(_issue("review_not_mapping", "review must be a mapping.", src, "review"))
    else:
        default = review.get("default")
        if default not in VALID_REVIEW_SETTINGS:
            issues.append(_issue("unknown_review_setting", f"Unknown review.default setting: {default!r}.", src, "review.default"))
        gates = review.get("gates")
        if not isinstance(gates, dict):
            issues.append(_issue("review_gates_not_mapping", "review.gates must be a mapping.", src, "review.gates"))
        else:
            for scale in VALID_SCALES:
                scale_cfg = gates.get(scale)
                if not isinstance(scale_cfg, dict):
                    issues.append(_issue("review_scale_not_mapping", f"review.gates.{scale} must be a mapping.", src, f"review.gates.{scale}"))
                    continue
                for key, value in scale_cfg.items():
                    allowed = _VALID_PRD_SETTINGS if key == "prd" else VALID_REVIEW_SETTINGS
                    if value not in allowed:
                        issues.append(_issue("unknown_gate_setting", f"Unknown review.gates.{scale}.{key} setting: {value!r}.", src, f"review.gates.{scale}.{key}"))

    dag = policy.get("dag")
    if not isinstance(dag, dict):
        issues.append(_issue("dag_not_mapping", "dag must be a mapping.", src, "dag"))
    else:
        if dag.get("format") != "yaml":
            issues.append(_issue("unsupported_dag_format", "dag.format must be 'yaml'.", src, "dag.format"))
        max_parallel = dag.get("max_parallel_engineers")
        if not isinstance(max_parallel, int) or max_parallel < 1:
            issues.append(_issue("invalid_max_parallel_engineers", "dag.max_parallel_engineers must be a positive integer.", src, "dag.max_parallel_engineers"))

    worktrees = policy.get("worktrees")
    if not isinstance(worktrees, dict):
        issues.append(_issue("worktrees_not_mapping", "worktrees must be a mapping.", src, "worktrees"))
    else:
        root = worktrees.get("root")
        if not isinstance(root, str) or not root.strip():
            issues.append(_issue("invalid_worktree_root", "worktrees.root must be a non-empty relative path.", src, "worktrees.root"))
        elif Path(root).is_absolute() or ".." in Path(root).parts:
            issues.append(_issue("unsafe_worktree_root", "worktrees.root must stay inside the workspace.", src, "worktrees.root"))
        if not isinstance(worktrees.get("enabled"), bool):
            issues.append(_issue("invalid_worktrees_enabled", "worktrees.enabled must be a boolean.", src, "worktrees.enabled"))
        prefix = worktrees.get("branch_prefix")
        if not isinstance(prefix, str) or not prefix.strip():
            issues.append(_issue("invalid_branch_prefix", "worktrees.branch_prefix must be a non-empty string.", src, "worktrees.branch_prefix"))

    return issues


def _default_policy_for_workspace(workspace_path: str | Path) -> dict[str, Any]:
    data = copy.deepcopy(DEFAULT_POLICY)
    workspace = Path(workspace_path).expanduser()
    if not data["project"].get("name"):
        data["project"]["name"] = workspace.name or "workspace"
    return data


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _issue(code: str, message: str, source: str, path: str) -> WorkflowValidationIssue:
    return WorkflowValidationIssue(code=code, message=message, path=f"{source}:{path}" if source else path)
