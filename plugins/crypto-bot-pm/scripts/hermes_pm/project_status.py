#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

repo_root_for_import = Path(__file__).resolve().parents[2]
if str(repo_root_for_import) not in sys.path:
    sys.path.insert(0, str(repo_root_for_import))

# Direct execution through hermes_pm_status.py can first resolve an unrelated
# third-party package named `scripts`.  Remove that shadowing package so later
# `scripts.hermes_pm.*` imports bind to this plugin's local scripts package.
_local_scripts_dir = repo_root_for_import / "scripts"
_loaded_scripts = sys.modules.get("scripts")
_loaded_scripts_paths = [Path(p).resolve() for p in getattr(_loaded_scripts, "__path__", [])]
if _loaded_scripts is not None and _local_scripts_dir.resolve() not in _loaded_scripts_paths:
    sys.modules.pop("scripts", None)


@dataclass(frozen=True)
class ProviderImportStatus:
    provider: str
    available: bool
    module: str | None = None
    error_type: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider": self.provider,
            "available": self.available,
        }
        if self.module:
            payload["module"] = self.module
        if self.error_type:
            payload["error_type"] = self.error_type
        if self.error:
            payload["error"] = _fallback_redact_text(self.error)
        return payload


def _fallback_redact_text(value: str) -> str:
    redacted = re.sub(
        r"(?i)(token|secret|password|private[_ -]?key|credential)(=|:)\S+",
        r"\1\2<redacted>",
        value,
    )
    return redacted


def _import_provider_attrs(
    provider: str,
    module_candidates: tuple[str, ...],
    attr_names: tuple[str, ...],
) -> tuple[dict[str, Any], ProviderImportStatus]:
    errors: list[str] = []
    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
            return (
                {attr_name: getattr(module, attr_name) for attr_name in attr_names},
                ProviderImportStatus(
                    provider=provider,
                    available=True,
                    module=module_name,
                ),
            )
        except (AttributeError, ImportError) as exc:
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
    return (
        {attr_name: None for attr_name in attr_names},
        ProviderImportStatus(
            provider=provider,
            available=False,
            module=module_candidates[0] if module_candidates else None,
            error_type="ImportUnavailable",
            error="; ".join(errors),
        ),
    )


def _provider_import_blocker(status: ProviderImportStatus) -> dict[str, str]:
    return {
        "endpoint": "<local-import>",
        "code": "provider_import_failed",
        "provider": status.provider,
        "module": status.module or "<unknown>",
        "error": _fallback_redact_text(
            status.error or f"{status.provider} provider is unavailable."
        ),
    }


_gitea_attrs, _gitea_import_status = _import_provider_attrs(
    "gitea_snapshot",
    ("scripts.hermes_pm.gitea_readonly_snapshot",),
    ("capture_gitea_snapshot",),
)
capture_gitea_snapshot = _gitea_attrs["capture_gitea_snapshot"]

_lifecycle_attrs, _lifecycle_import_status = _import_provider_attrs(
    "issue_lifecycle",
    ("scripts.hermes_pm.issue_lifecycle_status",),
    (
        "EXPECTED_PM_SEED_ISSUE_TITLE",
        "capture_issue_lifecycle_status",
        "compact_lifecycle_summary",
        "summarize_seed_issue_from_snapshot",
    ),
)
EXPECTED_PM_SEED_ISSUE_TITLE = (
    _lifecycle_attrs["EXPECTED_PM_SEED_ISSUE_TITLE"]
    or "[Hermes PM] Establish initial PM-managed backlog item"
)
capture_issue_lifecycle_status = _lifecycle_attrs["capture_issue_lifecycle_status"]
compact_lifecycle_summary = _lifecycle_attrs["compact_lifecycle_summary"]
summarize_seed_issue_from_snapshot = _lifecycle_attrs[
    "summarize_seed_issue_from_snapshot"
]

_work_state_attrs, _work_state_import_status = _import_provider_attrs(
    "work_state",
    ("scripts.hermes_pm.work_state",),
    ("build_work_state",),
)
build_work_state = _work_state_attrs["build_work_state"]

_ci_attrs, _ci_import_status = _import_provider_attrs(
    "ci_evidence",
    ("scripts.hermes_operator.gitea_ci_evidence_contracts",),
    ("build_gitea_ci_evidence_summary",),
)
build_gitea_ci_evidence_summary = _ci_attrs["build_gitea_ci_evidence_summary"]

_policy_attrs, _policy_import_status = _import_provider_attrs(
    "policy_audit",
    ("scripts.hermes_operator.policy_audit",),
    ("redact_text",),
)
redact_text = _policy_attrs["redact_text"] or _fallback_redact_text

PROVIDER_IMPORT_STATUS = {
    "gitea_snapshot": _gitea_import_status,
    "issue_lifecycle": _lifecycle_import_status,
    "work_state": _work_state_import_status,
    "ci_evidence": _ci_import_status,
    "policy_audit": _policy_import_status,
}


PM_STATUS_SCHEMA_VERSION = "hermes.pm.project_status.v1"
MANAGED_PROJECTS_SCHEMA_VERSION = "hermes.pm.managed_projects.v1"

DEFAULT_PROJECT_ID = "crypto_bot"
DEFAULT_FORBIDDEN_SURFACES = [
    "broker",
    "trading",
    "secrets",
    "runtime",
    "deployment",
    "live financial actions",
]

NON_ACTION_BOOLEANS = {
    "writes_files": False,
    "calls_gitea_write_api": False,
    "starts_runner": False,
    "runs_workflows": False,
    "deploys": False,
    "runtime_actions": False,
    "financial_actions": False,
    "secret_access": False,
    "branch_writer_invoked": False,
    "inspects_runtime_databases": False,
    "inspects_runtime_logs": False,
}

SENSITIVE_SEGMENTS = {
    ".aws",
    ".gnupg",
    ".ssh",
    "credential",
    "credentials",
    "keychain",
    "keys",
    "private_keys",
    "secret",
    "secrets",
    "token",
    "tokens",
}
RUNTIME_SEGMENTS = {
    "cache",
    "caches",
    "cookies",
    "data",
    "generated_ota_artifacts",
    "logs",
    "runtime",
}
FORBIDDEN_SUFFIXES = (
    ".asc",
    ".db",
    ".db-shm",
    ".db-wal",
    ".duckdb",
    ".env",
    ".gpg",
    ".key",
    ".log",
    ".p12",
    ".pem",
    ".pfx",
    ".sqlite",
    ".sqlite3",
)
SAFE_CHECKPOINT_GLOBS = (
    "docs/implementation/hermes_operator_checkpoint_*.md",
    "docs/implementation/hermes_pm_checkpoint_*.md",
)


class RefusedPathError(ValueError):
    """Raised when a status input points at a forbidden surface."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _redact_structure(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _redact_structure(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_structure(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_structure(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def _safe_repo_reference(repo_root: Path) -> str:
    try:
        if repo_root.resolve(strict=False) == Path.cwd().resolve(strict=False):
            return "."
    except OSError:
        pass
    return f"<repo-root>/{redact_text(repo_root.name)}"


def _path_looks_forbidden(path: Path) -> str | None:
    parts = [part.lower() for part in path.parts]
    for part in parts:
        if part in SENSITIVE_SEGMENTS:
            return f"path segment {part!r} is secret-like"
        if part in RUNTIME_SEGMENTS:
            return f"path segment {part!r} is runtime-like"
        if part.startswith(".env"):
            return "path targets an environment file"
    name = path.name.lower()
    if any(name.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
        return f"path suffix for {path.name!r} is forbidden"
    if "cookie" in name:
        return "path name is cookie-like"
    if "keychain" in name:
        return "path name is keychain-like"
    return None


def ensure_safe_input_path(path: Path, *, label: str) -> Path:
    reason = _path_looks_forbidden(path)
    if reason:
        raise RefusedPathError(f"Refusing {label}: {reason}.")
    return path


def _run_git(repo_root: Path, args: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, redact_text(str(exc))
    output = result.stdout.strip() if result.stdout else result.stderr.strip()
    return result.returncode == 0, redact_text(output)


def _git_summary(repo_root: Path) -> dict[str, Any]:
    branch_ok, branch = _run_git(repo_root, ["branch", "--show-current"])
    status_ok, status = _run_git(repo_root, ["status", "--short", "--branch"])
    latest_ok, latest = _run_git(repo_root, ["log", "--oneline", "-1"])
    status_lines = [line for line in status.splitlines() if line.strip()]
    dirty_lines = [
        line
        for line in status_lines
        if not line.startswith("## ") and line.strip()
    ]
    return {
        "branch": branch if branch_ok and branch else "<unknown>",
        "status_available": status_ok,
        "dirty": bool(dirty_lines) if status_ok else None,
        "status_short_redacted": status_lines[:20],
        "latest_commit_redacted": latest if latest_ok else "",
    }


def _checkpoint_sort_key(path: Path) -> tuple[int, str, float, str]:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    match = re.search(r"checkpoint_(\d+)([a-z]*)", path.name)
    number = int(match.group(1)) if match else -1
    suffix = match.group(2) if match else ""
    return (int(mtime), number, suffix, path.name)


def _latest_checkpoint_docs(repo_root: Path, *, limit: int = 6) -> list[dict[str, str]]:
    docs: list[Path] = []
    for pattern in SAFE_CHECKPOINT_GLOBS:
        docs.extend(repo_root.glob(pattern))
    docs = sorted({path for path in docs if path.is_file()}, key=_checkpoint_sort_key)
    latest = docs[-limit:]
    entries: list[dict[str, str]] = []
    for path in reversed(latest):
        try:
            rel = path.relative_to(repo_root).as_posix()
        except ValueError:
            rel = f"<repo-root>/{redact_text(path.name)}"
        title = ""
        try:
            for line in path.read_text(encoding="utf-8").splitlines()[:12]:
                if line.startswith("# "):
                    title = redact_text(line.removeprefix("# ").strip())
                    break
        except OSError:
            title = ""
        entries.append({"path": rel, "title": title})
    return entries


def _load_json_file(path: Path) -> dict[str, Any]:
    ensure_safe_input_path(path, label="JSON input path")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON input must be an object.")
    return payload


def _optional_issue_lifecycle(
    *,
    lifecycle_path: Path | None,
    live_gitea_read: bool,
) -> dict[str, Any] | None:
    if live_gitea_read:
        if capture_issue_lifecycle_status is None:
            return {
                "schema_version": "hermes.pm.issue_lifecycle_status.v1",
                "lifecycle_state": "unknown",
                "blockers": [
                    "Issue lifecycle module is unavailable in this environment."
                ],
            }
        return capture_issue_lifecycle_status()
    if lifecycle_path is None:
        return None
    return _load_json_file(lifecycle_path)


def _optional_gitea_snapshot(
    *,
    snapshot_path: Path | None,
    live_gitea_read: bool,
) -> dict[str, Any] | None:
    if live_gitea_read:
        if capture_gitea_snapshot is None:
            return {
                "schema_version": "hermes.pm.gitea_readonly_snapshot.v1",
                "provider_status": _gitea_import_status.as_dict(),
                "blockers": [_provider_import_blocker(_gitea_import_status)],
            }
        snapshot = capture_gitea_snapshot()
        if isinstance(snapshot, dict):
            snapshot.setdefault("provider_status", _gitea_import_status.as_dict())
        return snapshot
    if snapshot_path is None:
        return None
    return _load_json_file(snapshot_path)


def _gitea_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    issues = (
        snapshot.get("issues") if isinstance(snapshot.get("issues"), dict) else {}
    )
    prs = (
        snapshot.get("pull_requests")
        if isinstance(snapshot.get("pull_requests"), dict)
        else {}
    )
    checks = (
        snapshot.get("checks") if isinstance(snapshot.get("checks"), dict) else {}
    )
    workflows = (
        snapshot.get("workflows")
        if isinstance(snapshot.get("workflows"), dict)
        else {}
    )
    statuses = (
        checks.get("statuses") if isinstance(checks.get("statuses"), list) else []
    )
    return {
        "schema_version": snapshot.get("schema_version"),
        "base_url": snapshot.get("gitea_base_url"),
        "owner": snapshot.get("owner"),
        "repo": snapshot.get("repo"),
        "auth_used": bool(snapshot.get("auth_used")),
        "token_value_exposed": False,
        "http_methods_used": snapshot.get("http_methods_used") or [],
        "open_issue_count": int(issues.get("open_count") or 0),
        "recently_closed_issue_count": int(
            issues.get("recently_closed_count") or 0
        ),
        "open_pr_count": int(prs.get("open_count") or 0),
        "recently_closed_or_merged_pr_count": int(
            prs.get("recently_closed_or_merged_count") or 0
        ),
        "status_count": len(statuses),
        "combined_status": checks.get("combined_status") or {},
        "workflow_run_count": int(workflows.get("recent_run_count") or 0),
        "blockers": snapshot.get("blockers") or [],
        "warnings": snapshot.get("warnings") or [],
    }


def _work_state_from_snapshot(
    *,
    pm_status: dict[str, Any],
    gitea_snapshot: dict[str, Any],
    issue_lifecycle: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if build_work_state is None:
        return None
    return build_work_state(
        pm_status=pm_status,
        gitea_snapshot=gitea_snapshot,
        issue_lifecycle=issue_lifecycle,
    )


def _has_local_import_blocker(summary_or_snapshot: dict[str, Any]) -> bool:
    blockers = summary_or_snapshot.get("blockers")
    if not isinstance(blockers, list):
        return False
    return any(
        isinstance(blocker, dict) and blocker.get("endpoint") == "<local-import>"
        for blocker in blockers
    )


def _recommendation_class(
    *,
    gitea_summary: dict[str, Any],
    work_state: dict[str, Any] | None,
) -> str:
    if _has_local_import_blocker(gitea_summary):
        return "control_plane_regression"
    if work_state is None:
        return "stale_context"
    if work_state.get("blocked_items") or work_state.get("untriaged_items"):
        return "approval_required"
    ci_status = work_state.get("ci_status_summary")
    if isinstance(ci_status, dict) and not bool(
        ci_status.get("evidence_ready_for_future_workflow_trial")
    ):
        return "approval_required"
    if gitea_summary.get("blockers"):
        return "external_gitea_degraded"
    return "no_action_required"


def _fallback_next_pm_action(recommendation_class: str) -> str:
    if recommendation_class == "control_plane_regression":
        return (
            "Repair PM status provider import isolation before trusting PM/Kanban "
            "recommendations."
        )
    if recommendation_class == "external_gitea_degraded":
        return "Review live Gitea read blockers before planning mutations."
    if recommendation_class == "stale_context":
        return "Regenerate PM/Kanban context after provider health is restored."
    if recommendation_class == "approval_required":
        return "Resolve blocked PM items or request explicit approvals."
    return "No PM action is required from the current read-only status."


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and re.fullmatch(r"[-+]?\d+", stripped):
            return int(stripped)
    return None


def classify_pm_kanban_semantic_parity(
    *,
    pm_status: dict[str, Any],
    kanban_packet: dict[str, Any],
) -> dict[str, Any]:
    raw_pm_gitea = pm_status.get("gitea_pm_snapshot_summary")
    pm_gitea: dict[str, Any] = raw_pm_gitea if isinstance(raw_pm_gitea, dict) else {}
    raw_pm_lifecycle = pm_status.get("issue_lifecycle_summary")
    pm_lifecycle: dict[str, Any] = (
        raw_pm_lifecycle if isinstance(raw_pm_lifecycle, dict) else {}
    )
    raw_kanban_summary = kanban_packet.get("daily_summary")
    kanban_summary: dict[str, Any] = (
        raw_kanban_summary if isinstance(raw_kanban_summary, dict) else {}
    )
    blockers: list[dict[str, Any]] = []
    pm_open_issue_count = _safe_int(pm_gitea.get("open_issue_count"))
    kanban_open_issue_count = _safe_int(kanban_summary.get("open_issue_count"))
    if pm_open_issue_count is None or kanban_open_issue_count is None:
        blockers.append(
            {
                "code": "open_issue_count_unparseable",
                "pm_status_open_issue_count": pm_gitea.get("open_issue_count"),
                "kanban_open_issue_count": kanban_summary.get("open_issue_count"),
            }
        )
    elif pm_open_issue_count != kanban_open_issue_count:
        blockers.append(
            {
                "code": "open_issue_count_mismatch",
                "pm_status_open_issue_count": pm_open_issue_count,
                "kanban_open_issue_count": kanban_open_issue_count,
            }
        )
    if bool(pm_lifecycle.get("exists")) != bool(
        kanban_summary.get("seed_pm_issue_exists")
    ):
        blockers.append(
            {
                "code": "seed_issue_existence_mismatch",
                "pm_status_seed_exists": bool(pm_lifecycle.get("exists")),
                "kanban_seed_exists": bool(
                    kanban_summary.get("seed_pm_issue_exists")
                ),
            }
        )
    if _has_local_import_blocker(pm_gitea):
        blockers.append(
            {
                "code": "pm_status_local_import_blocker",
                "blockers": pm_gitea.get("blockers") or [],
            }
        )
    return {
        "consistent": not blockers,
        "recommendation_class": (
            "control_plane_regression" if blockers else "no_action_required"
        ),
        "blockers": blockers,
    }


def _approval_candidates(
    *,
    gitea_summary: dict[str, Any],
    work_state: dict[str, Any] | None,
) -> list[str]:
    candidates = [
        "Any Gitea issue, PR, label, card, project, or comment mutation "
        "requires explicit approval."
    ]
    if int(gitea_summary.get("open_pr_count") or 0):
        candidates.append(
            "Open PR attention may require approved PR labels or comments."
        )
    if work_state and work_state.get("untriaged_items"):
        candidates.append(
            "Untriaged items may require approved issue labels or board moves."
        )
    if work_state and work_state.get("blocked_items"):
        candidates.append(
            "Blocked items may require an approved follow-up or scope decision."
        )
    return candidates


def _default_project(repo_root: Path, project_id: str) -> dict[str, Any]:
    return {
        "project_id": project_id,
        "repo_path": str(repo_root),
        "gitea_remote": "http://127.0.0.1:3005/preston/crypto_bot.git",
        "authority_profile": "trading-control-plane",
        "default_mode": "read_only_pm",
        "forbidden_surfaces": list(DEFAULT_FORBIDDEN_SURFACES),
    }


def load_managed_project(
    *,
    registry_path: Path | None,
    project_id: str,
    repo_root_override: Path | None,
) -> dict[str, Any]:
    if registry_path is None:
        repo_root = (repo_root_override or Path.cwd()).resolve(strict=False)
        ensure_safe_input_path(repo_root, label="repo root")
        return _default_project(repo_root, project_id)
    payload = _load_json_file(registry_path)
    projects = payload.get("projects")
    if not isinstance(projects, list):
        raise ValueError("Managed project registry must contain a projects list.")
    for item in projects:
        if isinstance(item, dict) and item.get("project_id") == project_id:
            project = dict(item)
            if repo_root_override is not None:
                project["repo_path"] = str(repo_root_override)
            return project
    raise ValueError(f"Project {project_id!r} was not found in the registry.")


def _ci_evidence_summary(repo_root: Path) -> dict[str, Any]:
    if build_gitea_ci_evidence_summary is None:
        return {
            "available": False,
            "reason": "existing Hermes CI evidence module is unavailable",
        }
    mirror_map = (
        repo_root
        / "scripts/hermes_operator/gitea_action_mirror_map.example.json"
    )
    runner_inventory = (
        repo_root
        / "scripts/hermes_operator/gitea_runner_inventory.example.json"
    )
    mirror_evidence = (
        repo_root
        / (
            "scripts/hermes_operator/"
            "gitea_action_mirror_evidence.checkpoint20c.example.json"
        )
    )
    runner_evidence = (
        repo_root
        / (
            "scripts/hermes_operator/"
            "gitea_runner_registration_evidence.checkpoint22.example.json"
        )
    )
    try:
        summary = build_gitea_ci_evidence_summary(
            repo_root=repo_root,
            mirror_map_path=mirror_map if mirror_map.exists() else None,
            runner_inventory_path=(
                runner_inventory if runner_inventory.exists() else None
            ),
            action_mirror_evidence_path=(
                mirror_evidence if mirror_evidence.exists() else None
            ),
            runner_registration_evidence_path=(
                runner_evidence if runner_evidence.exists() else None
            ),
        )
    except Exception as exc:  # pragma: no cover - defensive status path
        return {"available": False, "reason": redact_text(str(exc))}
    return {
        "available": True,
        "required_mirrors_covered": bool(summary.get("required_mirrors_covered")),
        "action_mirror_evidence_valid": bool(
            summary.get("action_mirror_evidence_valid")
        ),
        "runner_registration_proven": bool(
            summary.get("runner_registration_proven")
        ),
        "runner_registration_evidence_valid": bool(
            summary.get("runner_registration_evidence_valid")
        ),
        "runner_online_for_future_workflow_trial": bool(
            summary.get("runner_online_for_future_workflow_trial")
        ),
        "evidence_ready_for_future_workflow_trial": bool(
            summary.get("evidence_ready_for_future_workflow_trial")
        ),
        "blockers": summary.get("blockers") or [],
        "warnings": summary.get("warnings") or [],
        "next_approval_needed": summary.get("next_approval_needed"),
    }


def _runner_readiness(ci_summary: dict[str, Any]) -> dict[str, Any]:
    if not ci_summary.get("available"):
        return {
            "registration": "unknown",
            "online": "unknown",
            "workflow_trial": "blocked",
        }
    registration = (
        "proven" if ci_summary.get("runner_registration_proven") else "unknown"
    )
    online = (
        "online"
        if ci_summary.get("runner_online_for_future_workflow_trial")
        else "offline"
    )
    workflow_trial = (
        "ready"
        if ci_summary.get("evidence_ready_for_future_workflow_trial")
        else "blocked"
    )
    return {
        "registration": registration,
        "online": online,
        "workflow_trial": workflow_trial,
        "approval_required_before_start": True,
    }


def build_project_status(
    *,
    project_registry: Path | None = None,
    project_id: str = DEFAULT_PROJECT_ID,
    repo_root: Path | None = None,
    gitea_snapshot: dict[str, Any] | None = None,
    issue_lifecycle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if repo_root is not None:
        ensure_safe_input_path(repo_root, label="repo root")
        repo_root = repo_root.resolve(strict=False)
    project = load_managed_project(
        registry_path=project_registry,
        project_id=project_id,
        repo_root_override=repo_root,
    )
    project_repo = Path(str(project.get("repo_path") or repo_root or Path.cwd()))
    ensure_safe_input_path(project_repo, label="managed project repo root")
    project_repo = project_repo.resolve(strict=False)
    git = _git_summary(project_repo)
    checkpoint_docs = _latest_checkpoint_docs(project_repo)
    ci_summary = _ci_evidence_summary(project_repo)
    runner_summary = _runner_readiness(ci_summary)
    blockers: list[str] = []
    warnings: list[str] = []
    if git.get("dirty") is True:
        blockers.append("Worktree is dirty; PM status remains read-only.")
    if runner_summary.get("workflow_trial") != "ready":
        blockers.append(
            "Workflow trial remains blocked until a future approval starts a "
            "runner and runs a controlled CI trial."
        )
    warnings.extend(str(item) for item in ci_summary.get("warnings") or [])
    forbidden_surfaces = project.get("forbidden_surfaces")
    if not isinstance(forbidden_surfaces, list) or not forbidden_surfaces:
        forbidden_surfaces = list(DEFAULT_FORBIDDEN_SURFACES)
    status = {
        "schema_version": PM_STATUS_SCHEMA_VERSION,
        "tool": "hermes_pm_status",
        "created_at": _utc_now(),
        "project": {
            "project_id": str(project.get("project_id") or project_id),
            "repo_root_redacted_or_relative": _safe_repo_reference(project_repo),
            "gitea_remote": redact_text(str(project.get("gitea_remote") or "")),
            "authority_profile": str(project.get("authority_profile") or ""),
            "default_mode": str(project.get("default_mode") or "read_only_pm"),
            "forbidden_surfaces": [str(item) for item in forbidden_surfaces],
        },
        "scope_distinction": {
            "hermes_pm_platform": (
                "Hermes PM platform owns work intake, triage, plans, "
                "delegation, evidence review, approval requests, and operator "
                "status."
            ),
            "managed_project_runtime": (
                "crypto_bot is managed as a product repository and example "
                "corpus; its daemon, broker, runtime, deployment, and trading "
                "surfaces remain out of scope for PM status."
            ),
        },
        "git": git,
        "latest_checkpoint_docs": checkpoint_docs,
        "known_blockers": sorted(dict.fromkeys(blockers)),
        "warnings": sorted(dict.fromkeys(redact_text(item) for item in warnings)),
        "ci_locality_readiness": ci_summary,
        "provider_import_status": {
            name: status.as_dict()
            for name, status in PROVIDER_IMPORT_STATUS.items()
        },
        "runner_readiness": runner_summary,
        "outstanding_approval_gates": [
            "branch_write requires scoped operator approval and evidence gates",
            "forge_write requires explicit operator approval before Gitea mutation",
            "runner start requires a future explicit checkpoint",
            "workflow trial requires a future explicit checkpoint",
            "deploy/runtime/secret/financial surfaces are forbidden here",
        ],
        "recommended_next_pm_action": (
            "Use the Hermes Telegram PM startup prompt, review this status, "
            "propose a PM backlog/Kanban sync plan, and ask for approval before "
            "any write or forge mutation."
        ),
        "non_action_booleans": dict(NON_ACTION_BOOLEANS),
    }
    if gitea_snapshot is not None:
        gitea_summary = _gitea_summary(gitea_snapshot)
        if compact_lifecycle_summary is not None and issue_lifecycle is not None:
            seed_summary = compact_lifecycle_summary(issue_lifecycle)
        elif summarize_seed_issue_from_snapshot is not None:
            seed_summary = summarize_seed_issue_from_snapshot(gitea_snapshot)
        else:
            seed_summary = {
                "issue_index": 1,
                "title": EXPECTED_PM_SEED_ISSUE_TITLE,
                "exists": False,
                "state": None,
                "lifecycle_state": "unknown",
                "duplicate_seed_issue_blocker": False,
            }
        work_state = _work_state_from_snapshot(
            pm_status=status,
            gitea_snapshot=gitea_snapshot,
            issue_lifecycle=issue_lifecycle,
        )
        untriaged_count = (
            len(work_state.get("untriaged_items") or []) if work_state else 0
        )
        blocked_count = (
            len(work_state.get("blocked_items") or []) if work_state else 0
        )
        recommendation_class = _recommendation_class(
            gitea_summary=gitea_summary,
            work_state=work_state,
        )
        next_pm_action = (
            (work_state.get("recommended_next_actions") or [None])[0]
            if work_state
            else None
        ) or _fallback_next_pm_action(recommendation_class)
        status["gitea_pm_snapshot_summary"] = gitea_summary
        status["issue_lifecycle_summary"] = {
            "issue_index": seed_summary.get("issue_index"),
            "issue_url": seed_summary.get("issue_url"),
            "title": seed_summary.get("title") or EXPECTED_PM_SEED_ISSUE_TITLE,
            "exists": bool(seed_summary.get("exists")),
            "state": seed_summary.get("state"),
            "lifecycle_state": seed_summary.get("lifecycle_state"),
            "duplicate_seed_issue_blocker": bool(
                seed_summary.get("duplicate_seed_issue_blocker")
            ),
            "source": seed_summary.get("source") or "gitea_snapshot",
        }
        status["pm_work_state_summary"] = {
            "project_id": str(project.get("project_id") or project_id),
            "open_issue_count": gitea_summary["open_issue_count"],
            "open_pr_count": gitea_summary["open_pr_count"],
            "seed_pm_issue_exists": bool(seed_summary.get("exists")),
            "seed_pm_issue_state": seed_summary.get("state"),
            "seed_pm_issue_lifecycle_state": seed_summary.get("lifecycle_state"),
            "duplicate_seed_issue_blocker": bool(
                seed_summary.get("duplicate_seed_issue_blocker")
            ),
            "untriaged_count": untriaged_count,
            "blocked_count": blocked_count,
            "ci_status_summary": (
                work_state.get("ci_status_summary") if work_state else {}
            ),
            "recommendation_class": recommendation_class,
            "approval_candidates": _approval_candidates(
                gitea_summary=gitea_summary,
                work_state=work_state,
            ),
            "next_pm_action": next_pm_action,
        }
        status["recommended_next_pm_action"] = (
            status["pm_work_state_summary"]["next_pm_action"]
        )
    return _redact_structure(status)


def format_project_status_text(report: dict[str, Any]) -> str:
    payload = _redact_structure(report if isinstance(report, dict) else {})
    project = payload.get("project") if isinstance(payload.get("project"), dict) else {}
    git = payload.get("git") if isinstance(payload.get("git"), dict) else {}
    ci = (
        payload.get("ci_locality_readiness")
        if isinstance(payload.get("ci_locality_readiness"), dict)
        else {}
    )
    runner = (
        payload.get("runner_readiness")
        if isinstance(payload.get("runner_readiness"), dict)
        else {}
    )
    gitea = (
        payload.get("gitea_pm_snapshot_summary")
        if isinstance(payload.get("gitea_pm_snapshot_summary"), dict)
        else {}
    )
    work_state = (
        payload.get("pm_work_state_summary")
        if isinstance(payload.get("pm_work_state_summary"), dict)
        else {}
    )
    issue_lifecycle = (
        payload.get("issue_lifecycle_summary")
        if isinstance(payload.get("issue_lifecycle_summary"), dict)
        else {}
    )
    blockers = payload.get("known_blockers") or []
    warnings = payload.get("warnings") or []
    lines = [
        "Hermes PM status",
        f"Project: {project.get('project_id') or '<unknown>'}",
        f"Mode: {project.get('default_mode') or 'read_only_pm'}",
        f"Branch: {git.get('branch') or '<unknown>'}",
        f"Worktree: {'dirty' if git.get('dirty') else 'clean'}",
        (
            "CI evidence: mirrors "
            + ("covered" if ci.get("required_mirrors_covered") else "unknown")
            + ", runner "
            + ("proven" if ci.get("runner_registration_proven") else "unknown")
        ),
        (
            "Runner: "
            f"{runner.get('registration') or 'unknown'}, "
            f"{runner.get('online') or 'unknown'}"
        ),
        f"Blockers: {len(blockers)}",
    ]
    if gitea:
        lines.extend(
            [
                (
                    "Gitea: "
                    f"{gitea.get('open_issue_count', 0)} open issues, "
                    f"{gitea.get('open_pr_count', 0)} open PRs, "
                    f"{gitea.get('status_count', 0)} statuses"
                ),
                (
                    "PM work state: "
                    f"{work_state.get('untriaged_count', 0)} untriaged, "
                    f"{work_state.get('blocked_count', 0)} blocked"
                ),
            ]
        )
    if issue_lifecycle:
        lines.append(
            "Seed issue: "
            + (
                "exists"
                if issue_lifecycle.get("exists")
                else "missing or unknown"
            )
            + f", {issue_lifecycle.get('lifecycle_state') or 'unknown'}"
        )
    for item in blockers[:4]:
        lines.append(f"- {redact_text(str(item))}")
    if warnings:
        lines.append(f"Warnings: {len(warnings)}")
        for item in warnings[:3]:
            lines.append(f"- {redact_text(str(item))}")
    latest_docs = payload.get("latest_checkpoint_docs") or []
    if latest_docs:
        lines.append("Latest docs:")
        for item in latest_docs[:3]:
            if not isinstance(item, dict):
                continue
            lines.append(f"- {item.get('path')}")
    lines.append(f"Next: {payload.get('recommended_next_pm_action')}")
    return redact_text("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only Hermes PM status reporter. It summarizes git state, "
            "checkpoint docs, local CI evidence, runner readiness, approval "
            "gates, and forbidden surfaces without writing files, mutating "
            "Gitea, starting runners, running workflows, deploying, inspecting "
            "secrets, or touching runtime/financial surfaces."
        )
    )
    parser.add_argument("--project-registry", type=Path)
    parser.add_argument("--project-id", default=DEFAULT_PROJECT_ID)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--gitea-snapshot", type=Path)
    parser.add_argument("--issue-lifecycle", type=Path)
    parser.add_argument(
        "--live-gitea-read",
        action="store_true",
        help="Include a live read-only Gitea snapshot using GET requests only.",
    )
    parser.add_argument("--format", choices=("json", "text"), default="json")
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument(
        "--describe-authority",
        action="store_true",
        help="Print this tool's Hermes PM authority metadata as JSON.",
    )
    return parser


OPERATOR_AUTHORITY_METADATA = {
    "tool": "hermes_pm_status",
    "authority_class": "read",
    "schema_version": PM_STATUS_SCHEMA_VERSION,
    "read_only": True,
    "mutation_capability": False,
    **NON_ACTION_BOOLEANS,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    indent = 2 if args.pretty else None
    if args.describe_authority:
        print(json.dumps(OPERATOR_AUTHORITY_METADATA, indent=indent, sort_keys=True))
        return 0
    try:
        gitea_snapshot = _optional_gitea_snapshot(
            snapshot_path=args.gitea_snapshot,
            live_gitea_read=args.live_gitea_read,
        )
        issue_lifecycle = _optional_issue_lifecycle(
            lifecycle_path=args.issue_lifecycle,
            live_gitea_read=args.live_gitea_read,
        )
        report = build_project_status(
            project_registry=args.project_registry,
            project_id=args.project_id,
            repo_root=args.repo_root,
            gitea_snapshot=gitea_snapshot,
            issue_lifecycle=issue_lifecycle,
        )
    except (RefusedPathError, ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    if args.format == "text":
        print(format_project_status_text(report))
    else:
        print(json.dumps(report, indent=indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
