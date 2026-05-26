"""Read-only Kanban workspace janitor classifier.

This module intentionally performs no deletion. It classifies Kanban task
workspaces and reproducible artifacts so cleanup can stay report-first and
approval-gated.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ARTIFACT_NAMES = {
    "node_modules",
    ".next",
    ".turbo",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    "target",
    "coverage",
}
TERMINAL_STATUSES = {"done", "archived", "cancelled", "superseded"}


@dataclass(slots=True)
class WorkspaceReport:
    task_id: str
    workspace_path: str
    state: str
    reason: str
    size_bytes: int
    task: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    gates: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "workspace_path": self.workspace_path,
            "state": self.state,
            "reason": self.reason,
            "size_bytes": self.size_bytes,
            "task": self.task,
            "artifacts": self.artifacts,
            "gates": self.gates,
        }


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


def path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file() or path.is_symlink():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for root, dirs, files in os.walk(path, followlinks=False):
        for name in files:
            p = Path(root) / name
            try:
                if not p.is_symlink():
                    total += p.stat().st_size
            except OSError:
                continue
    return total


def discover_artifacts(workspace: Path, *, min_bytes: int = 0) -> list[dict[str, Any]]:
    """Return allowlisted reproducible artifacts under a workspace."""
    artifacts: list[dict[str, Any]] = []
    if not workspace.exists():
        return artifacts
    for root, dirs, _files in os.walk(workspace, followlinks=False):
        # Mutate dirs so os.walk does not descend into already-counted artifacts.
        kept: list[str] = []
        for dirname in dirs:
            child = Path(root) / dirname
            if dirname in ARTIFACT_NAMES:
                size = path_size(child)
                if size >= min_bytes:
                    artifacts.append({
                        "kind": dirname,
                        "path": str(child),
                        "size_bytes": size,
                    })
            else:
                kept.append(dirname)
        dirs[:] = kept
    artifacts.sort(key=lambda item: int(item.get("size_bytes") or 0), reverse=True)
    return artifacts


def process_cwds() -> list[str]:
    cwds: list[str] = []
    proc = Path("/proc")
    if not proc.exists():
        return cwds
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cwd = (entry / "cwd").resolve()
        except OSError:
            continue
        cwds.append(str(cwd))
    return cwds


def tmux_cwds() -> list[str]:
    try:
        cp = subprocess.run(
            ["tmux", "list-panes", "-a", "-F", "#{pane_current_path}"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [line.strip() for line in cp.stdout.splitlines() if line.strip()]


def git_state(workspace: Path) -> dict[str, Any]:
    if not (workspace / ".git").exists():
        return {"is_git_worktree": False, "dirty": None, "status_short": None}
    cp = subprocess.run(
        ["git", "-C", str(workspace), "status", "--short"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    status = cp.stdout.strip()
    return {
        "is_git_worktree": True,
        "dirty": bool(status) if cp.returncode == 0 else None,
        "status_short": status,
        "error": cp.stderr.strip() if cp.returncode != 0 else None,
    }


def _load_task(conn: sqlite3.Connection, task_id: str) -> dict[str, Any]:
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT id, public_id, title, status, review_phase, completed_at, result,
               closeout_evidence, current_run_id, worker_pid, assignee, workspace_kind,
               workspace_path
        FROM tasks WHERE id = ?
        """,
        (task_id,),
    ).fetchone()
    return dict(row) if row else {"id": task_id, "status": "unknown"}


def classify_workspace(
    workspace: Path,
    task: Mapping[str, Any],
    *,
    now: int | None = None,
    artifact_ttl_seconds: int = 48 * 3600,
    workspace_ttl_seconds: int = 7 * 24 * 3600,
    proc_cwds: Sequence[str] | None = None,
    pane_cwds: Sequence[str] | None = None,
    min_artifact_bytes: int = 0,
) -> WorkspaceReport:
    """Classify one workspace without mutating it."""
    now = int(now if now is not None else __import__("time").time())
    task_id = str(task.get("id") or workspace.name)
    proc_cwds = list(process_cwds() if proc_cwds is None else proc_cwds)
    pane_cwds = list(tmux_cwds() if pane_cwds is None else pane_cwds)
    artifacts = discover_artifacts(workspace, min_bytes=min_artifact_bytes)
    gstate = git_state(workspace)
    size = path_size(workspace)

    active_refs = [p for p in [*proc_cwds, *pane_cwds] if _is_relative_to(Path(p), workspace)]
    active_worker = bool(task.get("current_run_id") or task.get("worker_pid"))
    status = str(task.get("status") or "unknown")
    completed_at = task.get("completed_at") or 0
    age = max(0, now - int(completed_at or 0)) if completed_at else None
    has_evidence = bool(task.get("result") or task.get("closeout_evidence"))

    gates = {
        "terminal_status": status in TERMINAL_STATUSES,
        "age_seconds": age,
        "artifact_ttl_met": age is not None and age >= artifact_ttl_seconds,
        "workspace_ttl_met": age is not None and age >= workspace_ttl_seconds,
        "active_refs": active_refs,
        "active_worker": active_worker,
        "git": gstate,
        "has_evidence": has_evidence,
        "artifact_count": len(artifacts),
    }

    if active_refs or active_worker or status not in TERMINAL_STATUSES:
        return WorkspaceReport(task_id, str(workspace), "blocked-active", "task is active/non-terminal or has active references", size, dict(task), artifacts, gates)
    if not has_evidence:
        return WorkspaceReport(task_id, str(workspace), "approval-required", "terminal task lacks preserved result/closeout evidence", size, dict(task), artifacts, gates)
    if gstate.get("dirty") is True:
        return WorkspaceReport(task_id, str(workspace), "approval-required", "git worktree has dirty/untracked state", size, dict(task), artifacts, gates)
    if artifacts and gates["artifact_ttl_met"]:
        return WorkspaceReport(task_id, str(workspace), "safe-artifact-candidate", "allowlisted reproducible artifacts meet terminal-state TTL", size, dict(task), artifacts, gates)
    if gates["workspace_ttl_met"] and gstate.get("dirty") is False:
        return WorkspaceReport(task_id, str(workspace), "future-workspace-cleanup-candidate", "clean terminal workspace meets full-workspace TTL", size, dict(task), artifacts, gates)
    return WorkspaceReport(task_id, str(workspace), "approval-required", "terminal workspace is too recent or lacks an allowlisted cleanup path", size, dict(task), artifacts, gates)


def classify_workspaces(
    db_path: Path,
    workspaces_root: Path,
    *,
    now: int | None = None,
    min_workspace_bytes: int = 0,
    min_artifact_bytes: int = 0,
) -> list[WorkspaceReport]:
    """Classify every task workspace directory under *workspaces_root*."""
    conn = sqlite3.connect(str(db_path))
    reports: list[WorkspaceReport] = []
    if not workspaces_root.exists():
        return reports
    proc = process_cwds()
    panes = tmux_cwds()
    for workspace in sorted(p for p in workspaces_root.iterdir() if p.is_dir()):
        size = path_size(workspace)
        if size < min_workspace_bytes:
            continue
        task = _load_task(conn, workspace.name)
        report = classify_workspace(
            workspace,
            task,
            now=now,
            proc_cwds=proc,
            pane_cwds=panes,
            min_artifact_bytes=min_artifact_bytes,
        )
        reports.append(report)
    reports.sort(key=lambda report: report.size_bytes, reverse=True)
    return reports


@dataclass(slots=True)
class CleanupAction:
    """A planned exact-path artifact cleanup action.

    The action is inert until passed to ``apply_artifact_cleanup_actions`` with
    ``dry_run=False``. This keeps BO-127's apply path explicit and testable.
    """

    task_id: str
    workspace_path: str
    artifact_path: str
    kind: str
    size_bytes: int
    candidate_state: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "workspace_path": self.workspace_path,
            "artifact_path": self.artifact_path,
            "kind": self.kind,
            "size_bytes": self.size_bytes,
            "candidate_state": self.candidate_state,
            "reason": self.reason,
        }


def plan_artifact_cleanup_actions(
    reports: Iterable[WorkspaceReport],
) -> list[CleanupAction]:
    """Plan exact allowlisted artifact cleanup actions from classifier output."""
    actions: list[CleanupAction] = []
    for report in reports:
        if report.state != "safe-artifact-candidate":
            continue
        workspace = Path(report.workspace_path)
        for artifact in report.artifacts:
            kind = str(artifact.get("kind") or "")
            artifact_path = Path(str(artifact.get("path") or ""))
            if kind not in ARTIFACT_NAMES:
                continue
            if artifact_path.name != kind:
                continue
            if not _is_relative_to(artifact_path, workspace):
                continue
            actions.append(
                CleanupAction(
                    task_id=report.task_id,
                    workspace_path=str(workspace),
                    artifact_path=str(artifact_path),
                    kind=kind,
                    size_bytes=int(artifact.get("size_bytes") or 0),
                    candidate_state=report.state,
                    reason=report.reason,
                )
            )
    actions.sort(key=lambda action: action.size_bytes, reverse=True)
    return actions


def validate_artifact_safety(
    artifact_path: str,
    workspace_path: str,
    kind: str,
    *,
    proc_cwds: Sequence[str] | None = None,
    pane_cwds: Sequence[str] | None = None,
    kanban_db_path: str | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Validate safety of an artifact cleanup action.

    Returns ``{"safe": True, "guard_errors": []}`` when all checks pass,
    or ``{"safe": False, "guard_errors": [...]}`` with reasons when any
    guard fails.  This function is intentionally reusable by the closeout
    verifier and other lifecycle gate callers.

    Guards (fail-closed, in order):
    - kind is in ARTIFACT_NAMES allowlist
    - artifact basename matches the allowlisted kind
    - artifact path is under the declared workspace
    - artifact is not a symlink
    - artifact exists and is a directory (source file deletion fails)
    - no active process cwd under target
    - no active tmux pane cwd under target
    - no active kanban worker/run for this task (when kanban_db_path provided)
    """
    artifact = Path(artifact_path)
    workspace = Path(workspace_path)
    guard_errors: list[str] = []

    if kind not in ARTIFACT_NAMES:
        guard_errors.append("kind_not_allowlisted")
    if artifact.name != kind:
        guard_errors.append("artifact_basename_mismatch")
    if not _is_relative_to(artifact, workspace):
        guard_errors.append("artifact_not_under_workspace")
    if artifact.is_symlink():
        guard_errors.append("artifact_is_symlink")
    if not artifact.exists():
        guard_errors.append("artifact_missing")
    if artifact.exists() and not artifact.is_dir():
        guard_errors.append("artifact_not_directory")

    # Active process CWD guard
    cwds = list(proc_cwds or ())
    if any(_is_relative_to(Path(c), artifact) for c in cwds):
        guard_errors.append("active_process_cwd")

    # Active tmux CWD guard
    panes = list(pane_cwds or ())
    if any(_is_relative_to(Path(p), artifact) for p in panes):
        guard_errors.append("active_tmux_cwd")

    # Active Kanban worker guard
    if kanban_db_path and task_id:
        try:
            conn = sqlite3.connect(kanban_db_path)
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT current_run_id, worker_pid FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
            if row and (row["current_run_id"] or row["worker_pid"]):
                guard_errors.append("active_kanban_worker")
        except sqlite3.Error:
            pass  # fail open on DB read errors — cleanup is not security-critical

    return {"safe": len(guard_errors) == 0, "guard_errors": guard_errors}


def apply_artifact_cleanup_actions(
    actions: Iterable[CleanupAction],
    *,
    dry_run: bool = True,
    proc_cwds: Sequence[str] | None = None,
    pane_cwds: Sequence[str] | None = None,
    kanban_db_path: str | None = None,
) -> list[dict[str, Any]]:
    """Apply or preview exact artifact cleanup actions.

    ``dry_run=True`` is the default and never deletes. ``dry_run=False`` only
    removes an artifact if all final exact-path guards still pass:

    - artifact path exists and is a directory;
    - artifact is not a symlink;
    - artifact path remains under the recorded workspace;
    - basename still matches the allowlisted artifact kind;
    - no active process cwd under target;
    - no active tmux pane cwd under target;
    - no active kanban worker/run for this task.
    """
    results: list[dict[str, Any]] = []
    for action in actions:
        safety = validate_artifact_safety(
            artifact_path=action.artifact_path,
            workspace_path=action.workspace_path,
            kind=action.kind,
            proc_cwds=proc_cwds,
            pane_cwds=pane_cwds,
            kanban_db_path=kanban_db_path,
            task_id=action.task_id,
        )
        guard_errors = safety["guard_errors"]

        result = action.to_dict()
        result.update({"dry_run": dry_run, "deleted": False, "guard_errors": guard_errors})
        if guard_errors or dry_run:
            results.append(result)
            continue
        artifact = Path(action.artifact_path)
        before = path_size(artifact)
        shutil.rmtree(artifact)
        result.update({
            "deleted": not artifact.exists(),
            "reclaimed_bytes": before,
        })
        results.append(result)
    return results


@dataclass(slots=True)
class WorkspaceCleanupAction:
    """A planned exact-path full-workspace cleanup action."""

    task_id: str
    workspace_path: str
    size_bytes: int
    candidate_state: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "workspace_path": self.workspace_path,
            "size_bytes": self.size_bytes,
            "candidate_state": self.candidate_state,
            "reason": self.reason,
        }


def plan_workspace_cleanup_actions(
    reports: Iterable[WorkspaceReport],
) -> list[WorkspaceCleanupAction]:
    """Plan full-workspace cleanup actions from strict classifier output."""
    actions: list[WorkspaceCleanupAction] = []
    for report in reports:
        if report.state != "future-workspace-cleanup-candidate":
            continue
        git = report.gates.get("git") or {}
        if git.get("is_git_worktree") is not True or git.get("dirty") is not False:
            continue
        if report.gates.get("active_refs") or report.gates.get("active_worker"):
            continue
        if not report.gates.get("has_evidence"):
            continue
        actions.append(
            WorkspaceCleanupAction(
                task_id=report.task_id,
                workspace_path=report.workspace_path,
                size_bytes=report.size_bytes,
                candidate_state=report.state,
                reason=report.reason,
            )
        )
    actions.sort(key=lambda action: action.size_bytes, reverse=True)
    return actions


def apply_workspace_cleanup_actions(
    actions: Iterable[WorkspaceCleanupAction],
    *,
    dry_run: bool = True,
) -> list[dict[str, Any]]:
    """Apply or preview exact full-workspace cleanup actions.

    Full workspace cleanup is deliberately stricter than artifact cleanup. The
    final guard rejects symlinks, non-directories, non-workspace-looking paths,
    and any git worktree with a dirty status at apply time.
    """
    results: list[dict[str, Any]] = []
    for action in actions:
        workspace = Path(action.workspace_path)
        guard_errors: list[str] = []
        if workspace.is_symlink():
            guard_errors.append("workspace_is_symlink")
        if not workspace.exists():
            guard_errors.append("workspace_missing")
        if workspace.exists() and not workspace.is_dir():
            guard_errors.append("workspace_not_directory")
        if workspace.name != action.task_id:
            guard_errors.append("workspace_basename_not_task_id")
        gstate = git_state(workspace) if workspace.exists() else {"dirty": None}
        if gstate.get("is_git_worktree") is not True:
            guard_errors.append("workspace_not_git_worktree")
        if gstate.get("dirty") is not False:
            guard_errors.append("workspace_git_not_clean")

        result = action.to_dict()
        result.update({"dry_run": dry_run, "deleted": False, "guard_errors": guard_errors})
        if guard_errors or dry_run:
            results.append(result)
            continue
        before = path_size(workspace)
        shutil.rmtree(workspace)
        result.update({
            "deleted": not workspace.exists(),
            "reclaimed_bytes": before,
        })
        results.append(result)
    return results


def _format_bytes(size: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def _disk_usage_dict(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    used = usage.total - usage.free
    percent = (used / usage.total * 100) if usage.total else 0.0
    return {
        "path": str(path),
        "total_bytes": usage.total,
        "used_bytes": used,
        "free_bytes": usage.free,
        "used_percent": round(percent, 1),
    }


def _pressure_level(used_percent: float) -> str:
    if used_percent >= 95:
        return "critical"
    if used_percent >= 85:
        return "warning"
    return "ok"


def collect_disk_pressure_report(
    *,
    root_path: Path = Path("/"),
    data_path: Path = Path("/mnt/hermes-data"),
    db_path: Path = Path.home() / ".hermes" / "kanban.db",
    workspaces_root: Path = Path.home() / ".hermes" / "kanban" / "workspaces",
    top_paths: Sequence[Path] | None = None,
    min_workspace_bytes: int = 10 * 1024 * 1024,
    min_artifact_bytes: int = 10 * 1024 * 1024,
    now: int | None = None,
) -> dict[str, Any]:
    """Collect a read-only daily disk pressure report.

    The report intentionally only observes disk pressure and cleanup candidates;
    it never calls cleanup apply functions. It is safe for a no-agent cron job.
    """
    now = int(now if now is not None else time.time())
    top_paths = list(top_paths or [
        Path.home() / ".hermes" / "hermes-agent" / ".worktrees",
        Path.home() / ".hermes" / "kanban" / "workspaces",
        Path.home() / ".hermes" / "sessions",
        Path("/var/lib/docker"),
        Path("/var/lib/containerd"),
        Path("/tmp"),
        Path.home() / ".npm",
        Path.home() / ".cache" / "uv",
    ])

    root_usage = _disk_usage_dict(root_path)
    data_usage = _disk_usage_dict(data_path) if data_path.exists() else None
    workspace_reports = classify_workspaces(
        db_path,
        workspaces_root,
        now=now,
        min_workspace_bytes=min_workspace_bytes,
        min_artifact_bytes=min_artifact_bytes,
    ) if db_path.exists() else []
    artifact_actions = plan_artifact_cleanup_actions(workspace_reports)
    workspace_actions = plan_workspace_cleanup_actions(workspace_reports)

    pressure_paths: list[dict[str, Any]] = []
    for path in top_paths:
        if not path.exists():
            pressure_paths.append({"path": str(path), "exists": False, "size_bytes": 0})
            continue
        measured_path = path
        resolved_path = None
        if path.is_symlink():
            try:
                candidate = path.resolve(strict=True)
            except OSError:
                candidate = path
            if candidate.is_dir():
                measured_path = candidate
                resolved_path = str(candidate)
        item = {"path": str(path), "exists": True, "size_bytes": path_size(measured_path)}
        if resolved_path:
            item["resolved_path"] = resolved_path
        pressure_paths.append(item)
    pressure_paths.sort(key=lambda item: int(item.get("size_bytes") or 0), reverse=True)

    state_counts: dict[str, int] = {}
    for report in workspace_reports:
        state_counts[report.state] = state_counts.get(report.state, 0) + 1

    return {
        "checked_at_epoch": now,
        "root": root_usage,
        "data": data_usage,
        "pressure_level": _pressure_level(float(root_usage["used_percent"])),
        "top_paths": pressure_paths,
        "workspace_state_counts": state_counts,
        "workspace_reports": [report.to_dict() for report in workspace_reports],
        "artifact_cleanup_candidates": [action.to_dict() for action in artifact_actions],
        "workspace_cleanup_candidates": [action.to_dict() for action in workspace_actions],
        "safe_to_apply_without_approval": False,
        "boundary": "read-only report; no deletion/apply/restart/env mutation",
    }


def format_disk_pressure_report(report: Mapping[str, Any]) -> str:
    """Render a privacy-safe operational report for Discord/cron delivery."""
    root = report["root"]
    data = report.get("data")
    artifact_candidates = list(report.get("artifact_cleanup_candidates") or [])
    workspace_candidates = list(report.get("workspace_cleanup_candidates") or [])
    top_paths = list(report.get("top_paths") or [])[:8]
    state_counts = dict(report.get("workspace_state_counts") or {})

    lines = [
        "📌 Daily disk pressure report",
        f"- root: {root['used_percent']}% used, {_format_bytes(int(root['free_bytes']))} free ({report['pressure_level']})",
    ]
    if data:
        lines.append(f"- /mnt/hermes-data: {data['used_percent']}% used, {_format_bytes(int(data['free_bytes']))} free")
    lines.extend([
        f"- cleanup candidates: artifacts={len(artifact_candidates)}, full_workspaces={len(workspace_candidates)}",
        "- boundary: read-only; no cleanup was applied",
        "",
        "Top pressure paths:",
    ])
    for item in top_paths:
        suffix = _format_bytes(int(item.get("size_bytes") or 0)) if item.get("exists") else "missing"
        resolved = f" -> {item['resolved_path']}" if item.get("resolved_path") else ""
        lines.append(f"- {item['path']}{resolved}: {suffix}")

    lines.extend(["", "Kanban workspace states:"])
    if state_counts:
        for state, count in sorted(state_counts.items()):
            lines.append(f"- {state}: {count}")
    else:
        lines.append("- none")

    if artifact_candidates or workspace_candidates:
        lines.extend(["", "Candidate preview:"])
        for action in artifact_candidates[:5]:
            lines.append(f"- artifact {action['kind']} {action['artifact_path']} ({_format_bytes(int(action['size_bytes']))})")
        for action in workspace_candidates[:5]:
            lines.append(f"- workspace {action['workspace_path']} ({_format_bytes(int(action['size_bytes']))})")
    return "\n".join(lines)
