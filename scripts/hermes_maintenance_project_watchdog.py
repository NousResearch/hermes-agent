#!/usr/bin/env python3
"""Downstream Hermes Maintenance project watchdog health snapshots.

This module is intentionally project-specific/downstream-only.  It does not wire
into the generic Hermes core; it gives the Hermes Maintenance coordinator a
small deterministic surface for evaluating live Kanban, Matrix, and GitHub
bindings from injected snapshots or injected adapter clients.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


STATUS_ORDER = {"working": 0, "waiting": 1, "blocked": 2}
WAITING_TASK_STATUSES = {"todo", "scheduled"}
WORKING_TASK_STATUSES = {"ready", "queue", "running", "in_progress", "review"}
BLOCKED_TASK_STATUSES = {"blocked"}


@dataclass(frozen=True)
class HealthFinding:
    surface: str
    code: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KanbanTaskSnapshot:
    task_id: str
    title: str
    status: str
    updated_at: float | None = None
    block_reason: str | None = None
    pr_refs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class KanbanSnapshot:
    db_readable: bool
    board_exists: bool
    board_slug: str
    dispatch_owner: str | None
    tasks: list[KanbanTaskSnapshot] = field(default_factory=list)
    db_path: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class MatrixSnapshot:
    project_slug: str
    coordinator_room_id: str | None
    project_room_id: str | None
    gateway_route_registered: bool
    profile_route_registered: bool
    strict_validator_ok: bool
    degraded_fallback: str | None = None
    route_registry_path: str | None = None
    validator_error: str | None = None


@dataclass(frozen=True)
class PullRequestSnapshot:
    number: int
    state: str
    head_branch: str
    linked_task_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GitHubSnapshot:
    repo_path: str
    remote_url: str | None
    remote_reachable: bool
    auth_ok: bool
    branch_name: str
    branch_upstream: str | None
    worktree_clean: bool
    open_prs: list[PullRequestSnapshot] = field(default_factory=list)
    kanban_task_id: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class ProjectHealthSnapshot:
    project_slug: str
    kanban: KanbanSnapshot
    matrix: MatrixSnapshot
    github: GitHubSnapshot


@dataclass(frozen=True)
class ProjectHealthSummary:
    project_slug: str
    status: str
    findings: list[HealthFinding]


class GitCommandClient:
    """Tiny injectable command client for GitHub adapter tests and live use."""

    def run(self, args: Sequence[str], cwd: str | Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603 - args are fixed by adapter callers.
            list(args),
            cwd=cwd,
            check=False,
            text=True,
            capture_output=True,
        )


def _finding(surface: str, code: str, status: str, message: str, **details: Any) -> HealthFinding:
    return HealthFinding(surface=surface, code=code, status=status, message=message, details={k: v for k, v in details.items() if v is not None})


def evaluate_kanban_health(snapshot: KanbanSnapshot, *, now: float | None = None, stale_seconds: int = 3600) -> list[HealthFinding]:
    findings: list[HealthFinding] = []
    now = time.time() if now is None else now

    if not snapshot.db_readable:
        findings.append(_finding("kanban", "kanban.db_unreadable", "blocked", "Kanban DB is not readable", db_path=snapshot.db_path, error=snapshot.error))
    if not snapshot.board_exists:
        findings.append(_finding("kanban", "kanban.board_missing", "blocked", "Kanban project board is missing", board_slug=snapshot.board_slug))
    if not snapshot.dispatch_owner:
        findings.append(_finding("kanban", "kanban.dispatch_owner_missing", "blocked", "Kanban board dispatch-owner metadata is missing", board_slug=snapshot.board_slug))

    waiting = [task.task_id for task in snapshot.tasks if task.status in WAITING_TASK_STATUSES]
    if waiting:
        findings.append(_finding("kanban", "kanban.waiting_tasks", "waiting", "Kanban tasks are waiting on non-human dependencies", task_ids=waiting))

    stale_ready = [
        task.task_id
        for task in snapshot.tasks
        if task.status in {"ready", "queue"} and task.updated_at is not None and now - task.updated_at > stale_seconds
    ]
    if stale_ready:
        findings.append(_finding("kanban", "kanban.stale_ready_tasks", "waiting", "Kanban tasks have been ready/queued longer than the stale threshold", task_ids=stale_ready, stale_seconds=stale_seconds))

    stale_running = [
        task.task_id
        for task in snapshot.tasks
        if task.status in {"running", "in_progress"} and task.updated_at is not None and now - task.updated_at > stale_seconds
    ]
    if stale_running:
        findings.append(_finding("kanban", "kanban.stale_running_tasks", "waiting", "Kanban tasks have been running without fresh updates longer than the stale threshold", task_ids=stale_running, stale_seconds=stale_seconds))

    review_required = [
        task.task_id
        for task in snapshot.tasks
        if task.status in BLOCKED_TASK_STATUSES and (task.block_reason or "").startswith("review-required:")
    ]
    if review_required:
        findings.append(_finding("kanban", "kanban.review_required", "blocked", "Kanban tasks require explicit human review", task_ids=review_required))

    human_blocked = [
        task.task_id
        for task in snapshot.tasks
        if task.status in BLOCKED_TASK_STATUSES and not (task.block_reason or "").startswith("review-required:")
    ]
    if human_blocked:
        findings.append(_finding("kanban", "kanban.human_blocked_tasks", "blocked", "Kanban tasks are blocked on human action", task_ids=human_blocked))

    task_pr_refs = {task.task_id: task.pr_refs for task in snapshot.tasks if task.pr_refs}
    dangling_pr_refs = [task_id for task_id, refs in task_pr_refs.items() if any(not ref for ref in refs)]
    if dangling_pr_refs:
        findings.append(_finding("kanban", "kanban.invalid_pr_refs", "waiting", "Kanban task/PR references contain empty values", task_ids=dangling_pr_refs))

    return findings


def evaluate_matrix_health(snapshot: MatrixSnapshot) -> list[HealthFinding]:
    findings: list[HealthFinding] = []
    if not snapshot.coordinator_room_id:
        findings.append(_finding("matrix", "matrix.coordinator_room_missing", "blocked", "Matrix coordinator room binding is missing", project_slug=snapshot.project_slug))
    if not snapshot.project_room_id:
        findings.append(_finding("matrix", "matrix.project_room_missing", "blocked", "Matrix project room binding is missing", project_slug=snapshot.project_slug))
    if not snapshot.gateway_route_registered:
        findings.append(_finding("matrix", "matrix.gateway_route_missing", "blocked", "Matrix gateway route registry is missing the project route", route_registry_path=snapshot.route_registry_path))
    if not snapshot.profile_route_registered:
        findings.append(_finding("matrix", "matrix.profile_route_missing", "blocked", "Matrix profile-local route registry is missing the project route", route_registry_path=snapshot.route_registry_path))
    if not snapshot.strict_validator_ok:
        findings.append(_finding("matrix", "matrix.strict_validator_failed", "blocked", "Matrix strict validator failed", error=snapshot.validator_error))
    if snapshot.degraded_fallback:
        findings.append(_finding("matrix", "matrix.degraded_fallback_active", "waiting", "Matrix degraded fallback is active", fallback=snapshot.degraded_fallback))
    return findings


def evaluate_github_health(snapshot: GitHubSnapshot) -> list[HealthFinding]:
    findings: list[HealthFinding] = []
    if not snapshot.remote_url:
        findings.append(_finding("github", "github.remote_missing", "blocked", "GitHub remote is missing", repo_path=snapshot.repo_path))
    if not snapshot.remote_reachable:
        findings.append(_finding("github", "github.remote_unreachable", "blocked", "GitHub remote is not reachable", remote_url=snapshot.remote_url, error=snapshot.error))
    if not snapshot.auth_ok:
        findings.append(_finding("github", "github.auth_failed", "blocked", "GitHub authentication is not valid for this repo", remote_url=snapshot.remote_url))
    if not snapshot.branch_upstream:
        findings.append(_finding("github", "github.upstream_missing", "waiting", "Git branch has no upstream tracking branch", branch=snapshot.branch_name))
    if not snapshot.worktree_clean:
        findings.append(_finding("github", "github.worktree_dirty", "waiting", "Git worktree contains uncommitted changes", repo_path=snapshot.repo_path, branch=snapshot.branch_name))

    branch_drift_prs = [pr.number for pr in snapshot.open_prs if pr.head_branch != snapshot.branch_name]
    if branch_drift_prs:
        findings.append(_finding("github", "github.pr_branch_drift", "waiting", "Open PR head branch differs from current watchdog branch", pr_numbers=branch_drift_prs, branch=snapshot.branch_name))

    if snapshot.kanban_task_id:
        task_drift_prs = [pr.number for pr in snapshot.open_prs if snapshot.kanban_task_id not in pr.linked_task_ids]
        if task_drift_prs:
            findings.append(_finding("github", "github.pr_task_drift", "waiting", "Open PRs are not linked to the current Kanban task", pr_numbers=task_drift_prs, task_id=snapshot.kanban_task_id))
    return findings


def evaluate_project_health(snapshot: ProjectHealthSnapshot) -> ProjectHealthSummary:
    findings = [
        *evaluate_kanban_health(snapshot.kanban),
        *evaluate_matrix_health(snapshot.matrix),
        *evaluate_github_health(snapshot.github),
    ]
    status = "working"
    for finding in findings:
        if STATUS_ORDER[finding.status] > STATUS_ORDER[status]:
            status = finding.status
    findings.sort(key=lambda finding: (STATUS_ORDER[finding.status], finding.surface, finding.code))
    return ProjectHealthSummary(project_slug=snapshot.project_slug, status=status, findings=findings)


def load_kanban_snapshot_from_db(db_path: str | Path, board_slug: str) -> KanbanSnapshot:
    """Best-effort live adapter for a project Kanban DB.

    The schema has evolved over time, so this adapter only depends on stable task
    columns and optional board metadata when present.  Tests can inject snapshots
    directly; live callers get explicit blocked findings when DB/board discovery
    fails instead of exceptions.
    """

    path = Path(db_path)
    try:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        return KanbanSnapshot(False, False, board_slug, None, db_path=str(path), error=str(exc))

    try:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        board_exists = True
        dispatch_owner = None
        if "boards" in tables:
            row = conn.execute("SELECT * FROM boards WHERE slug = ? LIMIT 1", (board_slug,)).fetchone()
            board_exists = row is not None
            if row is not None:
                for key in ("dispatch_owner", "owner", "created_by"):
                    if key in row.keys() and row[key]:
                        dispatch_owner = str(row[key])
                        break
                if dispatch_owner is None and "metadata" in row.keys() and row["metadata"]:
                    try:
                        metadata = json.loads(row["metadata"])
                    except json.JSONDecodeError:
                        metadata = {}
                    dispatch_owner = metadata.get("dispatch_owner") or metadata.get("owner")
        elif "tasks" not in tables:
            board_exists = False

        tasks: list[KanbanTaskSnapshot] = []
        if "tasks" in tables:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(tasks)")}
            select_cols = [col for col in ("id", "title", "status", "updated_at", "started_at", "created_at", "block_reason", "result") if col in columns]
            rows = conn.execute(f"SELECT {', '.join(select_cols)} FROM tasks").fetchall() if select_cols else []
            for row in rows:
                keys = row.keys()
                updated_at = row["updated_at"] if "updated_at" in keys else row["started_at"] if "started_at" in keys else row["created_at"] if "created_at" in keys else None
                block_reason = row["block_reason"] if "block_reason" in keys else row["result"] if "result" in keys else None
                pr_refs = _extract_pr_refs("\n".join(str(row[key] or "") for key in keys))
                tasks.append(KanbanTaskSnapshot(str(row["id"]), str(row["title"] if "title" in keys else ""), str(row["status"]), updated_at, block_reason, pr_refs))
        return KanbanSnapshot(True, board_exists, board_slug, dispatch_owner, tasks, db_path=str(path))
    except sqlite3.Error as exc:
        return KanbanSnapshot(False, False, board_slug, None, db_path=str(path), error=str(exc))
    finally:
        conn.close()


def load_matrix_snapshot_from_registry(project_slug: str, registry: Mapping[str, Any], *, strict_validator_ok: bool, validator_error: str | None = None) -> MatrixSnapshot:
    project = registry.get(project_slug, {}) if isinstance(registry, Mapping) else {}
    routes = registry.get("routes", {}) if isinstance(registry, Mapping) else {}
    profile_routes = registry.get("profile_routes", {}) if isinstance(registry, Mapping) else {}
    degraded = registry.get("degraded_fallbacks", {}) if isinstance(registry, Mapping) else {}
    return MatrixSnapshot(
        project_slug=project_slug,
        coordinator_room_id=project.get("coordinator_room_id"),
        project_room_id=project.get("project_room_id"),
        gateway_route_registered=project_slug in routes or bool(project.get("gateway_route_registered")),
        profile_route_registered=project_slug in profile_routes or bool(project.get("profile_route_registered")),
        strict_validator_ok=strict_validator_ok,
        degraded_fallback=degraded.get(project_slug) or project.get("degraded_fallback"),
        route_registry_path=registry.get("path") if isinstance(registry.get("path"), str) else None,
        validator_error=validator_error,
    )


def load_github_snapshot(repo_path: str | Path, *, kanban_task_id: str | None = None, client: GitCommandClient | None = None) -> GitHubSnapshot:
    client = client or GitCommandClient()
    repo = Path(repo_path)

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return client.run(("git", *args), cwd=repo)

    branch = git("branch", "--show-current")
    remote = git("remote", "get-url", "--push", "origin")
    upstream = git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    status = git("status", "--porcelain")
    reachable = git("ls-remote", "--exit-code", "origin", "HEAD")
    auth = client.run(("gh", "auth", "status"), cwd=repo)
    prs = _load_open_prs(repo, client)
    return GitHubSnapshot(
        repo_path=str(repo),
        remote_url=remote.stdout.strip() if remote.returncode == 0 else None,
        remote_reachable=reachable.returncode == 0,
        auth_ok=auth.returncode == 0,
        branch_name=branch.stdout.strip() if branch.returncode == 0 else "",
        branch_upstream=upstream.stdout.strip() if upstream.returncode == 0 else None,
        worktree_clean=status.returncode == 0 and status.stdout.strip() == "",
        open_prs=prs,
        kanban_task_id=kanban_task_id,
        error=(reachable.stderr or remote.stderr or auth.stderr).strip() or None,
    )


def _load_open_prs(repo: Path, client: GitCommandClient) -> list[PullRequestSnapshot]:
    result = client.run(("gh", "pr", "list", "--state", "open", "--json", "number,state,headRefName,body,title"), cwd=repo)
    if result.returncode != 0 or not result.stdout.strip():
        return []
    try:
        raw_prs = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    prs: list[PullRequestSnapshot] = []
    for raw in raw_prs:
        text = f"{raw.get('title', '')}\n{raw.get('body', '')}"
        prs.append(PullRequestSnapshot(int(raw["number"]), str(raw.get("state", "open")).lower(), str(raw.get("headRefName", "")), _extract_task_refs(text)))
    return prs


def _extract_task_refs(text: str) -> list[str]:
    import re

    return sorted(set(re.findall(r"t_[0-9a-f]{8}", text)))


def _extract_pr_refs(text: str) -> list[str]:
    import re

    return sorted(set(re.findall(r"https://github\.com/[^\s)]+/pull/\d+|PR\s*#\d+|pull/\d+", text)))


def _summary_to_json(summary: ProjectHealthSummary) -> str:
    return json.dumps(asdict(summary), indent=2, sort_keys=True)


def _project_health_snapshot_from_dict(data: Mapping[str, Any]) -> ProjectHealthSnapshot:
    kanban_data = dict(data["kanban"])
    kanban_data["tasks"] = [KanbanTaskSnapshot(**task) for task in kanban_data.get("tasks", [])]

    github_data = dict(data["github"])
    github_data["open_prs"] = [PullRequestSnapshot(**pr) for pr in github_data.get("open_prs", [])]

    return ProjectHealthSnapshot(
        project_slug=str(data["project_slug"]),
        kanban=KanbanSnapshot(**kanban_data),
        matrix=MatrixSnapshot(**data["matrix"]),
        github=GitHubSnapshot(**github_data),
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-json", type=Path, help="Evaluate an injected ProjectHealthSnapshot JSON file")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.snapshot_json:
        parser.error("--snapshot-json is required for deterministic CLI use")
    data = json.loads(args.snapshot_json.read_text(encoding="utf-8"))
    snapshot = _project_health_snapshot_from_dict(data)
    print(_summary_to_json(evaluate_project_health(snapshot)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
