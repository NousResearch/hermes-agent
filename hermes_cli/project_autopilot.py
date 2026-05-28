"""Deterministic filesystem-backed Project Autopilot V0.

V0 is intentionally boring:
- file-backed project homes
- kanban board remains execution truth
- project.json owns lifecycle/continuity metadata
- only project_mode=stacked-slices-one-pr is supported
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "project-autopilot/v0"
PROJECT_MODE_V0 = "stacked-slices-one-pr"
SUPPORTED_PROJECT_MODES = {PROJECT_MODE_V0}
PROJECTS_ROOT = Path.home() / "Documents" / "hermes-projects"
ACTIVE_PROJECTS_ROOT = PROJECTS_ROOT / "active"

REQUIRED_FILES = (
    "PROJECT.md",
    "STATUS.md",
    "SESSION-HANDOFF.md",
    "SESSION-LOG.md",
    "PARKING-LOT.md",
    "TASKS.md",
    "project.json",
)
REQUIRED_DIRS = ("status", "refs", "scratch", "artifacts")
TERMINAL_TASK_STATES = {"done", "archived", "superseded", "canceled"}
DEPENDENCY_SATISFYING_STATES = {"done"}

_BRANCH_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-_]{0,63}$")


class ProjectAutopilotError(RuntimeError):
    """Base error for deterministic Project Autopilot failures."""


class InvariantError(ProjectAutopilotError):
    """Raised when a project lifecycle invariant fails."""


def now_ts() -> int:
    return int(time.time())


def branch_to_worktree_path(*, worktree_namespace: Path, branch_name: str) -> Path:
    branch = branch_name.strip().strip("/")
    if not branch or branch.startswith("-") or ".." in branch.split("/"):
        raise ProjectAutopilotError(
            f"invalid branch name for worktree path: {branch_name!r}"
        )
    if not _BRANCH_RE.match(branch):
        raise ProjectAutopilotError(
            f"invalid branch name for worktree path: {branch_name!r}"
        )
    return worktree_namespace / Path(*branch.split("/"))


def _canonical_worktree_namespace(repo_org: str, repo_name: str) -> Path:
    return Path("/Users/vsletten/src") / repo_org / repo_name


def normalize_project_doc(
    *,
    slug: str,
    title: str,
    goal: str,
    board_slug: str,
    root_task_id: str,
    project_home: Path,
    repo_org: str,
    repo_name: str,
    canonical_checkout: Path,
    final_branch: str,
    default_worker: str = "codexapp",
) -> dict[str, Any]:
    if not _SLUG_RE.match(slug):
        raise ProjectAutopilotError(f"invalid project slug: {slug!r}")
    worktree_namespace = _canonical_worktree_namespace(repo_org, repo_name)
    final_worktree_path = branch_to_worktree_path(
        worktree_namespace=worktree_namespace,
        branch_name=final_branch,
    )
    ts = now_ts()
    return {
        "schema_version": SCHEMA_VERSION,
        "slug": slug,
        "title": title,
        "goal": goal,
        "project_type": "coding",
        "project_mode": PROJECT_MODE_V0,
        "state": "BOOTSTRAPPED",
        "board_slug": board_slug,
        "root_task_id": root_task_id,
        "project_home": str(project_home),
        "default_worker": default_worker,
        "repo": {
            "org": repo_org,
            "name": repo_name,
            "canonical_checkout": str(canonical_checkout),
            "worktree_namespace": str(worktree_namespace),
            "remote_name": "origin",
            "remote_url": None,
            "default_branch": "main",
            "pr_base_ref": "main",
            "provider": "github",
        },
        "branch_strategy": {
            "mode": PROJECT_MODE_V0,
            "base_ref": "origin/main",
            "final_branch": final_branch,
        },
        "final_worktree_path": str(final_worktree_path),
        "pr_url": None,
        "pr_requirement": {
            "required": True,
            "waived": False,
            "waived_by": None,
            "waived_at": None,
            "reason": None,
            "approval_evidence": None,
        },
        "execution_policy": {"parallelism": "none", "max_active_workers": 1},
        "cleanup_policy": {
            "mode": "manual-approval-required",
            "keep_final_pr_worktree": True,
            "remove_intermediate_worktrees_after_pr_open": False,
        },
        "cleanup": {
            "state": "not_started",
            "inventory_path": None,
            "approved_by": None,
            "approved_at": None,
            "completed_at": None,
            "targets": [],
        },
        "completion": {
            "done_by": None,
            "done_at": None,
            "done_basis": None,
            "evidence": [],
        },
        "created_at": ts,
        "updated_at": ts,
        "last_verified_at": None,
        "last_transition": None,
        "transition_log": [],
        "invariant_failures": [],
        "workspace_contracts": {},
        "task_graph": {"nodes": [], "edges": []},
    }


def validate_project_doc(doc: dict[str, Any]) -> None:
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ProjectAutopilotError("unsupported schema_version")
    if doc.get("project_mode") not in SUPPORTED_PROJECT_MODES:
        raise ProjectAutopilotError(
            f"unsupported project_mode: {doc.get('project_mode')!r}"
        )
    branch_strategy = doc.get("branch_strategy") or {}
    if branch_strategy.get("mode") not in SUPPORTED_PROJECT_MODES:
        raise ProjectAutopilotError(
            f"unsupported branch_strategy.mode: {branch_strategy.get('mode')!r}"
        )
    if "final_branch" in doc:
        raise ProjectAutopilotError(
            "top-level final_branch is not authoritative in V0"
        )

    required_top = [
        "slug",
        "title",
        "goal",
        "state",
        "board_slug",
        "root_task_id",
        "project_home",
        "repo",
        "branch_strategy",
        "final_worktree_path",
        "pr_requirement",
        "execution_policy",
        "cleanup",
        "transition_log",
    ]
    missing = [key for key in required_top if key not in doc]
    if missing:
        raise ProjectAutopilotError(f"missing project fields: {', '.join(missing)}")

    repo = doc["repo"]
    for key in (
        "org",
        "name",
        "canonical_checkout",
        "worktree_namespace",
        "remote_name",
        "default_branch",
        "pr_base_ref",
        "provider",
    ):
        if not repo.get(key):
            raise ProjectAutopilotError(f"missing repo.{key}")

    expected_policy = {"parallelism": "none", "max_active_workers": 1}
    if doc["execution_policy"] != expected_policy:
        raise ProjectAutopilotError(
            "V0 requires execution_policy parallelism=none max_active_workers=1"
        )


def project_home_for_slug(slug: str) -> Path:
    if not _SLUG_RE.match(slug):
        raise ProjectAutopilotError(f"invalid project slug: {slug!r}")
    return ACTIVE_PROJECTS_ROOT / slug


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_project_doc(project_home: Path) -> dict[str, Any]:
    doc = json.loads((project_home / "project.json").read_text(encoding="utf-8"))
    validate_project_doc(doc)
    return doc


def render_project_md(doc: dict[str, Any]) -> str:
    return f"""# {doc["title"]}

Goal: {doc["goal"]}

Board: `{doc["board_slug"]}`
Root task: `{doc["root_task_id"]}`
Project mode: `{doc["project_mode"]}`
Canonical repo: `{doc["repo"]["canonical_checkout"]}`
Final branch: `{doc["branch_strategy"]["final_branch"]}`
Final worktree: `{doc["final_worktree_path"]}`
"""


def _status_counts(task_graph: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in task_graph.get("nodes", []):
        status = str(node.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _find_next_executable(task_graph: dict[str, Any]) -> dict[str, Any] | None:
    nodes = list(task_graph.get("nodes", []))
    for desired_status in ("running", "ready"):
        for node in nodes:
            if node.get("status") == desired_status:
                return node
    return None


def render_status_md(
    doc: dict[str, Any],
    *,
    next_action: str | None = None,
    blocker: str | None = None,
    task_graph: dict[str, Any] | None = None,
) -> str:
    graph = task_graph or doc.get("task_graph") or {"nodes": [], "edges": []}
    next_task = _find_next_executable(graph)
    if next_action:
        action = next_action
    elif next_task:
        action = (
            f"Next executable task: `{next_task['id']}` {next_task['title']} "
            f"({next_task['status']})."
        )
    else:
        action = "No ready or running task found; inspect the board for blockers."
    blocker_text = blocker or "none"
    counts = _status_counts(graph)
    count_bits = ", ".join(
        f"{status}: {count}" for status, count in sorted(counts.items())
    )
    total_tasks = len(graph.get("nodes", []))
    task_summary = (
        f"Tasks: {total_tasks} total"
        + (f" ({count_bits})" if count_bits else "")
    )
    return f"""# Status: {doc["title"]}

State: {doc["state"]}
Project home: `{doc["project_home"]}`
Board: `{doc["board_slug"]}`
Root task: `{doc["root_task_id"]}`
PR: {doc.get("pr_url") or "not open"}
Blocker: {blocker_text}
{task_summary}

## Next action

{action}
"""


def render_handoff_md(
    doc: dict[str, Any],
    task_graph: dict[str, Any] | None = None,
) -> str:
    graph = task_graph or doc.get("task_graph") or {"nodes": [], "edges": []}
    next_task = _find_next_executable(graph)
    next_line = (
        f"- Next executable task: `{next_task['id']}` {next_task['title']}"
        if next_task
        else "- Next executable task: none"
    )
    task_lines = [
        f"- `{node['id']}` [{node['status']}] {node['title']}"
        for node in graph.get("nodes", [])
    ]
    task_snapshot = "\n".join(task_lines) if task_lines else "- no tasks cached"
    return f"""# Session Handoff: {doc["title"]}

State: {doc["state"]}

Restart checklist:
1. Verify this project home exists and `project.json` validates.
2. Inspect board `{doc["board_slug"]}` and root task `{doc["root_task_id"]}`.
3. Reconcile `TASKS.md` and `STATUS.md` from board truth before dispatching work.

## Task graph snapshot

{next_line}

{task_snapshot}
"""


def _task_to_node(task: Any, parents: list[str], children: list[str]) -> dict[str, Any]:
    return {
        "id": task.id,
        "title": task.title,
        "body": task.body,
        "assignee": task.assignee,
        "status": task.status,
        "priority": task.priority,
        "created_at": task.created_at,
        "started_at": task.started_at,
        "completed_at": task.completed_at,
        "workspace_kind": task.workspace_kind,
        "workspace_path": task.workspace_path,
        "branch_name": task.branch_name,
        "parents": parents,
        "children": children,
    }


def derive_task_graph(conn: Any, root_task_id: str) -> dict[str, Any]:
    from hermes_cli import kanban_db

    root = kanban_db.get_task(conn, root_task_id)
    if root is None:
        raise InvariantError(f"root task not found on kanban board: {root_task_id}")

    ordered_ids: list[str] = []
    seen: set[str] = set()
    queue = [root_task_id]
    while queue:
        task_id = queue.pop(0)
        if task_id in seen:
            continue
        seen.add(task_id)
        ordered_ids.append(task_id)
        queue.extend(
            child_id
            for child_id in kanban_db.child_ids(conn, task_id)
            if child_id not in seen
        )

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    workspace_contracts: dict[str, dict[str, str | None]] = {}
    for task_id in ordered_ids:
        task = kanban_db.get_task(conn, task_id)
        if task is None:
            continue
        parents = [pid for pid in kanban_db.parent_ids(conn, task_id) if pid in seen]
        children = [cid for cid in kanban_db.child_ids(conn, task_id) if cid in seen]
        nodes.append(_task_to_node(task, parents, children))
        edges.extend({"parent": pid, "child": task_id} for pid in parents)
        if task.workspace_kind == "worktree":
            workspace_contracts[task_id] = {
                "workspace_kind": task.workspace_kind,
                "workspace_path": task.workspace_path,
                "branch_name": task.branch_name,
            }

    return {
        "nodes": nodes,
        "edges": edges,
        "workspace_contracts": workspace_contracts,
    }


def render_tasks_md(doc: dict[str, Any], task_graph: dict[str, Any]) -> str:
    lines = [
        f"# Tasks: {doc['title']}",
        "",
        f"Root task: `{doc['root_task_id']}`",
        "",
        "| Task | Status | Assignee | Workspace |",
        "| --- | --- | --- | --- |",
    ]
    for node in task_graph.get("nodes", []):
        workspace = node.get("workspace_path") or node.get("workspace_kind") or ""
        lines.append(
            "| "
            f"`{node['id']}` {node['title']} | "
            f"{node['status']} | "
            f"{node.get('assignee') or ''} | "
            f"{workspace} |"
        )
    if task_graph.get("edges"):
        lines.extend(["", "## Dependencies", ""])
        for edge in task_graph["edges"]:
            lines.append(f"- `{edge['parent']}` blocks `{edge['child']}`")
    return "\n".join(lines) + "\n"


def sync_project_home(
    project_home: Path,
    *,
    db_path: Path | None = None,
    board: str | None = None,
) -> dict[str, Any]:
    from hermes_cli import kanban_db

    doc = load_project_doc(project_home)
    conn = kanban_db.connect(db_path, board=board or doc["board_slug"])
    try:
        graph_with_contracts = derive_task_graph(conn, doc["root_task_id"])
    finally:
        conn.close()

    task_graph = {
        "nodes": graph_with_contracts["nodes"],
        "edges": graph_with_contracts["edges"],
    }
    doc["task_graph"] = task_graph
    doc["workspace_contracts"] = graph_with_contracts["workspace_contracts"]
    doc["updated_at"] = now_ts()
    validate_project_doc(doc)

    (project_home / "TASKS.md").write_text(
        render_tasks_md(doc, task_graph),
        encoding="utf-8",
    )
    (project_home / "STATUS.md").write_text(
        render_status_md(doc, task_graph=task_graph),
        encoding="utf-8",
    )
    (project_home / "SESSION-HANDOFF.md").write_text(
        render_handoff_md(doc, task_graph),
        encoding="utf-8",
    )
    write_json(project_home / "project.json", doc)
    verify_project_home(project_home)
    return doc


def bootstrap_project_home(
    *,
    slug: str,
    title: str,
    goal: str,
    board_slug: str,
    root_task_id: str,
    project_home: Path | None,
    repo_org: str,
    repo_name: str,
    canonical_checkout: Path,
    final_branch: str,
    source_plan: Path | None = None,
) -> dict[str, Any]:
    project_home = project_home or project_home_for_slug(slug)
    project_home.mkdir(parents=True, exist_ok=True)
    for dirname in REQUIRED_DIRS:
        (project_home / dirname).mkdir(parents=True, exist_ok=True)

    doc = normalize_project_doc(
        slug=slug,
        title=title,
        goal=goal,
        board_slug=board_slug,
        root_task_id=root_task_id,
        project_home=project_home,
        repo_org=repo_org,
        repo_name=repo_name,
        canonical_checkout=canonical_checkout,
        final_branch=final_branch,
    )
    validate_project_doc(doc)

    (project_home / "PROJECT.md").write_text(
        render_project_md(doc),
        encoding="utf-8",
    )
    (project_home / "STATUS.md").write_text(
        render_status_md(doc),
        encoding="utf-8",
    )
    (project_home / "SESSION-HANDOFF.md").write_text(
        render_handoff_md(doc),
        encoding="utf-8",
    )
    (project_home / "SESSION-LOG.md").write_text(
        f"# Session Log: {title}\n\n- bootstrapped project home\n",
        encoding="utf-8",
    )
    (project_home / "PARKING-LOT.md").write_text(
        f"# Parking Lot: {title}\n\n",
        encoding="utf-8",
    )
    (project_home / "TASKS.md").write_text(
        f"# Tasks: {title}\n\nRoot task: `{root_task_id}`\n",
        encoding="utf-8",
    )
    if source_plan:
        target = project_home / "refs" / source_plan.name
        target.write_text(source_plan.read_text(encoding="utf-8"), encoding="utf-8")
    write_json(project_home / "project.json", doc)
    verify_project_home(project_home)
    return doc


def verify_project_home(project_home: Path) -> dict[str, Any]:
    missing = [rel for rel in REQUIRED_FILES if not (project_home / rel).exists()]
    missing_dirs = [rel for rel in REQUIRED_DIRS if not (project_home / rel).is_dir()]
    if missing or missing_dirs:
        raise InvariantError(
            f"missing project artifacts: files={missing} dirs={missing_dirs}"
        )
    doc = load_project_doc(project_home)
    status = (project_home / "STATUS.md").read_text(encoding="utf-8")
    if status.count("## Next action") != 1:
        raise InvariantError("STATUS.md must contain exactly one ## Next action")
    if str(project_home) != doc["project_home"]:
        raise InvariantError("project.json project_home does not match actual path")
    return doc
