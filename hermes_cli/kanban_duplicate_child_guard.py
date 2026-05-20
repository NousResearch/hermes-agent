"""Read-only duplicate child detector for Hermes Kanban task graphs.

The command-line entry point is ``kanban-duplicate-child-guard``.  It reads a
Kanban board, scans a root task's descendant graph, groups review/audit child
cards by scope markers, and emits a JSON receipt plus a dry-run convergence
plan.  It deliberately does not mutate the board: no task updates, comments,
links, or status transitions are performed from this module.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from hermes_cli import kanban_db as kb

SCHEMA_VERSION = "kanban-duplicate-child-guard:receipt:v1"
ENTRYPOINT = "kanban-duplicate-child-guard"
REVIEW_CHILD_RE = re.compile(
    r"(审计|审核|复审|终审|(?<![A-Za-z0-9_])(?:re-review|final[-_ ]?review|review|audit|auditor|closeout)(?![A-Za-z0-9_]))",
    re.IGNORECASE,
)
ISSUE_URL_RE = re.compile(r"github\.com/[^\s)]+/issues/(\d+)", re.IGNORECASE)
PULL_URL_RE = re.compile(r"github\.com/[^\s)]+/pull/(\d+)", re.IGNORECASE)
ISSUE_FIELD_RE = re.compile(r"(?:issue|issue_number|github issue)\s*[:=#\s]+#?(\d+)", re.IGNORECASE)
PR_FIELD_RE = re.compile(r"(?:pr|pull_request|pull request)\s*[:=#\s]+#?(\d+)", re.IGNORECASE)
HASH_ISSUE_RE = re.compile(r"(?<![\w/])#(\d{1,6})(?![\w-])")
HEAD_RE = re.compile(r"(?:head_sha|head|artifact_head_sha|commit_id)\s*[:=]\s*([A-Za-z0-9._/-]{6,80})", re.IGNORECASE)
REVIEW_SCOPE_RE = re.compile(r"(?:review_scope_key|review_scope)\s*[:=]\s*([A-Za-z0-9_.-]+)", re.IGNORECASE)
AUDIT_SCOPE_RE = re.compile(r"(?:audit_scope|audit_scope_key)\s*[:=]\s*([A-Za-z0-9_.-]+)", re.IGNORECASE)
PHASE_RE = re.compile(r"(?:phase|scope|stage|current_step_key)\s*[:=]\s*([A-Za-z0-9_.-]+)", re.IGNORECASE)
HASH_PHASE_RE = re.compile(r"#(\d+-[A-Za-z][A-Za-z0-9_.-]*)")
TERMINAL_STATUSES = {"done", "archived"}
UNRUN_STATUSES = {"triage", "todo", "scheduled", "ready"}


@dataclass(frozen=True)
class TaskNode:
    rowid: int
    id: str
    title: str
    body: str
    assignee: str
    status: str
    priority: int
    created_by: str
    created_at: int
    started_at: Optional[int]
    completed_at: Optional[int]
    workflow_template_id: str = ""
    current_step_key: str = ""


@dataclass
class TaskEvidence:
    comments: list[str] = field(default_factory=list)
    run_metadata: list[dict[str, Any]] = field(default_factory=list)
    run_count: int = 0
    completed_run_count: int = 0


@dataclass(frozen=True)
class GroupKey:
    issues: tuple[str, ...]
    prs: tuple[str, ...]
    head_shas: tuple[str, ...]
    assignee: str
    review_scope_key: str
    audit_scope: str
    phase_scope: str
    workflow_template_id: str
    current_step_key: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "issues": list(self.issues),
            "prs": list(self.prs),
            "head_shas": list(self.head_shas),
            "assignee": self.assignee,
            "review_scope_key": self.review_scope_key,
            "audit_scope": self.audit_scope,
            "phase_scope": self.phase_scope,
            "workflow_template_id": self.workflow_template_id,
            "current_step_key": self.current_step_key,
        }


def _normalize_issue(value: Optional[str | int]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    url_match = ISSUE_URL_RE.search(s)
    if url_match:
        return f"#{int(url_match.group(1))}"
    field_match = ISSUE_FIELD_RE.search(s)
    if field_match:
        return f"#{int(field_match.group(1))}"
    if s.startswith("#"):
        s = s[1:]
    if not s.isdigit():
        return str(value).strip()
    return f"#{int(s)}"


def _normalize_number_set(values: Iterable[str | int], prefix: str) -> tuple[str, ...]:
    out: set[str] = set()
    for value in values:
        if value is None:
            continue
        s = str(value).strip()
        if not s:
            continue
        if prefix and s.startswith(prefix):
            s = s[len(prefix):]
        if s.isdigit():
            out.add(f"{prefix}{int(s)}")
        else:
            out.add(s)
    return tuple(sorted(out))


def _first_match(regex: re.Pattern[str], text: str) -> str:
    match = regex.search(text)
    return match.group(1).strip() if match else ""


def _stringify_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def _task_search_text(task: TaskNode, evidence: Optional[TaskEvidence] = None) -> str:
    parts = [task.title, task.body, task.assignee, task.workflow_template_id, task.current_step_key]
    if evidence:
        parts.extend(evidence.comments)
        parts.extend(_stringify_json(meta) for meta in evidence.run_metadata)
    return "\n".join(p for p in parts if p)


def _review_child_search_text(task: TaskNode, evidence: Optional[TaskEvidence] = None) -> str:
    # Deliberately exclude assignee: a profile named "reviewer" is not enough to
    # prove that an arbitrary child task is an audit/review child.
    parts = [task.title, task.body, task.workflow_template_id, task.current_step_key]
    if evidence:
        parts.extend(evidence.comments)
        parts.extend(_stringify_json(meta) for meta in evidence.run_metadata)
    return "\n".join(p for p in parts if p)


def _extract_markers(task: TaskNode, evidence: TaskEvidence) -> dict[str, Any]:
    text = _task_search_text(task, evidence)
    metadata_text = "\n".join(_stringify_json(meta) for meta in evidence.run_metadata)
    issues = set(_normalize_issue(x) for x in ISSUE_URL_RE.findall(text))
    issues.update(_normalize_issue(x) for x in ISSUE_FIELD_RE.findall(text))
    # Title-style references such as "审计｜#155" are useful for grouping, but
    # avoid parsing pull URL numbers as issues by running URL/field regexes first
    # and then accepting generic #N markers only from the title/body text.
    title_body_text = "\n".join(p for p in (task.title, task.body) if p)
    issues.update(_normalize_issue(x) for x in HASH_ISSUE_RE.findall(title_body_text))
    issues.discard(None)

    prs = set(f"PR#{int(x)}" for x in PULL_URL_RE.findall(text))
    prs.update(f"PR#{int(x)}" for x in PR_FIELD_RE.findall(text))

    heads = {h for h in HEAD_RE.findall(text) if h}
    review_scope = _first_match(REVIEW_SCOPE_RE, text)
    audit_scope = _first_match(AUDIT_SCOPE_RE, text)
    phase = _first_match(PHASE_RE, text)
    if not phase:
        phase = _first_match(HASH_PHASE_RE, title_body_text)
    phase_scope = review_scope or audit_scope or phase or task.current_step_key

    # Metadata keys win when they are present as structured data, because issue
    # comments often contain several historical trace payloads.
    for meta in evidence.run_metadata:
        if not isinstance(meta, dict):
            continue
        for key in ("issue", "issue_number", "issue_url"):
            value = meta.get(key)
            if isinstance(value, str):
                issues.update(_normalize_issue(x) for x in ISSUE_URL_RE.findall(value))
                norm = _normalize_issue(value)
                if norm:
                    issues.add(norm)
            elif value is not None:
                norm = _normalize_issue(value)
                if norm:
                    issues.add(norm)
        pr_value = meta.get("pr_number") or meta.get("pr")
        if pr_value is not None:
            pr_norm = str(pr_value).strip().lstrip("#")
            if pr_norm.isdigit():
                prs.add(f"PR#{int(pr_norm)}")
        pr_state = meta.get("pr_state")
        if isinstance(pr_state, dict):
            pr_number = pr_state.get("number")
            if pr_number is not None and str(pr_number).isdigit():
                prs.add(f"PR#{int(pr_number)}")
        for key in ("head_sha", "artifact_head_sha", "commit_id", "merge_commit"):
            value = meta.get(key)
            if value:
                heads.add(str(value))
        review_scope = str(meta.get("review_scope_key") or review_scope or "")
        audit_scope = str(meta.get("audit_scope") or audit_scope or "")
        phase_scope = str(meta.get("phase_scope") or meta.get("current_step_key") or phase_scope or "")

    return {
        "issues": tuple(sorted(i for i in issues if i)),
        "prs": tuple(sorted(prs)),
        "head_shas": tuple(sorted(heads)),
        "review_scope_key": review_scope,
        "audit_scope": audit_scope,
        "phase_scope": phase_scope,
        "raw_marker_text_bytes": len(text.encode("utf-8")),
        "metadata_text_bytes": len(metadata_text.encode("utf-8")),
    }


def _group_key(task: TaskNode, evidence: TaskEvidence, markers: Optional[dict[str, Any]] = None) -> GroupKey:
    markers = markers if markers is not None else _extract_markers(task, evidence)
    return GroupKey(
        issues=markers["issues"],
        prs=markers["prs"],
        head_shas=markers["head_shas"],
        assignee=task.assignee or "",
        review_scope_key=markers["review_scope_key"],
        audit_scope=markers["audit_scope"] or "",
        phase_scope=markers["phase_scope"],
        workflow_template_id=task.workflow_template_id or "",
        current_step_key=task.current_step_key or "",
    )


def _has_duplicate_group_markers(markers: dict[str, Any]) -> bool:
    has_subject_marker = bool(markers["issues"] or markers["prs"] or markers["head_shas"])
    has_scope_marker = bool(markers["review_scope_key"] or markers["audit_scope"] or markers["phase_scope"])
    return has_subject_marker and has_scope_marker


def _missing_duplicate_group_markers(markers: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    if not (markers["issues"] or markers["prs"] or markers["head_shas"]):
        missing.append("issue_or_pr_or_head")
    if not (markers["review_scope_key"] or markers["audit_scope"] or markers["phase_scope"]):
        missing.append("review_scope_or_audit_scope_or_phase")
    return missing


def _is_review_child(task: TaskNode, evidence: TaskEvidence) -> bool:
    text = _review_child_search_text(task, evidence)
    return bool(REVIEW_CHILD_RE.search(text))


def _row_to_task(row: sqlite3.Row) -> TaskNode:
    keys = set(row.keys())
    return TaskNode(
        rowid=int(row["_rowid"] if "_rowid" in keys else 0),
        id=str(row["id"]),
        title=str(row["title"] or ""),
        body=str(row["body"] or ""),
        assignee=str(row["assignee"] or ""),
        status=str(row["status"] or ""),
        priority=int(row["priority"] or 0),
        created_by=str(row["created_by"] or ""),
        created_at=int(row["created_at"] or 0),
        started_at=(int(row["started_at"]) if row["started_at"] is not None else None),
        completed_at=(int(row["completed_at"]) if row["completed_at"] is not None else None),
        workflow_template_id=str(row["workflow_template_id"] or "") if "workflow_template_id" in keys else "",
        current_step_key=str(row["current_step_key"] or "") if "current_step_key" in keys else "",
    )


def _fetch_tasks(conn: sqlite3.Connection) -> dict[str, TaskNode]:
    rows = conn.execute("SELECT rowid AS _rowid, * FROM tasks ORDER BY created_at, rowid").fetchall()
    return {str(row["id"]): _row_to_task(row) for row in rows}


def _fetch_links(conn: sqlite3.Connection) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    children_by_parent: dict[str, list[str]] = defaultdict(list)
    parents_by_child: dict[str, list[str]] = defaultdict(list)
    rows = conn.execute("SELECT parent_id, child_id FROM task_links ORDER BY parent_id, child_id").fetchall()
    for row in rows:
        parent = str(row["parent_id"])
        child = str(row["child_id"])
        children_by_parent[parent].append(child)
        parents_by_child[child].append(parent)
    return dict(children_by_parent), dict(parents_by_child)


def _parse_run_metadata(raw: Any) -> Optional[dict[str, Any]]:
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return {"_unparsed_metadata": str(raw)}
    return data if isinstance(data, dict) else {"_metadata": data}


def _fetch_evidence(conn: sqlite3.Connection, task_ids: Iterable[str]) -> dict[str, TaskEvidence]:
    ids = list(task_ids)
    evidence = {task_id: TaskEvidence() for task_id in ids}
    if not ids:
        return evidence
    placeholders = ",".join("?" for _ in ids)
    for row in conn.execute(
        f"SELECT task_id, body FROM task_comments WHERE task_id IN ({placeholders}) ORDER BY created_at, id",
        ids,
    ).fetchall():
        evidence[str(row["task_id"])].comments.append(str(row["body"] or ""))
    for row in conn.execute(
        f"SELECT task_id, status, outcome, metadata FROM task_runs WHERE task_id IN ({placeholders}) ORDER BY started_at, id",
        ids,
    ).fetchall():
        task_id = str(row["task_id"])
        evidence[task_id].run_count += 1
        if row["outcome"] == "completed" or row["status"] == "done":
            evidence[task_id].completed_run_count += 1
        parsed = _parse_run_metadata(row["metadata"])
        if parsed is not None:
            evidence[task_id].run_metadata.append(parsed)
    return evidence


def _descendants(root_task_ids: Sequence[str], children_by_parent: dict[str, list[str]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    queue: deque[str] = deque(root_task_ids)
    while queue:
        task_id = queue.popleft()
        if task_id in seen:
            continue
        seen.add(task_id)
        ordered.append(task_id)
        queue.extend(children_by_parent.get(task_id, []))
    return ordered


def _task_matches_filters(
    task: TaskNode,
    evidence: TaskEvidence,
    *,
    issue: Optional[str] = None,
    pr: Optional[str] = None,
    head: Optional[str] = None,
    review_scope_key: Optional[str] = None,
    audit_scope: Optional[str] = None,
) -> bool:
    markers = _extract_markers(task, evidence)
    if issue and _normalize_issue(issue) not in markers["issues"]:
        return False
    if pr:
        pr_s = str(pr).strip().lstrip("#")
        pr_norm = f"PR#{int(pr_s)}" if pr_s.isdigit() else pr_s
        if pr_norm not in markers["prs"]:
            return False
    if head and head not in markers["head_shas"]:
        return False
    if review_scope_key and review_scope_key != markers["review_scope_key"]:
        return False
    if audit_scope and audit_scope != markers["audit_scope"]:
        return False
    return True


def _nearest_scan_roots_for_match(
    task_id: str,
    tasks: dict[str, TaskNode],
    parents_by_child: dict[str, list[str]],
    evidence: dict[str, TaskEvidence],
) -> list[str]:
    task = tasks[task_id]
    task_evidence = evidence.get(task_id, TaskEvidence())
    if not _is_review_child(task, task_evidence):
        return [task_id]

    roots: list[str] = []
    queue: deque[str] = deque(parents_by_child.get(task_id, []))
    seen: set[str] = set()
    while queue:
        parent_id = queue.popleft()
        if parent_id in seen or parent_id not in tasks:
            continue
        seen.add(parent_id)
        parent = tasks[parent_id]
        parent_evidence = evidence.get(parent_id, TaskEvidence())
        if not _is_review_child(parent, parent_evidence):
            roots.append(parent_id)
            continue
        queue.extend(parents_by_child.get(parent_id, []))
    return roots or [task_id]


def _has_selected_ancestor(
    task_id: str,
    selected: set[str],
    parents_by_child: dict[str, list[str]],
) -> bool:
    queue: deque[str] = deque(parents_by_child.get(task_id, []))
    seen: set[str] = set()
    while queue:
        parent_id = queue.popleft()
        if parent_id in seen:
            continue
        if parent_id in selected:
            return True
        seen.add(parent_id)
        queue.extend(parents_by_child.get(parent_id, []))
    return False


def _resolve_root_task_ids(
    tasks: dict[str, TaskNode],
    parents_by_child: dict[str, list[str]],
    evidence: dict[str, TaskEvidence],
    *,
    root_task_ids: Optional[Sequence[str]] = None,
    issue: Optional[str] = None,
    pr: Optional[str] = None,
    head: Optional[str] = None,
    review_scope_key: Optional[str] = None,
    audit_scope: Optional[str] = None,
) -> list[str]:
    if root_task_ids:
        missing = [task_id for task_id in root_task_ids if task_id not in tasks]
        if missing:
            raise ValueError(f"unknown root task(s): {', '.join(missing)}")
        return list(dict.fromkeys(root_task_ids))

    if not any([issue, pr, head, review_scope_key, audit_scope]):
        raise ValueError("provide at least one --root-task, --issue, --pr, --head, --review-scope-key, or --audit-scope")

    matching = [
        task_id
        for task_id, task in tasks.items()
        if _task_matches_filters(
            task,
            evidence.get(task_id, TaskEvidence()),
            issue=issue,
            pr=pr,
            head=head,
            review_scope_key=review_scope_key,
            audit_scope=audit_scope,
        )
    ]
    if not matching:
        raise ValueError("no tasks match the provided issue/PR/head/scope filters")

    candidate_roots: list[str] = []
    for task_id in matching:
        candidate_roots.extend(_nearest_scan_roots_for_match(task_id, tasks, parents_by_child, evidence))

    ordered_candidates = sorted(
        dict.fromkeys(candidate_roots),
        key=lambda tid: (tasks[tid].created_at, tasks[tid].rowid, tid),
    )
    selected = set(ordered_candidates)
    roots = [task_id for task_id in ordered_candidates if not _has_selected_ancestor(task_id, selected, parents_by_child)]
    return roots or ordered_candidates


def _task_member_dict(task: TaskNode, evidence: TaskEvidence, parents_by_child: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "task_id": task.id,
        "assignee": task.assignee,
        "status": task.status,
        "created_at": task.created_at,
        "completed_at": task.completed_at,
        "parents": parents_by_child.get(task.id, []),
        "run_count": evidence.run_count,
        "completed_run_count": evidence.completed_run_count,
        "is_unrun": _is_unrun(task, evidence),
        "is_terminal": _is_terminal(task),
    }


def _is_terminal(task: TaskNode) -> bool:
    return task.status in TERMINAL_STATUSES or task.completed_at is not None


def _is_unrun(task: TaskNode, evidence: TaskEvidence) -> bool:
    return task.status in UNRUN_STATUSES and evidence.run_count == 0


def _group_state(members: Sequence[TaskNode], evidence: dict[str, TaskEvidence]) -> str:
    if all(_is_terminal(task) for task in members):
        return "historical_duplicate"
    if all(_is_unrun(task, evidence.get(task.id, TaskEvidence())) for task in members):
        return "unrun_duplicate"
    if any(task.status == "running" for task in members):
        return "active_duplicate_manual_review"
    return "mixed_duplicate_manual_review"


def _canonical_member(members: Sequence[TaskNode]) -> TaskNode:
    return sorted(members, key=lambda task: (task.created_at, task.rowid, -task.priority, task.id))[0]


def _build_actions(duplicate_groups: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for group in duplicate_groups:
        state = group["group_state"]
        members = group["members"]
        if state == "historical_duplicate":
            actions.append(
                {
                    "action": "record_only_historical_duplicate",
                    "target_task_ids": [member["task_id"] for member in members],
                    "reason": "all duplicate children are terminal; detector records history only and does not back-edit old work",
                    "destructive": False,
                    "mutation_performed": False,
                }
            )
            continue
        if state == "unrun_duplicate":
            canonical = members[0]
            actions.append(
                {
                    "action": "keep_canonical_child",
                    "target_task_id": canonical["task_id"],
                    "reason": "oldest matching unrun child is the canonical survivor for a human-controlled convergence plan",
                    "destructive": False,
                    "mutation_performed": False,
                }
            )
            for redundant in members[1:]:
                actions.append(
                    {
                        "action": "safe_converge_unrun_duplicate",
                        "target_task_id": redundant["task_id"],
                        "canonical_task_id": canonical["task_id"],
                        "reason": "dry-run only: candidate duplicate has not run; a later controlled apply may block/archive/link after human approval",
                        "destructive": False,
                        "mutation_performed": False,
                    }
                )
            continue
        actions.append(
            {
                "action": "manual_review_required",
                "target_task_ids": [member["task_id"] for member in members],
                "reason": f"duplicate group is {state}; detector refuses automated convergence",
                "destructive": False,
                "mutation_performed": False,
            }
        )
    return actions


def _runtime_profile_smoke(actor_profile: Optional[str]) -> dict[str, Any]:
    actual_profile = os.environ.get("HERMES_PROFILE") or os.environ.get("HERMES_AGENT_PROFILE") or ""
    return {
        "entrypoint": ENTRYPOINT,
        "python_executable": sys.executable,
        "cwd": os.getcwd(),
        "actual_profile": actual_profile,
        "expected_actor_profile": actor_profile or "",
        "actor_profile_match": (not actor_profile) or actual_profile == actor_profile,
        "current_kanban_task_id": os.environ.get("HERMES_KANBAN_TASK", ""),
        "current_kanban_board": os.environ.get("HERMES_KANBAN_BOARD", ""),
        "current_kanban_db_set": bool(os.environ.get("HERMES_KANBAN_DB")),
    }


def build_duplicate_child_receipt(
    conn: sqlite3.Connection,
    *,
    root_task_ids: Optional[Sequence[str]] = None,
    issue: Optional[str] = None,
    pr: Optional[str] = None,
    head: Optional[str] = None,
    review_scope_key: Optional[str] = None,
    audit_scope: Optional[str] = None,
    actor_profile: Optional[str] = None,
    board: Optional[str] = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Build the duplicate-child guard receipt without mutating ``conn``."""
    tasks = _fetch_tasks(conn)
    children_by_parent, parents_by_child = _fetch_links(conn)
    all_evidence = _fetch_evidence(conn, tasks.keys())
    roots = _resolve_root_task_ids(
        tasks,
        parents_by_child,
        all_evidence,
        root_task_ids=root_task_ids,
        issue=issue,
        pr=pr,
        head=head,
        review_scope_key=review_scope_key,
        audit_scope=audit_scope,
    )
    graph_ids = _descendants(roots, children_by_parent)
    graph_tasks = [tasks[task_id] for task_id in graph_ids if task_id in tasks]
    graph_evidence = {task.id: all_evidence.get(task.id, TaskEvidence()) for task in graph_tasks}

    groups: dict[GroupKey, list[TaskNode]] = defaultdict(list)
    non_review_children: list[TaskNode] = []
    insufficiently_marked_review_children: list[dict[str, Any]] = []
    review_children_scanned = 0
    for task in graph_tasks:
        if task.id in roots:
            continue
        evidence = graph_evidence.get(task.id, TaskEvidence())
        if not _is_review_child(task, evidence):
            non_review_children.append(task)
            continue
        review_children_scanned += 1
        markers = _extract_markers(task, evidence)
        if not _has_duplicate_group_markers(markers):
            insufficiently_marked_review_children.append(
                {
                    "task_id": task.id,
                    "status": task.status,
                    "assignee": task.assignee,
                    "missing_markers": _missing_duplicate_group_markers(markers),
                }
            )
            continue
        groups[_group_key(task, evidence, markers)].append(task)

    insufficiently_marked_review_children.sort(
        key=lambda item: (
            tasks[item["task_id"]].created_at,
            tasks[item["task_id"]].rowid,
            item["task_id"],
        )
    )

    duplicate_groups: list[dict[str, Any]] = []
    non_duplicate_review_children: list[dict[str, Any]] = []
    for key, members in sorted(groups.items(), key=lambda item: (item[0].as_dict()["issues"], item[0].assignee, item[0].review_scope_key, item[0].phase_scope)):
        ordered_members = sorted(members, key=lambda task: (task.created_at, task.rowid, task.id))
        if len(ordered_members) <= 1:
            task = ordered_members[0]
            non_duplicate_review_children.append(
                {
                    "task_id": task.id,
                    "status": task.status,
                    "group_key": key.as_dict(),
                }
            )
            continue
        state = _group_state(ordered_members, graph_evidence)
        canonical = _canonical_member(ordered_members)
        duplicate_groups.append(
            {
                "group_key": key.as_dict(),
                "group_state": state,
                "canonical_task_id": canonical.id,
                "members": [
                    _task_member_dict(task, graph_evidence.get(task.id, TaskEvidence()), parents_by_child)
                    for task in ordered_members
                ],
            }
        )

    actions = _build_actions(duplicate_groups)
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "schema": SCHEMA_VERSION,
        "mode": "dry_run" if dry_run else "detect",
        "ok": True,
        "generated_at": generated_at,
        "input": {
            "root_task_ids": list(root_task_ids or []),
            "issue": _normalize_issue(issue) if issue else None,
            "pr": pr,
            "head": head,
            "review_scope_key": review_scope_key,
            "audit_scope": audit_scope,
        },
        "board": {
            "slug": board or os.environ.get("HERMES_KANBAN_BOARD") or "",
            "read_only": True,
        },
        "graph": {
            "root_task_ids": roots,
            "task_ids": graph_ids,
            "tasks_scanned": len(graph_tasks),
            "edges_scanned": sum(len(children_by_parent.get(task_id, [])) for task_id in graph_ids),
            "review_children_scanned": review_children_scanned,
            "non_review_children_scanned": len(non_review_children),
            "insufficiently_marked_review_children_scanned": len(insufficiently_marked_review_children),
        },
        "detector": {
            "grouping_fields": [
                "issues",
                "prs",
                "head_shas",
                "assignee",
                "review_scope_key",
                "audit_scope",
                "phase_scope",
                "workflow_template_id",
                "current_step_key",
            ],
            "duplicate_groups": duplicate_groups,
            "non_duplicate_review_children": non_duplicate_review_children,
            "insufficiently_marked_review_children": insufficiently_marked_review_children,
            "non_review_children": [
                {"task_id": task.id, "status": task.status}
                for task in non_review_children
            ],
        },
        "dry_run_plan": {
            "apply_enabled": False,
            "apply_supported": False,
            "summary": f"{len(duplicate_groups)} duplicate group(s), {len(actions)} dry-run action(s); no mutations performed",
            "actions": actions,
        },
        "receipt_contract": {
            "consumer_hint": "#159-B and #157 may consume detector.duplicate_groups plus dry_run_plan.actions; do not infer live mutation from this receipt",
            "idempotency_key_fields": ["schema", "graph.root_task_ids", "detector.duplicate_groups[].group_key"],
            "title_policy": "task titles are intentionally omitted from public receipts because they are user-controlled text",
        },
        "runtime_profile_smoke": _runtime_profile_smoke(actor_profile),
        "public_safety": {
            "no_mutations_performed": True,
            "write_targets": [],
            "destructive_apply_default_enabled": False,
            "raw_platform_locator_included": False,
            "user_controlled_titles_included": False,
            "credentials_or_tokens_touched": False,
        },
    }


def _open_readonly_connection(db_path: Optional[Path] = None, *, board: Optional[str] = None) -> sqlite3.Connection:
    path = db_path if db_path is not None else kb.kanban_db_path(board=board)
    if not path.exists():
        raise FileNotFoundError(f"kanban DB does not exist: {path}")
    uri = f"file:{path.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _write_or_print_receipt(receipt: dict[str, Any], *, receipt_file: Optional[str], json_output: bool) -> None:
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True)
    if receipt_file:
        Path(receipt_file).write_text(encoded + "\n", encoding="utf-8")
    if json_output:
        print(encoded)
    else:
        duplicates = len(receipt.get("detector", {}).get("duplicate_groups", []))
        actions = len(receipt.get("dry_run_plan", {}).get("actions", []))
        print(f"{ENTRYPOINT}: {duplicates} duplicate group(s), {actions} dry-run action(s); no mutations performed")
        if receipt_file:
            print(f"receipt: {receipt_file}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=ENTRYPOINT,
        description="Read-only detector for duplicate Kanban audit/review child tasks.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    detect = sub.add_parser("detect", help="scan a task graph and emit a JSON receipt")
    detect.add_argument("--root-task", action="append", default=[], help="root task id to scan; repeatable")
    detect.add_argument("--issue", help="GitHub issue filter, e.g. #155 or 155")
    detect.add_argument("--pr", help="GitHub PR number filter")
    detect.add_argument("--head", help="PR/head SHA filter")
    detect.add_argument("--review-scope-key", help="review_scope_key filter")
    detect.add_argument("--audit-scope", help="audit_scope filter")
    detect.add_argument("--actor-profile", help="expected worker/profile name for runtime-profile smoke evidence")
    detect.add_argument("--board", help="Kanban board slug; defaults to env/current board")
    detect.add_argument("--db-path", help="explicit Kanban SQLite DB path; opened read-only")
    detect.add_argument("--receipt-file", help="write the full JSON receipt to this path")
    detect.add_argument("--json", action="store_true", help="print the full JSON receipt to stdout")

    doctor = sub.add_parser("doctor", help="emit no-side-effect runtime/profile smoke evidence")
    doctor.add_argument("--actor-profile", help="expected worker/profile name")
    doctor.add_argument("--json", action="store_true", help="print JSON instead of a one-line summary")

    apply = sub.add_parser("apply", help="intentionally disabled placeholder for future controlled apply")
    apply.add_argument("--json", action="store_true", help="print JSON receipt")
    return parser


def _doctor_receipt(actor_profile: Optional[str]) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "mode": "doctor",
        "ok": True,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_profile_smoke": _runtime_profile_smoke(actor_profile),
        "public_safety": {
            "no_mutations_performed": True,
            "write_targets": [],
            "raw_platform_locator_included": False,
            "user_controlled_titles_included": False,
            "credentials_or_tokens_touched": False,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "doctor":
        receipt = _doctor_receipt(args.actor_profile)
        if args.json:
            print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            smoke = receipt["runtime_profile_smoke"]
            print(
                f"{ENTRYPOINT}: profile={smoke['actual_profile'] or '(unset)'} "
                f"expected={smoke['expected_actor_profile'] or '(none)'} "
                f"match={smoke['actor_profile_match']} no mutations performed"
            )
        return 0

    if args.command == "apply":
        receipt = {
            "schema": SCHEMA_VERSION,
            "mode": "apply_disabled",
            "ok": False,
            "error": "controlled apply is intentionally not implemented; detector/dry-run is read-only",
            "public_safety": {
                "no_mutations_performed": True,
                "write_targets": [],
                "destructive_apply_default_enabled": False,
            },
        }
        if args.json:
            print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(receipt["error"], file=sys.stderr)
        return 2

    try:
        db_path = Path(args.db_path).expanduser() if args.db_path else None
        with _open_readonly_connection(db_path, board=args.board) as conn:
            receipt = build_duplicate_child_receipt(
                conn,
                root_task_ids=args.root_task or None,
                issue=args.issue,
                pr=args.pr,
                head=args.head,
                review_scope_key=args.review_scope_key,
                audit_scope=args.audit_scope,
                actor_profile=args.actor_profile,
                board=args.board,
                dry_run=True,
            )
        _write_or_print_receipt(receipt, receipt_file=args.receipt_file, json_output=args.json)
        return 0
    except Exception as exc:
        error_receipt = {
            "schema": SCHEMA_VERSION,
            "mode": "dry_run",
            "ok": False,
            "error": str(exc),
            "public_safety": {"no_mutations_performed": True, "write_targets": []},
        }
        if getattr(args, "json", False):
            print(json.dumps(error_receipt, ensure_ascii=False, indent=2, sort_keys=True), file=sys.stderr)
        else:
            print(f"{ENTRYPOINT}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
