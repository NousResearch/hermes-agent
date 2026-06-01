"""Deterministic Self/Lineage status rendering for project profile notifications.

This module is intentionally small and dependency-light: it turns Kanban task
state plus an explicit profile lineage registry into the two generic lines that
project/profile notifications can paste without LLM reasoning.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from hermes_cli import kanban_db

LIVE_STATES = ("working", "waiting", "blocked", "dormant")
LIVE_STATE_SET = set(LIVE_STATES)

_WORKING_STATUSES = {"queue", "ready", "running", "in_progress", "review"}
_WAITING_STATUSES = {"todo", "scheduled"}
_BLOCKED_STATUSES = {"blocked"}
_DORMANT_STATUSES = {"triage"}
_INACTIVE_STATUSES = {"done", "archived"}

# Prefer active/spawnable work over hidden blockers in the aggregate, but include
# counts/evidence in text so a branch-local blocker is not lost.
_PRECEDENCE = ("working", "blocked", "waiting", "dormant")


@dataclass(frozen=True)
class TaskStatusInput:
    """Minimal task shape needed to compute notification status."""

    id: str
    assignee: str | None
    status: str
    title: str | None = None


@dataclass(frozen=True)
class ProfileStatus:
    state: str
    text: str
    counts: dict[str, int] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)

    def line(self, label: str) -> str:
        return f"{label}: {self.state.upper()} — {self.text}"


def kanban_status_to_live(status: str) -> str | None:
    """Map internal/legacy Kanban statuses to four live notification states.

    ``None`` means the task is completion history, not live availability.
    """

    normalized = str(status or "").strip().lower()
    if normalized in _WORKING_STATUSES:
        return "working"
    if normalized in _WAITING_STATUSES:
        return "waiting"
    if normalized in _BLOCKED_STATUSES:
        return "blocked"
    if normalized in _DORMANT_STATUSES:
        return "dormant"
    if normalized in _INACTIVE_STATUSES:
        return None
    # Unknown legacy statuses are safest as waiting: there is some active row, but
    # this renderer cannot prove it is spawnable, blocked on a human, or dormant.
    return "waiting"


def load_lineage_registry(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    profiles = data.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError("lineage registry must contain a 'profiles' object")
    return data


def descendants_for(profile: str, registry: Mapping[str, Any]) -> list[str]:
    profiles = registry.get("profiles", {})
    seen: set[str] = set()
    ordered: list[str] = []

    def visit(name: str) -> None:
        node = profiles.get(name, {})
        children = node.get("children", []) if isinstance(node, Mapping) else []
        if not isinstance(children, Sequence) or isinstance(children, (str, bytes)):
            return
        for child in children:
            child_name = str(child)
            if child_name in seen:
                continue
            seen.add(child_name)
            ordered.append(child_name)
            visit(child_name)

    visit(profile)
    return ordered


def _choose_state(counts: Mapping[str, int]) -> str:
    for state in _PRECEDENCE:
        if counts.get(state, 0) > 0:
            return state
    return "dormant"


def _group_tasks(tasks: Iterable[TaskStatusInput], profile: str) -> tuple[dict[str, int], dict[str, list[str]]]:
    counts = {state: 0 for state in LIVE_STATES}
    evidence = {state: [] for state in LIVE_STATES}
    for task in tasks:
        if task.assignee != profile:
            continue
        live = kanban_status_to_live(task.status)
        if live is None:
            continue
        counts[live] += 1
        evidence[live].append(task.id)
    return counts, evidence


def compute_self_status(profile: str, tasks: Iterable[TaskStatusInput]) -> ProfileStatus:
    counts, evidence_by_state = _group_tasks(tasks, profile)
    state = _choose_state(counts)
    evidence = evidence_by_state.get(state, [])[:5]
    active_total = sum(counts.values())

    if active_total == 0 or state == "dormant":
        return ProfileStatus(
            state="dormant",
            text=f"no active direct Kanban task for {profile}.",
            counts=counts,
            evidence=[],
        )

    pieces = [f"{counts[s]} direct {s}" for s in LIVE_STATES if counts[s]]
    if evidence:
        pieces.append("evidence: " + ", ".join(evidence))
    return ProfileStatus(state=state, text="; ".join(pieces) + ".", counts=counts, evidence=evidence)


def compute_lineage_status(profile: str, registry: Mapping[str, Any], tasks: Iterable[TaskStatusInput]) -> ProfileStatus:
    task_list = list(tasks)
    descendants = descendants_for(profile, registry)
    if not descendants:
        return ProfileStatus(
            state="dormant",
            text=f"no registered descendants for {profile}.",
            counts={state: 0 for state in LIVE_STATES},
            evidence=[],
        )

    counts = {state: 0 for state in LIVE_STATES}
    evidence_by_state = {state: [] for state in LIVE_STATES}
    for descendant in descendants:
        child_status = compute_self_status(descendant, task_list)
        counts[child_status.state] += 1
        evidence_by_state[child_status.state].extend(child_status.evidence)

    state = _choose_state(counts)
    evidence = evidence_by_state.get(state, [])[:5]
    pieces = [f"{counts[s]} descendant {s}" for s in LIVE_STATES if counts[s]]
    if evidence:
        pieces.append("evidence: " + ", ".join(evidence))
    return ProfileStatus(state=state, text="; ".join(pieces) + ".", counts=counts, evidence=evidence)


def render_status_lines(profile: str, registry: Mapping[str, Any], tasks: Iterable[TaskStatusInput]) -> tuple[str, str, dict[str, Any]]:
    task_list = list(tasks)
    self_status = compute_self_status(profile, task_list)
    lineage_status = compute_lineage_status(profile, registry, task_list)
    payload = {
        "project": registry.get("project"),
        "profile": profile,
        "self_status": self_status.state,
        "self_text": self_status.text,
        "self_counts": self_status.counts,
        "self_evidence": self_status.evidence,
        "lineage_status": lineage_status.state,
        "lineage_text": lineage_status.text,
        "lineage_counts": lineage_status.counts,
        "lineage_evidence": lineage_status.evidence,
    }
    return self_status.line("Self status"), lineage_status.line("Lineage status"), payload


def tasks_from_kanban(board: str | None = None, tenant: str | None = None) -> list[TaskStatusInput]:
    conn = kanban_db.connect(board=board)
    try:
        rows = kanban_db.list_tasks(conn, tenant=tenant, include_archived=False)
    finally:
        conn.close()
    return [TaskStatusInput(id=t.id, assignee=t.assignee, status=t.status, title=t.title) for t in rows]


def tasks_from_json(path: str | Path) -> list[TaskStatusInput]:
    with Path(path).expanduser().open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, Mapping):
        raw = raw.get("tasks", [])
    if not isinstance(raw, list):
        raise ValueError("tasks JSON must be a list or an object with a 'tasks' list")
    return [
        TaskStatusInput(
            id=str(item["id"]),
            assignee=item.get("assignee"),
            status=str(item.get("status", "")),
            title=item.get("title"),
        )
        for item in raw
        if isinstance(item, Mapping)
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render generic Self/Lineage status lines from Kanban state.")
    parser.add_argument("--profile", required=True, help="Profile to report, e.g. coordinator_hermes_maintenance")
    parser.add_argument("--registry", required=True, help="Path to project lineage registry JSON")
    parser.add_argument("--board", default=None, help="Kanban board slug; defaults to normal Hermes board resolution")
    parser.add_argument("--tenant", default=None, help="Optional tenant filter")
    parser.add_argument("--tasks-json", default=None, help="Fixture/input tasks JSON instead of reading the live Kanban DB")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of two text lines")
    args = parser.parse_args(argv)

    registry = load_lineage_registry(args.registry)
    tasks = tasks_from_json(args.tasks_json) if args.tasks_json else tasks_from_kanban(board=args.board, tenant=args.tenant)
    self_line, lineage_line, payload = render_status_lines(args.profile, registry, tasks)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(self_line)
        print(lineage_line)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
