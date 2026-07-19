"""Read-only, deterministic project-finalization evaluation.

This module derives a transition recommendation from one durable Kanban snapshot.
It intentionally owns neither task transitions nor project-finalization writes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import sqlite3
from typing import Any, Iterable, Optional

from hermes_cli.project_finalization_contract import (
    CHECKER_VERDICTS,
    MEMBERSHIP_KINDS,
    PROJECT_FINALIZATION_STATES,
    TERMINAL_OUTCOMES,
    ProjectFinalization,
    ProjectMember,
    get_project_finalization,
    list_project_finalizations,
    list_project_members,
)

EVALUATION_STATES: tuple[str, ...] = (
    "WAITING",
    "REPAIRABLE",
    "COMPLETE_ELIGIBLE",
    "BLOCKED",
    "FAILED",
    "MALFORMED",
)

_INTERNAL_FAILURE_OUTCOMES = frozenset({"crashed", "timed_out", "gave_up"})
_HUMAN_BLOCK_KINDS = frozenset({"needs_input", "capability"})


@dataclass(frozen=True)
class ProjectEvaluation:
    """A value-only transition assessment for one project generation."""

    board_id: str
    root_task_id: str
    generation: int
    snapshot_version: str
    evaluation_state: str
    terminal_outcome: Optional[str]
    required_task_ids: tuple[str, ...]
    optional_task_ids: tuple[str, ...]
    unfinished_task_ids: tuple[str, ...]
    successful_task_ids: tuple[str, ...]
    blocked_task_ids: tuple[str, ...]
    failed_task_ids: tuple[str, ...]
    checker_task_id: Optional[str]
    checker_verdict: Optional[str]
    repair_generation: int
    repair_budget: int
    repair_eligible: bool
    finalization_eligible: bool
    blocker: Optional[str]
    failure_reason: Optional[str]
    evidence_references: tuple[str, ...]
    candidate_snapshot_version: str = ""


def evaluate_project(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int | None = None,
    evaluation_time: int,
) -> ProjectEvaluation:
    """Evaluate a project from one SQLite read snapshot without writing.

    ``evaluation_time`` is an explicit, caller-owned deterministic input.  It
    is not interpreted as a wall clock by this module; it contributes to the
    snapshot identity so callers can distinguish otherwise identical evaluations
    performed for different transition windows.
    """
    if not isinstance(evaluation_time, int) or isinstance(evaluation_time, bool):
        raise ValueError("evaluation_time must be an integer")
    if generation is not None and (not isinstance(generation, int) or isinstance(generation, bool) or generation < 1):
        raise ValueError("generation must be an integer at least 1")

    started_transaction = not conn.in_transaction
    if started_transaction:
        conn.execute("BEGIN")
    try:
        snapshot = _read_snapshot(
            conn,
            board_id=board_id,
            root_task_id=root_task_id,
            generation=generation,
            evaluation_time=evaluation_time,
        )
        return _evaluate_snapshot(snapshot)
    finally:
        if started_transaction:
            conn.rollback()


def _read_snapshot(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int | None,
    evaluation_time: int,
) -> dict[str, Any]:
    """Read every evaluator input once while the caller owns a read snapshot."""
    requested = get_project_finalization(
        conn,
        board_id=board_id,
        root_task_id=root_task_id,
        generation=generation,
    )
    finalizations = tuple(
        item
        for item in list_project_finalizations(conn, board_id=board_id)
        if item.root_task_id == root_task_id
    )
    selected_generation = requested.generation if requested is not None else generation or 0
    members: tuple[ProjectMember, ...] = ()
    if requested is not None:
        members = tuple(
            sorted(
                list_project_members(
                    conn,
                    board_id=board_id,
                    root_task_id=root_task_id,
                    generation=requested.generation,
                ),
                key=lambda item: (item.task_id, item.membership_kind, bool(item.required), item.created_at),
            )
        )

    # Full canonical rows make the digest a real snapshot identity, rather
    # than a proxy based on the project-finalization schema version.
    rows = {
        "tasks": _rows(conn.execute("SELECT * FROM tasks ORDER BY id")),
        "task_links": _rows(
            conn.execute("SELECT parent_id, child_id FROM task_links ORDER BY parent_id, child_id")
        ),
        "task_runs": _rows(conn.execute("SELECT * FROM task_runs ORDER BY task_id, id")),
        "task_events": _rows(conn.execute("SELECT * FROM task_events ORDER BY task_id, id")),
    }
    snapshot = {
        "board_id": board_id,
        "root_task_id": root_task_id,
        "requested_generation": generation,
        "selected_generation": selected_generation,
        "evaluation_time": evaluation_time,
        "requested_finalization": _as_dict(requested),
        "finalizations": [_as_dict(item) for item in finalizations],
        "members": [_as_dict(item) for item in members],
        **rows,
    }
    snapshot["snapshot_version"] = _snapshot_version(snapshot)
    snapshot["candidate_snapshot_version"] = _candidate_snapshot_version(snapshot)
    return snapshot


def _evaluate_snapshot(snapshot: dict[str, Any]) -> ProjectEvaluation:
    board_id = snapshot["board_id"]
    root_task_id = snapshot["root_task_id"]
    finalization_data = snapshot["requested_finalization"]
    finalization = _finalization_from_data(finalization_data)
    members = tuple(_member_from_data(item) for item in snapshot["members"])
    tasks = {row["id"]: row for row in snapshot["tasks"]}

    if finalization is None:
        return _malformed(
            snapshot,
            generation=int(snapshot["selected_generation"]),
            reason="missing_project_finalization",
            blocker="project finalization row is missing",
        )

    malformed = _validate_project_snapshot(finalization, snapshot["finalizations"], members, tasks)
    if malformed is not None:
        return _malformed(snapshot, generation=finalization.generation, reason=malformed[0], blocker=malformed[1], finalization=finalization)

    admitted = finalization.admission_key is not None
    closure: set[str] = set()
    if not admitted:
        closure, graph_error = _ancestor_closure(root_task_id, snapshot["task_links"], tasks)
        if graph_error is not None:
            return _malformed(snapshot, generation=finalization.generation, reason=graph_error[0], blocker=graph_error[1], finalization=finalization)

    checker_task_id = finalization.final_checker_task_id
    checker_task = tasks.get(checker_task_id)
    checker_members = tuple(member for member in members if member.membership_kind == "checker")
    checker_authority_current = not checker_members or (
        len(checker_members) == 1
        and checker_members[0].task_id == checker_task_id
        and checker_members[0].required
    )

    implementation_required = set(closure)
    optional: set[str] = set()
    for member in members:
        if member.membership_kind == "checker":
            if member.task_id != checker_task_id:
                optional.add(member.task_id)
        elif admitted and member.membership_kind in {"required", "repair"}:
            implementation_required.add(member.task_id)
        elif not admitted and member.required:
            implementation_required.add(member.task_id)
        else:
            optional.add(member.task_id)
    required = set(implementation_required)
    required.add(checker_task_id)
    optional.difference_update(required)

    required_ids = tuple(sorted(required))
    optional_ids = tuple(sorted(optional))
    task_rows = {task_id: tasks[task_id] for task_id in required_ids if task_id in tasks}
    successful = tuple(sorted(task_id for task_id, task in task_rows.items() if task["status"] == "done"))
    unfinished = tuple(sorted(task_id for task_id in required_ids if task_id not in successful))
    blocked = tuple(
        sorted(
            task_id
            for task_id, task in task_rows.items()
            if task["status"] in {"blocked", "triage"}
        )
    )
    implementation_ids = tuple(sorted(implementation_required))
    implementation_failed = _failed_task_ids(
        implementation_ids, task_rows, snapshot["task_runs"]
    )
    failed = implementation_failed
    evidence = _evidence_references(required_ids, members, snapshot["task_runs"], snapshot["task_events"])

    common = dict(
        board_id=board_id,
        root_task_id=root_task_id,
        generation=finalization.generation,
        snapshot_version=snapshot["snapshot_version"],
        candidate_snapshot_version=snapshot["candidate_snapshot_version"],
        required_task_ids=required_ids,
        optional_task_ids=optional_ids,
        unfinished_task_ids=unfinished,
        successful_task_ids=successful,
        blocked_task_ids=blocked,
        failed_task_ids=failed,
        checker_task_id=finalization.final_checker_task_id,
        checker_verdict=finalization.checker_verdict,
        repair_generation=finalization.repair_generation,
        repair_budget=finalization.repair_budget,
        evidence_references=evidence,
    )

    durable_outcome = finalization.terminal_outcome
    if durable_outcome == "FAILED" or failed:
        return ProjectEvaluation(
            **common,
            evaluation_state="FAILED",
            terminal_outcome="FAILED",
            repair_eligible=False,
            finalization_eligible=False,
            blocker=None,
            failure_reason="unrecovered_internal_failure",
        )

    implementation_blocked = tuple(
        task_id for task_id in blocked if task_id in implementation_required
    )
    human_blocked = any(
        task_rows[task_id]["status"] == "triage"
        or task_rows[task_id].get("block_kind") in _HUMAN_BLOCK_KINDS
        or (
            task_rows[task_id]["status"] == "blocked"
            and task_rows[task_id].get("block_kind") is None
        )
        for task_id in implementation_blocked
    )
    if durable_outcome == "BLOCKED" or human_blocked:
        return ProjectEvaluation(
            **common,
            evaluation_state="BLOCKED",
            terminal_outcome="BLOCKED",
            repair_eligible=False,
            finalization_eligible=False,
            blocker=finalization.blocker_json or "required task needs external or human action",
            failure_reason="external_or_human_block",
        )

    unfinished_implementation = tuple(
        task_id for task_id in implementation_ids if task_id not in successful
    )
    if unfinished_implementation:
        return ProjectEvaluation(
            **common,
            evaluation_state="WAITING",
            terminal_outcome=None,
            repair_eligible=False,
            finalization_eligible=False,
            blocker=None,
            failure_reason=None,
        )

    if (
        checker_authority_current
        and checker_task is not None
        and checker_task["status"] in {"blocked", "triage"}
    ):
        return ProjectEvaluation(
            **common,
            evaluation_state="BLOCKED",
            terminal_outcome="BLOCKED",
            repair_eligible=False,
            finalization_eligible=False,
            blocker=finalization.blocker_json or "current project checker is blocked",
            failure_reason="checker_blocked",
        )

    if (
        not checker_authority_current
        or checker_task is None
        or checker_task["status"] != "done"
    ):
        return ProjectEvaluation(
            **common,
            evaluation_state="WAITING",
            terminal_outcome=None,
            repair_eligible=False,
            finalization_eligible=False,
            blocker=None,
            failure_reason="checker_required",
        )

    verdict = finalization.checker_verdict
    if verdict == "FAIL_TERMINAL":
        return ProjectEvaluation(
            **common,
            evaluation_state="BLOCKED",
            terminal_outcome="BLOCKED",
            repair_eligible=False,
            finalization_eligible=False,
            blocker=finalization.blocker_json or "final checker reported a terminal failure",
            failure_reason="checker_fail_terminal",
        )
    if verdict == "FAIL_REPAIRABLE":
        if finalization.repair_generation < finalization.repair_budget:
            return ProjectEvaluation(
                **common,
                evaluation_state="REPAIRABLE",
                terminal_outcome=None,
                repair_eligible=True,
                finalization_eligible=False,
                blocker=None,
                failure_reason="checker_fail_repairable",
            )
        return ProjectEvaluation(
            **common,
            evaluation_state="BLOCKED",
            terminal_outcome="BLOCKED",
            repair_eligible=False,
            finalization_eligible=False,
            blocker=finalization.blocker_json or "repair budget is exhausted",
            failure_reason="repair_budget_exhausted",
        )
    if verdict != "PASS":
        return ProjectEvaluation(
            **common,
            evaluation_state="WAITING",
            terminal_outcome=None,
            repair_eligible=False,
            finalization_eligible=False,
            blocker=None,
            failure_reason="checker_required",
        )

    return ProjectEvaluation(
        **common,
        evaluation_state="COMPLETE_ELIGIBLE",
        terminal_outcome="COMPLETE",
        repair_eligible=False,
        finalization_eligible=True,
        blocker=None,
        failure_reason=None,
    )


def _validate_project_snapshot(
    finalization: ProjectFinalization,
    finalizations: Iterable[dict[str, Any]],
    members: Iterable[ProjectMember],
    tasks: dict[str, dict[str, Any]],
) -> tuple[str, str] | None:
    if finalization.state not in PROJECT_FINALIZATION_STATES:
        return "invalid_finalization_state", "project finalization state is not in the frozen vocabulary"
    if finalization.terminal_outcome is not None and finalization.terminal_outcome not in TERMINAL_OUTCOMES:
        return "invalid_terminal_outcome", "project terminal outcome is not in the frozen vocabulary"
    if finalization.checker_verdict is not None and finalization.checker_verdict not in CHECKER_VERDICTS:
        return "invalid_checker_verdict", "checker verdict is not in the frozen vocabulary"
    active = [item for item in finalizations if item["terminal_outcome"] is None]
    if len(active) != 1:
        return "multiple_active_generations", "project identity has zero or multiple active generations"
    if finalization.root_task_id not in tasks:
        return "missing_root_task", "root task is missing from the durable board snapshot"
    for member in members:
        if member.membership_kind not in MEMBERSHIP_KINDS:
            return "inconsistent_membership", "member kind is not in the frozen vocabulary"
        if not isinstance(member.required, bool):
            return "inconsistent_membership", "member required flag is not boolean"
        if member.task_id not in tasks:
            if member.membership_kind == "checker":
                continue
            return "inconsistent_membership", "explicit member task is missing from the durable board snapshot"
    return None


def _ancestor_closure(
    root_task_id: str,
    links: Iterable[dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
) -> tuple[set[str], tuple[str, str] | None]:
    parents_by_child: dict[str, list[str]] = defaultdict(list)
    for link in links:
        parents_by_child[str(link["child_id"])].append(str(link["parent_id"]))
    for parents in parents_by_child.values():
        parents.sort()

    closure: set[str] = set()
    visiting: set[str] = set()

    def visit(task_id: str) -> tuple[str, str] | None:
        if task_id in visiting:
            return "dependency_cycle", f"dependency cycle reaches task {task_id}"
        if task_id in closure:
            return None
        if task_id not in tasks:
            return "missing_task_reference", f"dependency graph references missing task {task_id}"
        visiting.add(task_id)
        for parent_id in parents_by_child.get(task_id, []):
            error = visit(parent_id)
            if error is not None:
                return error
        visiting.remove(task_id)
        closure.add(task_id)
        return None

    error = visit(root_task_id)
    return closure, error


def _failed_task_ids(
    required_task_ids: Iterable[str],
    tasks: dict[str, dict[str, Any]],
    runs: Iterable[dict[str, Any]],
) -> tuple[str, ...]:
    latest: dict[str, dict[str, Any]] = {}
    for run in runs:
        task_id = str(run["task_id"])
        if task_id in tasks:
            latest[task_id] = run
    return tuple(
        sorted(
            task_id
            for task_id in required_task_ids
            if tasks[task_id]["status"] != "done"
            and (
                latest.get(task_id, {}).get("outcome") == "gave_up"
                or (
                    tasks[task_id]["status"] == "blocked"
                    and latest.get(task_id, {}).get("outcome")
                    in _INTERNAL_FAILURE_OUTCOMES
                )
            )
        )
    )


def _evidence_references(
    required_task_ids: Iterable[str],
    members: Iterable[ProjectMember],
    runs: Iterable[dict[str, Any]],
    events: Iterable[dict[str, Any]],
) -> tuple[str, ...]:
    required = set(required_task_ids)
    refs = {f"task:{task_id}" for task_id in required}
    refs.update(
        f"member:{member.task_id}:{member.membership_kind}:{int(member.required)}"
        for member in members
    )
    refs.update(f"run:{run['task_id']}:{run['id']}" for run in runs if run["task_id"] in required)
    refs.update(f"event:{event['task_id']}:{event['id']}" for event in events if event["task_id"] in required)
    return tuple(sorted(refs))


def _malformed(
    snapshot: dict[str, Any],
    *,
    generation: int,
    reason: str,
    blocker: str,
    finalization: ProjectFinalization | None = None,
) -> ProjectEvaluation:
    return ProjectEvaluation(
        board_id=snapshot["board_id"],
        root_task_id=snapshot["root_task_id"],
        generation=generation,
        snapshot_version=snapshot["snapshot_version"],
        candidate_snapshot_version=snapshot["candidate_snapshot_version"],
        evaluation_state="MALFORMED",
        terminal_outcome=None,
        required_task_ids=(),
        optional_task_ids=(),
        unfinished_task_ids=(),
        successful_task_ids=(),
        blocked_task_ids=(),
        failed_task_ids=(),
        checker_task_id=finalization.final_checker_task_id if finalization else None,
        checker_verdict=finalization.checker_verdict if finalization else None,
        repair_generation=finalization.repair_generation if finalization else 0,
        repair_budget=finalization.repair_budget if finalization else 0,
        repair_eligible=False,
        finalization_eligible=False,
        blocker=blocker,
        failure_reason=reason,
        evidence_references=(),
    )


def _rows(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
    return [dict(row) for row in cursor.fetchall()]


def _as_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return {name: getattr(value, name) for name in value.__dataclass_fields__}


def _finalization_from_data(data: dict[str, Any] | None) -> ProjectFinalization | None:
    return ProjectFinalization(**data) if data is not None else None


def _member_from_data(data: dict[str, Any]) -> ProjectMember:
    return ProjectMember(**data)


def _snapshot_version(snapshot: dict[str, Any]) -> str:
    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


_CANDIDATE_TASK_FIELDS: tuple[str, ...] = (
    "id",
    "title",
    "body",
    "assignee",
    "status",
    "result",
    "contract",
    "workspace_kind",
    "workspace_path",
    "branch_name",
    "block_kind",
    "block_recurrences",
    "consecutive_failures",
    "last_failure_error",
    "workflow_template_id",
    "current_step_key",
)

_CANDIDATE_RUN_FIELDS: tuple[str, ...] = (
    "id",
    "task_id",
    "profile",
    "step_key",
    "status",
    "started_at",
    "ended_at",
    "outcome",
    "summary",
    "metadata",
    "error",
)

_TRANSIENT_CANDIDATE_EVENT_KINDS = frozenset(
    {
        "assigned",
        "claimed",
        "claim_extended",
        "heartbeat",
        "reclaim_deferred",
        "reclaimed",
        "respawn_guarded",
        "spawned",
        "stale",
        "tip_scratch_workspace",
    }
)


def _candidate_snapshot_version(snapshot: dict[str, Any]) -> str:
    """Hash the implementation candidate, not the evaluator's live world.

    Candidate identity deliberately excludes evaluation time, dependency graph
    closure, task/run claim leases, checker authority and all project delivery
    state.  This lets a checker bind to one completed implementation candidate
    while registration, verdict recording and delivery continue to mutate the
    broader project snapshot.
    """
    members = [
        {
            "task_id": item["task_id"],
            "membership_kind": item["membership_kind"],
            "required": bool(item["required"]),
        }
        for item in snapshot["members"]
        if item["membership_kind"] in {"required", "repair"}
    ]
    members.sort(key=lambda item: (item["task_id"], item["membership_kind"], item["required"]))
    task_ids = {item["task_id"] for item in members}

    tasks = [
        {field: row.get(field) for field in _CANDIDATE_TASK_FIELDS}
        for row in snapshot["tasks"]
        if row["id"] in task_ids
    ]
    tasks.sort(key=lambda item: item["id"])

    runs = [
        {field: row.get(field) for field in _CANDIDATE_RUN_FIELDS}
        for row in snapshot["task_runs"]
        if row["task_id"] in task_ids
    ]
    runs.sort(key=lambda item: (item["task_id"], item["id"]))

    events = [
        {
            "id": row["id"],
            "task_id": row["task_id"],
            "run_id": row.get("run_id"),
            "kind": row["kind"],
            "payload": row.get("payload"),
        }
        for row in snapshot["task_events"]
        if row["task_id"] in task_ids
        and not str(row["kind"]).startswith("project_")
        and row["kind"] not in _TRANSIENT_CANDIDATE_EVENT_KINDS
        and not str(row["kind"]).startswith("notify_")
    ]
    events.sort(key=lambda item: (item["task_id"], item["id"]))

    candidate = {
        "board_id": snapshot["board_id"],
        "root_task_id": snapshot["root_task_id"],
        "generation": snapshot["selected_generation"],
        "members": members,
        "tasks": tasks,
        "task_runs": runs,
        "task_events": events,
    }
    return _snapshot_version(candidate)
