"""Kanban project admission and status commands.

This module is the presentation boundary for finalized Kanban projects.  Query
functions return privacy-safe dictionaries; rendering is kept separate so the
CLI and the existing ``/kanban`` gateway path share the same deterministic
results without exposing prompts, model bodies, or provider payloads.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable, Mapping

from hermes_cli import kanban_db as kb
from hermes_cli.project_board_projection import (
    active_project_projection,
    historical_project_projection,
)
from hermes_cli.project_finalization_contract import (
    get_project_finalization,
    list_project_finalizations,
    list_project_members,
)
from hermes_cli.project_finalizer import evaluate_project
from hermes_cli.project_retention_cleanup import plan_project_cleanup


_TERMINAL_TASK_STATUSES = frozenset({"done", "archived", "cancelled", "failed"})
_SENSITIVE_PATH_RE = re.compile(r"(?:^|[/\\.])(?:env|auth|credential|secret|token|private[_-]?key)(?:$|[/\\.])", re.I)


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _safe_text(value: Any, *, limit: int = 240) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    if not text:
        return None
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def _safe_failure(value: Any) -> str | None:
    """Keep only an accepted short failure excerpt, never prompt/body material."""
    text = _safe_text(value)
    if text is None:
        return None
    lowered = text.casefold()
    if any(marker in lowered for marker in ("prompt", "response body", "raw provider", "authorization", "bearer ", "api_key", "token=")):
        return "[redacted error]"
    return text


def _safe_path(value: Any) -> str | None:
    text = _safe_text(value, limit=500)
    if text is None:
        return None
    if _SENSITIVE_PATH_RE.search(text):
        return "[redacted path]"
    return text


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return {name: _jsonable(getattr(value, name)) for name in value.__dataclass_fields__}
    return _safe_text(value)


def _finalization_dict(finalization: Any) -> dict[str, Any]:
    return {
        "board_id": _safe_text(_value(finalization, "board_id")),
        "root_task_id": _safe_text(_value(finalization, "root_task_id")),
        "generation": int(_value(finalization, "generation", 0) or 0),
        "state": _safe_text(_value(finalization, "state")),
        "terminal_outcome": _safe_text(_value(finalization, "terminal_outcome")),
        "checker_task_id": _safe_text(_value(finalization, "final_checker_task_id")),
        "checker_verdict": _safe_text(_value(finalization, "checker_verdict")),
        "repair_generation": int(_value(finalization, "repair_generation", 0) or 0),
        "repair_budget": int(_value(finalization, "repair_budget", 0) or 0),
        "notification_policy": _safe_text(_value(finalization, "notification_policy")),
        "retention_days": int(_value(finalization, "retention_days", 0) or 0),
        "cleanup_after": _safe_text(_value(finalization, "cleanup_after")),
        "cleaned_at": _value(finalization, "cleaned_at"),
    }


def _task_summary(task: Any, *, membership_kind: str | None = None, required: bool | None = None) -> dict[str, Any]:
    result = {
        "task_id": _safe_text(_value(task, "id", _value(task, "task_id"))),
        "title": _safe_text(_value(task, "title")) or "",
        "status": _safe_text(_value(task, "status")) or "",
        "assignee": _safe_text(_value(task, "assignee")),
    }
    if membership_kind is not None:
        result["membership_kind"] = membership_kind
    if required is not None:
        result["required"] = bool(required)
    return result


def _member_summary(member: Any, task_map: Mapping[str, Any]) -> dict[str, Any]:
    task_id = _value(member, "task_id")
    task = task_map.get(task_id)
    if task is None:
        result = {
            "task_id": _safe_text(task_id),
            "title": "",
            "status": "missing",
            "assignee": None,
        }
    else:
        result = _task_summary(task)
    result.update(
        {
            "generation": int(_value(member, "generation", 0) or 0),
            "membership_kind": _safe_text(_value(member, "membership_kind")) or "unknown",
            "required": bool(_value(member, "required", False)),
        }
    )
    return result


def _tasks(conn: sqlite3.Connection) -> dict[str, Any]:
    return {task.id: task for task in kb.list_tasks(conn, include_archived=True, order_by="created")}


def _members_for(conn: sqlite3.Connection, finalization: Any) -> list[Any]:
    return list(
        list_project_members(
            conn,
            board_id=str(_value(finalization, "board_id")),
            root_task_id=str(_value(finalization, "root_task_id")),
            generation=int(_value(finalization, "generation")),
        )
    )


def _evaluation(finalization: Any, conn: sqlite3.Connection, *, evaluation_time: int = 0) -> Any:
    return evaluate_project(
        conn,
        board_id=str(_value(finalization, "board_id")),
        root_task_id=str(_value(finalization, "root_task_id")),
        generation=int(_value(finalization, "generation")),
        evaluation_time=evaluation_time,
    )


def _evaluation_dict(evaluation: Any, task_map: Mapping[str, Any]) -> dict[str, Any]:
    required = list(evaluation.required_task_ids)
    successful = list(evaluation.successful_task_ids)
    return {
        "state": evaluation.evaluation_state,
        "terminal_outcome": evaluation.terminal_outcome,
        "required_task_ids": required,
        "required_progress": {"completed": len(successful), "total": len(required)},
        "successful_task_ids": successful,
        "unfinished_task_ids": list(evaluation.unfinished_task_ids),
        "blocked_task_ids": list(evaluation.blocked_task_ids),
        "failed_task_ids": list(evaluation.failed_task_ids),
        "checker_task_id": evaluation.checker_task_id,
        "checker_verdict": evaluation.checker_verdict,
        "repair_generation": evaluation.repair_generation,
        "repair_budget": evaluation.repair_budget,
        "repair_eligible": evaluation.repair_eligible,
        "finalization_eligible": evaluation.finalization_eligible,
        "blocker": _safe_text(evaluation.blocker),
        "failure_reason": _safe_text(evaluation.failure_reason),
        "evidence_references": list(evaluation.evidence_references),
        "running_worker_count": sum(
            1 for task_id in set(required) if _value(task_map.get(task_id), "status") == "running"
        ),
    }


def _terminal_evaluation_dict(
    finalization: Any,
    members: Iterable[Any],
    task_map: Mapping[str, Any],
) -> dict[str, Any]:
    """Present a durable terminal generation without reopening active validation."""

    required = sorted(
        str(_value(member, "task_id"))
        for member in members
        if bool(_value(member, "required", False))
    )
    successful = [
        task_id
        for task_id in required
        if _value(task_map.get(task_id), "status") in {"done", "archived"}
    ]
    blocked = [
        task_id
        for task_id in required
        if _value(task_map.get(task_id), "status") in {"blocked", "triage"}
    ]
    failed = [
        task_id
        for task_id in required
        if _value(task_map.get(task_id), "status") in {"failed", "cancelled"}
    ]
    unfinished = [task_id for task_id in required if task_id not in successful]
    outcome = _safe_text(_value(finalization, "terminal_outcome"))
    blocker = None if outcome == "COMPLETE" else _safe_text(_value(finalization, "blocker_json"))
    return {
        "state": outcome,
        "terminal_outcome": outcome,
        "required_task_ids": required,
        "required_progress": {"completed": len(successful), "total": len(required)},
        "successful_task_ids": successful,
        "unfinished_task_ids": unfinished,
        "blocked_task_ids": blocked,
        "failed_task_ids": failed,
        "checker_task_id": _safe_text(_value(finalization, "final_checker_task_id")),
        "checker_verdict": _safe_text(_value(finalization, "checker_verdict")),
        "repair_generation": int(_value(finalization, "repair_generation", 0) or 0),
        "repair_budget": int(_value(finalization, "repair_budget", 0) or 0),
        "repair_eligible": False,
        "finalization_eligible": False,
        "blocker": blocker,
        "failure_reason": None,
        "evidence_references": [],
        "running_worker_count": 0,
    }


def _active_details(finalization: Any, members: Iterable[Any], task_map: Mapping[str, Any]) -> dict[str, Any]:
    details: dict[str, Any] = {"repair": None, "checker": None}
    for member in members:
        task = task_map.get(_value(member, "task_id"))
        kind = str(_value(member, "membership_kind", ""))
        if kind not in {"repair", "checker"} or task is None:
            continue
        if str(_value(task, "status", "")) in _TERMINAL_TASK_STATUSES:
            continue
        value = _task_summary(task, membership_kind=kind, required=bool(_value(member, "required", False)))
        details["repair" if kind == "repair" else "checker"] = value
    if details["checker"] is None:
        details["checker"] = {
            "task_id": _safe_text(_value(finalization, "final_checker_task_id")),
            "title": "",
            "status": _safe_text(_value(task_map.get(_value(finalization, "final_checker_task_id")), "status")) or "missing",
            "assignee": _safe_text(_value(task_map.get(_value(finalization, "final_checker_task_id")), "assignee")),
            "membership_kind": "checker",
            "required": True,
        }
    return details


def _project_identity(finalization: Any) -> dict[str, Any]:
    return {
        "board_id": _safe_text(_value(finalization, "board_id")),
        "root_task_id": _safe_text(_value(finalization, "root_task_id")),
        "generation": int(_value(finalization, "generation", 0) or 0),
    }


def _missing(root_task_id: str) -> dict[str, Any]:
    return {"ok": False, "found": False, "error": "project_not_found", "root_task_id": root_task_id}


def _get_finalization(conn: sqlite3.Connection, board_id: str, root_task_id: str, generation: int | None = None) -> Any:
    return get_project_finalization(
        conn,
        board_id=board_id,
        root_task_id=root_task_id,
        generation=generation,
    )


def _project_context(
    conn: sqlite3.Connection,
    finalization: Any,
    *,
    evaluation_time: int = 0,
) -> dict[str, Any]:
    task_map = _tasks(conn)
    members = _members_for(conn, finalization)
    evaluation = _evaluation(finalization, conn, evaluation_time=evaluation_time)
    return {
        "finalization": finalization,
        "task_map": task_map,
        "members": members,
        "evaluation": evaluation,
        "identity": _project_identity(finalization),
        "active": _active_details(finalization, members, task_map),
    }


def list_active_projects(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    evaluation_time: int = 0,
) -> dict[str, Any]:
    """Return active project summaries in stable root/generation order."""
    finalizations = [
        item
        for item in list_project_finalizations(conn, board_id=board_id)
        if _value(item, "terminal_outcome") is None
        and _value(item, "state") != "cleaned"
    ]
    task_map = _tasks(conn)
    all_members: list[Any] = []
    for finalization in finalizations:
        all_members.extend(_members_for(conn, finalization))
    projection = active_project_projection(
        tuple(task_map.values()),
        memberships=all_members,
        finalizations=finalizations,
    )
    projected_ids = set(projection.task_ids)
    projects: list[dict[str, Any]] = []
    for finalization in finalizations:
        context = _project_context(conn, finalization, evaluation_time=evaluation_time)
        evaluation = context["evaluation"]
        active = context["active"]
        projects.append(
            {
                **context["identity"],
                "root_title": _safe_text(_value(task_map.get(_value(finalization, "root_task_id")), "title")) or "",
                "generation": int(_value(finalization, "generation")),
                "evaluator_state": evaluation.evaluation_state,
                "required_progress": {
                    "completed": len(evaluation.successful_task_ids),
                    "total": len(evaluation.required_task_ids),
                },
                "running_worker_count": sum(
                    1 for task_id in projected_ids if _value(task_map.get(task_id), "status") == "running"
                    and task_id in {entry.task_id for entry in projection.entries if entry.root_task_id == _value(finalization, "root_task_id")}
                ),
                "active_repair": active["repair"],
                "active_checker": active["checker"],
                "checker_verdict": _safe_text(_value(finalization, "checker_verdict")),
                "blocker": _safe_text(evaluation.blocker or evaluation.failure_reason),
                "next_action": _next_action(evaluation),
            }
        )
    projects.sort(key=lambda item: (str(item["root_task_id"]), int(item["generation"])))
    return {"ok": True, "found": True, "board_id": board_id, "projects": projects}


def _next_action(evaluation: Any) -> str:
    return {
        "WAITING": "wait for required workers",
        "REPAIRABLE": "create the bounded repair generation",
        "COMPLETE_ELIGIBLE": "finalize and prepare the report",
        "BLOCKED": "resolve the blocker before retrying",
        "FAILED": "inspect the redacted failure and recovery path",
        "MALFORMED": "repair the malformed project identity",
    }.get(evaluation.evaluation_state, "inspect project state")


def project_status(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int | None = None,
    evaluation_time: int = 0,
) -> dict[str, Any]:
    finalization = _get_finalization(conn, board_id, root_task_id, generation)
    if finalization is None:
        return _missing(root_task_id)
    terminal_outcome = _safe_text(_value(finalization, "terminal_outcome"))
    if terminal_outcome is None:
        context = _project_context(conn, finalization, evaluation_time=evaluation_time)
        task_map = context["task_map"]
        evaluation = context["evaluation"]
        evaluator = _evaluation_dict(evaluation, task_map)
        blocker = _safe_text(evaluation.blocker or evaluation.failure_reason)
        next_action = _next_action(evaluation)
    else:
        task_map = _tasks(conn)
        members = _members_for(conn, finalization)
        context = {
            "identity": _project_identity(finalization),
            "active": _active_details(finalization, members, task_map),
        }
        evaluator = _terminal_evaluation_dict(finalization, members, task_map)
        blocker = evaluator["blocker"]
        next_action = "review terminal project history"
    root = task_map.get(root_task_id)
    return {
        "ok": True,
        "found": True,
        "identity": context["identity"],
        "root": _task_summary(root) if root is not None else {"task_id": root_task_id, "title": "", "status": "missing", "assignee": None},
        "finalization": _finalization_dict(finalization),
        "evaluator": evaluator,
        "active_repair": context["active"]["repair"],
        "active_checker": context["active"]["checker"],
        "artifact_state": {
            "report_recorded": bool(_value(finalization, "final_report_path") and _value(finalization, "final_report_sha256")),
            "manifest_recorded": bool(_value(finalization, "manifest_path") and _value(finalization, "manifest_sha256")),
        },
        "technical_terminal_outcome": terminal_outcome,
        "delivery": project_delivery_status(
            conn,
            board_id=board_id,
            root_task_id=root_task_id,
            generation=int(_value(finalization, "generation")),
        ),
        "cleanup": {
            "eligible_schedule": _safe_text(_value(finalization, "cleanup_after")),
            "state": _safe_text(_value(finalization, "state")),
            "cleaned_at": _value(finalization, "cleaned_at"),
        },
        "blocker": blocker,
        "next_action": next_action,
    }


def project_show(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int | None = None,
    evaluation_time: int = 0,
) -> dict[str, Any]:
    status = project_status(
        conn,
        board_id=board_id,
        root_task_id=root_task_id,
        generation=generation,
        evaluation_time=evaluation_time,
    )
    if not status.get("found"):
        return status
    finalization = _get_finalization(conn, board_id, root_task_id, status["identity"]["generation"])
    assert finalization is not None
    task_map = _tasks(conn)
    members = _members_for(conn, finalization)
    status["evidence_references"] = status["evaluator"]["evidence_references"]
    status["members"] = [
        _member_summary(member, task_map)
        for member in sorted(members, key=lambda item: (str(_value(item, "task_id")), str(_value(item, "membership_kind"))))
    ]
    return status


def _history_rows(conn: sqlite3.Connection, table: str, columns: str, *, board_id: str, root_task_id: str) -> list[dict[str, Any]]:
    # These tables are append-only history surfaces with no public aggregate
    # reader.  The table name is fixed by callers; all values are bound.
    try:
        rows = conn.execute(
            f"SELECT {columns} FROM {table} WHERE board_id = ? AND root_task_id = ? ORDER BY generation ASC, id ASC",
            (board_id, root_task_id),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    result: list[dict[str, Any]] = []
    for row in rows:
        item = {key: _jsonable(row[key]) for key in row.keys()}
        if "redacted_error" in item:
            item["redacted_error"] = _safe_failure(item["redacted_error"])
        result.append(item)
    return result


def _delivery_history(conn: sqlite3.Connection, *, board_id: str, root_task_id: str) -> list[dict[str, Any]]:
    try:
        rows = conn.execute(
            "SELECT generation, platform, attempt_number, delivery_state, accepted, "
            "provider_message_id, created_at, completed_at, next_retry_at "
            "FROM project_delivery_attempts WHERE board_id = ? AND root_task_id = ? "
            "ORDER BY generation ASC, attempt_number ASC, id ASC",
            (board_id, root_task_id),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        {
            "generation": int(row["generation"]),
            "platform": _safe_text(row["platform"]),
            "attempt_number": int(row["attempt_number"]),
            "delivery_state": _safe_text(row["delivery_state"]),
            "accepted": None if row["accepted"] is None else bool(row["accepted"]),
            "provider_message_id": _safe_text(row["provider_message_id"]),
            "created_at": row["created_at"],
            "completed_at": row["completed_at"],
            "next_retry_at": row["next_retry_at"],
        }
        for row in rows
    ]


def project_history(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
) -> dict[str, Any]:
    finalizations = [item for item in list_project_finalizations(conn, board_id=board_id) if _value(item, "root_task_id") == root_task_id]
    if not finalizations:
        return _missing(root_task_id)
    task_map = _tasks(conn)
    members: list[Any] = []
    for finalization in finalizations:
        members.extend(_members_for(conn, finalization))
    projection = historical_project_projection(tuple(task_map.values()), memberships=members, finalizations=finalizations)
    terminal_members: list[dict[str, Any]] = []
    archived_zero_run: list[dict[str, Any]] = []
    for entry in projection.entries:
        task = task_map.get(entry.task_id)
        if task is None:
            continue
        summary = _task_summary(task, membership_kind=entry.membership_kind)
        if entry.status in _TERMINAL_TASK_STATUSES:
            terminal_members.append(summary)
            if entry.status == "archived" and not kb.list_runs(conn, entry.task_id, include_active=True):
                archived_zero_run.append(summary)
    terminal_members.sort(key=lambda item: str(item["task_id"]))
    archived_zero_run.sort(key=lambda item: str(item["task_id"]))
    finalization_history = [_finalization_dict(item) for item in finalizations]
    finalization_history.sort(key=lambda item: int(item["generation"]))
    checker_verdicts = [
        {
            "generation": item["generation"],
            "checker_task_id": item["checker_task_id"],
            "verdict": item["checker_verdict"],
        }
        for item in finalization_history
    ]
    finalization_by_generation = {
        int(_value(item, "generation")): item for item in finalizations
    }
    artifacts = []
    for item in finalization_history:
        original = finalization_by_generation[item["generation"]]
        artifacts.append(
            {
                "generation": item["generation"],
                "report_path": _safe_path(_value(original, "final_report_path")),
                "report_sha256": _safe_text(_value(original, "final_report_sha256")),
                "manifest_path": _safe_path(_value(original, "manifest_path")),
                "manifest_sha256": _safe_text(_value(original, "manifest_sha256")),
                "usage_summary_recorded": bool(_value(original, "usage_summary_json")),
            }
        )
    return {
        "ok": True,
        "found": True,
        "identity": {"board_id": board_id, "root_task_id": root_task_id},
        "terminal_member_tasks": terminal_members,
        "prior_repairs": [
            _member_summary(member, task_map)
            for member in sorted(members, key=lambda item: (int(_value(item, "generation")), str(_value(item, "task_id"))))
            if _value(member, "membership_kind") == "repair"
        ],
        "checker_verdicts": checker_verdicts,
        "archived_zero_run_cards": archived_zero_run,
        "final_artifacts": artifacts,
        "delivery_attempts": _delivery_history(conn, board_id=board_id, root_task_id=root_task_id),
        "failure_envelopes": _history_rows(
            conn,
            "project_failure_envelopes",
            "id, generation, task_id, provider, model, failure_class, status_code, retry_after, redacted_error, error_fingerprint, created_at",
            board_id=board_id,
            root_task_id=root_task_id,
        ),
        "cleanup_journal": _history_rows(
            conn,
            "project_cleanup_journal",
            "id, generation, plan_sha256, mode, status, retention_cutoff, eligible_task_count, excluded_task_count, archived_task_count, evidence_path, created_at, executed_at, redacted_error",
            board_id=board_id,
            root_task_id=root_task_id,
        ),
    }


def _artifact(path: Any, expected_hash: Any) -> dict[str, Any]:
    safe = _safe_path(path)
    result = {"path": safe, "sha256": _safe_text(expected_hash), "exists": False, "hash_matches": False}
    if not isinstance(path, str) or not path:
        return result
    file_path = Path(path)
    if not file_path.is_file():
        return result
    result["exists"] = True
    actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
    result["actual_sha256"] = actual
    result["hash_matches"] = bool(expected_hash) and actual == expected_hash
    return result


def project_final_report(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int | None = None,
) -> dict[str, Any]:
    finalization = _get_finalization(conn, board_id, root_task_id, generation)
    if finalization is None:
        return _missing(root_task_id)
    usage = _value(finalization, "usage_summary_json")
    usage_result: dict[str, Any] = {"recorded": bool(usage)}
    if isinstance(usage, str) and usage and "\n" not in usage and len(usage) <= 500:
        usage_result["identity"] = _safe_path(usage) if "/" in usage or "\\" in usage else _safe_text(usage)
    return {
        "ok": True,
        "found": True,
        "identity": _project_identity(finalization),
        "report": _artifact(_value(finalization, "final_report_path"), _value(finalization, "final_report_sha256")),
        "manifest": _artifact(_value(finalization, "manifest_path"), _value(finalization, "manifest_sha256")),
        "usage_summary": usage_result,
        "technical_terminal_outcome": _safe_text(_value(finalization, "terminal_outcome")),
        "delivery": project_delivery_status(
            conn,
            board_id=board_id,
            root_task_id=root_task_id,
            generation=int(_value(finalization, "generation")),
        ),
    }


def _all_delivery_attempts(conn: sqlite3.Connection, *, board_id: str, root_task_id: str, generation: int) -> list[Any]:
    # The public ledger readers require a complete delivery identity, but the
    # frozen schema does not persist message_kind.  This bounded aggregate is
    # therefore the only read-only way to present project-level delivery state.
    try:
        rows = conn.execute(
            "SELECT platform, destination_reference, attempt_number, delivery_state, accepted, provider_message_id, redacted_error, created_at, completed_at, next_retry_at "
            "FROM project_delivery_attempts WHERE board_id = ? AND root_task_id = ? AND generation = ? "
            "ORDER BY attempt_number ASC, id ASC",
            (board_id, root_task_id, generation),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return rows


def project_delivery_status(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int | None = None,
) -> dict[str, Any]:
    finalization = _get_finalization(conn, board_id, root_task_id, generation)
    if finalization is None:
        return _missing(root_task_id)
    resolved_generation = int(_value(finalization, "generation"))
    rows = _all_delivery_attempts(conn, board_id=board_id, root_task_id=root_task_id, generation=resolved_generation)
    latest = rows[-1] if rows else None
    state = _safe_text(latest["delivery_state"]) if latest is not None else "not_recorded"
    ambiguity = state == "ambiguous"
    retry_state = "scheduled" if state == "retry_scheduled" else ("available" if state in {"rejected", "permanent_failure"} else "none")
    return {
        "ok": True,
        "found": True,
        "identity": _project_identity(finalization),
        "technical_project_outcome": _safe_text(_value(finalization, "terminal_outcome")) or "nonterminal",
        "delivery_state": state,
        "provider_message_id": _safe_text(latest["provider_message_id"]) if latest is not None else None,
        "retry_state": retry_state,
        "ambiguity": ambiguity,
        "attempt_count": len(rows),
        "latest_attempt_number": int(latest["attempt_number"]) if latest is not None else None,
        "redacted_error": _safe_text(latest["redacted_error"]) if latest is not None else None,
    }


def project_cleanup_preview(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int | None = None,
    now: datetime | str | None = None,
) -> dict[str, Any]:
    finalization = _get_finalization(conn, board_id, root_task_id, generation)
    if finalization is None:
        return _missing(root_task_id)
    plan = plan_project_cleanup(
        conn,
        board_id=board_id,
        root_task_id=root_task_id,
        generation=int(_value(finalization, "generation")),
        now=now or datetime.now(timezone.utc),
    )
    return {
        "ok": True,
        "found": True,
        "identity": _project_identity(finalization),
        "eligible": plan.eligible,
        "refusal_reasons": list(plan.refusal_reasons),
        "actions": [action.to_dict() for action in plan.actions],
        "eligible_task_ids": list(plan.eligible_task_ids),
        "excluded_task_ids": list(plan.excluded_task_ids),
        "evidence_paths": [_safe_path(path) for path in plan.evidence_paths],
        "retention_cutoff": plan.retention_cutoff,
        "plan_sha256": plan.plan_sha256,
        "already_applied": plan.already_applied,
    }


def project_admit(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    required_task_ids: Iterable[str],
    checker_profile: str,
    repair_budget: int = 1,
    retention_days: int = 3,
    notification_policy: str = "project_summary",
) -> dict[str, Any]:
    """Admit existing cards into the explicit project-finalization protocol."""
    # Kept local so status/history remain usable while an older installation
    # has only the read-side project tables.
    from hermes_cli.project_runtime_registration import admit_existing_project

    required = tuple(str(item) for item in required_task_ids)
    admitted = admit_existing_project(
        conn,
        board_id=board_id,
        root_task_id=root_task_id,
        required_task_ids=required,
        checker_profile=checker_profile,
        repair_profile="builder-gptluna",
        sealed_evidence_required=True,
        repair_budget=repair_budget,
        retention_days=retention_days,
        notification_policy=notification_policy,
    )
    finalization = admitted.finalization
    persisted_required = sorted(
        member.task_id
        for member in list_project_members(
            conn,
            board_id=board_id,
            root_task_id=root_task_id,
            generation=finalization.generation,
        )
        if member.membership_kind == "required" and member.required
    )
    return {
        "ok": True,
        "found": True,
        "disposition": admitted.disposition,
        "identity": _project_identity(finalization),
        "state": _safe_text(_value(finalization, "state")),
        "checker_profile": _safe_text(_value(finalization, "checker_profile")),
        "repair_worker_profile": _safe_text(_value(finalization, "repair_worker_profile")),
        "checker_task_id": _safe_text(_value(finalization, "final_checker_task_id")),
        "required_task_ids": persisted_required,
        "repair_budget": int(_value(finalization, "repair_budget", 0) or 0),
        "retention_days": int(_value(finalization, "retention_days", 0) or 0),
        "notification_policy": _safe_text(_value(finalization, "notification_policy")),
        "admission_key": _safe_text(admitted.admission_key),
    }


def _line(label: str, value: Any) -> str:
    if isinstance(value, bool):
        value = "yes" if value else "no"
    if value is None or value == "":
        value = "—"
    return f"{label}: {value}"


def _render_status(result: Mapping[str, Any], *, include_details: bool = False) -> str:
    if not result.get("found"):
        return f"Project {result.get('root_task_id', '—')}: not found"
    identity = result["identity"]
    evaluator = result["evaluator"]
    progress = evaluator["required_progress"]
    lines = [
        f"Project {identity['root_task_id']} (generation {identity.get('generation', '—')})",
        _line("Goal", result["root"].get("title")),
        _line("Board", identity.get("board_id")),
        _line("Evaluator", evaluator.get("state")),
        _line("Required progress", f"{progress['completed']}/{progress['total']}"),
        _line("Running workers", evaluator.get("running_worker_count")),
        _line("Active repair", (result.get("active_repair") or {}).get("task_id") if result.get("active_repair") else None),
        _line("Active checker", (result.get("active_checker") or {}).get("task_id") if result.get("active_checker") else None),
        _line("Checker verdict", evaluator.get("checker_verdict")),
        _line("Artifacts", f"report={'recorded' if result['artifact_state']['report_recorded'] else 'missing'}, manifest={'recorded' if result['artifact_state']['manifest_recorded'] else 'missing'}"),
        _line("Technical outcome", result.get("technical_terminal_outcome")),
        _line("Delivery", result.get("delivery", {}).get("delivery_state")),
        _line("Provider message", result.get("delivery", {}).get("provider_message_id")),
        _line("Cleanup", result.get("cleanup", {}).get("eligible_schedule")),
        _line("Blocker", result.get("blocker")),
        _line("Next action", result.get("next_action")),
    ]
    if include_details:
        lines.append("Evidence: " + ", ".join(result.get("evidence_references") or []) if result.get("evidence_references") else "Evidence: —")
        lines.append("Members:")
        for member in result.get("members", []):
            lines.append(
                f"  {member.get('task_id')} [{member.get('membership_kind')}] {member.get('status')} — {member.get('title') or '—'}"
            )
    return "\n".join(lines)


def render_project_result(result: Mapping[str, Any], *, mode: str = "status") -> str:
    """Render a query result as stable Telegram-friendly text or JSON."""
    if mode == "json":
        return json.dumps(_jsonable(result), ensure_ascii=False, indent=2, sort_keys=True)
    if mode == "list-active":
        projects = result.get("projects", [])
        if not projects:
            return f"Active projects on {result.get('board_id', '—')}: none"
        lines = [f"Active projects on {result.get('board_id', '—')}:"]
        for project in projects:
            progress = project["required_progress"]
            lines.append(
                f"- {project['root_task_id']} gen {project['generation']} | {project['evaluator_state']} | "
                f"required {progress['completed']}/{progress['total']} | running {project['running_worker_count']} | "
                f"repair {((project.get('active_repair') or {}).get('task_id') or '—')} | "
                f"checker {((project.get('active_checker') or {}).get('task_id') or '—')} "
                f"({project.get('checker_verdict') or '—'}) | blocker {project.get('blocker') or '—'} | "
                f"next {project.get('next_action') or '—'}"
            )
        return "\n".join(lines)
    if mode == "admit":
        identity = result["identity"]
        return "\n".join(
            [
                f"Project {result.get('disposition', 'admitted')}: {identity['root_task_id']} gen {identity['generation']}",
                _line("Board", identity.get("board_id")),
                _line("Checker profile", result.get("checker_profile")),
                _line("Repair profile", result.get("repair_worker_profile")),
                _line("Pending checker", result.get("checker_task_id")),
                _line("Required tasks", ", ".join(result.get("required_task_ids", []))),
                _line("Repair budget", result.get("repair_budget")),
                _line("Retention days", result.get("retention_days")),
                _line("Notification policy", result.get("notification_policy")),
                _line("Admission key", result.get("admission_key")),
            ]
        )
    if mode == "show":
        return _render_status(result, include_details=True)
    if mode == "status":
        return _render_status(result)
    if mode == "history":
        if not result.get("found"):
            return f"Project {result.get('root_task_id', '—')}: not found"
        return "\n".join(
            [
                f"Project history: {result['identity']['root_task_id']}",
                "Terminal members: " + str(len(result.get("terminal_member_tasks", []))),
                "Prior repairs: " + str(len(result.get("prior_repairs", []))),
                "Checker verdicts: " + str(len(result.get("checker_verdicts", []))),
                "Archived zero-run cards: " + str(len(result.get("archived_zero_run_cards", []))),
                "Final artifacts: " + str(len(result.get("final_artifacts", []))),
                "Delivery attempts: " + str(len(result.get("delivery_attempts", []))),
                "Failure envelopes: " + str(len(result.get("failure_envelopes", []))),
                "Cleanup journal: " + str(len(result.get("cleanup_journal", []))),
            ]
        )
    if mode == "final-report":
        if not result.get("found"):
            return f"Project {result.get('root_task_id', '—')}: not found"
        report = result["report"]
        manifest = result["manifest"]
        return "\n".join(
            [
                f"Final report: {result['identity']['root_task_id']} gen {result['identity']['generation']}",
                _line("Report path", report.get("path")),
                _line("Report SHA-256", report.get("sha256")),
                _line("Report present", report.get("exists")),
                _line("Manifest path", manifest.get("path")),
                _line("Manifest SHA-256", manifest.get("sha256")),
                _line("Manifest present", manifest.get("exists")),
                _line("Usage summary", result.get("usage_summary", {}).get("identity") or ("recorded" if result.get("usage_summary", {}).get("recorded") else "missing")),
                _line("Technical outcome", result.get("technical_terminal_outcome")),
                _line("Delivery", result.get("delivery", {}).get("delivery_state")),
            ]
        )
    if mode == "delivery-status":
        if not result.get("found"):
            return f"Project {result.get('root_task_id', '—')}: not found"
        return "\n".join(
            [
                f"Delivery status: {result['identity']['root_task_id']} gen {result['identity']['generation']}",
                _line("Technical outcome", result.get("technical_project_outcome")),
                _line("Delivery state", result.get("delivery_state")),
                _line("Provider message", result.get("provider_message_id")),
                _line("Retry state", result.get("retry_state")),
                _line("Ambiguity", result.get("ambiguity")),
                _line("Attempts", result.get("attempt_count")),
                _line("Redacted error", result.get("redacted_error")),
            ]
        )
    if mode == "cleanup-preview":
        if not result.get("found"):
            return f"Project {result.get('root_task_id', '—')}: not found"
        lines = [
            f"Cleanup preview: {result['identity']['root_task_id']} gen {result['identity']['generation']}",
            _line("Eligible", result.get("eligible")),
            _line("Plan SHA-256", result.get("plan_sha256")),
            _line("Retention cutoff", result.get("retention_cutoff")),
            _line("Actions", ", ".join(action["task_id"] for action in result.get("actions", [])) or "—"),
            _line("Refusals", ", ".join(result.get("refusal_reasons", [])) or "—"),
            _line("Already applied", result.get("already_applied")),
        ]
        return "\n".join(lines)
    raise ValueError(f"unknown project render mode: {mode}")


def dispatch_project_command(
    args: Any,
    conn: sqlite3.Connection,
    *,
    board_id: str | None = None,
) -> int:
    """Dispatch a parsed nested project command against an existing connection."""
    board = board_id or getattr(args, "board", None) or kb.get_current_board()
    action = getattr(args, "project_action", None)
    generation = getattr(args, "generation", None)
    root_task_id = getattr(args, "root_task_id", None)
    if action == "list-active":
        result = list_active_projects(conn, board_id=board)
    elif action == "admit":
        result = project_admit(
            conn,
            board_id=board,
            root_task_id=root_task_id,
            required_task_ids=getattr(args, "required_task_ids", ()),
            checker_profile=getattr(args, "checker_profile", ""),
            repair_budget=getattr(args, "repair_budget", 1),
            retention_days=getattr(args, "retention_days", 3),
            notification_policy=getattr(args, "notification_policy", "project_summary"),
        )
    elif action == "status":
        result = project_status(conn, board_id=board, root_task_id=root_task_id, generation=generation)
    elif action == "show":
        result = project_show(conn, board_id=board, root_task_id=root_task_id, generation=generation)
    elif action == "history":
        result = project_history(conn, board_id=board, root_task_id=root_task_id)
    elif action == "final-report":
        result = project_final_report(conn, board_id=board, root_task_id=root_task_id, generation=generation)
    elif action == "delivery-status":
        result = project_delivery_status(conn, board_id=board, root_task_id=root_task_id, generation=generation)
    elif action == "cleanup-preview":
        result = project_cleanup_preview(conn, board_id=board, root_task_id=root_task_id, generation=generation)
    else:
        print(f"kanban project: unknown action {action!r}")
        return 2
    mode = "json" if bool(getattr(args, "json", False)) else action
    print(render_project_result(result, mode=mode))
    return 0 if result.get("ok", True) else 1


__all__ = [
    "dispatch_project_command",
    "list_active_projects",
    "project_admit",
    "project_cleanup_preview",
    "project_delivery_status",
    "project_final_report",
    "project_history",
    "project_show",
    "project_status",
    "render_project_result",
]
