"""Phase 4 durable active/closed task registry.

This module is intentionally narrow. It records durable task lifecycle state for
Context Health, but it does not implement Phase 5 Task Boundary Firewall,
retrieval scoping, compact fallback changes, or transcript mutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
from typing import Any, Mapping, Sequence

from hermes_constants import get_hermes_home
from agent.context_health_policy import classify_task_boundary, normalize_context_health_policy


SCHEMA = "context_health_task_registry_v1"
SCHEMA_VERSION = 1
SENSITIVE_MARKERS = (
    "token",
    "password",
    "secret",
    "credential",
    "credentials",
    "api key",
    "private key",
    "connection string",
)


@dataclass(frozen=True)
class TaskRegistryDecision:
    action: str
    effective_task_id: str | None
    reason: str
    registry_snapshot: Mapping[str, Any] | None = None
    linked_task_id: str | None = None
    hold_response: str | None = None
    event: Mapping[str, Any] | None = None
    safe_snapshot: Mapping[str, Any] = field(default_factory=dict)


class TaskRegistryFailure(Exception):
    """Typed failure for enabled registry paths that must fail closed."""

    def __init__(self, reason: str, user_response: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.user_response = user_response or _safe_failure_response(reason)


def default_registry_root() -> Path:
    return get_hermes_home() / "context-health" / "task-registry"


def resolve_task_for_turn(
    *,
    policy: Any = None,
    registry_root: str | Path | None = None,
    session_id: str | None = None,
    incoming_task_id: str | None = None,
    user_message: str = "",
    explicit_continuation_refs: Sequence[str] = (),
    ambiguous_relation: bool = False,
    turn_id: str | None = None,
) -> TaskRegistryDecision:
    """Resolve the durable task id for the current turn.

    Disabled mode is complete pass-through: no file read/write and no registry
    mutation. Enabled mode records only safe metadata and pointer fields.
    """

    if not _registry_enabled(policy):
        return TaskRegistryDecision(
            action="use_original",
            effective_task_id=incoming_task_id,
            reason="task_registry_disabled",
            registry_snapshot=None,
        )

    root = Path(registry_root) if registry_root is not None else default_registry_root()
    snapshot = _load_registry(root)
    active_task_id = _safe_str(snapshot.get("active_task_id"))
    tasks = _tasks(snapshot)
    closed_task_ids = [
        task_id
        for task_id, rec in tasks.items()
        if isinstance(rec, Mapping) and str(rec.get("status") or "") in {"closed", "archived"}
    ]

    boundary_policy = normalize_context_health_policy(_raw_context_health(policy))
    boundary_decision = classify_task_boundary(
        user_message=user_message,
        active_task_id=active_task_id,
        closed_task_ids=closed_task_ids,
        explicit_continuation_refs=explicit_continuation_refs,
        ambiguous_relation=ambiguous_relation,
        policy=boundary_policy,
    )

    if boundary_decision.action == "hold":
        return TaskRegistryDecision(
            action="hold",
            effective_task_id=None,
            reason=boundary_decision.reason,
            registry_snapshot=_safe_snapshot(snapshot),
            hold_response=_safe_hold_response(boundary_decision.reason),
            safe_snapshot=_safe_snapshot(snapshot),
        )

    if boundary_decision.action == "continue_task":
        target = boundary_decision.linked_task_id or incoming_task_id or active_task_id or _new_task_id()
        event_name = "task_continued"
        updated = _upsert_task(
            snapshot,
            task_id=target,
            status="active",
            session_id=session_id,
            turn_id=turn_id,
            reason=boundary_decision.reason,
            linked_task_id=boundary_decision.linked_task_id,
        )
        try:
            event = _append_event(
                root,
                event=event_name,
                task_id=target,
                session_id=session_id,
                turn_id=turn_id,
                status="active",
                reason=boundary_decision.reason,
                linked_task_id=boundary_decision.linked_task_id,
            )
            _write_registry(root, updated)
        except TaskRegistryFailure:
            raise
        except Exception as exc:
            raise TaskRegistryFailure("task_registry_write_failure") from exc
        return TaskRegistryDecision(
            action="continue_task",
            effective_task_id=target,
            reason=boundary_decision.reason,
            registry_snapshot=_safe_snapshot(updated),
            linked_task_id=boundary_decision.linked_task_id,
            event=event,
            safe_snapshot=_safe_snapshot(updated),
        )

    # Default/new task path. If the caller supplied a task id, preserve it;
    # otherwise create a durable id here before Phase 2 intake runs.
    new_task_id = incoming_task_id or _new_task_id()
    updated = _close_prior_active_if_different(snapshot, new_task_id, session_id=session_id, turn_id=turn_id)
    updated = _upsert_task(
        updated,
        task_id=new_task_id,
        status="active",
        session_id=session_id,
        turn_id=turn_id,
        reason=boundary_decision.reason,
    )
    try:
        event = _append_event(
            root,
            event="task_created",
            task_id=new_task_id,
            session_id=session_id,
            turn_id=turn_id,
            status="active",
            reason=boundary_decision.reason,
        )
        _write_registry(root, updated)
    except TaskRegistryFailure:
        raise
    except Exception as exc:
        raise TaskRegistryFailure("task_registry_write_failure") from exc
    return TaskRegistryDecision(
        action="new_task",
        effective_task_id=new_task_id,
        reason=boundary_decision.reason,
        registry_snapshot=_safe_snapshot(updated),
        event=event,
        safe_snapshot=_safe_snapshot(updated),
    )


def record_task_event(
    *,
    registry_root: str | Path | None = None,
    event: str,
    task_id: str,
    session_id: str | None = None,
    status: str | None = None,
    reason: str | None = None,
    turn_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    root = Path(registry_root) if registry_root is not None else default_registry_root()
    snapshot = _load_registry(root)
    updated = _upsert_task(
        snapshot,
        task_id=task_id,
        status=status or _status_from_event(event),
        session_id=session_id,
        turn_id=turn_id,
        reason=reason or event,
        **_safe_extra(extra),
    )
    event_record = _append_event(
        root,
        event=event,
        task_id=task_id,
        session_id=session_id,
        turn_id=turn_id,
        status=status or _status_from_event(event),
        reason=reason,
        **_safe_extra(extra),
    )
    _write_registry(root, updated)
    return event_record


def import_workspec_task_state(
    *,
    registry_root: str | Path | None = None,
    task_id: str,
    session_id: str | None = None,
    task_state_path: str | Path,
    turn_id: str | None = None,
) -> dict[str, Any]:
    path = Path(task_state_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    status = str(data.get("status") or "").lower()
    closed = status in {"completed", "verified", "closed"}
    event = "task_closed" if closed else "task_state_observed"
    record = record_task_event(
        registry_root=registry_root,
        event=event,
        task_id=task_id,
        session_id=session_id,
        turn_id=turn_id,
        status="closed" if closed else "active",
        reason="completed_task_state" if closed else "task_state_observed",
        task_state_path=str(path),
    )
    record["task_state_path"] = str(path)
    return record


def record_completed_workspec_state(
    *,
    registry_root: str | Path | None = None,
    task_id: str,
    session_id: str | None = None,
    task_state_path: str | Path,
    turn_id: str | None = None,
) -> dict[str, Any]:
    return import_workspec_task_state(
        registry_root=registry_root,
        task_id=task_id,
        session_id=session_id,
        task_state_path=task_state_path,
        turn_id=turn_id,
    )


def registry_enabled(policy: Any = None) -> bool:
    return _registry_enabled(policy)


def _registry_enabled(policy: Any) -> bool:
    raw = _raw_context_health(policy)
    if not isinstance(raw, Mapping):
        return False
    if not raw.get("enabled") or not raw.get("runtime_behavior_enabled"):
        return False
    task_boundary = raw.get("task_boundary")
    task_registry = raw.get("task_registry")
    return bool(
        isinstance(task_boundary, Mapping)
        and task_boundary.get("enabled")
        and isinstance(task_registry, Mapping)
        and task_registry.get("enabled")
    )


def _raw_context_health(policy: Any) -> Mapping[str, Any]:
    if isinstance(policy, Mapping):
        if "context_health" in policy and isinstance(policy.get("context_health"), Mapping):
            return policy.get("context_health") or {}
        return policy
    return {}


def _load_registry(root: Path) -> dict[str, Any]:
    path = root / "registry.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.setdefault("schema", SCHEMA)
            data.setdefault("schema_version", SCHEMA_VERSION)
            data.setdefault("active_task_id", None)
            data.setdefault("tasks", {})
            return data
    except FileNotFoundError:
        pass
    except json.JSONDecodeError as exc:
        raise TaskRegistryFailure("task_registry_corrupt_snapshot") from exc
    except Exception as exc:
        raise TaskRegistryFailure("task_registry_read_failure") from exc
    return _empty_registry()


def _empty_registry() -> dict[str, Any]:
    now = _now()
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at": now,
        "updated_at": now,
        "active_task_id": None,
        "tasks": {},
    }


def _write_registry(root: Path, snapshot: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    data = dict(snapshot)
    data["schema"] = SCHEMA
    data["schema_version"] = SCHEMA_VERSION
    data["updated_at"] = _now()
    tmp = root / "registry.json.tmp"
    dst = root / "registry.json"
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(dst)


def _append_event(root: Path, **event: Any) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    record = {"ts": _now(), **_safe_extra(event)}
    with (root / "events.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return record


def _upsert_task(
    snapshot: Mapping[str, Any],
    *,
    task_id: str,
    status: str,
    session_id: str | None = None,
    turn_id: str | None = None,
    reason: str | None = None,
    linked_task_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    updated = dict(snapshot)
    updated.setdefault("schema", SCHEMA)
    updated.setdefault("schema_version", SCHEMA_VERSION)
    updated.setdefault("created_at", _now())
    tasks = dict(_tasks(updated))
    existing = dict(tasks.get(task_id) or {})
    existing.setdefault("task_id", task_id)
    existing.setdefault("created_at", _now())
    existing["status"] = status
    existing["updated_at"] = _now()
    if session_id:
        existing["session_id"] = session_id
    if turn_id:
        existing["latest_turn_id"] = turn_id
    if reason:
        existing["last_reason"] = _safe_str(reason)
    if linked_task_id:
        links = list(existing.get("linked_task_ids") or [])
        if linked_task_id not in links:
            links.append(linked_task_id)
        existing["linked_task_ids"] = links
    for key, value in _safe_extra(extra).items():
        existing[key] = value
    tasks[task_id] = existing
    updated["tasks"] = tasks
    if status == "active":
        updated["active_task_id"] = task_id
    elif updated.get("active_task_id") == task_id:
        updated["active_task_id"] = None
    updated["updated_at"] = _now()
    return updated


def _close_prior_active_if_different(
    snapshot: Mapping[str, Any],
    new_task_id: str,
    *,
    session_id: str | None,
    turn_id: str | None,
) -> dict[str, Any]:
    active = _safe_str(snapshot.get("active_task_id"))
    if not active or active == new_task_id:
        return dict(snapshot)
    return _upsert_task(
        snapshot,
        task_id=active,
        status="closed",
        session_id=session_id,
        turn_id=turn_id,
        reason="superseded_by_new_independent_task",
    )


def _safe_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for task_id, rec in _tasks(snapshot).items():
        if not isinstance(rec, Mapping):
            continue
        safe = {
            "task_id": _safe_str(rec.get("task_id") or task_id),
            "status": _safe_str(rec.get("status")),
            "session_id": _safe_str(rec.get("session_id")),
            "latest_turn_id": _safe_str(rec.get("latest_turn_id")),
            "task_state_path": _safe_str(rec.get("task_state_path")),
            "workspec_path": _safe_str(rec.get("workspec_path")),
            "current_pin_path": _safe_str(rec.get("current_pin_path")),
            "linked_task_ids": [
                _safe_str(v) for v in rec.get("linked_task_ids", []) if _safe_str(v)
            ],
        }
        tasks[task_id] = {k: v for k, v in safe.items() if v not in ("", None, [])}
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "active_task_id": _safe_str(snapshot.get("active_task_id")) or None,
        "tasks": tasks,
    }


def _tasks(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    tasks = snapshot.get("tasks")
    return dict(tasks) if isinstance(tasks, Mapping) else {}


def _safe_failure_response(reason: str) -> str:
    return (
        "Context Health HOLD: durable task registry could not be resolved "
        "safely, so Hermes did not fall back to unregistered task-id flow. "
        f"Reason: {_safe_str(reason)[:120]}"
    )


def _safe_hold_response(reason: str) -> str:
    return f"Context Health HOLD: task relation is ambiguous. Reason: {_safe_str(reason)[:120]}"


def _safe_extra(values: Mapping[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            safe[key] = _safe_str(value) if isinstance(value, str) else value
        elif isinstance(value, (list, tuple)):
            safe[key] = [_safe_str(v) for v in value if isinstance(v, str)]
    return safe


def _safe_str(value: Any) -> str:
    text = str(value or "")
    lowered = text.lower()
    if any(marker in lowered for marker in SENSITIVE_MARKERS):
        return "[REDACTED]"
    return text[:500]


def _status_from_event(event: str) -> str:
    if event in {"task_closed", "task_archived"}:
        return "closed" if event == "task_closed" else "archived"
    if event in {"task_created", "task_continued", "task_reopened"}:
        return "active"
    return "active"


def _new_task_id() -> str:
    return f"task-{uuid.uuid4().hex[:12]}"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
