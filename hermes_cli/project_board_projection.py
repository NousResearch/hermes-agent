"""Deterministic active and historical projections for project-linked board work.

The projection is intentionally read-side and pure.  A task is eligible for a
project view only when it is named by an explicit project-membership record;
titles, assignees, and board-wide guesses are never used to infer ownership.
Callers may supply dictionaries, dataclasses, or other objects with matching
attributes, which keeps this boundary usable with snapshots and injected test
stores without adding a new persistence schema.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping


TERMINAL_TASK_STATUSES = frozenset({"done", "archived", "cancelled", "failed"})
ACTIVE_TASK_STATUSES = frozenset({"triage", "todo", "scheduled", "ready", "running", "blocked", "review"})


@dataclass(frozen=True)
class ProjectionEntry:
    """One task retained by a projection."""

    task_id: str
    title: str = ""
    status: str = ""
    membership_kind: str = "required"
    root_task_id: str | None = None
    generation: int | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "status": self.status,
            "membership_kind": self.membership_kind,
            "root_task_id": self.root_task_id,
            "generation": self.generation,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ProjectProjection:
    """Stable result of :func:`build_project_projection`."""

    historical: bool
    entries: tuple[ProjectionEntry, ...] = field(default_factory=tuple)
    excluded: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(entry.task_id for entry in self.entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "historical": self.historical,
            "entries": [entry.to_dict() for entry in self.entries],
            "excluded": [dict(item) for item in self.excluded],
            "task_ids": list(self.task_ids),
        }


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _as_items(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping) or isinstance(value, (str, bytes)):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _aware_datetime(value: Any, *, name: str) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        try:
            result = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    else:
        raise ValueError(f"{name} must be timezone-aware")
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return result


def _task_created_or_finished(item: Any) -> datetime | None:
    for key in ("completed_at", "finalized_at", "created_at", "updated_at"):
        raw = _value(item, key)
        if raw is None:
            continue
        if isinstance(raw, datetime):
            return raw if raw.tzinfo is not None and raw.utcoffset() is not None else None
        if isinstance(raw, (int, float)):
            return datetime.fromtimestamp(raw, tz=timezone.utc)
        if isinstance(raw, str):
            try:
                return _aware_datetime(raw, name=key)
            except ValueError:
                return None
    return None


def _is_superseded(task: Any) -> bool:
    if bool(_value(task, "superseded", False)):
        return True
    if _value(task, "is_current", True) is False:
        return True
    return str(_value(task, "status", "")).casefold() == "superseded"


def _run_count(task: Any) -> int:
    raw = _value(task, "run_count")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0
    runs = _value(task, "runs")
    return len(_as_items(runs)) if runs is not None else 0


def _normal_memberships(memberships: Iterable[Any]) -> dict[str, tuple[Any, ...]]:
    result: dict[str, list[Any]] = {}
    for member in _as_items(memberships):
        task_id = _value(member, "task_id")
        root_id = _value(member, "root_task_id")
        if not isinstance(task_id, str) or not task_id or not isinstance(root_id, str) or not root_id:
            continue
        result.setdefault(root_id, []).append(member)
    return {key: tuple(value) for key, value in result.items()}


def _normal_roots(finalizations: Iterable[Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for finalization in _as_items(finalizations):
        root_id = _value(finalization, "root_task_id")
        if isinstance(root_id, str) and root_id:
            result[root_id] = finalization
    return result


def _root_is_recent(finalization: Any, *, now: datetime, max_age: timedelta | None) -> bool:
    if max_age is None:
        return True
    if not isinstance(max_age, timedelta) or max_age.total_seconds() < 0:
        raise ValueError("active_root_max_age must be a non-negative timedelta")
    raw = _value(finalization, "finalized_at")
    if raw is None:
        raw = _value(finalization, "updated_at")
    if raw is None:
        return True
    if isinstance(raw, (int, float)):
        stamp = datetime.fromtimestamp(raw, tz=timezone.utc)
    else:
        stamp = _aware_datetime(raw, name="finalized_at")
    return now - stamp <= max_age


def build_project_projection(
    tasks: Iterable[Any],
    *,
    memberships: Iterable[Any] = (),
    finalizations: Iterable[Any] = (),
    historical: bool = False,
    now: datetime | None = None,
    active_root_max_age: timedelta | None = None,
    project_id: str | None = None,
) -> ProjectProjection:
    """Build an active-only or explicit historical project projection.

    ``memberships`` must contain records with ``root_task_id`` and ``task_id``.
    The root task is included from the finalization identity even when it has no
    separate member row.  Historical mode retains every explicitly named task;
    active mode applies the documented worker/checker/repair/root filters.
    """
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    task_map: dict[str, Any] = {}
    for task in _as_items(tasks):
        task_id = _value(task, "id", _value(task, "task_id"))
        if isinstance(task_id, str) and task_id:
            task_map[task_id] = task

    members_by_root = _normal_memberships(memberships)
    roots = _normal_roots(finalizations)
    if not roots:
        roots = {root_id: None for root_id in members_by_root}

    entries: list[ProjectionEntry] = []
    excluded: list[dict[str, Any]] = []
    for root_id in sorted(set(roots) | set(members_by_root)):
        finalization = roots.get(root_id)
        if project_id is not None and _value(finalization, "project_id") not in (None, project_id):
            continue
        if not historical and finalization is not None and not _root_is_recent(
            finalization, now=current, max_age=active_root_max_age
        ):
            excluded.append({"root_task_id": root_id, "reason": "terminal_root_expired"})
            continue

        member_by_task: dict[str, Any] = {}
        for member in members_by_root.get(root_id, ()):
            task_id = _value(member, "task_id")
            if isinstance(task_id, str) and task_id:
                member_by_task[task_id] = member
        # Root identity is explicit in the finalization record, not inferred
        # from a title or a board-wide task scan.
        member_by_task.setdefault(root_id, None)

        for task_id in sorted(member_by_task):
            task = task_map.get(task_id)
            member = member_by_task[task_id]
            if task is None:
                excluded.append({"task_id": task_id, "root_task_id": root_id, "reason": "missing_task"})
                continue
            status = str(_value(task, "status", ""))
            kind = str(_value(member, "membership_kind", "root" if task_id == root_id else "required"))
            reason = "historical_explicit_membership" if historical else "active_membership"
            retain = historical
            if not historical:
                if task_id == root_id:
                    retain = status not in TERMINAL_TASK_STATUSES or _root_is_recent(
                        finalization, now=current, max_age=active_root_max_age
                    )
                    reason = "active_root"
                elif kind == "required":
                    retain = status not in TERMINAL_TASK_STATUSES
                    reason = "unfinished_required"
                elif kind == "repair":
                    retain = not _is_superseded(task) and status not in TERMINAL_TASK_STATUSES
                    reason = "current_repair"
                elif kind == "checker":
                    retain = not _is_superseded(task) and status not in TERMINAL_TASK_STATUSES
                    reason = "current_checker"
                elif kind == "support":
                    retain = status == "running" or status == "blocked"
                    reason = "running_or_blocked_support"
                else:
                    retain = status in ACTIVE_TASK_STATUSES
                    reason = "active_member"
                if status == "blocked" and not _is_superseded(task):
                    retain = True
                    reason = "unresolved_blocked"
                if status == "archived" and _run_count(task) == 0:
                    retain = False
                    reason = "archived_zero_run_placeholder"
                if kind == "checker" and status in {"done", "archived"}:
                    retain = False
                    reason = "completed_checker"
            if retain:
                entries.append(
                    ProjectionEntry(
                        task_id=task_id,
                        title=str(_value(task, "title", "")),
                        status=status,
                        membership_kind=kind,
                        root_task_id=root_id,
                        generation=_value(member, "generation", _value(finalization, "generation")),
                        reason=reason,
                    )
                )
            else:
                excluded.append({"task_id": task_id, "root_task_id": root_id, "reason": reason})

    entries.sort(key=lambda item: (item.root_task_id or "", item.task_id, item.membership_kind))
    excluded.sort(key=lambda item: (str(item.get("root_task_id", "")), str(item.get("task_id", "")), item["reason"]))
    return ProjectProjection(historical=historical, entries=tuple(entries), excluded=tuple(excluded))


def active_project_projection(tasks: Iterable[Any], **kwargs: Any) -> ProjectProjection:
    """Build the default active projection."""
    kwargs["historical"] = False
    return build_project_projection(tasks, **kwargs)


def historical_project_projection(tasks: Iterable[Any], **kwargs: Any) -> ProjectProjection:
    """Build the explicit historical projection."""
    kwargs["historical"] = True
    return build_project_projection(tasks, **kwargs)


# Descriptive aliases used by callers that treat this module as a board view.
project_board_projection = build_project_projection
build_active_projection = active_project_projection
build_historical_projection = historical_project_projection

__all__ = [
    "ACTIVE_TASK_STATUSES", "TERMINAL_TASK_STATUSES", "ProjectionEntry", "ProjectProjection",
    "build_project_projection", "active_project_projection", "historical_project_projection",
    "project_board_projection", "build_active_projection", "build_historical_projection",
]
