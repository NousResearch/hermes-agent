"""Read-only Ship's Crew baseline metrics for a Kanban board.

The aggregator intentionally derives a snapshot from existing board rows. It
never inserts events, updates tasks, or infers absent routing/context fields.
Unknown values are grouped under the explicit ``unknown`` bucket.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from typing import Any, Iterable, Mapping


_FAILURE_OUTCOMES = frozenset({"blocked", "crashed", "failed", "gave_up", "spawn_failed", "timed_out"})
_RATE_LIMIT_TERMS = ("rate_limit", "rate-limited", "ratelimit")
_QUOTA_TERMS = ("quota", "budget_exhausted", "capacity_exhausted")
_PROTOCOL_TERMS = ("protocol_violation", "protocol_error", "invalid_protocol")


def _row(row: Any, name: str, default: Any = None) -> Any:
    if row is None:
        return default
    try:
        keys = row.keys()
        if name in keys:
            return row[name]
    except Exception:
        pass
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _unknown(value: Any) -> str:
    value = str(value or "").strip()
    return value or "unknown"


def _inc(counter: Counter[str], value: Any) -> None:
    counter[_unknown(value)] += 1


def _metadata(run: Any, task: Any, event: Any = None) -> dict[str, Any]:
    """Merge only explicitly stored metadata; later sources win."""
    merged: dict[str, Any] = {}
    merged.update(_json_object(_row(task, "metadata")))
    merged.update(_json_object(_row(run, "metadata")))
    merged.update(_json_object(_row(event, "payload")))
    return merged


def _field(run: Any, task: Any, metadata: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in metadata and metadata[name] not in (None, ""):
            return metadata[name]
        value = _row(run, name)
        if value not in (None, ""):
            return value
        value = _row(task, name)
        if value not in (None, ""):
            return value
    return None


def _event_matches(kind: Any, terms: tuple[str, ...]) -> bool:
    text = str(kind or "").casefold()
    return any(term in text for term in terms)


def _sum_numeric(values: Iterable[Any]) -> int:
    total = 0
    for value in values:
        try:
            total += max(0, int(value or 0))
        except (TypeError, ValueError):
            continue
    return total


def aggregate_metrics(tasks: Iterable[Any], runs: Iterable[Any], comments: Iterable[Any], events: Iterable[Any], *, board: str = "default") -> dict[str, Any]:
    """Return a JSON-serializable, read-only baseline snapshot."""
    task_rows = list(tasks)
    run_rows = list(runs)
    comment_rows = list(comments)
    event_rows = list(events)

    tasks_by_profile: Counter[str] = Counter()
    tasks_by_status: Counter[str] = Counter()
    runs_by_profile: Counter[str] = Counter()
    runs_by_outcome: Counter[str] = Counter()
    models: Counter[str] = Counter()
    efforts: Counter[str] = Counter()
    quota_domains: Counter[str] = Counter()
    output_classes: Counter[str] = Counter()
    task_runs: Counter[str] = Counter()
    task_goal_mode: set[str] = set()
    run_goal_mode = 0
    retries = 0

    tasks_by_id = {}
    for task in task_rows:
        task_id = str(_row(task, "id", ""))
        tasks_by_id[task_id] = task
        _inc(tasks_by_profile, _row(task, "assignee"))
        _inc(tasks_by_status, _row(task, "status"))
        if bool(_row(task, "goal_mode", 0)):
            task_goal_mode.add(task_id)

    runs_by_task: defaultdict[str, int] = defaultdict(int)
    body_bytes = _sum_numeric(len(str(_row(t, "body", "") or "").encode("utf-8")) for t in task_rows)
    parent_summary_bytes = 0
    assembled_context_bytes = 0
    output_bytes = 0
    inline_output_bytes = 0
    for run in run_rows:
        task_id = str(_row(run, "task_id", ""))
        task = tasks_by_id.get(task_id)
        runs_by_task[task_id] += 1
        _inc(runs_by_profile, _row(run, "profile"))
        outcome = _row(run, "outcome") or _row(run, "status")
        _inc(runs_by_outcome, outcome)
        metadata = _metadata(run, task)
        model = _field(run, task, metadata, "model", "model_name", "route_model", "model_override")
        effort = _field(run, task, metadata, "reasoning_effort", "effort")
        quota = _field(run, task, metadata, "quota_domain", "quota")
        _inc(models, model)
        _inc(efforts, effort)
        _inc(quota_domains, quota)
        output_class = _field(run, task, metadata, "output_class", "output_classification")
        _inc(output_classes, output_class)
        if bool(_field(run, task, metadata, "goal_mode")):
            run_goal_mode += 1
        parent_summary_bytes += _sum_numeric((metadata.get("parent_summary_bytes"), metadata.get("parent_result_bytes")))
        assembled_context_bytes += _sum_numeric((metadata.get("assembled_context_bytes"), metadata.get("context_bytes")))
        output_bytes += _sum_numeric((metadata.get("output_bytes"), metadata.get("result_bytes")))
        inline_output_bytes += _sum_numeric((metadata.get("inline_output_bytes"),))

    retries = sum(max(0, count - 1) for count in runs_by_task.values())
    comment_bytes = _sum_numeric(len(str(_row(c, "body", "") or "").encode("utf-8")) for c in comment_rows)

    protocol_violations = 0
    rate_limit_events = 0
    quota_block_events = 0
    for event in event_rows:
        kind = _row(event, "kind")
        if _event_matches(kind, _PROTOCOL_TERMS):
            protocol_violations += 1
        if _event_matches(kind, _RATE_LIMIT_TERMS):
            rate_limit_events += 1
        if _event_matches(kind, _QUOTA_TERMS):
            quota_block_events += 1

    queue_waits: list[int] = []
    durations: list[int] = []
    for run in run_rows:
        task = tasks_by_id.get(str(_row(run, "task_id", "")))
        started = _row(run, "started_at")
        created = _row(task, "created_at")
        ended = _row(run, "ended_at")
        if started is not None and created is not None:
            queue_waits.append(max(0, int(started) - int(created)))
        if started is not None and ended is not None:
            durations.append(max(0, int(ended) - int(started)))

    def timing(values: list[int]) -> dict[str, Any]:
        return {
            "count": len(values),
            "total_seconds": sum(values),
            "max_seconds": max(values) if values else 0,
            "unknown": len(run_rows) - len(values),
        }

    return {
        "schema_version": "ship-crew/diagnostics/v1",
        "board": board,
        "read_only": True,
        "tasks": {
            "total": len(task_rows),
            "by_profile": dict(sorted(tasks_by_profile.items())),
            "by_status": dict(sorted(tasks_by_status.items())),
        },
        "runs": {
            "total": len(run_rows),
            "by_profile": dict(sorted(runs_by_profile.items())),
            "by_outcome": dict(sorted(runs_by_outcome.items())),
            "retries": retries,
            "failure_outcomes": sum(runs_by_outcome.get(outcome, 0) for outcome in _FAILURE_OUTCOMES),
            "goal_mode_runs": run_goal_mode,
            "goal_mode_tasks": len(task_goal_mode),
        },
        "protocol_violations": protocol_violations,
        "context_bytes": {
            "task_body": body_bytes,
            "parent_summary": parent_summary_bytes,
            "comments": comment_bytes,
            "assembled_context": assembled_context_bytes,
        },
        "output": {
            "by_class": dict(sorted(output_classes.items())),
            "total_bytes": output_bytes,
            "inline_bytes": inline_output_bytes,
        },
        "routing": {
            "model": dict(sorted(models.items())),
            "reasoning_effort": dict(sorted(efforts.items())),
            "quota_domain": dict(sorted(quota_domains.items())),
        },
        "events": {
            "rate_limit": rate_limit_events,
            "quota_block": quota_block_events,
        },
        "timing": {
            "queue_to_start": timing(queue_waits),
            "run_duration": timing(durations),
        },
    }


def aggregate_connection(conn: Any, *, board: str = "default") -> dict[str, Any]:
    """Aggregate existing rows without opening a write transaction."""
    tasks = conn.execute("SELECT * FROM tasks ORDER BY id").fetchall()
    runs = conn.execute("SELECT * FROM task_runs ORDER BY id").fetchall()
    comments = conn.execute("SELECT * FROM task_comments ORDER BY id").fetchall()
    events = conn.execute("SELECT * FROM task_events ORDER BY id").fetchall()
    return aggregate_metrics(tasks, runs, comments, events, board=board)
