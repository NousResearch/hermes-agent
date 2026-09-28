"""Deterministic, detection-only Kanban watchdog.

This module deliberately reuses the diagnostics rule engine as its source of
truth and the existing task-event notifier as its delivery mechanism.  It never
claims, reopens, restarts, deploys, or otherwise mutates work state: its only
write is one ``watchdog_alert`` event per newly-active condition.

The small state table is bounded by retention pruning.  A fingerprint remains
active while its condition is present, so a periodic caller cannot re-notify on
every tick.  When the condition clears it is marked resolved; a later recurrence
emits exactly one new event.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any, Iterable

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_diagnostics as diagnostics

# The liveness card owns the implementation of the running-worker diagnostic.
# Accept both names during its rollout without duplicating process-identity
# checks here.  This watchdog only routes the diagnostic it is given.
_RUNNING_WITHOUT_WORKER_KINDS = frozenset({"dead_running", "running_without_live_worker"})
_WATCHED_DIAGNOSTIC_KINDS = frozenset({"stranded_in_ready", "stranded_in_review"}) | _RUNNING_WITHOUT_WORKER_KINDS
_DEPLOYMENT_MARKERS = ("deployment-required", "deployment required", "requires deployment")


@dataclass(frozen=True)
class WatchdogAlert:
    task_id: str
    kind: str
    severity: str
    detail: str


@dataclass(frozen=True)
class WatchdogResult:
    new_alerts: list[WatchdogAlert]
    resolved_count: int
    pruned_count: int


def _metadata(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        raw = payload.get("metadata", payload)
        return raw if isinstance(raw, dict) else {}
    if isinstance(payload, str):
        try:
            return _metadata(json.loads(payload))
        except (TypeError, ValueError):
            return {}
    return {}


def _deployment_required(task: Any) -> bool:
    text = "\n".join(str(task[key] or "") for key in ("title", "body") if key in task.keys()).lower()
    return any(marker in text for marker in _DEPLOYMENT_MARKERS)


def _activation_verified(events: Iterable[Any]) -> bool:
    for event in events:
        if event["kind"] != "completed":
            continue
        metadata = _metadata(event["payload"])
        if metadata.get("live_verified") is True:
            return True
        # Compatible with the W/T/D/V vocabulary without owning its schema.
        if str(metadata.get("state") or "").upper() == "V":
            return True
        activation = metadata.get("activation")
        if isinstance(activation, dict) and activation.get("live_verified") is True:
            return True
    return False


def _conditions(conn, *, now: int, config: dict) -> list[WatchdogAlert]:
    rows = list(conn.execute("SELECT * FROM tasks WHERE status != 'archived'").fetchall())
    if not rows:
        return []
    ids = [row["id"] for row in rows]
    placeholders = ",".join("?" for _ in ids)
    event_rows = list(conn.execute(
        f"SELECT * FROM task_events WHERE task_id IN ({placeholders}) ORDER BY id", ids
    ).fetchall())
    run_rows = list(conn.execute(
        f"SELECT * FROM task_runs WHERE task_id IN ({placeholders}) ORDER BY id", ids
    ).fetchall())
    events_by: dict[str, list[Any]] = {task_id: [] for task_id in ids}
    runs_by: dict[str, list[Any]] = {task_id: [] for task_id in ids}
    for event in event_rows:
        events_by.setdefault(event["task_id"], []).append(event)
    for run in run_rows:
        runs_by.setdefault(run["task_id"], []).append(run)

    found: list[WatchdogAlert] = []
    graphs = kb.task_graph_contexts(conn, ids)
    for task in rows:
        task_id = task["id"]
        for diagnostic in diagnostics.compute_task_diagnostics(
            task, events_by[task_id], runs_by[task_id], now=now, config=config, graph=graphs.get(task_id)
        ):
            if diagnostic.kind in _WATCHED_DIAGNOSTIC_KINDS:
                found.append(WatchdogAlert(task_id, diagnostic.kind, diagnostic.severity, diagnostic.detail))
        if task["status"] == "done" and _deployment_required(task) and not _activation_verified(events_by[task_id]):
            found.append(WatchdogAlert(
                task_id, "activation_missing", "error",
                "Deployment-required work is marked done but has no live activation evidence. "
                "Route a deployment/verification continuation; this watchdog will not restart or deploy it.",
            ))
    return found


def run_watchdog(conn, *, now: int | None = None, config: dict | None = None, retention_days: int = 30) -> WatchdogResult:
    """Detect and route new stale conditions, returning quiet empty results when healthy.

    ``conn`` is intentionally caller-supplied so cron/CLI and fixture tests use
    the same board connection.  Notification delivery remains asynchronous in
    the existing gateway notifier after the inserted event commits.
    """
    now = int(time.time() if now is None else now)
    config = config or {}
    conditions = _conditions(conn, now=now, config=config)
    active = {(alert.task_id, alert.kind): alert for alert in conditions}
    new_alerts: list[WatchdogAlert] = []
    with kb.write_txn(conn):
        stored = {
            (row["task_id"], row["kind"]): row
            for row in conn.execute("SELECT * FROM kanban_watchdog_alerts").fetchall()
        }
        for key, alert in active.items():
            previous = stored.get(key)
            if previous is None:
                conn.execute(
                    "INSERT INTO kanban_watchdog_alerts "
                    "(task_id, kind, first_seen_at, last_seen_at, notified_at, resolved_at) VALUES (?, ?, ?, ?, ?, NULL)",
                    (alert.task_id, alert.kind, now, now, now),
                )
                kb._append_event(conn, alert.task_id, "watchdog_alert", {
                    "kind": alert.kind, "severity": alert.severity, "detail": alert.detail,
                    "source": "kanban_watchdog", "detected_at": now,
                })
                new_alerts.append(alert)
            elif previous["resolved_at"] is not None:
                conn.execute(
                    "UPDATE kanban_watchdog_alerts SET first_seen_at=?, last_seen_at=?, notified_at=?, resolved_at=NULL "
                    "WHERE task_id=? AND kind=?",
                    (now, now, now, alert.task_id, alert.kind),
                )
                kb._append_event(conn, alert.task_id, "watchdog_alert", {
                    "kind": alert.kind, "severity": alert.severity, "detail": alert.detail,
                    "source": "kanban_watchdog", "detected_at": now,
                })
                new_alerts.append(alert)
            else:
                conn.execute(
                    "UPDATE kanban_watchdog_alerts SET last_seen_at=? WHERE task_id=? AND kind=?",
                    (now, alert.task_id, alert.kind),
                )

        resolved_count = 0
        for key, row in stored.items():
            if key not in active and row["resolved_at"] is None:
                cur = conn.execute(
                    "UPDATE kanban_watchdog_alerts SET resolved_at=? WHERE task_id=? AND kind=? AND resolved_at IS NULL",
                    (now, key[0], key[1]),
                )
                resolved_count += int(cur.rowcount or 0)
        cutoff = now - max(1, int(retention_days)) * 86400
        cur = conn.execute(
            "DELETE FROM kanban_watchdog_alerts WHERE resolved_at IS NOT NULL AND resolved_at < ?", (cutoff,)
        )
        pruned_count = int(cur.rowcount or 0)
    return WatchdogResult(new_alerts, resolved_count, pruned_count)
