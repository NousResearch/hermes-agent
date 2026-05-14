"""Read-only Kanban duration and throughput metrics.

This module powers ``hermes kanban metrics``. It intentionally does not use
``kanban_db.connect()`` because that helper initializes/migrates schemas and
sets WAL pragmas. Metrics must be safe against a live board, so this module
opens SQLite with ``mode=ro`` and only runs SELECT/PRAGMA reads.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import quote

TERMINAL_STATUSES = {"done", "archived"}
TERMINAL_EVENTS = {"completed", "archived", "operator_reconciled_done"}
ACTIVE_OUTCOMES = {"completed", "blocked", "crashed", "timed_out", "spawn_failed", "reclaimed", "gave_up", "failed"}
KNOWN_EVENT_KINDS = {
    "created", "claimed", "spawned", "completed", "blocked", "unblocked", "crashed",
    "timed_out", "reclaimed", "gave_up", "spawn_failed", "promoted", "linked",
    "archived", "heartbeat", "auto_remediated", "completion_blocked_failed_gate",
    "completion_blocked_hallucination", "operator_reconciled_done", "assigned",
    "reassigned", "reprioritized", "commented", "edited", "unlinked", "ready",
    "priority", "spawn_auto_blocked",
}
GATE_STAGES = {"review", "verification", "deploy", "live_verify"}
AGING_BUCKETS = [
    ("<15m", 0, 15 * 60),
    ("15-60m", 15 * 60, 60 * 60),
    ("1-4h", 60 * 60, 4 * 60 * 60),
    ("4-24h", 4 * 60 * 60, 24 * 60 * 60),
    (">24h", 24 * 60 * 60, None),
]
RESERVE_TEXT_RE = re.compile(r"\b(ci|test|tests|docs?|cleanup|clean-up|refactor|lint|flake|typing|maintenance)\b", re.I)


def _rowdict(row: sqlite3.Row) -> dict[str, Any]:
    return {k: row[k] for k in row.keys()}


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _duration(start: Optional[int], end: Optional[int]) -> Optional[int]:
    if start is None or end is None:
        return None
    return max(0, int(end) - int(start))


def _overlap_seconds(start: Optional[int], end: Optional[int], window_start: int, window_end: int) -> int:
    if start is None or end is None:
        return 0
    return max(0, min(int(end), window_end) - max(int(start), window_start))


def _truncate(text: Optional[str], limit: int = 96) -> str:
    s = (text or "").replace("\n", " ").strip()
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 1)].rstrip() + "…"


def _parse_json_maybe(raw: Any) -> tuple[Any, bool]:
    if raw in (None, ""):
        return None, True
    try:
        return json.loads(raw), True
    except Exception:
        return None, False


def _parse_window(value: str) -> int:
    s = str(value or "7d").strip().lower()
    m = re.fullmatch(r"(\d+)([smhdw])", s)
    if not m:
        raise ValueError("window must look like 24h, 7d, 30d, 2w, or seconds with s")
    n = int(m.group(1))
    unit = m.group(2)
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]
    return n * mult


def _percentile(values: Iterable[Optional[int]], q: float) -> Optional[int]:
    vals = sorted(int(v) for v in values if v is not None)
    if not vals:
        return None
    # Nearest-rank percentile, documented and deterministic.
    rank = max(1, math.ceil((q / 100.0) * len(vals)))
    return vals[min(len(vals) - 1, rank - 1)]


def _fmt_dur(seconds: Optional[int]) -> str:
    if seconds is None:
        return ""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    rem_m = minutes % 60
    if hours < 24:
        return f"{hours}h {rem_m}m" if rem_m else f"{hours}h"
    days = hours // 24
    rem_h = hours % 24
    return f"{days}d {rem_h}h" if rem_h else f"{days}d"


def _open_readonly(db_path: Path, immutable: bool = False) -> sqlite3.Connection:
    path = Path(db_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Kanban DB not found: {path}")
    uri = f"file:{quote(str(path))}?mode=ro"
    if immutable:
        uri += "&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    # Defensive: this should already be query_only in mode=ro, but make intent visible.
    conn.execute("PRAGMA query_only=ON")
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return set()


def _select_existing(conn: sqlite3.Connection, table: str, wanted: list[str]) -> list[dict[str, Any]]:
    cols = _table_columns(conn, table)
    if not cols:
        return []
    selected = [c for c in wanted if c in cols]
    if not selected:
        return []
    rows = [_rowdict(r) for r in conn.execute(f"SELECT {', '.join(selected)} FROM {table}")]
    for row in rows:
        for col in wanted:
            row.setdefault(col, None)
    return rows


def _infer_stage(task: dict[str, Any], parents: list[str], tasks_by_id: dict[str, dict[str, Any]]) -> tuple[str, str, list[str], list[str]]:
    title = (task.get("title") or "").lower()
    body = (task.get("body") or "").lower()
    assignee = (task.get("assignee") or "").lower()
    skills_raw = task.get("skills") or ""
    skills = skills_raw.lower() if isinstance(skills_raw, str) else json.dumps(skills_raw).lower()
    evidence: list[str] = []
    assumptions: list[str] = []

    def hit(stage: str, confidence: str, why: str) -> tuple[str, str, list[str], list[str]]:
        evidence.append(why)
        return stage, confidence, evidence, assumptions

    if any(tok in title for tok in ["remediate", "follow-up", "fix review findings", "re-review", "re-verify"]):
        return hit("remediation", "high", "title signals remediation/rework")
    if any(tok in title for tok in ["live verify", "production verify", "post-deploy", "go/no-go"]):
        return hit("live_verify", "high", "title signals live verification")
    if assignee == "pm" or any(tok in title for tok in ["spec", "plan", "synthesize", "triage", "strategy"]):
        return hit("spec/planning", "high", "assignee/title signals planning")
    if assignee in {"reviewer", "security-reviewer"} or any(tok in title for tok in ["review", "security review", "code review"]):
        return hit("review", "high", "assignee/title signals review")
    if assignee == "verifier" or any(tok in title for tok in ["verify", "qa", "smoke", "acceptance"]):
        return hit("verification", "high", "assignee/title signals verification")
    if assignee == "ops" or any(tok in title for tok in ["deploy", "release", "rollout", "ship"]):
        return hit("deploy", "high", "assignee/title signals deploy")
    if assignee in {"backend-eng", "frontend-eng", "coder"} or any(tok in title for tok in ["implement", "build", "fix", "remediate"]):
        return hit("implementation", "high", "assignee/title signals implementation")
    if "review" in skills:
        return hit("review", "medium", "skills mention review")
    if "acceptance" in body or "pass/fail" in body:
        return hit("verification", "medium", "body mentions gate/testing")
    if parents:
        parent_stages = []
        for pid in parents:
            p = tasks_by_id.get(pid) or {}
            if (p.get("assignee") or "").lower() == "pm" or "spec" in (p.get("title") or "").lower():
                parent_stages.append("spec/planning")
        if parent_stages:
            assumptions.append("implementation_inferred_from_planning_parent")
            return "implementation", "medium", ["parent appears to be planning/spec"], assumptions
    assumptions.append("stage_unknown_no_strong_signal")
    return "unknown", "low", [], assumptions


@dataclass
class LoadedBoard:
    tasks: list[dict[str, Any]]
    links: list[dict[str, Any]]
    events: list[dict[str, Any]]
    runs: list[dict[str, Any]]
    unknown_event_kinds: Counter = field(default_factory=Counter)
    payload_parse_errors: int = 0


def _load_board(conn: sqlite3.Connection) -> LoadedBoard:
    tasks = _select_existing(conn, "tasks", [
        "id", "title", "body", "assignee", "status", "priority", "created_by", "created_at",
        "started_at", "completed_at", "workspace_kind", "workspace_path", "tenant", "result",
        "current_run_id", "skills", "max_runtime_seconds",
    ])
    links = _select_existing(conn, "task_links", ["parent_id", "child_id"])
    events = _select_existing(conn, "task_events", ["id", "task_id", "run_id", "kind", "payload", "created_at"])
    runs = _select_existing(conn, "task_runs", [
        "id", "task_id", "profile", "step_key", "status", "started_at", "ended_at", "outcome",
        "summary", "metadata", "error", "max_runtime_seconds",
    ])
    unknown: Counter = Counter()
    payload_errors = 0
    for ev in events:
        kind = ev.get("kind")
        if kind not in KNOWN_EVENT_KINDS:
            unknown[kind or ""] += 1
        payload, ok = _parse_json_maybe(ev.get("payload"))
        ev["payload_obj"] = payload
        if not ok:
            payload_errors += 1
    return LoadedBoard(tasks, links, events, runs, unknown, payload_errors)


def _active_from_runs(task_id: str, runs_by_task: dict[str, list[dict[str, Any]]], now: int) -> tuple[dict[str, int], dict[str, int], bool]:
    durations = defaultdict(int)
    run_counts = Counter()
    open_interval = False
    for run in runs_by_task.get(task_id, []):
        start = _coerce_int(run.get("started_at"))
        end = _coerce_int(run.get("ended_at")) or (now if run.get("status") == "running" else None)
        if start is None:
            continue
        if run.get("status") == "running" and run.get("ended_at") is None:
            open_interval = True
        sec = _duration(start, end) or 0
        outcome = (run.get("outcome") or run.get("status") or "running").lower()
        run_counts[outcome] += 1
        run_counts["total"] += 1
        durations["active_seconds_total"] += sec
        if outcome == "completed" or run.get("status") == "done":
            durations["active_completed_seconds"] += sec
        elif outcome == "blocked":
            durations["active_blocked_attempt_seconds"] += sec
        elif outcome == "crashed":
            durations["active_crashed_seconds"] += sec
        elif outcome == "timed_out":
            durations["active_timed_out_seconds"] += sec
        elif outcome == "reclaimed":
            durations["active_reclaimed_seconds"] += sec
        elif outcome in {"spawn_failed", "gave_up", "failed"}:
            durations["active_failed_seconds"] += sec
        elif outcome == "running":
            durations["active_running_seconds"] += sec
    for key in ["completed", "blocked", "crashed", "timed_out", "spawn_failed", "reclaimed", "gave_up", "failed", "running", "total"]:
        run_counts.setdefault(key, 0)
    return dict(durations), dict(run_counts), open_interval


def _blocked_intervals(task_id: str, events_by_task: dict[str, list[dict[str, Any]]], status: str, now: int) -> tuple[int, list[dict[str, Any]], bool]:
    evs = sorted(events_by_task.get(task_id, []), key=lambda e: (_coerce_int(e.get("created_at")) or 0, e.get("id") or 0))
    intervals: list[dict[str, Any]] = []
    start: Optional[int] = None
    open_interval = False
    for ev in evs:
        kind = ev.get("kind")
        ts = _coerce_int(ev.get("created_at"))
        if ts is None:
            continue
        if kind == "blocked" and start is None:
            start = ts
        elif kind in {"unblocked", "claimed", "completed", "archived"} and start is not None:
            intervals.append({"start": start, "end": ts, "seconds": _duration(start, ts) or 0, "open_interval": False})
            start = None
    if start is not None:
        intervals.append({"start": start, "end": now, "seconds": _duration(start, now) or 0, "open_interval": True})
        open_interval = True
    elif status == "blocked" and not intervals:
        # Legacy rows may lack blocked events; fall back to created_at handled by caller if needed.
        pass
    return sum(i["seconds"] for i in intervals), intervals, open_interval


def _first_claim(task: dict[str, Any], events: list[dict[str, Any]], runs: list[dict[str, Any]]) -> Optional[int]:
    candidates = []
    if task.get("started_at"):
        candidates.append(_coerce_int(task.get("started_at")))
    candidates.extend(_coerce_int(r.get("started_at")) for r in runs if r.get("started_at") is not None)
    candidates.extend(_coerce_int(e.get("created_at")) for e in events if e.get("kind") == "claimed")
    vals = [c for c in candidates if c is not None]
    return min(vals) if vals else None


def _terminal_at(task: dict[str, Any], events: list[dict[str, Any]], runs: list[dict[str, Any]]) -> Optional[int]:
    if task.get("completed_at"):
        return _coerce_int(task.get("completed_at"))
    vals = [_coerce_int(e.get("created_at")) for e in events if e.get("kind") in TERMINAL_EVENTS]
    vals += [_coerce_int(r.get("ended_at")) for r in runs if (r.get("outcome") == "completed" or r.get("status") == "done")]
    vals = [v for v in vals if v is not None]
    if vals:
        return max(vals)
    return None


def _weak_components(task_ids: list[str], links: list[dict[str, Any]]) -> list[list[str]]:
    adj: dict[str, set[str]] = {tid: set() for tid in task_ids}
    for link in links:
        p, c = link.get("parent_id"), link.get("child_id")
        if p in adj and c in adj:
            adj[p].add(c)
            adj[c].add(p)
    seen: set[str] = set()
    comps: list[list[str]] = []
    for tid in sorted(task_ids):
        if tid in seen:
            continue
        q = deque([tid]); seen.add(tid); comp = []
        while q:
            cur = q.popleft(); comp.append(cur)
            for nxt in sorted(adj[cur]):
                if nxt not in seen:
                    seen.add(nxt); q.append(nxt)
        comps.append(sorted(comp))
    return comps


def compute_metrics(
    db_path: Path,
    *,
    window_seconds: int = 7 * 86400,
    generated_at: Optional[int] = None,
    immutable: bool = False,
    reserve_priority_threshold: int = 0,
    reserve_default_minutes: int = 15,
) -> dict[str, Any]:
    now = int(generated_at or time.time())
    window_start = now - int(window_seconds)
    with _open_readonly(db_path, immutable=immutable) as conn:
        board = _load_board(conn)

    tasks_by_id = {str(t["id"]): t for t in board.tasks}
    parent_map: dict[str, list[str]] = defaultdict(list)
    child_map: dict[str, list[str]] = defaultdict(list)
    for link in board.links:
        p, c = link.get("parent_id"), link.get("child_id")
        if p in tasks_by_id and c in tasks_by_id:
            parent_map[c].append(p)
            child_map[p].append(c)
    events_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in board.events:
        events_by_task[str(ev.get("task_id"))].append(ev)
    runs_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in board.runs:
        runs_by_task[str(run.get("task_id"))].append(run)

    task_metrics_by_id: dict[str, dict[str, Any]] = {}
    assumptions: list[str] = ["percentiles_use_nearest_rank"]
    unknowns: list[str] = []
    if board.payload_parse_errors:
        unknowns.append(f"payload_parse_errors={board.payload_parse_errors}")
    if board.unknown_event_kinds:
        unknowns.append("unknown_event_kinds_present")

    # Terminal timestamps first for dependency readiness.
    terminal_map = {tid: _terminal_at(task, events_by_task.get(tid, []), runs_by_task.get(tid, [])) for tid, task in tasks_by_id.items()}

    for tid in sorted(tasks_by_id):
        task = tasks_by_id[tid]
        parents = sorted(parent_map.get(tid, []))
        children = sorted(child_map.get(tid, []))
        evs = events_by_task.get(tid, [])
        runs = runs_by_task.get(tid, [])
        stage, confidence, evidence, stage_assumptions = _infer_stage(task, parents, tasks_by_id)
        first_claimed_at = _first_claim(task, evs, runs)
        terminal = terminal_map.get(tid)
        created_at = _coerce_int(task.get("created_at")) or 0
        promoted_times = sorted(_coerce_int(e.get("created_at")) for e in evs if e.get("kind") == "promoted" and e.get("created_at") is not None)
        board_promoted_at = promoted_times[0] if promoted_times else None
        dependency_ready_at: Optional[int] = None
        dependency_wait_seconds = 0
        waiting_parent_ids = [p for p in parents if terminal_map.get(p) is None]
        if not parents:
            dependency_ready_at = created_at
        elif not waiting_parent_ids:
            dependency_ready_at = max(terminal_map[p] or created_at for p in parents)
            dependency_wait_seconds = max(0, dependency_ready_at - created_at)
        else:
            dependency_wait_seconds = max(0, now - created_at)
        ready_at = dependency_ready_at if dependency_ready_at is not None else None
        if board_promoted_at is not None and dependency_ready_at is not None:
            if abs(board_promoted_at - dependency_ready_at) > 1:
                stage_assumptions.append("board_promoted_at_differs_from_dependency_ready_at")
            ready_at = min(board_promoted_at, dependency_ready_at)
        elif board_promoted_at is not None:
            ready_at = board_promoted_at
        created_to_first = _duration(created_at, first_claimed_at)
        ready_to_first = _duration(ready_at, first_claimed_at) if ready_at is not None else None
        active_durs, run_counts, active_open = _active_from_runs(tid, runs_by_task, now)
        active_assumptions = []
        if not runs and first_claimed_at is not None:
            terminal_events = [e for e in evs if e.get("kind") in {"completed", "blocked", "crashed", "timed_out", "reclaimed", "gave_up", "archived"}]
            if terminal_events:
                end = min(_coerce_int(e.get("created_at")) for e in terminal_events if e.get("created_at") is not None)
                active_durs["active_seconds_total"] = _duration(first_claimed_at, end) or 0
                active_assumptions.append("active_time_from_events_fallback")
        blocked_total, blocked_intervals, blocked_open = _blocked_intervals(tid, events_by_task, task.get("status") or "", now)
        if task.get("status") == "blocked" and blocked_total == 0:
            fallback_start = first_claimed_at or created_at
            blocked_total = _duration(fallback_start, now) or 0
            blocked_intervals.append({"start": fallback_start, "end": now, "seconds": blocked_total, "open_interval": True, "assumption": "blocked_status_without_event"})
            blocked_open = True
            active_assumptions.append("blocked_status_without_event")
        ready_queue_age = 0
        if first_claimed_at is None and ready_at is not None and not waiting_parent_ids and task.get("status") in {"todo", "ready"}:
            ready_queue_age = max(0, now - ready_at)
        active_open_age = 0
        if task.get("status") == "running" and first_claimed_at is not None:
            active_open_age = max(0, now - first_claimed_at)
        open_age = 0 if task.get("status") in TERMINAL_STATUSES else max(0, now - created_at)
        is_rework = stage == "remediation" or any(e.get("kind") in {"auto_remediated", "completion_blocked_failed_gate", "completion_blocked_hallucination"} for e in evs)
        durations = {
            "created_to_first_claim_seconds": created_to_first,
            "ready_to_first_claim_seconds": ready_to_first,
            "active_seconds_total": active_durs.get("active_seconds_total", 0),
            "active_completed_seconds": active_durs.get("active_completed_seconds", 0),
            "active_blocked_attempt_seconds": active_durs.get("active_blocked_attempt_seconds", 0),
            "active_crashed_seconds": active_durs.get("active_crashed_seconds", 0),
            "active_timed_out_seconds": active_durs.get("active_timed_out_seconds", 0),
            "active_reclaimed_seconds": active_durs.get("active_reclaimed_seconds", 0),
            "active_failed_seconds": active_durs.get("active_failed_seconds", 0),
            "blocked_seconds_total": blocked_total,
            "dependency_wait_seconds": dependency_wait_seconds,
            "ready_queue_age_seconds": ready_queue_age,
            "active_open_age_seconds": active_open_age,
            "rework_active_seconds": active_durs.get("active_seconds_total", 0) if is_rework else 0,
            "rework_wait_seconds": ((ready_to_first or 0) + dependency_wait_seconds) if is_rework else 0,
            "open_age_seconds": open_age,
        }
        metric = {
            "task_id": tid,
            "title": _truncate(task.get("title")),
            "assignee": task.get("assignee"),
            "status": task.get("status"),
            "priority": task.get("priority") or 0,
            "created_by": task.get("created_by"),
            "stage": stage,
            "stage_confidence": confidence,
            "stage_evidence": evidence,
            "parents": parents,
            "children": children,
            "created_at": created_at,
            "started_at": _coerce_int(task.get("started_at")),
            "ready_at": ready_at,
            "dependency_ready_at": dependency_ready_at,
            "board_promoted_at": board_promoted_at,
            "first_claimed_at": first_claimed_at,
            "terminal_at": terminal,
            "durations": durations,
            "run_counts": run_counts,
            "waiting_parent_ids": sorted(waiting_parent_ids),
            "blocked_intervals": blocked_intervals[:5],
            "open_interval": bool(active_open or blocked_open or task.get("status") not in TERMINAL_STATUSES),
            "rework": bool(is_rework),
            "assumptions": sorted(set(stage_assumptions + active_assumptions)),
        }
        task_metrics_by_id[tid] = metric

    graphs: list[dict[str, Any]] = []
    for comp in _weak_components(sorted(tasks_by_id), board.links):
        roots = [tid for tid in comp if not set(parent_map.get(tid, [])) & set(comp)]
        leaves = [tid for tid in comp if not set(child_map.get(tid, [])) & set(comp)]
        first_root_created = min((task_metrics_by_id[t]["created_at"] for t in roots), default=None)
        all_leaves_terminal = all(task_metrics_by_id[t]["status"] in TERMINAL_STATUSES and task_metrics_by_id[t]["terminal_at"] for t in leaves)
        last_terminal = max((task_metrics_by_id[t]["terminal_at"] or 0 for t in leaves), default=0) if all_leaves_terminal else None
        cycle_end = last_terminal if last_terminal is not None else now
        status_counts = Counter(task_metrics_by_id[t]["status"] for t in comp)
        stages = sorted(set(task_metrics_by_id[t]["stage"] for t in comp))
        assignees = sorted(set(a for a in (task_metrics_by_id[t]["assignee"] for t in comp) if a))
        open_candidates = [task_metrics_by_id[t] for t in comp if task_metrics_by_id[t]["status"] not in TERMINAL_STATUSES]
        bottleneck = None
        if open_candidates:
            oldest = max(open_candidates, key=lambda m: m["durations"].get("open_age_seconds") or 0)
            bottleneck = {"task_id": oldest["task_id"], "stage": oldest["stage"], "open_age_seconds": oldest["durations"].get("open_age_seconds", 0)}
        graphs.append({
            "graph_id": sorted(roots)[0] if roots else comp[0],
            "task_ids": comp,
            "root_task_ids": sorted(roots),
            "leaf_task_ids": sorted(leaves),
            "task_count": len(comp),
            "status_counts": dict(sorted(status_counts.items())),
            "assignees": assignees,
            "stages_present": stages,
            "first_root_created_at": first_root_created,
            "last_terminal_at": last_terminal,
            "cycle_time_seconds": _duration(first_root_created, cycle_end),
            "critical_path_seconds": _duration(first_root_created, cycle_end),
            "critical_path_assumption": "approximated_as_component_cycle_time",
            "open_interval": not all_leaves_terminal,
            "blocked_seconds_total": sum(task_metrics_by_id[t]["durations"]["blocked_seconds_total"] for t in comp),
            "active_seconds_total": sum(task_metrics_by_id[t]["durations"]["active_seconds_total"] for t in comp),
            "dependency_wait_seconds_total": sum(task_metrics_by_id[t]["durations"]["dependency_wait_seconds"] for t in comp),
            "ready_queue_seconds_total": sum((task_metrics_by_id[t]["durations"].get("ready_to_first_claim_seconds") or task_metrics_by_id[t]["durations"].get("ready_queue_age_seconds") or 0) for t in comp),
            "rework_seconds_total": sum(task_metrics_by_id[t]["durations"].get("rework_active_seconds", 0) + task_metrics_by_id[t]["durations"].get("rework_wait_seconds", 0) for t in comp),
            "current_bottleneck": bottleneck,
        })
    graphs.sort(key=lambda g: g["graph_id"])

    tasks = [task_metrics_by_id[tid] for tid in sorted(task_metrics_by_id)]
    recent_tasks = [t for t in tasks if (t["terminal_at"] or t["created_at"] or now) >= window_start or t["status"] not in TERMINAL_STATUSES]

    def group_summary(key: str, vals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for t in vals:
            groups[str(t.get(key) or "(none)")].append(t)
        out = []
        for name in sorted(groups):
            rows = groups[name]
            active = [r["durations"]["active_seconds_total"] for r in rows]
            queue = [r["durations"].get("ready_to_first_claim_seconds") for r in rows]
            out.append({
                key: name,
                "count": len(rows),
                "completed_count": sum(1 for r in rows if r["status"] in TERMINAL_STATUSES),
                "p50_active_seconds": _percentile(active, 50),
                "p90_active_seconds": _percentile(active, 90),
                "max_active_seconds": max(active) if active else None,
                "p50_queue_seconds": _percentile(queue, 50),
                "p90_queue_seconds": _percentile(queue, 90),
                "blocked_seconds_total": sum(r["durations"]["blocked_seconds_total"] for r in rows),
                "open_running_count": sum(1 for r in rows if r["status"] == "running"),
                "oldest_open_age_seconds": max((r["durations"]["open_age_seconds"] for r in rows if r["status"] not in TERMINAL_STATUSES), default=0),
            })
        return out

    waiting_on_parents = [t for t in tasks if t["waiting_parent_ids"] and t["status"] in {"todo", "ready"}]
    ready_current = [t for t in tasks if t["status"] == "ready" and not t["waiting_parent_ids"]]
    blocked_current = [t for t in tasks if t["status"] == "blocked"]
    gates = [t for t in tasks if t["stage"] in GATE_STAGES and t["status"] in {"ready", "running", "blocked"}]
    gates.sort(key=lambda t: max(t["durations"].get("ready_queue_age_seconds") or 0, t["durations"].get("active_open_age_seconds") or 0, t["durations"].get("blocked_seconds_total") or 0), reverse=True)

    failed_gate_tasks = [t for t in tasks if t["stage"] in {"review", "verification"} and any(e.get("kind") in {"completion_blocked_failed_gate", "completion_blocked_hallucination"} for e in events_by_task.get(t["task_id"], []))]
    rework_tasks = [t for t in tasks if t["rework"]]
    rework_durations = [t["durations"].get("rework_active_seconds", 0) + t["durations"].get("rework_wait_seconds", 0) for t in rework_tasks]

    completed_tasks_in_window = [t for t in tasks if t["terminal_at"] and t["terminal_at"] >= window_start and t["status"] in TERMINAL_STATUSES]
    completed_graphs_in_window = [g for g in graphs if (not g["open_interval"] and g["last_terminal_at"] and g["last_terminal_at"] >= window_start)]
    days = max(window_seconds / 86400.0, 1 / 86400.0)

    wip_by_status = Counter(t["status"] for t in tasks if t["status"] not in TERMINAL_STATUSES)
    wip_by_assignee_stage: dict[str, Counter] = defaultdict(Counter)
    aging = {name: 0 for name, _, _ in AGING_BUCKETS}
    for t in tasks:
        if t["status"] in TERMINAL_STATUSES:
            continue
        wip_by_assignee_stage[str(t["assignee"] or "(none)")][t["stage"]] += 1
        age = t["durations"].get("open_age_seconds", 0) or 0
        for name, lo, hi in AGING_BUCKETS:
            if age >= lo and (hi is None or age < hi):
                aging[name] += 1
                break

    stage_p50 = {row["stage"]: row.get("p50_active_seconds") for row in group_summary("stage", recent_tasks)}
    reserve_candidates = []
    for t in ready_current:
        full_task = tasks_by_id[t["task_id"]]
        text = f"{full_task.get('title') or ''} {full_task.get('body') or ''}"
        if int(t.get("priority") or 0) <= reserve_priority_threshold and t["stage"] in {"implementation", "verification", "unknown"} and RESERVE_TEXT_RE.search(text):
            max_runtime = _coerce_int(full_task.get("max_runtime_seconds"))
            estimate_seconds = min(
                int(stage_p50.get(t["stage"]) or reserve_default_minutes * 60),
                int(max_runtime or reserve_default_minutes * 60),
            )
            reserve_candidates.append({"task_id": t["task_id"], "title": t["title"], "stage": t["stage"], "estimated_minutes": max(1, math.ceil(estimate_seconds / 60))})
    reserve_minutes = sum(c["estimated_minutes"] for c in reserve_candidates)
    blocks = reserve_minutes // 15
    coverage_status = "empty" if blocks == 0 else "thin" if blocks < 2 else "healthy" if blocks <= 6 else "overstocked"

    summaries = {
        "active_by_assignee": group_summary("assignee", recent_tasks),
        "active_by_stage": group_summary("stage", recent_tasks),
        "queue_by_assignee": group_summary("assignee", recent_tasks),
        "queue_by_stage": group_summary("stage", recent_tasks),
        "blocked": {
            "current_count": len(blocked_current),
            "current_tasks": [{"task_id": t["task_id"], "title": t["title"], "assignee": t["assignee"], "stage": t["stage"], "blocked_seconds": t["durations"]["blocked_seconds_total"]} for t in sorted(blocked_current, key=lambda x: x["durations"]["blocked_seconds_total"], reverse=True)[:10]],
            "interval_count": sum(1 for t in tasks for i in t.get("blocked_intervals", []) if _overlap_seconds(i.get("start"), i.get("end"), window_start, now) > 0),
            "total_blocked_seconds": sum(_overlap_seconds(i.get("start"), i.get("end"), window_start, now) for t in tasks for i in t.get("blocked_intervals", [])),
        },
        "dependency_wait": {
            "current_waiting_count": len(waiting_on_parents),
            "current_waiting_tasks": [{"task_id": t["task_id"], "title": t["title"], "waiting_parent_ids": t["waiting_parent_ids"], "dependency_wait_seconds": t["durations"]["dependency_wait_seconds"]} for t in sorted(waiting_on_parents, key=lambda x: x["durations"]["dependency_wait_seconds"], reverse=True)[:10]],
            "p50_dependency_wait_seconds": _percentile((t["durations"]["dependency_wait_seconds"] for t in recent_tasks), 50),
            "p90_dependency_wait_seconds": _percentile((t["durations"]["dependency_wait_seconds"] for t in recent_tasks), 90),
        },
        "rework": {
            "failed_gate_count": len(failed_gate_tasks),
            "remediation_task_count": len(rework_tasks),
            "remediation_loop_count": len(failed_gate_tasks) if failed_gate_tasks else len(rework_tasks),
            "open_loop_count": sum(1 for t in rework_tasks if t["status"] not in TERMINAL_STATUSES),
            "p50_rework_seconds": _percentile(rework_durations, 50),
            "p90_rework_seconds": _percentile(rework_durations, 90),
            "open_rework_tasks": [{"task_id": t["task_id"], "title": t["title"], "stage": t["stage"], "open_age_seconds": t["durations"]["open_age_seconds"]} for t in rework_tasks if t["status"] not in TERMINAL_STATUSES][:10],
        },
        "graph_cycle": {
            "completed_graph_count": sum(1 for g in graphs if not g["open_interval"]),
            "open_graph_count": sum(1 for g in graphs if g["open_interval"]),
            "p50_completed_cycle_seconds": _percentile((g["cycle_time_seconds"] for g in graphs if not g["open_interval"]), 50),
            "p90_completed_cycle_seconds": _percentile((g["cycle_time_seconds"] for g in graphs if not g["open_interval"]), 90),
            "oldest_open_graphs": sorted(({"graph_id": g["graph_id"], "task_count": g["task_count"], "cycle_time_seconds": g["cycle_time_seconds"], "current_bottleneck": g["current_bottleneck"]} for g in graphs if g["open_interval"]), key=lambda x: x["cycle_time_seconds"] or 0, reverse=True)[:10],
        },
        "throughput": {
            "tasks_completed_per_day": round(len(completed_tasks_in_window) / days, 3),
            "graphs_completed_per_day": round(len(completed_graphs_in_window) / days, 3),
            "completed_task_count": len(completed_tasks_in_window),
            "completed_graph_count": len(completed_graphs_in_window),
        },
        "wip": {
            "counts_by_status": dict(sorted(wip_by_status.items())),
            "counts_by_assignee_stage": {a: dict(sorted(c.items())) for a, c in sorted(wip_by_assignee_stage.items())},
            "aging_buckets": aging,
            "ready_queue_count": len(ready_current),
            "todo_waiting_on_parents_count": len(waiting_on_parents),
        },
        "reserve_backlog": {
            "reserve_priority_threshold": reserve_priority_threshold,
            "reserve_backlog_ready_count": len(reserve_candidates),
            "reserve_backlog_estimated_minutes": reserve_minutes,
            "coverage_15m_blocks": blocks,
            "coverage_status": coverage_status,
            "candidates": reserve_candidates[:10],
        },
    }
    oldest_waiting_gates = [{
        "task_id": t["task_id"],
        "title": t["title"],
        "assignee": t["assignee"],
        "stage": t["stage"],
        "status": t["status"],
        "ready_queue_age_seconds": t["durations"].get("ready_queue_age_seconds", 0),
        "active_open_age_seconds": t["durations"].get("active_open_age_seconds", 0),
        "blocked_seconds_total": t["durations"].get("blocked_seconds_total", 0),
    } for t in gates[:10]]

    recommendations = []
    if blocked_current:
        recommendations.append("Escalate oldest blocked gates before seeding new implementation work.")
    if summaries["reserve_backlog"]["coverage_status"] in {"empty", "thin"}:
        recommendations.append("Reserve backlog is thin; seed low-priority CI/test/docs cleanup only after high-priority Zach work is covered.")
    if summaries["rework"]["open_loop_count"]:
        recommendations.append("Open rework loops exist; prefer remediation/re-review clarity over new feature starts.")
    if not recommendations:
        recommendations.append("No immediate Kanban flow alarm detected in this window.")

    result = {
        "generated_at": now,
        "db_path": str(Path(db_path).expanduser()),
        "window": {"seconds": window_seconds, "start": window_start, "end": now},
        "assumptions": assumptions,
        "unknowns": unknowns,
        "unknown_event_kinds": dict(sorted(board.unknown_event_kinds.items())),
        "tasks": tasks,
        "graphs": graphs,
        "summaries": summaries,
        "oldest_waiting_gates": oldest_waiting_gates,
        "shepherd_recommendations": recommendations,
    }
    return result


def render_markdown(metrics: dict[str, Any]) -> str:
    window = metrics["window"]
    lines = [
        f"# Kanban metrics ({window['seconds']}s window, generated_at={metrics['generated_at']})",
        "",
        "## Executive summary",
    ]
    throughput = metrics["summaries"]["throughput"]
    wip = metrics["summaries"]["wip"]
    reserve = metrics["summaries"]["reserve_backlog"]
    rework = metrics["summaries"]["rework"]
    blocked = metrics["summaries"]["blocked"]
    lines += [
        f"- Tasks completed/day: {throughput['tasks_completed_per_day']} ({throughput['completed_task_count']} completed tasks in window).",
        f"- Graphs completed/day: {throughput['graphs_completed_per_day']} ({throughput['completed_graph_count']} completed graphs in window).",
        f"- Current WIP: {sum(wip['counts_by_status'].values())} open tasks; statuses {wip['counts_by_status']}.",
        f"- Blocked: {blocked['current_count']} current tasks; {_fmt_dur(blocked['total_blocked_seconds'])} blocked time in window.",
        f"- Rework: {rework['remediation_loop_count']} remediation/rework tasks; {rework['open_loop_count']} open.",
        f"- Reserve coverage: {reserve['coverage_status']} ({reserve['coverage_15m_blocks']} x 15m blocks, {reserve['reserve_backlog_ready_count']} ready candidates).",
    ]
    if metrics.get("unknown_event_kinds"):
        lines.append(f"- Unknown event kinds observed: {metrics['unknown_event_kinds']}.")
    lines += ["", "## Active by assignee", "| Assignee | Count | Done | p50 active | p90 active | Oldest open |", "|---|---:|---:|---:|---:|---:|"]
    for r in metrics["summaries"]["active_by_assignee"]:
        lines.append(f"| {r['assignee']} | {r['count']} | {r['completed_count']} | {_fmt_dur(r['p50_active_seconds'])} | {_fmt_dur(r['p90_active_seconds'])} | {_fmt_dur(r['oldest_open_age_seconds'])} |")
    lines += ["", "## Active by stage", "| Stage | Count | p50 active | p90 active | p50 queue | p90 queue | blocked |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in metrics["summaries"]["active_by_stage"]:
        lines.append(f"| {r['stage']} | {r['count']} | {_fmt_dur(r['p50_active_seconds'])} | {_fmt_dur(r['p90_active_seconds'])} | {_fmt_dur(r['p50_queue_seconds'])} | {_fmt_dur(r['p90_queue_seconds'])} | {_fmt_dur(r['blocked_seconds_total'])} |")
    lines += ["", "## WIP aging", "| Bucket | Count |", "|---|---:|"]
    for bucket, count in wip["aging_buckets"].items():
        lines.append(f"| {bucket} | {count} |")
    lines += ["", "## Oldest waiting gates", "| Task | Stage | Status | Assignee | Age |", "|---|---|---|---|---:|"]
    for g in metrics["oldest_waiting_gates"]:
        age = max(g.get("ready_queue_age_seconds") or 0, g.get("active_open_age_seconds") or 0, g.get("blocked_seconds_total") or 0)
        lines.append(f"| {g['task_id']} {g['title']} | {g['stage']} | {g['status']} | {g.get('assignee') or ''} | {_fmt_dur(age)} |")
    lines += ["", "## Reserve coverage", "| Ready candidates | Estimated minutes | 15m blocks | Status |", "|---:|---:|---:|---|"]
    lines.append(f"| {reserve['reserve_backlog_ready_count']} | {reserve['reserve_backlog_estimated_minutes']} | {reserve['coverage_15m_blocks']} | {reserve['coverage_status']} |")
    lines += ["", "## Shepherd recommendations"]
    lines += [f"- {rec}" for rec in metrics.get("shepherd_recommendations", [])]
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute read-only Kanban duration/throughput metrics")
    p.add_argument("--db", required=True, help="Path to kanban SQLite DB; opened read-only with mode=ro")
    p.add_argument("--window", default="7d", help="Recent window, e.g. 24h, 7d, 30d (default: 7d)")
    p.add_argument("--format", choices=["json", "markdown", "both"], default="both")
    p.add_argument("--json-out", default=None, help="Write JSON to this file instead of stdout")
    p.add_argument("--markdown-out", default=None, help="Write markdown to this file instead of stdout")
    p.add_argument("--generated-at", type=int, default=None, help="Fixed Unix timestamp for deterministic tests/snapshots")
    p.add_argument("--immutable", action="store_true", help="Open with SQLite immutable=1, appropriate for copied fixtures")
    p.add_argument("--reserve-priority-threshold", type=int, default=0)
    p.add_argument("--reserve-default-minutes", type=int, default=15)
    return p


def run_metrics_command(args: argparse.Namespace) -> int:
    try:
        window_seconds = _parse_window(args.window)
        metrics = compute_metrics(
            Path(args.db),
            window_seconds=window_seconds,
            generated_at=args.generated_at,
            immutable=bool(getattr(args, "immutable", False)),
            reserve_priority_threshold=int(getattr(args, "reserve_priority_threshold", 0)),
            reserve_default_minutes=int(getattr(args, "reserve_default_minutes", 15)),
        )
    except Exception as exc:
        print(f"kanban metrics: {exc}")
        return 1

    json_text = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    md_text = render_markdown(metrics)
    fmt = args.format
    if fmt in {"json", "both"}:
        if args.json_out:
            Path(args.json_out).write_text(json_text, encoding="utf-8")
        else:
            print(json_text, end="")
    if fmt in {"markdown", "both"}:
        if args.markdown_out:
            Path(args.markdown_out).write_text(md_text, encoding="utf-8")
        else:
            if fmt == "both":
                print("---")
            print(md_text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_metrics_command(build_arg_parser().parse_args()))
