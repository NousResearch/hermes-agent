"""Operator visibility CLI — query tasks and structured logs.

Milestone C: gives operators (and Hermes itself) a fast way to inspect
what the agent has been doing, how it's performing, and where things went wrong.

Usage (standalone):
    python3 -m hermes_cli.ops tasks [--limit N] [--status STATUS] [--session SESSION]
    python3 -m hermes_cli.ops events [--limit N] [--task TASK_ID] [--type EVENT_TYPE]
    python3 -m hermes_cli.ops loops [--limit N]
    python3 -m hermes_cli.ops slow [--limit N] [--threshold MS]
    python3 -m hermes_cli.ops summary [--session SESSION]

Programmatic use (from Hermes tool calls or scripts):
    from hermes_cli.ops import list_tasks, get_task, recent_events, slow_tools, task_summary
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

def _hermes_home() -> Path:
    return Path(os.getenv("HERMES_HOME", str(Path.home() / ".hermes")))


def _db_path() -> Path:
    return _hermes_home() / "state.db"


def _log_path() -> Path:
    return _hermes_home() / "logs" / "structured.jsonl"


def _ensure_state_schema() -> None:
    """Run SessionDB schema reconciliation for existing state.db files."""
    db_path = _db_path()
    if not db_path.exists():
        return
    try:
        from hermes_state import SessionDB
        db = SessionDB(db_path=db_path)
        db.close()
    except Exception:
        # Query helpers below surface any remaining DB errors in-band.
        pass


# =============================================================================
# Task queries (state.db)
# =============================================================================

def list_tasks(
    limit: int = 20,
    status: Optional[str] = None,
    session_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return recent tasks from state.db, newest first.

    Args:
        limit: Max number of tasks to return.
        status: Filter by status ("running", "completed", "failed", "interrupted").
        session_id: Filter by session ID (partial match).

        Returns:
        List of task dicts with keys: task_id, session_id, status, model_used,
        current_step, started_at, updated_at, token_usage, checkpoint_data, error_info.
"""
    import sqlite3

    db_path = _db_path()
    if not db_path.exists():
        return []
    _ensure_state_schema()

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        clauses = []
        params: List[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if session_id:
            clauses.append("session_id LIKE ?")
            params.append(f"%{session_id}%")

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)

        rows = conn.execute(
            f"""
            SELECT task_id, session_id, status, model_used, current_step,
                   started_at, updated_at, token_usage, checkpoint_data, error_info
            FROM tasks
            {where}
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        conn.close()

        results = []
        for row in rows:
            d = dict(row)
            for field in ("token_usage", "checkpoint_data"):
                if d.get(field):
                    try:
                        d[field] = json.loads(d[field])
                    except Exception:
                        pass
            results.append(d)
        return results

    except Exception as e:
        return [{"error": str(e)}]


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    """Return a single task record by task_id."""
    # Use direct query for exact match
    import sqlite3
    db_path = _db_path()
    if not db_path.exists():
        return None
    _ensure_state_schema()
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
        conn.close()
        if not row:
            return None
        d = dict(row)
        for field in ("token_usage", "checkpoint_data"):
            if d.get(field):
                try:
                    d[field] = json.loads(d[field])
                except Exception:
                    pass
        return d
    except Exception:
        return None


# =============================================================================
# Structured log queries (~/.hermes/logs/structured.jsonl)
# =============================================================================

def recent_events(
    limit: int = 50,
    event_type: Optional[str] = None,
    task_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return recent structured log events, newest first.

    Args:
        limit: Max number of events.
        event_type: Filter by event type ("task_start", "task_end",
            "tool_call", "tool_result", "model_call", "loop_detected").
        task_id: Filter to a specific task.
    """
    log_path = _log_path()
    if not log_path.exists():
        return []

    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    events: List[Dict[str, Any]] = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if event_type and ev.get("event") != event_type:
            continue
        if task_id and ev.get("task_id") != task_id:
            continue
        events.append(ev)
        if len(events) >= limit:
            break

    return events


def _get_slow_threshold() -> float:
    """Read slow-tool threshold from config. Falls back to 3000ms if config unavailable."""
    try:
        from hermes_cli.config import load_config
        return float(load_config().get("observability", {}).get("slow_tool_threshold_ms", 3000))
    except Exception:
        return 3000.0


def slow_tools(
    limit: int = 10,
    threshold_ms: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Return the slowest tool_result events above threshold_ms.

    threshold_ms defaults to observability.slow_tool_threshold_ms from config
    (typically 3000ms). Pass an explicit value to override for a specific call.

    Useful for perf analysis: which tools are bottlenecking the agent?
    """
    if threshold_ms is None:
        threshold_ms = _get_slow_threshold()
    log_path = _log_path()
    if not log_path.exists():
        return []

    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    slow: List[Dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("event") != "tool_result":
            continue
        dur = ev.get("duration_ms") or 0
        if dur >= threshold_ms:
            slow.append(ev)

    slow.sort(key=lambda e: e.get("duration_ms", 0), reverse=True)
    return slow[:limit]


def loop_events(limit: int = 20) -> List[Dict[str, Any]]:
    """Return recent loop_detected events for debugging spinning agents."""
    return recent_events(limit=limit, event_type="loop_detected")


def task_summary(session_id: Optional[str] = None) -> Dict[str, Any]:
    """Return aggregate stats for recent tasks.

    Returns:
        {
            "total": N,
            "by_status": {"completed": N, "failed": N, "interrupted": N, "running": N},
            "by_error": {"context_length_cannot_compress": N, ...},
            "avg_tokens_in": N,
            "avg_tokens_out": N,
            "models_used": {"qwen3.5:9b": N, ...},
        }
    """
    tasks = list_tasks(limit=200, session_id=session_id)
    if not tasks or (len(tasks) == 1 and "error" in tasks[0]):
        return {"error": "No tasks found"}

    by_status: Dict[str, int] = {}
    by_error: Dict[str, int] = {}
    models: Dict[str, int] = {}
    tokens_in: List[int] = []
    tokens_out: List[int] = []

    for t in tasks:
        status = t.get("status") or "unknown"
        by_status[status] = by_status.get(status, 0) + 1

        err = t.get("error_info")
        if err:
            by_error[err] = by_error.get(err, 0) + 1

        model = t.get("model_used") or "unknown"
        models[model] = models.get(model, 0) + 1

        usage = t.get("token_usage")
        if isinstance(usage, dict):
            if usage.get("input"):
                tokens_in.append(int(usage["input"]))
            if usage.get("output"):
                tokens_out.append(int(usage["output"]))

    return {
        "total": len(tasks),
        "by_status": by_status,
        "by_error": by_error,
        "avg_tokens_in": int(sum(tokens_in) / len(tokens_in)) if tokens_in else 0,
        "avg_tokens_out": int(sum(tokens_out) / len(tokens_out)) if tokens_out else 0,
        "models_used": models,
    }


def recent_projects(
    limit: int = 8,
    source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return recent project-like session summaries for the dashboard.

    Hermes treats a titled session as the unit of work a human thinks of as a
    "project". This helper keeps the dashboard responsive by deriving a compact
    overview from the existing sessions/tasks tables rather than introducing a
    new schema.
    """
    db_path = _db_path()
    if not db_path.exists():
        return []

    try:
        from hermes_state import SessionDB

        db = SessionDB(db_path=db_path)
        try:
            sessions = db.list_sessions_rich(source=source, limit=max(limit, 50))
        finally:
            db.close()

        tasks = list_tasks(limit=500, session_id=None)
        aggregates: Dict[str, Dict[str, Any]] = {}

        for task in tasks:
            session_id = task.get("session_id")
            if not session_id:
                continue
            bucket = aggregates.setdefault(
                session_id,
                {
                    "task_count": 0,
                    "by_status": {},
                    "latest_task": None,
                },
            )
            bucket["task_count"] += 1

            status = task.get("status") or "unknown"
            bucket["by_status"][status] = bucket["by_status"].get(status, 0) + 1

            if bucket["latest_task"] is None:
                bucket["latest_task"] = {
                    "task_id": task.get("task_id"),
                    "status": task.get("status"),
                    "model_used": task.get("model_used"),
                    "current_step": task.get("current_step"),
                    "updated_at": task.get("updated_at"),
                    "error_info": task.get("error_info"),
                }

        projects: List[Dict[str, Any]] = []
        for session in sessions:
            session_id = session.get("id")
            bucket = aggregates.get(session_id, {"task_count": 0, "by_status": {}, "latest_task": None})
            latest_task = bucket["latest_task"] or {}
            title = session.get("title") or ""
            preview = session.get("preview") or ""
            display_name = title or (preview[:48] if preview else (session_id[:10] + "…" if session_id else "Untitled"))

            projects.append(
                {
                    "session_id": session_id,
                    "title": title,
                    "display_name": display_name,
                    "source": session.get("source"),
                    "started_at": session.get("started_at"),
                    "last_active": session.get("last_active"),
                    "message_count": session.get("message_count") or 0,
                    "tool_call_count": session.get("tool_call_count") or 0,
                    "preview": preview,
                    "task_count": bucket["task_count"],
                    "task_status_counts": bucket["by_status"],
                    "latest_task": latest_task,
                    "pinned_at": session.get("pinned_at"),
                    "is_pinned": bool(session.get("pinned_at")),
                }
            )

        return projects[:limit]

    except Exception as e:
        return [{"error": str(e)}]


def recent_chats(limit: int = 20, source: Optional[str] = None) -> Dict[str, Any]:
    """Return historical chat threads plus metadata for the dashboard."""
    db_path = _db_path()
    if not db_path.exists():
        return {"items": [], "meta": {"total_historical": 0, "visible_count": 0, "pinned_count": 0, "limit": limit}, "source": source}

    try:
        from hermes_state import SessionDB

        db = SessionDB(db_path=db_path)
        try:
            sessions = db.list_sessions_rich(source=source, ended_only=True, limit=max(limit * 4, 50))
            where_clauses = ["ended_at IS NOT NULL"]
            params: List[Any] = []
            if source:
                where_clauses.append("source = ?")
                params.append(source)
            where_sql = f"WHERE {' AND '.join(where_clauses)}"
            with db._lock:
                counts_row = db._conn.execute(
                    f"""
                    SELECT
                        COUNT(*) AS total_historical,
                        COALESCE(SUM(CASE WHEN pinned_at IS NOT NULL THEN 1 ELSE 0 END), 0) AS pinned_count
                    FROM sessions
                    {where_sql}
                    """,
                    params,
                ).fetchone()
        finally:
            db.close()

        chats: List[Dict[str, Any]] = []
        for session in sessions:
            session["session_id"] = session.get("id")
            session["is_pinned"] = bool(session.get("pinned_at"))
            session["display_name"] = session.get("title") or session.get("preview") or session.get("id")
            chats.append(session)

        chats.sort(
            key=lambda s: (
                0 if s.get("is_pinned") else 1,
                -(float(s.get("last_active") or s.get("started_at") or 0)),
            )
        )
        visible = chats[:limit]
        total_historical = int(counts_row["total_historical"] if counts_row else 0)
        pinned_count = int(counts_row["pinned_count"] if counts_row else 0)
        return {
            "items": visible,
            "meta": {
                "total_historical": total_historical,
                "visible_count": len(visible),
                "pinned_count": pinned_count,
                "limit": limit,
            },
            "source": source,
        }

    except Exception as e:
        return {
            "items": [],
            "meta": {"total_historical": 0, "visible_count": 0, "pinned_count": 0, "limit": limit},
            "source": source,
            "error": str(e),
        }


# =============================================================================
# Pretty-print helpers
# =============================================================================

def _fmt_ts(ts) -> str:
    if not ts:
        return "—"
    try:
        # timestamps may be Unix floats or "YYYY-MM-DD HH:MM:SS" strings
        if isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            return dt.strftime("%m-%d %H:%M:%S")
        dt = datetime.strptime(str(ts)[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return dt.strftime("%m-%d %H:%M:%S")
    except Exception:
        return str(ts)[:16]


def _status_icon(status: str) -> str:
    return {
        "completed": "✅",
        "failed": "❌",
        "interrupted": "⚡",
        "running": "🔄",
    }.get(status, "❓")


def _print_tasks(tasks: List[Dict[str, Any]]) -> None:
    if not tasks:
        print("No tasks found.")
        return
    if len(tasks) == 1 and "error" in tasks[0]:
        print(f"Error: {tasks[0]['error']}")
        return

    header = f"{'Icon':<4} {'Status':<12} {'Model':<22} {'Step':<20} {'Updated':<14} {'Error':<30} Task ID"
    print(header)
    print("─" * len(header))
    for t in tasks:
        status = t.get("status") or "?"
        icon = _status_icon(status)
        model = (t.get("model_used") or "—")[:21]
        step = (t.get("current_step") or "—")[:19]
        updated = _fmt_ts(t.get("updated_at"))
        err = (t.get("error_info") or "")[:29]
        task_id = (t.get("task_id") or "")[:36]
        print(f"{icon:<4} {status:<12} {model:<22} {step:<20} {updated:<14} {err:<30} {task_id}")


def _print_events(events: List[Dict[str, Any]]) -> None:
    if not events:
        print("No events found.")
        return
    for ev in events:
        ts = _fmt_ts(ev.get("ts"))
        etype = (ev.get("event") or "?")[:18]
        tool = (ev.get("tool_name") or ev.get("model") or "")[:22]
        extra = ""
        if ev.get("duration_ms"):
            extra = f" [{ev['duration_ms']:.0f}ms]"
        if ev.get("error"):
            extra += f" err={ev['error'][:40]}"
        if ev.get("count"):
            extra += f" ×{ev['count']}"
        task = (ev.get("task_id") or "")[:8]
        print(f"{ts}  {etype:<18}  {tool:<22}  [{task}]{extra}")


def _print_summary(s: Dict[str, Any]) -> None:
    if "error" in s:
        print(f"Summary error: {s['error']}")
        return
    print(f"Total tasks: {s['total']}")
    print()
    print("By status:")
    for status, count in sorted(s["by_status"].items(), key=lambda x: -x[1]):
        print(f"  {_status_icon(status)} {status}: {count}")
    if s.get("by_error"):
        print()
        print("Failure reasons:")
        for err, count in sorted(s["by_error"].items(), key=lambda x: -x[1]):
            print(f"  ❌ {err}: {count}")
    print()
    print(f"Avg tokens in : {s['avg_tokens_in']:,}")
    print(f"Avg tokens out: {s['avg_tokens_out']:,}")
    if s.get("models_used"):
        print()
        print("Models used:")
        for model, count in sorted(s["models_used"].items(), key=lambda x: -x[1]):
            print(f"  {model}: {count}")


def _print_task(task: Dict[str, Any]) -> None:
    if not task:
        print("Task not found.")
        return
    if "error" in task:
        print(f"Task error: {task['error']}")
        return
    print(f"Task ID     : {task.get('task_id')}")
    print(f"Session ID  : {task.get('session_id') or '—'}")
    print(f"Status      : {task.get('status') or '—'}")
    print(f"Model       : {task.get('model_used') or '—'}")
    print(f"Current step: {task.get('current_step') or '—'}")
    print(f"Started at  : {_fmt_ts(task.get('started_at'))}")
    print(f"Updated at  : {_fmt_ts(task.get('updated_at'))}")
    print(f"Error info  : {task.get('error_info') or '—'}")
    if task.get('checkpoint_data') is not None:
        print(f"Checkpoint  : {json.dumps(task['checkpoint_data'], indent=2, sort_keys=True)}")
    if task.get('token_usage') is not None:
        print(f"Token usage : {json.dumps(task['token_usage'], indent=2, sort_keys=True)}")
    if task.get('approvals_needed') is not None:
        print(f"Approvals   : {json.dumps(task['approvals_needed'], indent=2, sort_keys=True)}")
    if task.get('artifacts') is not None:
        print(f"Artifacts   : {json.dumps(task['artifacts'], indent=2, sort_keys=True)}")


def _fmt_command_ts(ts: Any) -> str:
    if not ts:
        return "—"
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)[:16]


def _print_eval_help() -> None:
    print("Usage:")
    print("  eval list")
    print("  eval run <suite-or-case>")
    print("  eval recent [limit]")
    print("  eval show <run_id>")


def _handle_eval(argv: List[str]) -> None:
    argv = list(argv or [])
    if not argv or argv[0] in {"help", "-h", "--help"}:
        _print_eval_help()
        return

    subcommand = argv[0]
    if subcommand == "list":
        from agent.evals.cases import list_cases, list_suites

        print("Suites:")
        for suite in list_suites():
            print(f"  {suite}")
        print("Cases:")
        for case in list_cases():
            print(f"  {case.id} - {getattr(case, 'name', '')}")
        return

    if subcommand == "run":
        if len(argv) < 2:
            print("Usage: eval run <suite-or-case>")
            return
        target = argv[1]
        from agent.evals.cases import list_suites
        from agent.evals import runner as eval_runner
        from agent.evals.storage import EvalStore

        try:
            summary = eval_runner.run_suite(target) if target in list_suites() else eval_runner.run_single(target)
        except KeyError:
            print(f"Unknown suite or case: {target}")
            return
        except Exception as exc:
            print(f"Eval run failed: {exc}")
            return

        store = EvalStore()
        try:
            store.save_run(summary)
        finally:
            try:
                store.close()
            except Exception:
                pass
        print(
            f"Eval run {summary.run_id}: {summary.suite_name} "
            f"{summary.passed_count}/{summary.case_count} passed, avg={summary.avg_score}"
        )
        return

    if subcommand == "recent":
        limit = 10
        if len(argv) > 1:
            try:
                limit = int(argv[1])
            except ValueError:
                print("Usage: eval recent [limit]")
                return
        from agent.evals.storage import EvalStore

        store = EvalStore()
        try:
            rows = store.list_runs(limit=limit)
        finally:
            try:
                store.close()
            except Exception:
                pass
        if not rows:
            print("No eval runs found.")
            return
        for row in rows:
            print(
                f"{row.get('id')}  {row.get('suite_name')}  "
                f"{row.get('passed_count', 0)}/{row.get('case_count', 0)}  "
                f"avg={row.get('avg_score', 0)}  {_fmt_command_ts(row.get('created_at'))}"
            )
        return

    if subcommand == "show":
        if len(argv) < 2:
            print("Usage: eval show <run_id>")
            return
        from agent.evals.storage import EvalStore

        store = EvalStore()
        try:
            run = store.get_run_with_results(argv[1])
        finally:
            try:
                store.close()
            except Exception:
                pass
        if not run:
            print(f"Eval run not found: {argv[1]}")
            return
        print(
            f"Eval run {run.get('id')}: {run.get('suite_name')} "
            f"{run.get('passed_count', 0)}/{run.get('case_count', 0)} passed, "
            f"avg={run.get('avg_score', 0)}"
        )
        for result in run.get("case_results") or []:
            status = result.get("status")
            status_value = getattr(status, "value", status)
            icon = "✓" if status_value == "passed" else "✗"
            failure = result.get("failure_summary") or ""
            suffix = f" — {failure}" if failure else ""
            print(
                f"  {icon} {result.get('case_id')} score={result.get('total_score', 0)} "
                f"{result.get('duration_ms', 0)}ms{suffix}"
            )
        return

    _print_eval_help()


def _print_failures_help() -> None:
    print("Usage:")
    print("  failures recent [limit]")
    print("  failures top [limit]")
    print("  failures show <fingerprint>")


def _handle_failures(argv: List[str]) -> None:
    argv = list(argv or [])
    if not argv or argv[0] in {"help", "-h", "--help"}:
        _print_failures_help()
        return

    subcommand = argv[0]
    from agent.failure_analysis.storage import FailureStore

    if subcommand == "recent":
        limit = 20
        if len(argv) > 1:
            try:
                limit = int(argv[1])
            except ValueError:
                print("Usage: failures recent [limit]")
                return
        store = FailureStore()
        try:
            rows = store.list_recent(limit=limit)
        finally:
            try:
                store.close()
            except Exception:
                pass
        if not rows:
            print("No failures found.")
            return
        for row in rows:
            kind = f"{row.get('failure_type')}.{row.get('failure_subtype')}"
            print(
                f"{row.get('id')}  {row.get('severity', 'medium')}  "
                f"{kind}  {row.get('summary', '')}  fp={str(row.get('fingerprint', ''))[:16]}"
            )
        return

    if subcommand == "top":
        limit = 10
        if len(argv) > 1:
            try:
                limit = int(argv[1])
            except ValueError:
                print("Usage: failures top [limit]")
                return
        store = FailureStore()
        try:
            rows = store.top_fingerprints(limit=limit)
        finally:
            try:
                store.close()
            except Exception:
                pass
        if not rows:
            print("No recurring failures found.")
            return
        for row in rows:
            kind = f"{row.get('failure_type')}.{row.get('failure_subtype')}"
            print(
                f"{row.get('count', 0)}x  {kind}  fp={row.get('fingerprint')}  {row.get('summary', '')}"
            )
        return

    if subcommand == "show":
        if len(argv) < 2:
            print("Usage: failures show <fingerprint>")
            return
        fingerprint = argv[1]
        store = FailureStore()
        try:
            rows = store.get_by_fingerprint(fingerprint)
        finally:
            try:
                store.close()
            except Exception:
                pass
        print(f"Fingerprint: {fingerprint}")
        print(f"Occurrences: {len(rows)}")
        for row in rows:
            kind = f"{row.get('failure_type')}.{row.get('failure_subtype')}"
            print(
                f"  {row.get('severity', 'medium')}  {kind}  {row.get('summary', '')}  "
                f"source={row.get('source_surface') or '—'} "
                f"eval={row.get('eval_run_id') or '—'} "
                f"session={row.get('session_id') or '—'}"
            )
        return

    _print_failures_help()


# =============================================================================
# CLI entry point
# =============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="hermes-ops",
        description="Hermes operator visibility — query tasks and structured logs",
    )
    sub = parser.add_subparsers(dest="command")

    # tasks subcommand
    p_tasks = sub.add_parser("tasks", help="List recent tasks from state.db")
    p_tasks.add_argument("--limit", type=int, default=20, help="Max tasks (default 20)")
    p_tasks.add_argument("--status", help="Filter by status (completed/failed/interrupted/running)")
    p_tasks.add_argument("--session", help="Filter by session ID (partial match)")
    p_tasks.add_argument("--json", action="store_true", dest="as_json", help="Output raw JSON")

    # task subcommand — inspect a single task in detail
    p_task = sub.add_parser("task", help="Inspect one task by task_id")
    p_task.add_argument("task_id", help="Task ID to inspect")
    p_task.add_argument("--json", action="store_true", dest="as_json", help="Output raw JSON")

    # events subcommand
    p_ev = sub.add_parser("events", help="Query structured log events")
    p_ev.add_argument("--limit", type=int, default=50, help="Max events (default 50)")
    p_ev.add_argument("--task", help="Filter by task ID")
    p_ev.add_argument("--type", dest="event_type", help="Filter by event type (tool_call, tool_result, model_call, loop_detected, ...)")
    p_ev.add_argument("--json", action="store_true", dest="as_json", help="Output raw JSON")

    # loops subcommand
    p_loops = sub.add_parser("loops", help="Show recent loop_detected events")
    p_loops.add_argument("--limit", type=int, default=20)
    p_loops.add_argument("--json", action="store_true", dest="as_json")

    # slow subcommand
    p_slow = sub.add_parser("slow", help="Show slowest tool executions")
    p_slow.add_argument("--limit", type=int, default=10)
    p_slow.add_argument("--threshold", type=float, default=None, help="Min ms to include (default: observability.slow_tool_threshold_ms from config)")
    p_slow.add_argument("--json", action="store_true", dest="as_json")

    # summary subcommand
    p_sum = sub.add_parser("summary", help="Show aggregate task stats")
    p_sum.add_argument("--session", help="Filter by session ID")
    p_sum.add_argument("--json", action="store_true", dest="as_json")

    # serve subcommand — starts the live web dashboard
    p_serve = sub.add_parser("serve", help="Start the live operator dashboard at http://localhost:7788")
    p_serve.add_argument("--port", type=int, default=7788, help="Port to listen on (default 7788)")
    p_serve.add_argument("--host", default="127.0.0.1", help="Host to bind (default 127.0.0.1)")

    # schedule subcommand — manage ~/.hermes/schedule.yaml
    p_sched = sub.add_parser("schedule", help="Manage scheduled Hermes tasks (Milestone H)")
    p_sched.add_argument("--status", action="store_true", help="Show status of all scheduled tasks")
    p_sched.add_argument("--once", action="store_true", help="Dispatch any due tasks and exit")
    p_sched.add_argument("--dry-run", action="store_true", help="Show what would run without executing")
    p_sched.add_argument("--daemon", action="store_true", help="Run as daemon, polling every --poll seconds")
    p_sched.add_argument("--poll", type=int, default=300, help="Daemon poll interval in seconds (default 300)")
    p_sched.add_argument("--json", action="store_true", dest="as_json", help="Output raw JSON (--status only)")

    # report subcommand — generate perf benchmarking reports
    p_report = sub.add_parser("report", help="Generate a performance benchmarking report (Milestone I)")
    p_report.add_argument("--days", type=int, default=7, help="Lookback window in days (default 7)")
    p_report.add_argument("--output", help="Output path (default: ~/.hermes/reports/perf-YYYY-MM-DD.md)")
    p_report.add_argument("--json", action="store_true", dest="as_json", help="Output raw JSON instead of markdown")

    p_review = sub.add_parser("review", help="Self-improvement review: analyse patterns and propose config fixes (Milestone K)")
    p_review.add_argument("--days", type=int, default=7, help="Analysis window in days (default 7)")
    p_review.add_argument("--apply", action="store_true", help="Apply safe proposals after interactive confirmation")
    p_review.add_argument("--json", action="store_true", dest="as_json", help="Output raw JSON instead of markdown")

    p_eval = sub.add_parser("eval", help="Run or inspect local Hermes behavioral evals")
    p_eval.add_argument("eval_args", nargs=argparse.REMAINDER, help="eval subcommand and arguments")

    p_failures = sub.add_parser("failures", help="Inspect normalized failure-analysis records")
    p_failures.add_argument("failure_args", nargs=argparse.REMAINDER, help="failure subcommand and arguments")

    args = parser.parse_args(argv)

    if args.command == "tasks":
        tasks = list_tasks(limit=args.limit, status=args.status, session_id=args.session)
        if args.as_json:
            print(json.dumps(tasks, indent=2))
        else:
            _print_tasks(tasks)

    elif args.command == "task":
        task = get_task(args.task_id)
        if args.as_json:
            print(json.dumps(task, indent=2) if task else "null")
        else:
            _print_task(task or {})

    elif args.command == "events":
        events = recent_events(limit=args.limit, event_type=args.event_type, task_id=args.task)
        if args.as_json:
            print(json.dumps(events, indent=2))
        else:
            _print_events(events)

    elif args.command == "loops":
        events = loop_events(limit=args.limit)
        if args.as_json:
            print(json.dumps(events, indent=2))
        else:
            print(f"Recent loop detections ({len(events)}):")
            _print_events(events)

    elif args.command == "slow":
        resolved_threshold = args.threshold if args.threshold is not None else _get_slow_threshold()
        events = slow_tools(limit=args.limit, threshold_ms=resolved_threshold)
        if args.as_json:
            print(json.dumps(events, indent=2))
        else:
            print(f"Slowest tool calls (≥{resolved_threshold:.0f}ms):")
            _print_events(events)

    elif args.command == "summary":
        s = task_summary(session_id=args.session)
        if args.as_json:
            print(json.dumps(s, indent=2))
        else:
            _print_summary(s)

    elif args.command == "serve":
        from hermes_cli.dashboard import serve
        serve(port=args.port, host=args.host)

    elif args.command == "schedule":
        from hermes_cli.scheduler import status as sched_status, run_once, run_daemon
        if args.status or (not args.once and not args.dry_run and not args.daemon):
            rows = sched_status()
            if args.as_json:
                print(json.dumps(rows, indent=2))
            else:
                if not rows:
                    print("No tasks in ~/.hermes/schedule.yaml")
                else:
                    print(f"{'ID':<20} {'Name':<30} {'State':<10} {'Every':<10} {'Next run'}")
                    print("-" * 80)
                    for r in rows:
                        print(f"{r['id']:<20} {r['name']:<30} {r['state']:<10} {r['interval_hours']}h{'':<4} {r['next_run'] or '—'}")
        elif args.once or args.dry_run:
            counts = run_once(dry_run=args.dry_run)
            print(f"Done — due: {counts['due']}, dispatched: {counts['dispatched']}, failed: {counts['failed']}")
        elif args.daemon:
            run_daemon(poll_seconds=args.poll)

    elif args.command == "report":
        from hermes_cli.report import generate_report
        report = generate_report(days=args.days, output_path=args.output)
        if args.as_json:
            print(json.dumps(report["data"], indent=2, default=str))
        else:
            print(report["markdown"])
            if report.get("saved_to"):
                print(f"\nSaved to: {report['saved_to']}")

    elif args.command == "review":
        from hermes_cli.self_review import analyze, propose, render_report, save_proposal, apply_proposals
        findings = analyze(days=args.days)
        proposals = propose(findings)
        if args.as_json:
            print(json.dumps({"findings": findings, "proposals": proposals}, indent=2, sort_keys=True, default=str))
        else:
            saved = save_proposal(proposals)
            print(render_report(findings, proposals))
            if saved:
                print(f"\nProposal saved to: {saved}")
        if args.apply:
            response = input(
                "\nApply proposed changes to ~/.hermes/config.yaml and ~/.hermes/schedule.yaml? [y/N]: "
            ).strip()
            if response.lower() in {"y", "yes"}:
                apply_proposals(proposals, dry_run=False)
            else:
                print("Aborted — no changes applied.")

    elif args.command == "eval":
        _handle_eval(args.eval_args)

    elif args.command == "failures":
        _handle_failures(args.failure_args)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
