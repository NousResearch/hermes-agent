"""CLI handler for /swarm commands."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from swarm.types import SwarmConfig, SwarmTask, SwarmResult, TaskState, WorkerState
from swarm.scheduler import SwarmScheduler
from swarm.worker import WorkerPool
from swarm.monitor import SwarmMonitor
from swarm.roles import list_roles, get_role, PLANNER


# ---------------------------------------------------------------------------
# Persistent history of swarm runs
# ---------------------------------------------------------------------------

_HISTORY_DIR = Path.home() / ".hermes" / "swarm" / "history"


def _save_run(run_id: str, goal: str, scheduler: SwarmScheduler, monitor: SwarmMonitor) -> Path:
    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    summary = monitor.get_summary()
    payload = {
        "run_id": run_id,
        "goal": goal,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "scheduler": scheduler.to_dict(),
        "summary": summary,
    }
    path = _HISTORY_DIR / f"{run_id}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def _load_history(limit: int = 10) -> list[dict[str, Any]]:
    if not _HISTORY_DIR.exists():
        return []
    files = sorted(_HISTORY_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    runs = []
    for f in files[:limit]:
        try:
            runs.append(json.loads(f.read_text()))
        except Exception:
            continue
    return runs


# ---------------------------------------------------------------------------
# Active swarm state (module-level singleton for the CLI session)
# ---------------------------------------------------------------------------

class _SwarmSession:
    """Holds the active swarm state for the current CLI session."""

    def __init__(self) -> None:
        self.scheduler: SwarmScheduler | None = None
        self.pool: WorkerPool | None = None
        self.monitor: SwarmMonitor | None = None
        self.config: SwarmConfig | None = None
        self.goal: str = ""
        self.run_id: str = ""
        self.running: bool = False
        self._thread: threading.Thread | None = None

    @property
    def active(self) -> bool:
        return self.scheduler is not None

    def reset(self) -> None:
        self.scheduler = None
        self.pool = None
        self.monitor = None
        self.config = None
        self.goal = ""
        self.run_id = ""
        self.running = False
        self._thread = None


_session = _SwarmSession()


# ---------------------------------------------------------------------------
# Task decomposition (simple heuristic — real planner would use LLM)
# ---------------------------------------------------------------------------

def _decompose_goal(goal: str) -> list[SwarmTask]:
    """Split a goal into tasks.

    This is a placeholder decomposition.  A production implementation would
    send the goal to a planner-role LLM and parse the structured output.
    For now we create a plan -> execute -> verify chain.
    """
    plan_task = SwarmTask(
        name="plan",
        prompt=f"Create a detailed plan to accomplish: {goal}",
        role="planner",
    )
    exec_task = SwarmTask(
        name="execute",
        prompt=f"Execute the plan for: {goal}",
        deps=[plan_task.id],
        role="executor",
    )
    verify_task = SwarmTask(
        name="verify",
        prompt=f"Verify the results for: {goal}",
        deps=[exec_task.id],
        role="verifier",
    )
    return [plan_task, exec_task, verify_task]


# ---------------------------------------------------------------------------
# Swarm execution loop
# ---------------------------------------------------------------------------

def _run_swarm_loop(session: _SwarmSession) -> None:
    """Background loop that schedules ready tasks onto available workers."""
    scheduler = session.scheduler
    pool = session.pool
    monitor = session.monitor

    while session.running and not scheduler.is_complete():
        ready = scheduler.get_ready_tasks()
        for task in ready:
            worker = pool.get_available_worker()
            if worker is None:
                break
            scheduler.mark_running(task.id, worker.id)
            monitor.record_event("task_started", task_id=task.id, worker_id=worker.id)
            monitor.record_task_transition(task.id, "pending", "running")
            pool.assign_task(worker.id, task)

        # Poll for completed tasks
        time.sleep(0.2)
        status = pool.get_status()
        for wid, winfo in status.get("workers", {}).items():
            if winfo["state"] == "idle" and winfo["task"] is None:
                continue
            # Task completion is handled inside WorkerPool.assign_task thread

    # Final bookkeeping
    session.running = False
    if monitor:
        monitor.record_event("swarm_finished")
        try:
            _save_run(session.run_id, session.goal, scheduler, monitor)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _cmd_run(args: str, cli_instance: Any) -> None:
    goal = args.strip()
    if not goal:
        print("Usage: /swarm run <goal>")
        print("  Example: /swarm run refactor the auth module into separate files")
        return

    if _session.running:
        print("[swarm] A swarm is already running. Use /swarm cancel first.")
        return

    _session.reset()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    config = SwarmConfig()
    scheduler = SwarmScheduler()
    pool = WorkerPool(config)
    monitor = SwarmMonitor()

    # Decompose
    tasks = _decompose_goal(goal)
    for t in tasks:
        scheduler.add_task(t)

    # Spin up workers
    pool.scale_up(config.max_workers, backend="local")

    # Store session
    _session.scheduler = scheduler
    _session.pool = pool
    _session.monitor = monitor
    _session.config = config
    _session.goal = goal
    _session.run_id = run_id
    _session.running = True

    monitor.record_event("swarm_started", data={"goal": goal, "run_id": run_id})

    print(f"\n[swarm] Starting swarm run: {run_id}")
    print(f"  Goal:    {goal}")
    print(f"  Tasks:   {len(tasks)}")
    print(f"  Workers: {config.max_workers}")
    print()

    for t in tasks:
        dep_str = f" (deps: {', '.join(t.deps)})" if t.deps else ""
        print(f"  [{t.id}] {t.name} — role={t.role}{dep_str}")
    print()

    # Launch background loop
    thread = threading.Thread(target=_run_swarm_loop, args=(_session,), daemon=True, name="swarm-loop")
    _session._thread = thread
    thread.start()

    print("[swarm] Swarm is running in the background.")
    print("  Use /swarm status to check progress.")
    print()


def _cmd_status(args: str, cli_instance: Any) -> None:
    if not _session.active:
        print("[swarm] No active swarm. Use /swarm run <goal> to start one.")
        return

    scheduler = _session.scheduler
    pool = _session.pool
    monitor = _session.monitor

    sched_status = scheduler.to_dict()
    pool_status = pool.get_status()
    summary = monitor.get_summary()

    hr = "-" * 60
    print()
    print(hr)
    print(f"  SWARM STATUS — run {_session.run_id}")
    print(hr)
    print(f"  Goal:     {_session.goal}")
    print(f"  Running:  {_session.running}")
    print()

    # Task table
    print("  TASKS")
    print(f"  {'ID':<14} {'Name':<16} {'State':<12} {'Role':<10} {'Worker':<12}")
    print(f"  {'-'*14} {'-'*16} {'-'*12} {'-'*10} {'-'*12}")
    for t in sched_status["tasks"]:
        wid = t.get("worker_id") or "-"
        print(f"  {t['id']:<14} {t['name']:<16} {t['state']:<12} {t['role']:<10} {wid:<12}")
    print()

    # Workers
    print(f"  WORKERS ({pool_status['total']} total)")
    for wid, winfo in pool_status.get("workers", {}).items():
        task_str = f"task={winfo['task']}" if winfo["task"] else "idle"
        print(f"    {wid}: {winfo['name']} [{winfo['backend']}] {winfo['state']} — {task_str}")
    print()

    # Cost
    print(f"  Total cost:   ${summary.get('total_cost_usd', 0):.6f}")
    print(f"  Total tokens: {summary.get('total_tokens', 0):,}")
    print(f"  Events:       {summary.get('total_events', 0)}")
    print(hr)
    print()


def _cmd_cancel(args: str, cli_instance: Any) -> None:
    if not _session.active:
        print("[swarm] No active swarm to cancel.")
        return

    _session.running = False

    # Cancel pending tasks
    scheduler = _session.scheduler
    for t in scheduler.to_dict()["tasks"]:
        if t["state"] in ("pending", "queued"):
            try:
                scheduler.mark_failed(t["id"], "cancelled by user")
            except Exception:
                pass

    # Shutdown workers
    if _session.pool:
        try:
            _session.pool.shutdown()
        except Exception:
            pass

    if _session.monitor:
        _session.monitor.record_event("swarm_cancelled")
        try:
            _save_run(_session.run_id, _session.goal, scheduler, _session.monitor)
        except Exception:
            pass

    print(f"[swarm] Swarm run {_session.run_id} cancelled.")
    print()


def _cmd_plan(args: str, cli_instance: Any) -> None:
    goal = args.strip()
    if not goal:
        print("Usage: /swarm plan <goal>")
        print("  Shows the decomposed plan without executing.")
        return

    tasks = _decompose_goal(goal)

    hr = "-" * 60
    print()
    print(hr)
    print("  SWARM PLAN (dry run)")
    print(hr)
    print(f"  Goal: {goal}")
    print(f"  Tasks: {len(tasks)}")
    print()
    print(f"  {'#':<4} {'ID':<14} {'Name':<16} {'Role':<10} {'Dependencies'}")
    print(f"  {'-'*4} {'-'*14} {'-'*16} {'-'*10} {'-'*20}")
    for i, t in enumerate(tasks, 1):
        deps = ", ".join(t.deps) if t.deps else "(none)"
        print(f"  {i:<4} {t.id:<14} {t.name:<16} {t.role:<10} {deps}")
    print()
    print("  Execution order: plan -> execute -> verify")
    print("  Use /swarm run <goal> to execute this plan.")
    print(hr)
    print()


def _cmd_workers(args: str, cli_instance: Any) -> None:
    if not _session.active or _session.pool is None:
        print("[swarm] No active worker pool. Start a swarm with /swarm run <goal>.")
        print()
        print("  Worker pool configuration defaults:")
        config = SwarmConfig()
        print(f"    Max workers:    {config.max_workers}")
        print(f"    Max retries:    {config.max_retries}")
        print(f"    Timeout:        {config.timeout_seconds}s")
        print(f"    Budget limit:   ${config.budget_limit_usd:.2f}")
        print()
        return

    pool_status = _session.pool.get_status()
    states = pool_status.get("states", {})

    hr = "-" * 60
    print()
    print(hr)
    print("  WORKER POOL STATUS")
    print(hr)
    print(f"  Total workers: {pool_status['total']}")
    for state, count in sorted(states.items()):
        print(f"    {state}: {count}")
    print()

    workers = pool_status.get("workers", {})
    if workers:
        print(f"  {'ID':<14} {'Name':<20} {'Backend':<8} {'State':<10} {'Task'}")
        print(f"  {'-'*14} {'-'*20} {'-'*8} {'-'*10} {'-'*14}")
        for wid, w in workers.items():
            task = w.get("task") or "-"
            print(f"  {wid:<14} {w['name']:<20} {w['backend']:<8} {w['state']:<10} {task}")
    print(hr)
    print()


def _cmd_history(args: str, cli_instance: Any) -> None:
    runs = _load_history(limit=10)
    if not runs:
        print("[swarm] No swarm run history found.")
        return

    hr = "-" * 60
    print()
    print(hr)
    print("  SWARM RUN HISTORY (most recent first)")
    print(hr)
    print()

    for run in runs:
        run_id = run.get("run_id", "?")
        goal = run.get("goal", "?")
        finished = run.get("finished_at", "?")
        summary = run.get("summary", {})
        total_tasks = summary.get("total_tasks", 0)
        completed = summary.get("completed", 0)
        failed = summary.get("failed", 0)
        cost = summary.get("total_cost_usd", 0.0)

        print(f"  Run:       {run_id}")
        print(f"  Goal:      {goal}")
        print(f"  Finished:  {finished}")
        print(f"  Tasks:     {total_tasks} total, {completed} completed, {failed} failed")
        print(f"  Cost:      ${cost:.6f}")
        print()

    print(hr)
    print()


def _cmd_help(args: str, cli_instance: Any) -> None:
    hr = "-" * 60
    print()
    print(hr)
    print("  /swarm — Multi-agent swarm orchestration")
    print(hr)
    print()
    print("  Subcommands:")
    print()
    print("    /swarm run <goal>      Decompose goal into tasks and execute the swarm")
    print("    /swarm plan <goal>     Show decomposed plan without executing")
    print("    /swarm status          Show current swarm status (tasks, workers, cost)")
    print("    /swarm cancel          Cancel the running swarm")
    print("    /swarm workers         Show worker pool status")
    print("    /swarm history         Show recent swarm runs")
    print("    /swarm help            Show this help message")
    print()
    print("  Roles available:")
    for role in list_roles():
        print(f"    {role.name:<12} {role.description}")
    print()
    print(hr)
    print()


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_SUBCOMMANDS: dict[str, tuple[Any, str]] = {
    "run":     (_cmd_run,     "Decompose goal into tasks and execute"),
    "plan":    (_cmd_plan,    "Show decomposed plan without executing"),
    "status":  (_cmd_status,  "Show current swarm status"),
    "cancel":  (_cmd_cancel,  "Cancel the running swarm"),
    "workers": (_cmd_workers, "Show worker pool status"),
    "history": (_cmd_history, "Show recent swarm runs"),
    "help":    (_cmd_help,    "Show all subcommands"),
}


def handle_swarm_command(args: str, cli_instance: Any) -> None:
    """Dispatch /swarm subcommands.

    Called from the CLI's process_command method with everything after '/swarm '.

    Args:
        args: The subcommand and its arguments (e.g. "run refactor auth module").
        cli_instance: The HermesCLI instance (for accessing agent, config, etc.).
    """
    parts = args.strip().split(maxsplit=1)
    subcmd = parts[0].lower() if parts else "help"
    sub_args = parts[1] if len(parts) > 1 else ""

    handler_info = _SUBCOMMANDS.get(subcmd)
    if handler_info is None:
        print(f"[swarm] Unknown subcommand: {subcmd}")
        print(f"  Available: {', '.join(sorted(_SUBCOMMANDS))}")
        print("  Run /swarm help for details.")
        return

    handler, _ = handler_info
    handler(sub_args, cli_instance)
