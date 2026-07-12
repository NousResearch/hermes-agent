"""Hermes autonomous development runner.

The runner is deliberately deterministic.  LLM workers still do creative
implementation work, but this module owns the durable state machine that keeps
Hermes moving after restarts, desktop reloads, failed CI, or an exhausted
single conversation turn.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any
from uuid import uuid4

from hermes_delivery import (
    artifact_path,
    delivery_root,
    load_state as load_delivery_state,
    run_capability_audit,
    run_gate,
    update_stage,
)


RUNNER_SCHEMA = "hermes.autonomous.v1"
DEFAULT_BOARD = "default"
DEFAULT_GOAL_TURNS = 120
WATCHDOG_JOB_NAME = "Hermes Autonomous Watchdog"
WATCHDOG_SCRIPT_NAME = "autonomous_watchdog.py"
ACTIVE_TASK_STATUSES = {"todo", "scheduled", "ready", "running", "review"}
UNFINISHED_TASK_STATUSES = ACTIVE_TASK_STATUSES | {"blocked"}


@dataclass(frozen=True)
class TickOptions:
    create_tasks: bool = False
    dispatch: bool = False
    board: str = DEFAULT_BOARD
    assignee: str | None = None
    goal_max_turns: int = DEFAULT_GOAL_TURNS
    workspace_path: str | None = None
    repo: str | None = None
    pr: str | None = None
    ci_ref: str | None = None


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def runner_state_path() -> Path:
    return delivery_root() / "autonomous-runner.json"


def anti_shell_report_path() -> Path:
    return artifact_path("anti-shell/report.json")


def heartbeat_path() -> Path:
    return artifact_path("autonomous-heartbeat.json")


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return fallback
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback
    return data if isinstance(data, dict) else fallback


def _initial_runner_state() -> dict[str, Any]:
    return {
        "schema": RUNNER_SCHEMA,
        "created_at": _utcnow(),
        "updated_at": _utcnow(),
        "enabled": True,
        "mode": "supervised-autonomous",
        "no_approval_cards": True,
        "board": DEFAULT_BOARD,
        "goal_mode": True,
        "goal_max_turns": DEFAULT_GOAL_TURNS,
        "last_tick_at": "",
        "last_tick_id": "",
        "last_next_action": "",
        "last_blocked_reason": "",
        "heartbeat": {},
        "ticks": [],
    }


def load_runner_state() -> dict[str, Any]:
    state = _read_json(runner_state_path(), _initial_runner_state())
    state.setdefault("schema", RUNNER_SCHEMA)
    state.setdefault("created_at", _utcnow())
    state.setdefault("updated_at", _utcnow())
    state.setdefault("enabled", True)
    state.setdefault("mode", "supervised-autonomous")
    state.setdefault("no_approval_cards", True)
    state.setdefault("board", DEFAULT_BOARD)
    state.setdefault("goal_mode", True)
    state.setdefault("goal_max_turns", DEFAULT_GOAL_TURNS)
    state.setdefault("last_tick_at", "")
    state.setdefault("last_tick_id", "")
    state.setdefault("last_next_action", "")
    state.setdefault("last_blocked_reason", "")
    state.setdefault("heartbeat", {})
    state.setdefault("ticks", [])
    return state


def save_runner_state(state: dict[str, Any]) -> dict[str, Any]:
    state["updated_at"] = _utcnow()
    _atomic_write_json(runner_state_path(), state)
    return state


def _run_command(args: list[str], cwd: str | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        return {"ok": False, "args": args, "exit_code": -1, "stdout": "", "stderr": str(exc)}
    return {
        "ok": result.returncode == 0,
        "args": args,
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _git_status(repo: str | None) -> dict[str, Any]:
    cwd = repo or os.getcwd()
    status = _run_command(["git", "status", "--porcelain"], cwd=cwd)
    branch = _run_command(["git", "branch", "--show-current"], cwd=cwd)
    return {
        "repo": cwd,
        "available": status["exit_code"] == 0,
        "dirty": bool(status.get("stdout")),
        "branch": branch.get("stdout", ""),
        "porcelain": status.get("stdout", ""),
        "error": status.get("stderr") if status["exit_code"] != 0 else "",
    }


def _artifact_exists(relative: str) -> bool:
    path = artifact_path(relative)
    return path.exists() and path.stat().st_size > 0


def run_anti_shell_check(repo: str | None = None) -> dict[str, Any]:
    """Check that the delivery loop has real evidence, not just green labels."""

    git = _git_status(repo)
    delivery = load_delivery_state()
    stages = {stage["id"]: stage for stage in delivery.get("stages", [])}
    deploy_evidence = run_gate("deploy", update_state=False)
    deploy_verified = deploy_evidence.get("status") == "passed"
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add("delivery_ledger_exists", artifact_path("state.json").exists(), str(artifact_path("state.json")))
    add("spec_artifact_exists", _artifact_exists("spec.md"), "spec.md")
    add("task_artifact_exists", _artifact_exists("tasks.md"), "tasks.md")
    add("dispatch_has_job", _artifact_exists("jobs") or stages.get("dispatch", {}).get("status") in {"ready", "done"}, "jobs/*.json")
    add("pr_monitor_artifact", _artifact_exists("pr-status.json"), "pr-status.json")
    add("ci_monitor_artifact", _artifact_exists("ci/latest.json") or _artifact_exists("ci-latest.json"), "ci latest")
    add("acceptance_artifact", _artifact_exists("acceptance/report.md"), "acceptance/report.md")
    add("deploy_run_recorded", deploy_verified, f"deploy gate: {deploy_evidence.get('status')}")
    add("git_repo_available", bool(git["available"]), git["repo"])
    change_detail = git["porcelain"] or ("deploy gate verified" if deploy_verified else "no code/deploy evidence")
    add("git_has_real_change_or_deploy", bool(git["dirty"] or deploy_verified), change_detail)

    failed = [check for check in checks if not check["ok"]]
    report = {
        "schema": "hermes.anti-shell.v1",
        "generated_at": _utcnow(),
        "status": "passed" if not failed else "failed",
        "passed": len(checks) - len(failed),
        "failed": len(failed),
        "checks": checks,
        "git": git,
        "failure_action": "回 Stage3/Phase2 补真实代码、PR、CI、验收或部署证据" if failed else "allow_next_gate",
    }
    _atomic_write_json(anti_shell_report_path(), report)
    if failed:
        update_stage("anti_shell", "failed", "Anti-Shell failed; real evidence missing")
    else:
        update_stage("anti_shell", "done", "Anti-Shell passed")
    return report


def _next_stage_action(audit: dict[str, Any]) -> tuple[str, str]:
    for result in audit.get("results", []):
        if result.get("status") != "passed":
            return result.get("stage_id", "unknown"), "gate_failed"
    return "deploy", "all_gates_passed"


def _create_goal_task(
    *,
    title: str,
    body: str,
    board: str,
    assignee: str | None,
    goal_max_turns: int,
    workspace_path: str | None,
    idempotency_key: str,
) -> tuple[str, bool, dict[str, Any]]:
    from hermes_cli import kanban_db as kb

    with kb.connect(board=board) as conn:
        active = conn.execute(
            """
            SELECT id, status, created_at, completed_at, idempotency_key
              FROM tasks
             WHERE idempotency_key LIKE ?
               AND status != 'archived'
             ORDER BY created_at DESC
            """,
            (f"{idempotency_key}%",),
        ).fetchall()
        for row in active:
            if str(row["status"]) in ACTIVE_TASK_STATUSES:
                task = {
                    "id": str(row["id"]),
                    "status": str(row["status"]),
                    "created_at": int(row["created_at"] or 0),
                    "completed_at": int(row["completed_at"] or 0),
                    "idempotency_key": str(row["idempotency_key"] or ""),
                    "reused_active": True,
                }
                return task["id"], False, task
        latest = active[0] if active else None
        exact = conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ? AND status != 'archived' "
            "ORDER BY created_at DESC LIMIT 1",
            (idempotency_key,),
        ).fetchone()
        effective_key = idempotency_key
        if exact:
            effective_key = f"{idempotency_key}:retry:{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}:{uuid4().hex[:6]}"
        task_id = kb.create_task(
            conn,
            title=title,
            body=body,
            assignee=assignee,
            created_by="hermes-autonomous",
            workspace_kind="dir" if workspace_path else "scratch",
            workspace_path=workspace_path,
            idempotency_key=effective_key,
            goal_mode=True,
            goal_max_turns=goal_max_turns,
            initial_status="running",
            board=board,
        )
    return task_id, True, {
        "id": task_id,
        "status": "ready",
        "idempotency_key": effective_key,
        "reused_active": False,
        "previous_task": {
            "id": str(latest["id"]),
            "status": str(latest["status"]),
            "idempotency_key": str(latest["idempotency_key"] or ""),
        }
        if latest
        else None,
    }


def _dispatch_once(board: str) -> dict[str, Any]:
    from hermes_cli import kanban_db as kb

    with kb.connect(board=board) as conn:
        result = kb.dispatch_once(conn, board=board)
    if is_dataclass(result):
        return asdict(result)
    if isinstance(result, dict):
        return result
    return {"result": repr(result)}


def _unfinished_autonomous_tasks(board: str) -> list[dict[str, Any]]:
    from hermes_cli import kanban_db as kb

    with kb.connect(board=board) as conn:
        rows = conn.execute(
            """
            SELECT id, title, status, assignee, created_at, started_at, completed_at,
                   current_run_id, last_heartbeat_at, worker_pid, idempotency_key
              FROM tasks
             WHERE idempotency_key LIKE 'hermes-autonomous:%'
               AND status NOT IN ('done', 'archived')
             ORDER BY created_at DESC
             LIMIT 20
            """
        ).fetchall()
    return [
        {
            "id": str(row["id"]),
            "title": str(row["title"]),
            "status": str(row["status"]),
            "assignee": str(row["assignee"] or ""),
            "created_at": int(row["created_at"] or 0),
            "started_at": int(row["started_at"] or 0),
            "completed_at": int(row["completed_at"] or 0),
            "current_run_id": int(row["current_run_id"] or 0),
            "last_heartbeat_at": int(row["last_heartbeat_at"] or 0),
            "worker_pid": int(row["worker_pid"] or 0),
            "idempotency_key": str(row["idempotency_key"] or ""),
        }
        for row in rows
    ]


def _write_heartbeat(data: dict[str, Any]) -> dict[str, Any]:
    heartbeat = {
        "schema": "hermes.autonomous.heartbeat.v1",
        "updated_at": _utcnow(),
        **data,
    }
    _atomic_write_json(heartbeat_path(), heartbeat)
    return heartbeat


def autonomous_status() -> dict[str, Any]:
    state = load_runner_state()
    anti_shell = _read_json(anti_shell_report_path(), {})
    heartbeat = _read_json(heartbeat_path(), state.get("heartbeat", {}))
    delivery = load_delivery_state()
    return {
        "ok": True,
        "runner": state,
        "heartbeat": heartbeat,
        "delivery_summary": {
            "stages": [
                {
                    "id": stage.get("id"),
                    "status": stage.get("status"),
                    "notes": stage.get("notes", ""),
                }
                for stage in delivery.get("stages", [])
            ],
            "deploy_runs": len(delivery.get("deploy", {}).get("runs", [])),
        },
        "anti_shell": anti_shell,
        "state_path": str(runner_state_path()),
        "anti_shell_path": str(anti_shell_report_path()),
        "heartbeat_path": str(heartbeat_path()),
    }


def autonomous_tick(options: TickOptions | None = None) -> dict[str, Any]:
    opts = options or TickOptions()
    state = load_runner_state()
    tick_id = f"hat-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"

    audit = run_capability_audit()
    anti_shell = run_anti_shell_check(opts.repo)
    next_stage, reason = _next_stage_action(audit)
    all_gates_passed = reason == "all_gates_passed"
    unfinished_tasks = _unfinished_autonomous_tasks(opts.board)
    blocked_tasks = [task for task in unfinished_tasks if task.get("status") == "blocked"]
    created_tasks: list[dict[str, Any]] = []
    dispatch_result: dict[str, Any] | None = None

    if opts.create_tasks and blocked_tasks:
        blocked = blocked_tasks[0]
        stage = str(blocked.get("idempotency_key", "hermes-autonomous:recovery")).split(":", 2)[1] or "recovery"
        title = f"Hermes autonomous: retry blocked {stage} task"
        body = (
            "You are a Hermes autonomous no-idle recovery worker. A previous autonomous task "
            "is blocked while the delivery loop is still not allowed to pretend everything is finished.\n\n"
            f"Blocked task: {blocked['id']} ({blocked['title']})\n"
            f"Blocked status: {blocked['status']}\n"
            "Required behavior: inspect the blocked task evidence, unblock by doing the missing repo/local work, "
            "write durable evidence into ~/.hermes/delivery, then re-run the relevant Hermes delivery/autonomous checks."
        )
        task_id, created, task_info = _create_goal_task(
            title=title,
            body=body,
            board=opts.board,
            assignee=opts.assignee,
            goal_max_turns=max(1, int(opts.goal_max_turns or DEFAULT_GOAL_TURNS)),
            workspace_path=opts.workspace_path,
            idempotency_key=f"hermes-autonomous:recovery:{stage}",
        )
        created_tasks.append(
            {
                "id": task_id,
                "title": title,
                "stage": stage,
                "created": created,
                "status": str(task_info.get("status", "")),
                "idempotency_key": str(task_info.get("idempotency_key", "")),
                "previous_task": blocked,
            }
        )
    elif opts.create_tasks and not all_gates_passed:
        title = f"Hermes autonomous: fix {next_stage} gate"
        body = (
            "You are a Hermes autonomous goal worker. Continue until the gate is real, "
            "not merely claimed complete.\n\n"
            f"Failed stage: {next_stage}\n"
            f"Reason: {reason}\n"
            f"Anti-Shell status: {anti_shell['status']}\n"
            "No-idle rule: before ending any turn, check the same gate again. "
            "If it is still failing, write the missing evidence and continue; do not claim completion.\n"
            "Required behavior: inspect artifacts, implement missing work, run verification, "
            "write evidence into ~/.hermes/delivery, and do not ask the user for repo-only work."
        )
        task_id, created, task_info = _create_goal_task(
            title=title,
            body=body,
            board=opts.board,
            assignee=opts.assignee,
            goal_max_turns=max(1, int(opts.goal_max_turns or DEFAULT_GOAL_TURNS)),
            workspace_path=opts.workspace_path,
            idempotency_key=f"hermes-autonomous:{next_stage}",
        )
        created_tasks.append(
            {
                "id": task_id,
                "title": title,
                "stage": next_stage,
                "created": created,
                "status": str(task_info.get("status", "")),
                "idempotency_key": str(task_info.get("idempotency_key", "")),
                "previous_task": task_info.get("previous_task"),
            }
        )

    if opts.dispatch:
        dispatch_result = _dispatch_once(opts.board)

    spawned = 0
    if isinstance(dispatch_result, dict):
        try:
            spawned = int(dispatch_result.get("spawned") or 0)
        except (TypeError, ValueError):
            spawned = 0
    heartbeat_mode = "all_green" if all_gates_passed else "active"
    if blocked_tasks:
        heartbeat_mode = "blocked_task_needs_retry"
    wake_agent = (not all_gates_passed or bool(blocked_tasks)) and (
        any(task.get("created") for task in created_tasks) or spawned > 0
    )
    if (not all_gates_passed or blocked_tasks) and created_tasks and not any(task.get("created") for task in created_tasks) and spawned <= 0:
        heartbeat_mode = "waiting_on_active_task"
    heartbeat = _write_heartbeat(
        {
            "mode": heartbeat_mode,
            "wake_agent": wake_agent,
            "next_action": next_stage,
            "reason": reason,
            "audit_failed": int(audit.get("failed", 0) or 0),
            "anti_shell_failed": int(anti_shell.get("failed", 0) or 0),
            "created_tasks": created_tasks,
            "unfinished_tasks": unfinished_tasks,
            "blocked_tasks": blocked_tasks,
            "dispatch_spawned": spawned,
            "board": opts.board,
            "no_idle_enforced": True,
            "message": "all gates passed; watchdog may stay silent"
            if all_gates_passed and not blocked_tasks
            else "no-idle blocked task recovery active; Hermes must retry or expose the blockage"
            if blocked_tasks
            else "no-idle active; gate is not green so Hermes must keep an active worker or create a retry",
        }
    )

    tick = {
        "tick_id": tick_id,
        "created_at": _utcnow(),
        "next_action": next_stage,
        "reason": reason,
        "audit": {
            "passed": audit.get("passed", 0),
            "failed": audit.get("failed", 0),
            "total": audit.get("total", 0),
        },
        "anti_shell": {
            "status": anti_shell.get("status"),
            "passed": anti_shell.get("passed", 0),
            "failed": anti_shell.get("failed", 0),
        },
        "created_tasks": created_tasks,
        "dispatch": dispatch_result,
        "heartbeat": heartbeat,
        "no_approval_cards": True,
        "goal_mode": True,
        "no_idle_enforced": True,
    }
    ticks = state.setdefault("ticks", [])
    ticks.insert(0, tick)
    state["ticks"] = ticks[:50]
    state["last_tick_at"] = tick["created_at"]
    state["last_tick_id"] = tick_id
    state["last_next_action"] = next_stage
    state["last_blocked_reason"] = "" if reason == "all_gates_passed" else reason
    state["board"] = opts.board
    state["goal_max_turns"] = opts.goal_max_turns
    state["heartbeat"] = heartbeat
    save_runner_state(state)
    return {"ok": True, "tick": tick, "runner": state, "anti_shell": anti_shell}


def _scripts_dir() -> Path:
    return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes").expanduser() / "scripts"


def _watchdog_script_content(*, repo: str, workspace_path: str, board: str, assignee: str | None, goal_max_turns: int) -> str:
    payload = {
        "repo": repo,
        "workspace_path": workspace_path,
        "board": board,
        "assignee": assignee,
        "goal_max_turns": goal_max_turns,
    }
    return (
        "from __future__ import annotations\n"
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        f"PAYLOAD = {json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "repo = PAYLOAD.get('repo') or ''\n"
        "if repo and repo not in sys.path:\n"
        "    sys.path.insert(0, repo)\n\n"
        "from hermes_autonomous import TickOptions, autonomous_tick\n\n"
        "result = autonomous_tick(TickOptions(\n"
        "    create_tasks=True,\n"
        "    dispatch=True,\n"
        "    board=PAYLOAD.get('board') or 'default',\n"
        "    assignee=PAYLOAD.get('assignee') or None,\n"
        "    goal_max_turns=int(PAYLOAD.get('goal_max_turns') or 120),\n"
        "    workspace_path=PAYLOAD.get('workspace_path') or repo or None,\n"
        "    repo=repo or None,\n"
        "))\n"
        "tick = result.get('tick', {})\n"
        "created = [t for t in tick.get('created_tasks', []) if t.get('created')]\n"
        "heartbeat = tick.get('heartbeat') or {}\n"
        "dispatch = tick.get('dispatch') or {}\n"
        "spawned_raw = dispatch.get('spawned') if isinstance(dispatch, dict) else 0\n"
        "spawned = len(spawned_raw) if isinstance(spawned_raw, list) else int(spawned_raw or 0)\n"
        "if not bool(heartbeat.get('wake_agent')) and not created and spawned <= 0:\n"
        "    print(json.dumps({'wakeAgent': False}))\n"
        "else:\n"
        "    print(json.dumps({\n"
        "        'wakeAgent': True,\n"
        "        'runner': 'hermes-autonomous',\n"
        "        'tick_id': tick.get('tick_id'),\n"
        "        'next_action': tick.get('next_action'),\n"
        "        'heartbeat_mode': heartbeat.get('mode'),\n"
        "        'created_tasks': created,\n"
        "        'spawned': spawned,\n"
        "    }, ensure_ascii=False))\n"
    )


def install_watchdog(
    *,
    schedule: str = "every 2m",
    repo: str | None = None,
    workspace_path: str | None = None,
    board: str = DEFAULT_BOARD,
    assignee: str | None = None,
    goal_max_turns: int = DEFAULT_GOAL_TURNS,
) -> dict[str, Any]:
    """Install or replace the no-agent cron watchdog for autonomous ticks."""

    repo_path = str(Path(repo or os.getcwd()).expanduser().resolve())
    workspace = str(Path(workspace_path or repo_path).expanduser().resolve())
    scripts = _scripts_dir()
    scripts.mkdir(parents=True, exist_ok=True)
    script_path = scripts / WATCHDOG_SCRIPT_NAME
    script_path.write_text(
        _watchdog_script_content(
            repo=repo_path,
            workspace_path=workspace,
            board=board,
            assignee=assignee,
            goal_max_turns=max(1, int(goal_max_turns or DEFAULT_GOAL_TURNS)),
        ),
        encoding="utf-8",
    )

    from cron import jobs as cron_jobs

    removed: list[str] = []
    for job in cron_jobs.list_jobs(include_disabled=True):
        if job.get("name") == WATCHDOG_JOB_NAME:
            if cron_jobs.remove_job(str(job["id"])):
                removed.append(str(job["id"]))

    job = cron_jobs.create_job(
        prompt="Hermes autonomous watchdog",
        schedule=schedule,
        name=WATCHDOG_JOB_NAME,
        deliver="local",
        script=WATCHDOG_SCRIPT_NAME,
        workdir=repo_path,
        no_agent=True,
    )
    state = load_runner_state()
    state["watchdog"] = {
        "installed": True,
        "job_id": job["id"],
        "job_name": WATCHDOG_JOB_NAME,
        "schedule": schedule,
        "script": str(script_path),
        "repo": repo_path,
        "workspace_path": workspace,
        "board": board,
        "assignee": assignee or "",
        "goal_max_turns": max(1, int(goal_max_turns or DEFAULT_GOAL_TURNS)),
        "installed_at": _utcnow(),
        "replaced_job_ids": removed,
    }
    save_runner_state(state)
    return {"ok": True, "job": job, "script": str(script_path), "removed": removed, "runner": state}


def watchdog_status() -> dict[str, Any]:
    from cron import jobs as cron_jobs

    jobs = [
        job
        for job in cron_jobs.list_jobs(include_disabled=True)
        if job.get("name") == WATCHDOG_JOB_NAME
    ]
    return {
        "ok": True,
        "installed": bool(jobs),
        "jobs": jobs,
        "script": str(_scripts_dir() / WATCHDOG_SCRIPT_NAME),
        "runner": load_runner_state(),
    }
