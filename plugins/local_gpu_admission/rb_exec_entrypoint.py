"""Wait for a task-bound Resource Balancer GPU lease before execing a worker.

The process keeps its PID across ``exec``. That lets Kanban record the final
Hermes worker PID immediately after Popen while the RB lease remains owned by
the same process identity for the worker's lifetime.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sqlite3
import sys
import time

_LOGGER = logging.getLogger(__name__)


def _scope_contains_unit(scope_unit: str, cgroup_text: str) -> bool:
    if not scope_unit or not scope_unit.endswith(".scope"):
        return False
    for line in cgroup_text.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        # cgroup v1 and v2 both identify the systemd unit as a complete path
        # component. Substring checks can accept a different unit or a crafted
        # ancestor component that merely embeds the expected name.
        if scope_unit in fields[2].strip("/").split("/"):
            return True
    return False


def _task_is_still_claimed(db_path: str, context: dict[str, str], pid: int,
                           scope_unit: str = "", cgroup_text: str | None = None) -> bool:
    path = Path(db_path).resolve(strict=True)
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=1)
    conn.row_factory = sqlite3.Row
    try:
        task_row = conn.execute("SELECT * FROM tasks WHERE id=?", (context["task_id"],)).fetchone()
        run = conn.execute(
            "SELECT task_id,status,claim_lock,worker_pid FROM task_runs WHERE id=?",
            (int(context["run_id"]),),
        ).fetchone()
    finally:
        conn.close()
    if task_row is None or run is None:
        return False
    from hermes_cli.kanban_db import Task
    from hermes_cli.kanban_local_model_scope import capability_id_from_body, task_run_contract_sha256

    task = Task.from_row(task_row)
    identity_matches = bool(
        task.status == "running"
        and str(task.current_run_id) == context["run_id"]
        and task.claim_lock == context["claim_lock"]
        and run["task_id"] == task.id
        and run["status"] == "running"
        and run["claim_lock"] == task.claim_lock
        and capability_id_from_body(task.body) == context["capability_id"]
        and task_run_contract_sha256(task) == context["task_contract_sha256"]
    )
    if not identity_matches:
        return False
    if run["worker_pid"] == pid:
        return True
    if not run["worker_pid"] or not scope_unit:
        return False
    if cgroup_text is None:
        try:
            cgroup_text = Path("/proc/self/cgroup").read_text(encoding="utf-8")
        except OSError:
            return False
    if not _scope_contains_unit(scope_unit, cgroup_text):
        return False
    from hermes_cli.kanban_db_dispatch import _process_fingerprint
    started_at = _process_fingerprint(pid)
    if not started_at:
        return False
    with sqlite3.connect(path, timeout=1) as write_conn:
        write_conn.row_factory = sqlite3.Row
        try:
            write_conn.execute("BEGIN IMMEDIATE")
            changed_task = write_conn.execute(
                "UPDATE tasks SET worker_pid=?,worker_started_at=? "
                "WHERE id=? AND status='running' AND current_run_id=? AND claim_lock=? AND worker_pid=?",
                (pid, started_at, task.id, int(context["run_id"]), context["claim_lock"], run["worker_pid"]),
            ).rowcount
            changed_run = write_conn.execute(
                "UPDATE task_runs SET worker_pid=?,worker_started_at=? "
                "WHERE id=? AND task_id=? AND status='running' AND claim_lock=? AND worker_pid=?",
                (pid, started_at, int(context["run_id"]), task.id, context["claim_lock"], run["worker_pid"]),
            ).rowcount
            if changed_task != 1 or changed_run != 1:
                write_conn.rollback()
                return False
            from hermes_cli import kanban_db as kb
            kb._append_event(write_conn, task.id, "worker_registered",
                             {"pid": pid, "started_at": started_at}, run_id=int(context["run_id"]))
            write_conn.commit()
        except (OSError, sqlite3.Error, TypeError, ValueError):
            write_conn.rollback()
            return False
    return True


def _bind_context(state_dir: str, job_id: str, context: dict[str, str]) -> None:
    db_path = Path(state_dir).expanduser().resolve(strict=True) / "queue.sqlite3"
    conn = sqlite3.connect(db_path, timeout=1)
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hermes_local_gpu_reservations (
                    job_id TEXT PRIMARY KEY,
                    context TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            row = conn.execute(
                "SELECT 1 FROM jobs WHERE id=? AND class='gpu' AND state='running'", (job_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("active Resource Balancer reservation disappeared before worker exec")
            conn.execute(
                "INSERT INTO hermes_local_gpu_reservations(job_id,context,created_at) VALUES (?,?,?)",
                (job_id, json.dumps(context, sort_keys=True, separators=(",", ":")), time.time()),
            )
    finally:
        conn.close()


def _worker_exec_env(job_id: str) -> dict[str, str]:
    from tools.environments.local import served_profile_child_env

    env = served_profile_child_env(
        base=os.environ, target_home=os.environ.get("HERMES_HOME"), inherit_credentials=True,
    )
    env["RESOURCE_BALANCER_ADMISSION_JOB_ID"] = str(job_id)
    return env


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rb-python-path", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--kanban-db", required=True)
    parser.add_argument("--context-json", required=True)
    parser.add_argument("--scope-unit", default="")
    parser.add_argument("--claim-wait-seconds", type=float, default=10.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command or not command[0]:
        parser.error("a worker command is required after --")
    try:
        context = json.loads(args.context_json)
        required = {"task_id", "run_id", "claim_lock", "capability_id", "task_contract_sha256"}
        if type(context) is not dict or set(context) != required or not all(context.values()):
            raise ValueError("task/run reservation context is incomplete")
        rb_root = str(Path(args.rb_python_path).expanduser().resolve(strict=True))
        if not (Path(rb_root) / "resource_balancer" / "admission.py").is_file():
            raise ValueError("Resource Balancer package path is invalid")
        sys.path.insert(0, rb_root)
        from resource_balancer.admission import admission

        with admission("gpu", state_dir=args.state_dir, origin="hermes-kanban-local-model") as job_id:
            _bind_context(args.state_dir, str(job_id), context)
            deadline = time.monotonic() + max(0.1, args.claim_wait_seconds)
            while not _task_is_still_claimed(args.kanban_db, context, os.getpid(), args.scope_unit):
                if time.monotonic() >= deadline:
                    raise RuntimeError("dispatcher did not bind this PID to the claimed task/run")
                time.sleep(0.05)
            env = _worker_exec_env(str(job_id))
            os.execvpe(command[0], command, env)
    except Exception as exc:
        _LOGGER.exception("Local GPU worker dispatch refused before exec")
        print(f"Local GPU dispatch refused before worker exec: {exc}", file=sys.stderr)
        return 75
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
