"""Hermes Symphony bridge.

A small, durable integration layer between conversational entry points and the
existing Hermes Kanban dispatcher.  It deliberately does *not* embed OpenAI's
experimental Symphony implementation into Hermes core.  Instead it keeps a
persistent development-task projection, creates an isolated Kanban worktree
for eligible tasks, and reconciles its state from the proven Kanban lifecycle.

This makes the runner replaceable: a future external Symphony service can own
execution without changing Telegram, gateway, or task identity contracts.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, replace as dc_replace
from pathlib import Path
from typing import Any, Iterator, Optional

from hermes_constants import get_hermes_home
from hermes_cli import kanban_db as kb
from hermes_cli import web_git
from agent.redact import redact_sensitive_text

STATUSES = frozenset({
    "draft", "needs_clarification", "ready_for_development", "queued",
    "provisioning_workspace", "running", "validating", "awaiting_human_review",
    "completed", "failed", "cancelled", "paused",
})
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
_ACTIVE_STATUSES = frozenset({"queued", "provisioning_workspace", "running", "validating"})

_WORKFLOW_TEMPLATE = """---
name: symphony-development-workflow
version: 1
---

# Workflow разработки

1. Сначала изучи исходную задачу, `AGENTS.md` и существующий код проекта.
2. Меняй только связанные с задачей файлы. Политика: не выполняй production deploy,
   migration или merge без отдельного явного разрешения.
3. Сначала напиши или обнови регрессионный тест. Не обходи failing test
   фиктивными моками.
4. Запусти релевантные тесты и проверки, сохрани фактические команды и
   краткие результаты. Зафиксируй ограничения и нерешённые вопросы.
5. Перед handoff создай machine-readable result artifact JSON с изменёнными
   файлами, тестами, branch, commit/PR (если есть) и безопасным итогом.
   Artifact обязателен для Symphony completion: `.symphony/$HERMES_KANBAN_TASK-result.json`,
   с `task_id`, непустым `tests` и фактическим `exit_code: 0` для каждой проверки.
6. Никогда не записывай секреты, токены, ключи, пароли или credentials в
   код, результат, логи, PR или сообщения.
"""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS development_tasks (
  task_id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  source_text TEXT NOT NULL,
  technical_brief TEXT,
  project_id TEXT,
  repository TEXT,
  telegram_chat_id TEXT,
  telegram_thread_id TEXT,
  requester_id TEXT,
  status TEXT NOT NULL,
  priority INTEGER NOT NULL DEFAULT 0,
  idempotency_key TEXT UNIQUE,
  kanban_board TEXT,
  kanban_task_id TEXT UNIQUE,
  workspace_path TEXT,
  branch_name TEXT,
  pr_url TEXT,
  test_summary TEXT,
  result_summary TEXT,
  error TEXT,
  retry_count INTEGER NOT NULL DEFAULT 0,
  next_retry_at INTEGER,
  last_action TEXT,
  progress_detail TEXT,
  ci_state TEXT,
  ci_checks TEXT,
  source_issue_id TEXT,
  source_issue_url TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_development_tasks_status
  ON development_tasks(status, priority, created_at);
CREATE INDEX IF NOT EXISTS idx_development_tasks_origin
  ON development_tasks(telegram_chat_id, telegram_thread_id, created_at);
CREATE TABLE IF NOT EXISTS development_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  payload TEXT,
  created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_development_events_task
  ON development_events(task_id, id);
CREATE TABLE IF NOT EXISTS agent_runs (
  run_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  kanban_task_id TEXT,
  workspace_path TEXT,
  branch_name TEXT,
  worker_pid INTEGER,
  status TEXT NOT NULL,
  started_at INTEGER NOT NULL,
  finished_at INTEGER,
  last_progress_at INTEGER NOT NULL,
  result_artifact TEXT,
  test_summary TEXT,
  error TEXT
);
CREATE INDEX IF NOT EXISTS idx_agent_runs_task ON agent_runs(task_id, started_at DESC);
CREATE TABLE IF NOT EXISTS delivery_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  delivered_at INTEGER NOT NULL DEFAULT 0,
  claimed_at INTEGER NOT NULL DEFAULT 0,
  payload TEXT
);
CREATE INDEX IF NOT EXISTS idx_delivery_events_task ON delivery_events(task_id, id);
"""


@dataclass(frozen=True)
class DevelopmentTask:
    task_id: str
    title: str
    source_text: str
    technical_brief: Optional[str]
    project_id: Optional[str]
    repository: Optional[str]
    telegram_chat_id: Optional[str]
    telegram_thread_id: Optional[str]
    requester_id: Optional[str]
    status: str
    priority: int
    idempotency_key: Optional[str]
    kanban_board: Optional[str]
    kanban_task_id: Optional[str]
    workspace_path: Optional[str]
    branch_name: Optional[str]
    pr_url: Optional[str]
    test_summary: Optional[str]
    result_summary: Optional[str]
    error: Optional[str]
    retry_count: int
    next_retry_at: Optional[int]
    last_action: Optional[str]
    progress_detail: Optional[str]
    ci_state: Optional[str]
    ci_checks: Optional[str]
    source_issue_id: Optional[str]
    source_issue_url: Optional[str]
    created_at: int
    updated_at: int

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "DevelopmentTask":
        return cls(**dict(row))


@dataclass(frozen=True)
class AgentRun:
    run_id: str
    task_id: str
    kanban_task_id: Optional[str]
    workspace_path: Optional[str]
    branch_name: Optional[str]
    worker_pid: Optional[int]
    status: str
    started_at: int
    finished_at: Optional[int]
    last_progress_at: int
    result_artifact: Optional[str]
    test_summary: Optional[str]
    error: Optional[str]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "AgentRun":
        return cls(**dict(row))


class DevelopmentRunner:
    """Replaceable execution boundary for durable development tasks."""

    def create_task(self, **kwargs: Any) -> DevelopmentTask:
        return create_task(**kwargs)

    def enqueue(self, task_id: str, **kwargs: Any) -> DevelopmentTask:
        return enqueue(task_id, **kwargs)

    def get_status(self, task_id: str) -> DevelopmentTask:
        with connect() as conn:
            task = _get(conn, task_id)
        if task is None:
            raise ValueError(f"development task {task_id} not found")
        return task

    def cancel(self, task_id: str) -> DevelopmentTask:
        return cancel(task_id)

    def resume(self, task_id: str) -> DevelopmentTask:
        return resume(task_id)

    def reconcile(self, task_id: Optional[str] = None) -> list[DevelopmentTask]:
        return reconcile(task_id)

    def run_once(self, *, spawn_fn=None) -> list[str]:
        raise NotImplementedError


class SymphonyRunner(DevelopmentRunner):
    """Kanban-backed runner with one durable AgentRun per development task."""

    def __init__(self, *, max_concurrency: int = 1):
        self.max_concurrency = max(1, int(max_concurrency))

    def run_once(self, *, spawn_fn=None) -> list[str]:
        reconcile_agent_runs()
        now = int(time.time())
        queued = [
            task for task in list_tasks(status="queued", limit=100)
            if task.next_retry_at is None or task.next_retry_at <= now
        ]
        by_board: dict[str, list[DevelopmentTask]] = {}
        for task in queued:
            if task.kanban_board and task.kanban_task_id:
                by_board.setdefault(task.kanban_board, []).append(task)
        dispatched: list[str] = []
        for board, tasks in by_board.items():
            by_kanban = {task.kanban_task_id: task for task in tasks if task.kanban_task_id}

            def record_spawn(kanban_task, workspace_path: str, board: str) -> Optional[int]:
                task = by_kanban.get(getattr(kanban_task, "id", None))
                if task is None:
                    return None
                if os.path.isdir(workspace_path):
                    ensure_workflow_policy(workspace_path)
                with connect() as conn:
                    workspace_event = conn.execute(
                        "SELECT 1 FROM development_events WHERE task_id=? AND kind='workspace.created' LIMIT 1",
                        (task.task_id,),
                    ).fetchone()
                    if workspace_event is None and os.path.isdir(workspace_path):
                        _event(conn, task.task_id, "workspace.created", {
                            "workspace_path": str(Path(workspace_path).resolve())[:500],
                            "branch_name": task.branch_name,
                        })
                    existing = latest_agent_run(task.task_id)
                    if existing and existing.status == "running":
                        return existing.worker_pid
                    try:
                        claimed = _transition(
                            conn, task.task_id, "provisioning_workspace", kind="agent.claimed",
                            payload={"kanban_task_id": task.kanban_task_id},
                            workspace_path=workspace_path,
                            expected_status="queued",
                        )
                    except ValueError as exc:
                        if "invalid Symphony transition" not in str(exc) and "conditional Symphony transition lost" not in str(exc):
                            raise
                        return None
                    if claimed.status != "provisioning_workspace":
                        return None
                    now = int(time.time())
                    run_id = f"run_{uuid.uuid4().hex[:12]}"
                    if existing and existing.status == "queued":
                        try:
                            pid = (
                                spawn_fn(kanban_task, workspace_path, board)
                                if spawn_fn
                                else kb._default_spawn(kanban_task, workspace_path, board=board)
                            )
                        except Exception as exc:
                            conn.execute(
                                "UPDATE agent_runs SET status='failed', finished_at=?, error=? WHERE run_id=?",
                                (int(time.time()), str(exc), existing.run_id),
                            )
                            raise
                        conn.execute(
                            "UPDATE agent_runs SET status='running', worker_pid=?, workspace_path=?, branch_name=?, last_progress_at=? WHERE run_id=?",
                            (pid, workspace_path, task.branch_name, int(time.time()), existing.run_id),
                        )
                        _transition(
                            conn, task.task_id, "running", kind="agent.started",
                            payload={"run_id": existing.run_id, "worker_pid": pid},
                            workspace_path=workspace_path,
                        )
                        _event(conn, task.task_id, "agent.started", {"run_id": existing.run_id, "worker_pid": pid})
                        dispatched.append(task.task_id)
                        return pid
                    conn.execute("BEGIN IMMEDIATE")
                    try:
                        conn.execute(
                            """INSERT INTO agent_runs(
                               run_id, task_id, kanban_task_id, workspace_path,
                               branch_name, status, started_at, last_progress_at
                               ) VALUES (?, ?, ?, ?, ?, 'starting', ?, ?)""",
                            (run_id, task.task_id, task.kanban_task_id,
                             workspace_path, task.branch_name, now, now),
                        )
                        _transition(
                            conn, task.task_id, "running", kind="agent.starting",
                            payload={"run_id": run_id}, workspace_path=workspace_path,
                        )
                        conn.execute("COMMIT")
                    except Exception:
                        conn.execute("ROLLBACK")
                        raise
                    try:
                        pid = (
                            spawn_fn(kanban_task, workspace_path, board)
                            if spawn_fn
                            else kb._default_spawn(kanban_task, workspace_path, board=board)
                        )
                    except Exception as exc:
                        with connect() as failed_conn:
                            failed_conn.execute(
                                "UPDATE agent_runs SET status='failed', finished_at=?, error=? WHERE run_id=?",
                                (int(time.time()), str(exc), run_id),
                            )
                            _transition(
                                failed_conn, task.task_id, "failed", kind="agent.failed",
                                payload={"run_id": run_id}, error=str(exc),
                            )
                        raise
                    with connect() as conn:
                        conn.execute("BEGIN IMMEDIATE")
                        try:
                            conn.execute(
                                "UPDATE agent_runs SET status='running', worker_pid=?, last_progress_at=? WHERE run_id=?",
                                (pid, int(time.time()), run_id),
                            )
                            _event(conn, task.task_id, "agent.started", {"run_id": run_id, "worker_pid": pid})
                            conn.execute("COMMIT")
                        except Exception:
                            conn.execute("ROLLBACK")
                            raise
                dispatched.append(task.task_id)
                return pid

            with kb.connect_closing(board=board) as conn:
                kb.dispatch_once(conn, board=board, spawn_fn=record_spawn, max_spawn=self.max_concurrency)
        return dispatched


def ensure_workflow_policy(workspace: Path | str) -> Path:
    """Provision the minimal coding-agent policy without overwriting operator work."""
    root = Path(workspace)
    if not root.is_dir():
        raise ValueError("workspace must be an existing directory")
    workflow = root / "WORKFLOW.md"
    if not workflow.exists():
        workflow.write_text(_WORKFLOW_TEMPLATE, encoding="utf-8")
    return workflow


def _redact_artifact_value(value: Any) -> Any:
    if isinstance(value, str):
        return _redacted_summary(value)
    if isinstance(value, list):
        return [_redact_artifact_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _redact_artifact_value(item) for key, item in value.items()}
    return value


def write_result_artifact(
    workspace: Path | str,
    *,
    task_id: str,
    changed_files: list[str],
    tests: list[dict[str, Any]],
    summary: str,
    limitations: list[str],
    branch: Optional[str] = None,
    pr_url: Optional[str] = None,
) -> Path:
    """Write a redacted, machine-readable worker handoff inside its worktree."""
    root = Path(workspace).resolve()
    if not root.is_dir():
        raise ValueError("workspace must be an existing directory")
    safe_task_id = "".join(char for char in task_id if char.isalnum() or char in {"_", "-"})
    if not safe_task_id:
        raise ValueError("task_id must contain safe filename characters")
    artifact_dir = root / ".symphony"
    artifact_dir.mkdir(exist_ok=True)
    if artifact_dir.resolve() != artifact_dir:
        raise ValueError("result artifact directory is outside the workspace")
    artifact = artifact_dir / f"{safe_task_id}-result.json"
    payload = _redact_artifact_value({
        "task_id": task_id,
        "branch": branch,
        "pr_url": pr_url,
        "changed_files": changed_files,
        "tests": tests,
        "summary": summary,
        "limitations": limitations,
        "created_at": int(time.time()),
    })
    _validate_result_payload(task_id, payload)
    temp = artifact.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(artifact)
    return artifact


def _validate_result_payload(task_id: str, payload: Any) -> dict[str, Any]:
    """Validate an in-memory artifact before it can reach the filesystem."""
    required = {"task_id", "changed_files", "tests", "summary", "limitations", "created_at"}
    if not isinstance(payload, dict) or payload.get("task_id") != task_id or not required.issubset(payload):
        raise ValueError("result artifact has invalid schema or task binding")
    if not isinstance(payload["changed_files"], list) or not isinstance(payload["tests"], list):
        raise ValueError("result artifact has invalid lists")
    for changed_file in payload["changed_files"]:
        if not isinstance(changed_file, str) or not changed_file.strip():
            raise ValueError("result artifact has invalid changed file")
        candidate = Path(changed_file)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("result artifact contains a path outside the workspace")
    tests = payload["tests"]
    if not tests or any(
        not isinstance(test, dict)
        or not isinstance(test.get("command"), str)
        or not test.get("command", "").strip()
        or not isinstance(test.get("exit_code"), int)
        or test["exit_code"] != 0
        for test in tests
    ):
        raise ValueError("result artifact lacks passing test evidence")
    return payload


def _write_failure_artifact(
    workspace: Path | str,
    *,
    task_id: str,
    tests: list[dict[str, Any]],
    summary: str,
    branch: Optional[str] = None,
    pr_url: Optional[str] = None,
) -> Path:
    """Persist bounded diagnostic evidence that is explicitly non-completing."""
    root = Path(workspace).resolve()
    if not root.is_dir():
        raise ValueError("workspace must be an existing directory")
    safe_task_id = "".join(char for char in task_id if char.isalnum() or char in {"_", "-"})
    if not safe_task_id:
        raise ValueError("task_id must contain safe filename characters")
    artifact_dir = root / ".symphony"
    artifact_dir.mkdir(exist_ok=True)
    if artifact_dir.resolve() != artifact_dir:
        raise ValueError("failure artifact directory is outside the workspace")
    artifact = artifact_dir / f"{safe_task_id}-failure.json"
    payload = _redact_artifact_value({
        "task_id": task_id, "branch": branch, "pr_url": pr_url,
        "tests": tests, "summary": summary, "failure": True,
        "created_at": int(time.time()),
    })
    temp = artifact.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(artifact)
    return artifact
def validate_result_artifact(task_id: str, workspace: Path | str) -> dict[str, Any]:
    """Validate the worker handoff schema and workspace/task binding."""
    root = Path(workspace).resolve()
    artifact = root / ".symphony" / f"{task_id}-result.json"
    resolved_artifact = artifact.resolve(strict=False)
    try:
        resolved_artifact.relative_to(root)
    except ValueError as exc:
        raise ValueError("result artifact is outside the workspace") from exc
    if not artifact.is_file():
        raise ValueError("result artifact is missing")
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("result artifact is not valid JSON") from exc
    return _validate_result_payload(task_id, payload)




def run_worker_checks(task_id: str, *, commands: list[list[str]], timeout_seconds: int = 600) -> dict[str, Any]:
    """Run declared checks in the task workspace and create the completion artifact."""
    if not commands:
        raise ValueError("at least one worker check command is required")
    with connect() as conn:
        task = _get(conn, task_id)
    if task is None:
        raise ValueError(f"development task {task_id} not found")
    if task.status not in {"running", "validating"}:
        raise ValueError("worker checks require a running or validating task")
    if not task.workspace_path:
        raise ValueError("worker checks require a task workspace")
    workspace = Path(task.workspace_path).resolve()
    if not workspace.is_dir():
        raise ValueError("task workspace does not exist")
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            current = _get(conn, task_id)
            if current is None:
                raise ValueError(f"development task {task_id} not found")
            if current.status == "running":
                _transition(conn, task_id, "validating", kind="validation.started", payload={"checks": len(commands)})
            else:
                _event(conn, task_id, "validation.started", {"checks": len(commands)})
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    results: list[dict[str, Any]] = []
    for command in commands:
        if not isinstance(command, list) or not command or any(not isinstance(part, str) or not part for part in command):
            raise ValueError("worker commands must be non-empty argument lists")
        try:
            completed = subprocess.run(
                command, cwd=workspace, capture_output=True, text=True,
                timeout=max(1, int(timeout_seconds)), check=False,
            )
            exit_code = int(completed.returncode)
            output = _redacted_summary((completed.stdout or "") + (completed.stderr or ""))[:1000]
        except subprocess.TimeoutExpired as exc:
            exit_code = 124
            output = _redacted_summary(str(exc))[:1000]
        results.append({"command": " ".join(command)[:400], "exit_code": exit_code, "summary": output})
        audit_command = list(command)
        if audit_command and ("\\" in audit_command[0] or "/" in audit_command[0]):
            audit_command[0] = Path(audit_command[0]).name
        with connect() as conn:
            _event(conn, task_id, "validation.check", {
                "command": _redacted_summary(" ".join(audit_command))[:400],
                "cwd": "task workspace",
                "exit_code": exit_code,
                "timed_out": exit_code == 124,
            })
        record_progress(task_id, action=" ".join(command)[:240], detail=f"exit_code={exit_code}")
        if exit_code != 0:
            break
    checks_passed = all(item["exit_code"] == 0 for item in results)
    artifact = (
        write_result_artifact(
            workspace, task_id=task_id, changed_files=[], tests=results,
            summary="Worker checks completed", limitations=[], branch=task.branch_name, pr_url=task.pr_url,
        )
        if checks_passed else _write_failure_artifact(
            workspace, task_id=task_id, tests=results,
            summary="Worker checks failed", branch=task.branch_name, pr_url=task.pr_url,
        )
    )
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            _event(conn, task_id, "validation.finished", {
                "passed": checks_passed,
                "checks": len(results),
                "artifact": str(artifact),
            })
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    if checks_passed:
        complete_agent_run(
            task_id,
            test_summary=_redacted_summary("; ".join(
                f"{item['command']} -> exit_code={item['exit_code']}"
                + (f": {item['summary']}" if item["summary"] else "")
                for item in results
            )),
            result_summary="Worker checks passed", evidence_path=str(artifact),
        )
    else:
        with connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                failed_task = _get(conn, task_id)
                failed_run = _get_latest_agent_run_conn(conn, task_id)
                if failed_run is not None and failed_run.status not in {"failed", "cancelled", "completed"}:
                    now = int(time.time())
                    conn.execute(
                        "UPDATE agent_runs SET status='failed', finished_at=?, last_progress_at=?, result_artifact=?, error=? WHERE run_id=?",
                        (now, now, str(artifact), "worker checks failed", failed_run.run_id),
                    )
                    _event(conn, task_id, "agent.failed", {
                        "run_id": failed_run.run_id,
                        "artifact": str(artifact),
                    })
                if failed_task is not None and failed_task.status in {"running", "validating"}:
                    _transition(conn, task_id, "failed", kind="task.failed", payload={
                        "artifact": str(artifact), "checks": len(results),
                    }, result_summary="Worker checks failed")
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
    return {"artifact": str(artifact), "tests": results}


def _get_latest_agent_run_conn(conn: sqlite3.Connection, task_id: str) -> Optional[AgentRun]:
    row = conn.execute(
        "SELECT * FROM agent_runs WHERE task_id = ? ORDER BY started_at DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    return AgentRun.from_row(row) if row else None


def latest_agent_run(task_id: str) -> Optional[AgentRun]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM agent_runs WHERE task_id = ? ORDER BY started_at DESC LIMIT 1", (task_id,)).fetchone()
        return AgentRun.from_row(row) if row else None


def _update_agent_run(task_id: str, *, status: str, summary: Optional[str] = None,
                      error: Optional[str] = None, artifact: Optional[str] = None) -> Optional[AgentRun]:
    """Persist a worker handoff without exposing secrets or raw logs."""
    run = latest_agent_run(task_id)
    if run is None:
        return None
    now = int(time.time())
    finished = now if status in {"completed", "failed", "cancelled"} else None
    with connect() as conn:
        conn.execute(
            """UPDATE agent_runs
               SET status=?, finished_at=COALESCE(?, finished_at),
                   last_progress_at=?, result_artifact=COALESCE(?, result_artifact),
                   test_summary=COALESCE(?, test_summary), error=COALESCE(?, error)
               WHERE run_id=?""",
            (status, finished, now, artifact, summary, error, run.run_id),
        )
        payload = {"run_id": run.run_id, "status": status}
        if summary:
            payload["summary"] = summary[:400]
        _event(conn, task_id, f"agent.{status}", payload)
        return latest_agent_run(task_id)


def complete_agent_run(task_id: str, *, test_summary: str, result_summary: str,
                       evidence_path: Optional[str] = None) -> DevelopmentTask:
    """Close the Symphony projection only after the worker reports evidence."""
    if not test_summary.strip() or not result_summary.strip():
        raise ValueError("test_summary and result_summary are required")
    with connect() as check_conn:
        check_task = _get(check_conn, task_id)
    if check_task is None:
        raise ValueError(f"development task {task_id} not found")
    workspace_path = (check_task.workspace_path or "").strip()
    if not workspace_path:
        raise ValueError("task must have a resolved workspace before completion")
    workspace = Path(workspace_path).resolve()
    artifact_path = workspace / ".symphony" / f"{task_id}-result.json"
    if evidence_path:
        evidence = Path(evidence_path).resolve()
        if not artifact_path.is_file() and evidence != artifact_path:
            raise ValueError("result artifact is required before completion")
    validate_result_artifact(task_id, workspace)
    canonical_evidence_path = str(artifact_path)
    if evidence_path and Path(evidence_path).resolve() != artifact_path:
        raise ValueError("evidence_path must be the in-workspace result artifact")
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            task = _get(conn, task_id)
            if task is None:
                raise ValueError(f"development task {task_id} not found")
            if task.status not in {"running", "validating"}:
                raise ValueError("only running or validating tasks can complete")
            run = _get_latest_agent_run_conn(conn, task_id)
            if run is None:
                raise ValueError("active AgentRun not found")
            now = int(time.time())
            conn.execute(
                """UPDATE agent_runs SET status='completed', finished_at=?, last_progress_at=?,
                   result_artifact=COALESCE(?, result_artifact), test_summary=? WHERE run_id=?""",
                (now, now, canonical_evidence_path, test_summary, run.run_id),
            )
            _event(conn, task_id, "agent.completed", {"run_id": run.run_id, "summary": test_summary[:400]})
            _event(conn, task_id, "agent.finished", {"run_id": run.run_id, "summary": test_summary[:400]})
            result = _transition(
                conn, task_id, "awaiting_human_review", kind="task.evidence_ready",
                payload={"evidence_path": canonical_evidence_path}, test_summary=test_summary,
                result_summary=result_summary,
            )
            conn.execute("COMMIT")
            return result
        except Exception:
            conn.execute("ROLLBACK")
            raise


def _redacted_summary(value: str) -> str:
    return redact_sensitive_text(str(value or ""), force=True)[:4000]


def publish_for_review(
    task_id: str,
    *,
    commit_message: str,
    test_summary: str,
    result_summary: str,
) -> DevelopmentTask:
    """Commit, push, and open/reuse a GitHub PR; never merge or deploy.

    The durable Kanban card is the canonical review handoff. This thin adapter
    only projects its safe, concise result into Symphony for Telegram/CLI.
    """
    if not commit_message.strip() or not test_summary.strip() or not result_summary.strip():
        raise ValueError("commit_message, test_summary and result_summary are required")
    with connect() as conn:
        task = _get(conn, task_id)
    if task is None:
        raise ValueError(f"development task {task_id} not found")
    if task.status not in {"running", "validating"}:
        raise ValueError("only running or validating tasks can be published for review")
    if not task.kanban_task_id or not task.kanban_board or not task.workspace_path:
        raise ValueError("a running Kanban worktree is required before publication")

    workspace = task.workspace_path
    evidence_artifact = Path(workspace) / ".symphony" / f"{task.task_id}-result.json"
    if not evidence_artifact.is_file():
        raise ValueError("result artifact is required before publishing for review")
    validate_result_artifact(task.task_id, workspace)
    ci = web_git.review_ci_snapshot(workspace)
    if ci.get("state") in {"pending", "fail"}:
        raise ValueError(f"CI checks are not passing: {ci.get('state')}")
    safe_tests = _redacted_summary(test_summary)
    safe_result = _redacted_summary(result_summary)
    safe_commit_message = _redacted_summary(commit_message)
    ship = web_git.review_ship_info(workspace)
    if not ship.get("ghReady"):
        raise ValueError("GitHub CLI is unavailable or not authenticated for this worktree")
    pr = ship.get("pr")
    if not pr:
        web_git.review_commit(workspace, safe_commit_message, push=True)
        pr = web_git.review_create_pr(workspace)
    pr_url = str((pr or {}).get("url") or "").strip()
    if not pr_url:
        raise ValueError("GitHub did not return a pull request URL")

    with kb.connect_closing(board=task.kanban_board) as kconn:
        kanban_task = kb.get_task(kconn, task.kanban_task_id)
        if kanban_task is None:
            raise ValueError("linked Kanban task no longer exists")
        metadata = {
            "branch": task.branch_name,
            "pr_url": pr_url,
            "tests": safe_tests,
        }
        if not kb.request_review(
            kconn,
            task.kanban_task_id,
            summary=safe_result,
            metadata=metadata,
            expected_run_id=getattr(kanban_task, "current_run_id", None),
        ):
            raise ValueError("Kanban refused the review handoff; the active worker state changed")

    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            active_run = _get_latest_agent_run_conn(conn, task_id)
            if active_run is not None:
                now = int(time.time())
                conn.execute(
                    """UPDATE agent_runs SET status='review_requested', finished_at=?,
                       last_progress_at=?, test_summary=? WHERE run_id=?""",
                    (now, now, safe_tests, active_run.run_id),
                )
            result = _transition(
                conn, task_id, "awaiting_human_review", kind="task.review_requested",
                payload={"pr_url": pr_url, "ci_state": ci.get("state")}, pr_url=pr_url,
                test_summary=safe_tests, result_summary=safe_result,
                ci_state=ci.get("state"), ci_checks=json.dumps(ci.get("checks", [])[:50], ensure_ascii=False),
            )
            _event(conn, task_id, "pr.created", {"pr_url": pr_url, "ci_state": ci.get("state")})
            conn.execute("COMMIT")
            return result
        except Exception:
            conn.execute("ROLLBACK")
            raise


def reconcile_agent_runs(*, stale_after_seconds: int = 300) -> list[DevelopmentTask]:
    """Recover durable starts whose worker stopped reporting progress."""
    cutoff = int(time.time()) - max(1, int(stale_after_seconds))
    changed: list[DevelopmentTask] = []
    with connect() as conn:
        orphaned = conn.execute(
            "SELECT task_id FROM development_tasks "
            "WHERE status='provisioning_workspace' AND updated_at < ?",
            (cutoff,),
        ).fetchall()
        for row in orphaned:
            task = _get(conn, row["task_id"])
            if task is None:
                continue
            conn.execute("BEGIN IMMEDIATE")
            try:
                changed.append(_transition(
                    conn, task.task_id, "failed", kind="agent.recovered",
                    payload={"reason": "workspace_claim_expired"},
                    error="Workspace claim expired before AgentRun creation; task can be resumed.",
                ))
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        rows = conn.execute(
            "SELECT task_id, run_id, worker_pid FROM agent_runs "
            "WHERE status IN ('starting','running','validating') AND last_progress_at < ?",
            (cutoff,),
        ).fetchall()
        for row in rows:
            task = _get(conn, row["task_id"])
            if task is None or task.status in TERMINAL_STATUSES:
                continue
            worker_pid = row["worker_pid"]
            if worker_pid:
                _terminate_worker_process(int(worker_pid))
            conn.execute("BEGIN IMMEDIATE")
            try:
                now = int(time.time())
                conn.execute(
                    "UPDATE agent_runs SET status='failed', finished_at=?, error=? WHERE run_id=?",
                    (now, "worker heartbeat expired", row["run_id"]),
                )
                changed.append(_transition(
                    conn, task.task_id, "failed", kind="agent.recovered",
                    payload={"run_id": row["run_id"], "reason": "heartbeat_expired"},
                    error="Worker heartbeat expired; task can be resumed.",
                ))
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
    return changed


def _ensure_agent_run_from_kanban(conn: sqlite3.Connection, task: DevelopmentTask, kanban_task: Any) -> None:
    """Project a Kanban worker into AgentRun exactly once during reconciliation."""
    existing = conn.execute(
        "SELECT run_id, status FROM agent_runs WHERE task_id = ? AND kanban_task_id = ?",
        (task.task_id, task.kanban_task_id),
    ).fetchone()
    if existing:
        if existing["status"] == "queued":
            now = int(time.time())
            conn.execute(
                "UPDATE agent_runs SET status='running', worker_pid=?, workspace_path=?, branch_name=?, last_progress_at=? WHERE run_id=?",
                (getattr(kanban_task, "worker_pid", None), getattr(kanban_task, "workspace_path", None), getattr(kanban_task, "branch_name", None), now, existing["run_id"]),
            )
            _event(conn, task.task_id, "agent.started", {"source": "kanban", "run_id": existing["run_id"]})
        return
    now = int(time.time())
    conn.execute(
        """INSERT INTO agent_runs(
           run_id, task_id, kanban_task_id, workspace_path, branch_name,
           worker_pid, status, started_at, last_progress_at
           ) VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?)""",
        (
            f"run_{uuid.uuid4().hex[:12]}", task.task_id, task.kanban_task_id,
            getattr(kanban_task, "workspace_path", None),
            getattr(kanban_task, "branch_name", None),
            getattr(kanban_task, "worker_pid", None), now, now,
        ),
    )
    _event(conn, task.task_id, "agent.projected", {"source": "kanban"})


def _finish_agent_run_from_kanban(conn: sqlite3.Connection, task: DevelopmentTask, kanban_task: Any) -> None:
    """Store the Kanban handoff as evidence without inventing test results."""
    now = int(time.time())
    summary = getattr(kanban_task, "result", None)
    conn.execute(
        """UPDATE agent_runs SET status='completed', finished_at=?, last_progress_at=?,
           test_summary=COALESCE(test_summary, ?)
           WHERE task_id = ? AND status IN ('starting','running')""",
        (now, now, summary, task.task_id),
    )


def symphony_home() -> Path:
    return Path(get_hermes_home()) / "symphony"


def db_path() -> Path:
    return symphony_home() / "symphony.db"


@contextlib.contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        for attempt in range(20):
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == 19:
                    raise
                time.sleep(0.05 * (attempt + 1))
        conn.executescript(_SCHEMA)
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(development_tasks)")}
        for name, definition in (("next_retry_at", "INTEGER"), ("last_action", "TEXT"), ("progress_detail", "TEXT"), ("ci_state", "TEXT"), ("ci_checks", "TEXT"), ("source_issue_id", "TEXT"), ("source_issue_url", "TEXT")):
            if name not in columns:
                conn.execute(f"ALTER TABLE development_tasks ADD COLUMN {name} {definition}")
        agent_indexes = conn.execute("PRAGMA index_list(agent_runs)").fetchall()
        if any(row["unique"] and "kanban_task_id" in {col["name"] for col in conn.execute(f"PRAGMA index_info('{row['name']}')")} for row in agent_indexes):
            conn.execute("ALTER TABLE agent_runs RENAME TO agent_runs_legacy")
            conn.executescript("""
                CREATE TABLE agent_runs (
                  run_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, kanban_task_id TEXT,
                  workspace_path TEXT, branch_name TEXT, worker_pid INTEGER, status TEXT NOT NULL,
                  started_at INTEGER NOT NULL, finished_at INTEGER, last_progress_at INTEGER NOT NULL,
                  result_artifact TEXT, test_summary TEXT, error TEXT
                );
                INSERT INTO agent_runs(run_id, task_id, kanban_task_id, workspace_path, branch_name, worker_pid,
                  status, started_at, finished_at, last_progress_at, result_artifact, test_summary, error)
                SELECT run_id, task_id, kanban_task_id, workspace_path, branch_name, worker_pid,
                  status, started_at, finished_at, last_progress_at, result_artifact, test_summary, error
                FROM agent_runs_legacy;
                DROP TABLE agent_runs_legacy;
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_runs_task ON agent_runs(task_id, started_at DESC)")
        delivery_columns = {row["name"] for row in conn.execute("PRAGMA table_info(delivery_events)")}
        if "claimed_at" not in delivery_columns:
            conn.execute("ALTER TABLE delivery_events ADD COLUMN claimed_at INTEGER NOT NULL DEFAULT 0")
        yield conn
    finally:
        conn.close()


def _event(conn: sqlite3.Connection, task_id: str, kind: str, payload: Optional[dict[str, Any]] = None) -> None:
    conn.execute(
        "INSERT INTO development_events(task_id, kind, payload, created_at) VALUES (?, ?, ?, ?)",
        (task_id, kind, json.dumps(payload, ensure_ascii=False) if payload else None, int(time.time())),
    )
    conn.execute(
        "INSERT INTO delivery_events(task_id, kind, delivered_at, payload) VALUES (?, ?, 0, ?)",
        (task_id, kind, json.dumps(payload, ensure_ascii=False) if payload else None),
    )


def claim_delivery_events(*, limit: int = 20, claim_ttl_seconds: int = 120) -> list[dict[str, Any]]:
    """Atomically claim undelivered or stale-claimed lifecycle notifications."""
    now = int(time.time())
    stale_before = now - max(1, int(claim_ttl_seconds))
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            rows = conn.execute(
                "SELECT id, task_id, kind, payload FROM delivery_events "
                "WHERE delivered_at = 0 OR (delivered_at = -1 AND claimed_at <= ?) "
                "ORDER BY id LIMIT ?",
                (stale_before, max(1, min(int(limit), 100))),
            ).fetchall()
            ids = [row["id"] for row in rows]
            if ids:
                conn.execute(
                    f"UPDATE delivery_events SET delivered_at = -1, claimed_at = ? WHERE id IN ({','.join('?' for _ in ids)})",
                    [now, *ids],
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    result = []
    for row in rows:
        try:
            payload = json.loads(row["payload"] or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {}
        result.append({"id": row["id"], "task_id": row["task_id"], "kind": row["kind"], "payload": payload})
    return result


def mark_delivery_delivered(event_id: int) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE delivery_events SET delivered_at = ?, claimed_at = 0 WHERE id = ? AND delivered_at = -1",
            (int(time.time()), int(event_id)),
        )


def release_delivery_event(event_id: int) -> None:
    """Return an unsent claim to the durable outbox after a delivery failure."""
    with connect() as conn:
        conn.execute(
            "UPDATE delivery_events SET delivered_at = 0, claimed_at = 0 WHERE id = ? AND delivered_at = -1",
            (int(event_id),),
        )


def status_report(*, limit: int = 20) -> dict[str, Any]:
    """Return durable queue/status facts without reading worker logs or secrets."""
    with connect() as conn:
        counts = {row["status"]: int(row["n"]) for row in conn.execute(
            "SELECT status, COUNT(*) AS n FROM development_tasks GROUP BY status"
        )}
        rows = conn.execute(
            """SELECT t.task_id, t.title, t.status, t.branch_name, t.workspace_path, t.pr_url,
                      t.test_summary, t.error, t.retry_count, t.next_retry_at, t.last_action,
                      t.progress_detail, r.last_progress_at, r.status AS run_status
                 FROM development_tasks t
                 LEFT JOIN agent_runs r ON r.run_id = (
                     SELECT run_id FROM agent_runs WHERE task_id=t.task_id ORDER BY started_at DESC LIMIT 1
                 )
                 ORDER BY t.updated_at DESC LIMIT ?""",
            (max(1, min(int(limit), 100)),),
        ).fetchall()
    active_statuses = {"queued", "provisioning_workspace", "running", "validating"}
    recent = [dict(row) for row in rows]
    active = [row for row in recent if row["status"] in active_statuses]
    return {
        "counts": counts,
        "queue_size": sum(counts.get(status, 0) for status in {"queued", "ready_for_development"}),
        "active": active,
        "recent": recent,
    }


def record_progress(
    task_id: str,
    *,
    action: str,
    detail: str = "",
    run_id: Optional[str] = None,
) -> DevelopmentTask:
    """Persist a bounded, redacted worker progress update.

    ``run_id`` is optional for compatibility with operator-side updates, but
    worker callbacks should provide it so a stale worker cannot refresh a
    newer retry's heartbeat.
    """
    safe_action = _redacted_summary(action)[:240]
    safe_detail = _redacted_summary(detail)[:600]
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            task = _get(conn, task_id)
            if task is None:
                raise ValueError(f"development task {task_id} not found")
            run = _get_latest_agent_run_conn(conn, task_id)
            if run_id is not None:
                if run is None or run.run_id != run_id:
                    raise ValueError("agent run is no longer current")
            if run and run.status in {"starting", "running"}:
                conn.execute("UPDATE agent_runs SET last_progress_at=? WHERE run_id=?", (int(time.time()), run.run_id))
            result = _transition(conn, task_id, task.status, kind="agent.progress",
                                 payload={"action": safe_action, "detail": safe_detail},
                                 last_action=safe_action, progress_detail=safe_detail)
            conn.execute("COMMIT")
            return result
        except Exception:
            conn.execute("ROLLBACK")
            raise


def schedule_retry(task_id: str, *, reason: str, transient: bool, max_retries: int = 3) -> DevelopmentTask:
    """Retry only explicitly transient failures and stop at a finite bound."""
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            task = _get(conn, task_id)
            if task is None:
                raise ValueError(f"development task {task_id} not found")
            safe_reason = _redacted_summary(reason)
            allowed = transient and task.retry_count < max(0, int(max_retries))
            if allowed:
                retry = task.retry_count + 1
                next_at = int(time.time()) + min(3600, 30 * (2 ** (retry - 1)))
                result = _transition(conn, task_id, "queued", kind="agent.retry_scheduled",
                                     payload={"reason": safe_reason, "retry": retry},
                                     retry_count=retry, next_retry_at=next_at, error=safe_reason)
            else:
                result = _transition(conn, task_id, "failed", kind="task.failed",
                                     payload={"reason": safe_reason, "transient": bool(transient)}, error=safe_reason)
            conn.execute("COMMIT")
            return result
        except Exception:
            conn.execute("ROLLBACK")
            raise


def _get(conn: sqlite3.Connection, task_id: str) -> Optional[DevelopmentTask]:
    row = conn.execute("SELECT * FROM development_tasks WHERE task_id = ?", (task_id,)).fetchone()
    return DevelopmentTask.from_row(row) if row else None


def create_task(
    *, title: str, source_text: str, technical_brief: Optional[str] = None,
    project_id: Optional[str] = None, repository: Optional[str] = None,
    requester_id: Optional[str] = None, telegram_chat_id: Optional[str] = None,
    telegram_thread_id: Optional[str] = None, priority: int = 0,
    idempotency_key: Optional[str] = None, board: Optional[str] = None,
    source_issue_id: Optional[str] = None, source_issue_url: Optional[str] = None,
) -> DevelopmentTask:
    """Persist a task in draft state; duplicate inbound events return the same row."""
    title = title.strip()
    source_text = source_text.strip()
    if not title or not source_text:
        raise ValueError("title and source_text are required")
    now = int(time.time())
    with connect() as conn:
        if idempotency_key:
            existing = conn.execute(
                "SELECT * FROM development_tasks WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
            if existing:
                return DevelopmentTask.from_row(existing)
        task_id = f"dev_{uuid.uuid4().hex[:12]}"
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                """INSERT INTO development_tasks(
                    task_id,title,source_text,technical_brief,project_id,repository,
                    telegram_chat_id,telegram_thread_id,requester_id,status,priority,
                    idempotency_key,kanban_board,source_issue_id,source_issue_url,created_at,updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'draft', ?, ?, ?, ?, ?, ?, ?)""",
                (task_id, title, source_text, technical_brief, project_id, repository,
                 telegram_chat_id, telegram_thread_id, requester_id, int(priority),
                 idempotency_key, board, source_issue_id, source_issue_url, now, now),
            )
            _event(conn, task_id, "task.created", {"status": "draft"})
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return _get(conn, task_id)  # type: ignore[return-value]


def _assert_transition(current: str, target: str) -> None:
    allowed = {
        "draft": {"needs_clarification", "ready_for_development", "cancelled"},
        "needs_clarification": {"draft", "ready_for_development", "cancelled"},
        "ready_for_development": {"queued", "cancelled"},
        "queued": {"provisioning_workspace", "running", "failed", "cancelled", "paused"},
        "provisioning_workspace": {"running", "failed", "cancelled"},
        "running": {"validating", "awaiting_human_review", "completed", "failed", "cancelled", "paused"},
        "validating": {"awaiting_human_review", "completed", "failed", "cancelled", "paused"},
        "awaiting_human_review": {"completed", "queued", "failed", "cancelled"},
        "paused": {"queued", "cancelled"},
        "failed": {"queued", "cancelled"},
        "completed": set(),
        "cancelled": set(),
    }
    if target not in allowed.get(current, set()):
        raise ValueError(f"invalid Symphony transition: {current} -> {target}")


def _transition(conn: sqlite3.Connection, task_id: str, target: str, *, kind: str, payload: Optional[dict[str, Any]] = None, expected_status: Optional[str] = None, **fields: Any) -> DevelopmentTask:
    current = _get(conn, task_id)
    if current is None:
        raise ValueError(f"development task {task_id} not found")
    if target not in STATUSES:
        raise ValueError(f"unknown development status {target!r}")
    if current.status != target:
        _assert_transition(current.status, target)
    now = int(time.time())
    values = {"status": target, "updated_at": now, **fields}
    columns = ", ".join(f"{key} = ?" for key in values)
    where = "task_id = ?"
    params = [*values.values(), task_id]
    if expected_status is not None:
        where += " AND status = ?"
        params.append(expected_status)
    cursor = conn.execute(f"UPDATE development_tasks SET {columns} WHERE {where}", params)
    if expected_status is not None and cursor.rowcount != 1:
        raise ValueError(f"conditional Symphony transition lost: expected {expected_status}")
    _event(conn, task_id, kind, {"status": target, **(payload or {})})
    return _get(conn, task_id)  # type: ignore[return-value]


def mark_ready(task_id: str, *, brief: Optional[str] = None) -> DevelopmentTask:
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            task = _get(conn, task_id)
            if task is None:
                raise ValueError(f"development task {task_id} not found")
            if not (brief or task.technical_brief or task.source_text).strip():
                raise ValueError("a task needs a brief before it can be developed")
            result = _transition(conn, task_id, "ready_for_development", kind="task.ready", technical_brief=brief or task.technical_brief)
            conn.execute("COMMIT")
            return result
        except Exception:
            conn.execute("ROLLBACK")
            raise


def _prepared_brief(task: DevelopmentTask) -> str:
    """Produce a deterministic, non-LLM preparation contract from the source task."""
    source = _redacted_summary(task.source_text)
    return (
        "## Технический brief\n"
        f"Реализовать минимальное изменение по исходной задаче: {source}\n\n"
        "## Критерии готовности\n"
        "- Изменение ограничено указанным проектом и задачей.\n"
        "- Добавлен или обновлён целевой regression-test.\n"
        "- Целевые проверки выполнены, результаты сохранены в artifact.\n"
        "- Подготовлены branch, commit и PR только после проверок.\n\n"
        "## Риски и вопросы\n"
        "- Требования требуют review перед merge; production действия запрещены.\n"
        "- Если проектный контракт или тестовая команда неясны, задача должна быть остановлена с вопросом."
    )


def prepare_task(task_id: str) -> DevelopmentTask:
    """Move a sufficiently scoped draft into the explicit development-ready state."""
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            task = _get(conn, task_id)
            if task is None:
                raise ValueError(f"development task {task_id} not found")
            if task.status == "ready_for_development":
                conn.execute("COMMIT")
                return task
            if task.status not in {"draft", "needs_clarification"}:
                raise ValueError("only draft or needs_clarification tasks can be prepared")
            if not task.project_id:
                raise ValueError("a registered Hermes project is required before preparation")
            if len(task.source_text.strip()) < 8:
                result = _transition(conn, task_id, "needs_clarification", kind="task.needs_clarification",
                                     payload={"reason": "source_too_short"})
                conn.execute("COMMIT")
                return result
            brief = task.technical_brief or _prepared_brief(task)
            result = _transition(conn, task_id, "ready_for_development", kind="task.prepared",
                                 technical_brief=brief)
            conn.execute("COMMIT")
            return result
        except Exception:
            conn.execute("ROLLBACK")
            raise


def answer_clarification(task_id: str, answer: str) -> DevelopmentTask:
    """Persist a clarification answer and re-prepare a blocked task."""
    answer = redact_sensitive_text(str(answer or "").strip())
    if len(answer) < 8:
        raise ValueError("clarification answer is too short")
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            task = _get(conn, task_id)
            if task is None:
                raise ValueError(f"development task {task_id} not found")
            if task.status != "needs_clarification":
                raise ValueError("only tasks needing clarification can receive an answer")
            source = f"{task.source_text.strip()}\n\nУточнение пользователя:\n{answer}"
            result = _transition(
                conn, task_id, "ready_for_development", kind="task.clarification_answered",
                payload={"answer": answer[:500]}, source_text=source,
                technical_brief=_prepared_brief(dc_replace(task, source_text=source)),
            )
            conn.execute("COMMIT")
            return result
        except Exception:
            conn.execute("ROLLBACK")
            raise
def enqueue(task_id: str, *, board: Optional[str] = None, assignee: Optional[str] = None) -> DevelopmentTask:
    """Create exactly one linked Kanban worktree task for a ready development task."""
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            task = _get(conn, task_id)
            if task is None:
                raise ValueError(f"development task {task_id} not found")
            if task.kanban_task_id:
                return task
            if task.status != "ready_for_development":
                raise ValueError("only ready_for_development tasks can be queued")
            if not task.project_id:
                raise ValueError("a registered Hermes project is required before a task can be queued")
            selected_board = board or task.kanban_board or "development"
            effective_assignee = assignee
            if not effective_assignee:
                from hermes_cli.profiles import get_active_profile_name
                effective_assignee = get_active_profile_name()
            kb.create_board(selected_board, name="Development tasks", description="Hermes Symphony development queue", icon="🛠")
            with kb.connect_closing(board=selected_board) as kconn:
                body = (
                    f"## Исходная задача\n{task.source_text}\n\n"
                    f"## Технический brief\n{task.technical_brief or task.source_text}\n\n"
                    "## Symphony policy\n"
                    "Работай только в назначенном workspace. Не делай merge, production deploy "
                    "или миграции без отдельного подтверждения. Запусти релевантные тесты и "
                    "сохрани фактические результаты в отчёте. Перед `kanban_complete` "
                    "создай Symphony evidence через параметр `evidence`: передай "
                    "непустой `tests` со всеми фактическими командами и `exit_code: 0`, "
                    "а также changed_files/limitations."
                )
                kanban_id = kb.create_task(
                    kconn, title=f"[Symphony {task.task_id}] {task.title}", body=body,
                    assignee=effective_assignee, created_by="symphony", workspace_kind="worktree",
                    project_id=task.project_id, priority=task.priority,
                    idempotency_key=f"symphony:{task.task_id}", initial_status="blocked",
                )
                kb.unblock_task(kconn, kanban_id)
                ktask = kb.get_task(kconn, kanban_id)
                # A fresh Kanban task may be ready immediately; no direct spawning happens here.
            result = _transition(conn, task_id, "queued", kind="task.queued", payload={"kanban_task_id": kanban_id, "board": selected_board}, kanban_task_id=kanban_id, kanban_board=selected_board, workspace_path=ktask.workspace_path if ktask else None, branch_name=ktask.branch_name if ktask else None)
            now = int(time.time())
            conn.execute(
                """INSERT OR IGNORE INTO agent_runs(
                   run_id, task_id, kanban_task_id, workspace_path, branch_name,
                   status, started_at, last_progress_at
                   ) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?)""",
                (
                    f"run_{uuid.uuid4().hex[:12]}", task_id, kanban_id,
                    ktask.workspace_path if ktask else None,
                    ktask.branch_name if ktask else None, now, now,
                ),
            )
            _event(conn, task_id, "agent.queued", {"kanban_task_id": kanban_id})
            conn.execute("COMMIT")
            return result
        except Exception:
            conn.execute("ROLLBACK")
            raise


def _map_kanban_status(status: str) -> str:
    return {
        "ready": "queued", "todo": "queued", "scheduled": "queued", "running": "running",
        "review": "awaiting_human_review", "done": "completed", "blocked": "paused",
        "archived": "cancelled",
    }.get(status, "failed")


def reconcile(task_id: Optional[str] = None) -> list[DevelopmentTask]:
    """Refresh persistent projections from Kanban without launching a worker."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT task_id FROM development_tasks WHERE kanban_task_id IS NOT NULL" + (" AND task_id = ?" if task_id else ""),
            ((task_id,) if task_id else ()),
        ).fetchall()
        refreshed: list[DevelopmentTask] = []
        for row in rows:
            task = _get(conn, row["task_id"])
            if task is None or not task.kanban_task_id or not task.kanban_board:
                continue
            try:
                with kb.connect_closing(board=task.kanban_board) as kconn:
                    ktask = kb.get_task(kconn, task.kanban_task_id)
            except Exception as exc:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    refreshed.append(_transition(conn, task.task_id, "failed", kind="task.failed", payload={"reason": "kanban_unavailable"}, error=f"Kanban unavailable: {exc}"))
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                continue
            if ktask is None:
                continue
            target = _map_kanban_status(ktask.status)
            current = _get(conn, task.task_id)
            if current is None or current.status == target or current.status in TERMINAL_STATUSES:
                if current is not None and target == "running":
                    _ensure_agent_run_from_kanban(conn, current, ktask)
                if current is not None and target == "completed" and current.status != "completed":
                    _finish_agent_run_from_kanban(conn, current, ktask)
                    _event(conn, task.task_id, "task.completed", {
                        "source": "kanban",
                        "kanban_task_id": task.kanban_task_id,
                    })
                refreshed.append(current or task)
                continue
            # Some runner states are intentionally compacted into Kanban's smaller state machine.
            allowed = target in {"queued", "running", "awaiting_human_review", "completed", "paused", "cancelled"}
            if not allowed:
                continue
            conn.execute("BEGIN IMMEDIATE")
            try:
                fields: dict[str, Any] = {"workspace_path": ktask.workspace_path, "branch_name": ktask.branch_name}
                if target == "completed":
                    try:
                        validate_result_artifact(task.task_id, ktask.workspace_path or task.workspace_path or "")
                    except ValueError as exc:
                        conn.execute("ROLLBACK")
                        with connect() as event_conn:
                            _event(event_conn, task.task_id, "task.completion_blocked", {
                                "reason": "missing_or_invalid_evidence",
                                "detail": str(exc)[:240],
                            })
                        refreshed.append(_get(conn, task.task_id) or task)
                        continue
                    fields["result_summary"] = ktask.result
                refreshed_task = _transition(conn, task.task_id, target, kind=f"kanban.{ktask.status}", **fields)
                if target == "running":
                    _ensure_agent_run_from_kanban(conn, refreshed_task, ktask)
                elif target == "completed":
                    _finish_agent_run_from_kanban(conn, refreshed_task, ktask)
                    _event(conn, task.task_id, "task.completed", {
                        "source": "kanban",
                        "kanban_task_id": task.kanban_task_id,
                    })
                refreshed.append(refreshed_task)
                conn.execute("COMMIT")
            except ValueError:
                conn.execute("ROLLBACK")
                refreshed.append(_get(conn, task.task_id) or task)
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return refreshed


def pause(task_id: str) -> DevelopmentTask:
    with connect() as conn:
        task = _get(conn, task_id)
        if task is None:
            raise ValueError(f"development task {task_id} not found")
        if task.status in TERMINAL_STATUSES:
            raise ValueError("terminal tasks cannot be paused")
        _assert_transition(task.status, "paused")
        if task.kanban_task_id and task.kanban_board:
            with kb.connect_closing(board=task.kanban_board) as kconn:
                kb.stop_task_for_user(kconn, task.kanban_task_id, reason="Paused from Symphony")
        conn.execute("BEGIN IMMEDIATE")
        try:
            result = _transition(conn, task_id, "paused", kind="task.paused")
            conn.execute(
                "UPDATE agent_runs SET status='paused', finished_at=NULL WHERE task_id=? AND status IN ('starting','running','validating')",
                (task_id,),
            )
            conn.execute("COMMIT")
            return result
        except Exception:
            conn.execute("ROLLBACK")
            raise


def resume(task_id: str) -> DevelopmentTask:
    with connect() as conn:
        task = _get(conn, task_id)
        if task is None:
            raise ValueError(f"development task {task_id} not found")
        if task.status not in {"paused", "failed"}:
            raise ValueError("only paused or failed tasks can be resumed")
        if task.kanban_task_id and task.kanban_board:
            with kb.connect_closing(board=task.kanban_board) as kconn:
                kb.unblock_task(kconn, task.kanban_task_id)
        conn.execute("BEGIN IMMEDIATE")
        try:
            result = _transition(conn, task_id, "queued", kind="task.resumed", retry_count=task.retry_count + 1, error=None)
            conn.execute("COMMIT")
            return result
        except Exception:
            conn.execute("ROLLBACK")
            raise


def cancel(task_id: str) -> DevelopmentTask:
    with connect() as conn:
        task = _get(conn, task_id)
        if task is None:
            raise ValueError(f"development task {task_id} not found")
        if task.status in TERMINAL_STATUSES:
            return task
        run = _get_latest_agent_run_conn(conn, task_id)
        if run and run.worker_pid and int(run.worker_pid) != os.getpid():
            _terminate_worker_process(int(run.worker_pid))
        if task.kanban_task_id and task.kanban_board:
            with kb.connect_closing(board=task.kanban_board) as kconn:
                kb.stop_task_for_user(kconn, task.kanban_task_id, reason="Cancelled from Symphony")
                kb.archive_task(kconn, task.kanban_task_id)
        conn.execute("BEGIN IMMEDIATE")
        try:
            result = _transition(conn, task_id, "cancelled", kind="task.cancelled")
            conn.execute("COMMIT")
            return result
        except Exception:
            conn.execute("ROLLBACK")
            raise


def _terminate_worker_process(pid: int) -> None:
    """Terminate only the recorded worker process."""
    if pid <= 0 or pid == os.getpid():
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill.exe", "/PID", str(pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return


def list_tasks(
    *, status: Optional[str] = None, limit: int = 20,
    telegram_chat_id: Optional[str] = None, telegram_thread_id: Optional[str] = None,
    requester_id: Optional[str] = None,
) -> list[DevelopmentTask]:
    with connect() as conn:
        clauses: list[str] = []
        values: list[Any] = []
        if status:
            clauses.append("status = ?")
            values.append(status)
        if telegram_chat_id is not None:
            clauses.append("telegram_chat_id = ?")
            values.append(str(telegram_chat_id))
        if telegram_thread_id is not None:
            clauses.append("COALESCE(telegram_thread_id, '') = ?")
            values.append(str(telegram_thread_id or ""))
        if requester_id is not None:
            clauses.append("requester_id = ?")
            values.append(str(requester_id))
        sql = "SELECT * FROM development_tasks"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY priority ASC, created_at DESC LIMIT ?"
        values.append(max(1, min(int(limit), 100)))
        return [DevelopmentTask.from_row(row) for row in conn.execute(sql, values).fetchall()]


def list_events(task_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
    """Return bounded, redacted lifecycle events for operator diagnostics."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, kind, payload, created_at FROM development_events WHERE task_id=? ORDER BY id DESC LIMIT ?",
            (task_id, max(1, min(int(limit), 200))),
        ).fetchall()
    events = []
    for row in rows:
        try:
            payload = json.loads(row["payload"] or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {}
        events.append({"id": row["id"], "kind": row["kind"], "payload": _redact_artifact_value(payload), "created_at": row["created_at"]})
    return events


def reconcile_ci(task_id: str) -> bool:
    """Refresh one PR's CI snapshot and emit a delivery event only on change."""
    with connect() as conn:
        task = _get(conn, task_id)
    if task is None:
        raise ValueError(f"development task {task_id} not found")
    if not task.workspace_path or not task.pr_url:
        raise ValueError("CI reconciliation requires workspace and PR")
    snapshot = web_git.review_ci_snapshot(task.workspace_path)
    if not isinstance(snapshot, dict):
        raise ValueError("CI snapshot is unavailable")
    state = str(snapshot.get("state") or "unknown").lower()
    checks = snapshot.get("checks") if isinstance(snapshot.get("checks"), list) else []
    safe_checks = _redact_artifact_value(checks[:50])
    encoded = json.dumps(safe_checks, ensure_ascii=False, sort_keys=True)
    with connect() as conn:
        current = _get(conn, task_id)
        if current is None:
            raise ValueError(f"development task {task_id} not found")
        if current.ci_state == state and current.ci_checks == encoded:
            return False
        conn.execute("BEGIN IMMEDIATE")
        try:
            _transition(
                conn, task_id, current.status, kind="ci.updated",
                payload={"state": state, "checks": safe_checks},
                ci_state=state, ci_checks=encoded,
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return True


def reconcile_ci_all(*, limit: int = 50) -> int:
    """Refresh CI for review tasks; failures are isolated per task."""
    tasks = [task for task in list_tasks(status="awaiting_human_review", limit=limit) if task.pr_url]
    changed = 0
    for task in tasks:
        try:
            changed += int(reconcile_ci(task.task_id))
        except ValueError:
            continue
    return changed


def format_task(task: DevelopmentTask) -> str:
    lines = [f"🛠 {task.task_id}", f"{task.title}", f"Статус: {task.status}"]
    if task.project_id:
        lines.append(f"Проект: {task.project_id}")
    if task.branch_name:
        lines.append(f"Ветка: {task.branch_name}")
    if task.workspace_path:
        lines.append(f"Workspace: {task.workspace_path}")
    if task.pr_url:
        lines.append(f"PR: {task.pr_url}")
    if task.test_summary:
        lines.append(f"Тесты: {task.test_summary}")
    if task.result_summary:
        lines.append(f"Итог: {task.result_summary}")
    if task.error:
        lines.append(f"Ошибка: {task.error}")
    if task.last_action:
        lines.append(f"Последнее действие: {task.last_action}")
    if task.progress_detail:
        lines.append(f"Прогресс: {task.progress_detail}")
    if task.ci_state:
        lines.append(f"CI: {task.ci_state}")
    if task.source_issue_url:
        lines.append(f"Исходная задача: {task.source_issue_url}")
    if task.next_retry_at:
        lines.append(f"Следующая попытка: {datetime.fromtimestamp(task.next_retry_at, tz=timezone.utc).isoformat()}")
    return "\n".join(lines)


def build_parser(parent: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = parent.add_parser("symphony", help="Durable development-task bridge backed by Hermes Kanban")
    parser.add_argument("--json", action="store_true")
    sub = parser.add_subparsers(dest="symphony_action", required=True)
    create = sub.add_parser("create", help="Create a development task in draft")
    create.add_argument("title")
    create.add_argument("--body", required=True, help="Original task text")
    create.add_argument("--brief")
    create.add_argument("--project")
    create.add_argument("--repo")
    create.add_argument("--board")
    create.add_argument("--priority", type=int, default=0)
    create.add_argument("--idempotency-key")
    ready = sub.add_parser("ready", help="Mark a draft task ready for development")
    ready.add_argument("task_id")
    ready.add_argument("--brief")
    prepare = sub.add_parser("prepare", help="Prepare brief, criteria, and risks")
    prepare.add_argument("task_id")
    clarify = sub.add_parser("answer-clarification", help="Answer a task clarification request")
    clarify.add_argument("task_id")
    clarify.add_argument("answer", nargs="+", help="Clarification text")
    enqueue_p = sub.add_parser("enqueue", help="Queue a ready task through Hermes Kanban")
    enqueue_p.add_argument("task_id")
    enqueue_p.add_argument("--board")
    enqueue_p.add_argument("--assignee")
    for name, help_text in (("show", "Show task"), ("pause", "Pause task"), ("resume", "Resume task"), ("cancel", "Cancel task")):
        p = sub.add_parser(name, help=help_text); p.add_argument("task_id")
    reconcile_p = sub.add_parser("reconcile", help="Refresh task state from Kanban")
    reconcile_p.add_argument("task_id", nargs="?")
    run_p = sub.add_parser("run-once", help="Run one durable Symphony/Kanban dispatch tick")
    run_p.add_argument("--max-concurrency", type=int, default=1)
    events_p = sub.add_parser("events", help="Show safe lifecycle events for a task")
    events_p.add_argument("task_id")
    events_p.add_argument("--limit", type=int, default=50)
    ci_p = sub.add_parser("reconcile-ci", help="Refresh GitHub CI state for a PR task")
    ci_p.add_argument("task_id", nargs="?")
    status_p = sub.add_parser("status", help="Show safe queue, worker, and PR status")
    status_p.add_argument("--limit", type=int, default=20)
    publish_p = sub.add_parser("publish-review", help="Commit/push and open or reuse a GitHub PR for human review")
    publish_p.add_argument("task_id")
    publish_p.add_argument("--commit-message", required=True)
    publish_p.add_argument("--test-summary", required=True)
    publish_p.add_argument("--result-summary", required=True)
    list_p = sub.add_parser("list", help="List development tasks")
    list_p.add_argument("--status", choices=sorted(STATUSES))
    list_p.add_argument("--limit", type=int, default=20)
    return parser


def command(args: argparse.Namespace) -> int:
    action = args.symphony_action
    if action == "create":
        result: Any = create_task(title=args.title, source_text=args.body, technical_brief=args.brief, project_id=args.project, repository=args.repo, priority=args.priority, idempotency_key=args.idempotency_key, board=args.board)
    elif action == "ready": result = mark_ready(args.task_id, brief=args.brief)
    elif action == "prepare": result = prepare_task(args.task_id)
    elif action == "answer-clarification": result = answer_clarification(args.task_id, " ".join(args.answer))
    elif action == "enqueue": result = enqueue(args.task_id, board=args.board, assignee=args.assignee)
    elif action == "show":
        with connect() as conn:
            result = _get(conn, args.task_id)
        if result is None:
            raise ValueError(f"development task {args.task_id} not found")
    elif action == "pause": result = pause(args.task_id)
    elif action == "resume": result = resume(args.task_id)
    elif action == "cancel": result = cancel(args.task_id)
    elif action == "reconcile": result = reconcile(args.task_id)
    elif action == "run-once": result = SymphonyRunner(max_concurrency=args.max_concurrency).run_once()
    elif action == "events": result = list_events(args.task_id, limit=args.limit)
    elif action == "reconcile-ci": result = reconcile_ci(args.task_id) if args.task_id else reconcile_ci_all()
    elif action == "status": result = status_report(limit=args.limit)
    elif action == "publish-review": result = publish_for_review(
        args.task_id, commit_message=args.commit_message,
        test_summary=args.test_summary, result_summary=args.result_summary,
    )
    elif action == "list": result = list_tasks(status=args.status, limit=args.limit)
    else: raise ValueError(f"unknown Symphony action {action!r}")
    if args.json:
        if isinstance(result, list): print(json.dumps([asdict(item) if hasattr(item, "__dataclass_fields__") else item for item in result], ensure_ascii=False))
        else: print(json.dumps(asdict(result) if hasattr(result, "__dataclass_fields__") else result, ensure_ascii=False))
    elif isinstance(result, list):
        print("\n\n".join(format_task(item) if isinstance(item, DevelopmentTask) else str(item) for item in result) if result else "Нет задач разработки.")
    elif isinstance(result, dict): print(json.dumps(result, ensure_ascii=False, indent=2))
    else: print(format_task(result))
    return 0


def run_slash(rest: str) -> str:
    """Run a gateway slash-style argument string without mutating sys.argv."""
    parser = argparse.ArgumentParser(prog="/symphony", add_help=False)
    parent = parser.add_subparsers(dest="command", required=True)
    symphony_parser = build_parser(parent)
    # Make `run_slash('create ...')` work by injecting the top-level command.
    parsed = parser.parse_args(["symphony", *__import__("shlex").split(rest)])
    from io import StringIO
    from contextlib import redirect_stdout
    stream = StringIO()
    with redirect_stdout(stream):
        command(parsed)
    return stream.getvalue().strip()
