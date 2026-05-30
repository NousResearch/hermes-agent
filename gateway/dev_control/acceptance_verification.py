"""Advisory independent verification for Dev execution plan criteria."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from gateway.dev_control.acceptance_criteria import allowlisted_command, normalize_acceptance_criteria
from hermes_state import DEFAULT_DB_PATH, apply_wal_with_fallback


VERIFY_PROFILE_ID = "workspace.test"
VERIFY_PERMISSIONS = "verify"
VERIFICATION_OBJECT = "hermes.dev_verification_results"
VERIFICATION_BLOCK_RE = re.compile(
    r"```(?:json)?\s*(?:DEV_VERIFICATION_RESULTS|dev_verification_results)?\s*(\{.*?\})\s*```",
    re.IGNORECASE | re.DOTALL,
)
SUMMARY_RE = re.compile(
    r"\b(\d+\s+(passed|failed|skipped|errors?|warnings?)|passed|failed|success|build complete|tests? passed)\b",
    re.IGNORECASE,
)
TERMINAL_RUN_STATUSES = {"completed", "failed", "skipped", "needs_attention"}
TERMINAL_SESSION_STATUSES = {"completed", "complete", "done", "success", "succeeded", "failed", "errored", "error", "terminated", "killed", "exited"}
UNRUNNABLE_OUTPUT_RES = [
    re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    for pattern in (
        r"(^|\n)\s*make: \*\*\* No rule to make target\b",
        r"(^|\n|:\s*)[A-Za-z0-9_./-]+:\s*command not found\b",
        r"(^|\n).*\bNo such file or directory\b",
        r"(^|\n).*\bPermission denied\b",
        r"(^|\n).*\bnot executable\b",
    )
]
TRANSCRIPT_EXIT_RE = re.compile(
    r"(?:"
    r'"?exit_code"?'
    r"|exit[_ ]code"
    r"|\b"
    r"exit(?:ed)?(?:\s+\w+){0,4}?\s+with\s+(?:code|status)"
    r"|exit(?:ed)?(?:\s+with)?\s+(?:code|status)"
    r"|returncode"
    r"|exited"
    r")\s*[:=]?\s*(?P<code>-?\d+)\b",
    re.IGNORECASE,
)
UNFENCED_RESULTS_OBJECT_RE = re.compile(r"\{[^{}]*\"object\"\s*:\s*\"hermes\.dev_verification_results\".*\}", re.IGNORECASE | re.DOTALL)
DEFAULT_PROJECT_WORKDIRS = {
    "OrynWorkspace": "apps/oryn-workspace",
    "OrynPlatform": ".",
    "Oryn": ".",
}


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS dev_verification_runs (
    verification_run_id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL,
    task_id TEXT,
    target_type TEXT NOT NULL,
    status TEXT NOT NULL,
    verdict TEXT,
    acceptance_verification_score REAL,
    counts TEXT NOT NULL,
    results TEXT NOT NULL,
    executable_commands TEXT NOT NULL,
    verified_against TEXT NOT NULL,
    verification_session_id TEXT,
    verification_runtime TEXT,
    worker_launch_profile_id TEXT,
    warnings TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    completed_at REAL
);

CREATE INDEX IF NOT EXISTS idx_dev_verification_runs_plan
    ON dev_verification_runs(plan_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_dev_verification_runs_task
    ON dev_verification_runs(plan_id, task_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_dev_verification_runs_session
    ON dev_verification_runs(verification_session_id);
"""


class DevVerificationStore:
    """Persistence for advisory Dev verification runs."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        apply_wal_with_fallback(self._conn, db_label="state.db")
        self._lock = threading.Lock()
        with self._conn:
            self._conn.executescript(SCHEMA_SQL)

    def close(self) -> None:
        self._conn.close()

    def create_run(
        self,
        *,
        plan_id: str,
        task_id: Optional[str],
        target_type: str,
        status: str,
        results: list[Dict[str, Any]],
        executable_commands: list[Dict[str, Any]],
        verified_against: Dict[str, Any],
        warnings: Optional[list[str]] = None,
        verification_session_id: Optional[str] = None,
        verification_runtime: Optional[str] = None,
        worker_launch_profile_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = time.time()
        run = {
            "object": "hermes.dev_verification_run",
            "verification_run_id": f"devver-{uuid.uuid4().hex[:10]}",
            "plan_id": str(plan_id or "").strip(),
            "task_id": str(task_id or "").strip() or None,
            "target_type": target_type,
            "status": status,
            **_verdict_payload(results),
            "results": results,
            "executable_commands": executable_commands,
            "verified_against": verified_against,
            "verification_session_id": verification_session_id,
            "verification_runtime": verification_runtime,
            "worker_launch_profile_id": worker_launch_profile_id,
            "warnings": warnings or [],
            "created_at": now,
            "updated_at": now,
            "completed_at": now if status in TERMINAL_RUN_STATUSES else None,
        }
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO dev_verification_runs (
                    verification_run_id, plan_id, task_id, target_type, status,
                    verdict, acceptance_verification_score, counts, results,
                    executable_commands, verified_against, verification_session_id,
                    verification_runtime, worker_launch_profile_id, warnings,
                    created_at, updated_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _row_values(run),
            )
        return self.get_run(run["verification_run_id"]) or run

    def update_run(self, verification_run_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        current = self.get_run(verification_run_id)
        if not current:
            raise KeyError(f"Dev verification run not found: {verification_run_id}")
        merged = {**current, **updates}
        if "results" in updates:
            merged.update(_verdict_payload(merged.get("results") or []))
        merged["updated_at"] = time.time()
        if merged.get("status") in TERMINAL_RUN_STATUSES and not merged.get("completed_at"):
            merged["completed_at"] = merged["updated_at"]
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE dev_verification_runs
                SET plan_id = ?, task_id = ?, target_type = ?, status = ?,
                    verdict = ?, acceptance_verification_score = ?, counts = ?,
                    results = ?, executable_commands = ?, verified_against = ?,
                    verification_session_id = ?, verification_runtime = ?,
                    worker_launch_profile_id = ?, warnings = ?, created_at = ?,
                    updated_at = ?, completed_at = ?
                WHERE verification_run_id = ?
                """,
                (*_row_values(merged)[1:], verification_run_id),
            )
        return self.get_run(verification_run_id) or merged

    def get_run(self, verification_run_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            """
            SELECT *
            FROM dev_verification_runs
            WHERE verification_run_id = ?
            """,
            (str(verification_run_id or "").strip(),),
        ).fetchone()
        return _run_from_row(row) if row else None

    def list_runs(
        self,
        *,
        plan_id: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[Dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if plan_id:
            clauses.append("plan_id = ?")
            params.append(str(plan_id).strip())
        if task_id:
            clauses.append("task_id = ?")
            params.append(str(task_id).strip())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(int(limit or 50), 200)))
        rows = self._conn.execute(
            f"""
            SELECT *
            FROM dev_verification_runs
            {where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [_run_from_row(row) for row in rows]

    def latest_for_task(self, *, plan_id: str, task_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            """
            SELECT *
            FROM dev_verification_runs
            WHERE plan_id = ? AND task_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (plan_id, task_id),
        ).fetchone()
        return _run_from_row(row) if row else None


def classify_acceptance_criteria(criteria: Iterable[Any]) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], list[str]]:
    """Return reconciled seed results and executable commands for raw criteria."""

    results: list[Dict[str, Any]] = []
    executable: list[Dict[str, Any]] = []
    warnings: list[str] = []
    for index, criterion in enumerate(criteria or [], start=1):
        normalized = _criterion_from_raw(criterion, index)
        criterion_id = normalized["criterion_id"]
        method = normalized["verification_method"]
        if method not in {"test", "command"} or not normalized["machine_checkable"]:
            results.append({
                **normalized,
                "status": "manual_required" if method in {"manual", "probe"} else "unchecked",
                "command_run": None,
                "exit_code": None,
                "passed": None,
                "output_excerpt": "",
                "notes": "Criterion is not executable in v1 verification.",
                "warnings": [],
            })
            continue
        allowed, command, reason = allowlisted_command(normalized["verification_detail"])
        if not allowed:
            warning = f"{criterion_id}: {reason}"
            warnings.append(warning)
            results.append({
                **normalized,
                "status": "unverifiable",
                "command_run": normalized["verification_detail"],
                "exit_code": None,
                "passed": None,
                "output_excerpt": "",
                "notes": reason,
                "warnings": [reason],
            })
            continue
        command_payload = {
            "criterion_id": criterion_id,
            "statement": normalized["statement"],
            "verification_method": method,
            "command": command,
        }
        executable.append(command_payload)
        results.append({
            **normalized,
            "status": "pending",
            "command_run": command,
            "exit_code": None,
            "passed": None,
            "output_excerpt": "",
            "notes": "",
            "warnings": [],
        })
    return results, executable, warnings


def parse_verification_results(text: Optional[str]) -> Dict[str, Any]:
    value = str(text or "")
    if not value.strip():
        return _empty_parse("missing", "Worker output did not include DEV_VERIFICATION_RESULTS.")
    match = _find_verification_block(value)
    if not match:
        unfenced = _parse_unfenced_results_object(value)
        if unfenced:
            return unfenced
        if "DEV_VERIFICATION_RESULTS" in value.upper():
            return _empty_parse("invalid", "DEV_VERIFICATION_RESULTS marker was present but no JSON object could be extracted.")
        return _empty_parse("missing", "Worker output did not include DEV_VERIFICATION_RESULTS.")
    raw_json = match.group(1).strip()
    return _parse_results_json(raw_json)


def parse_transcript_verification_results(text: Optional[str], executable_commands: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    value = str(text or "")
    if not value.strip():
        return _empty_parse("missing", "Worker transcript is empty.")
    results: list[Dict[str, Any]] = []
    for command in executable_commands or []:
        command_text = str(command.get("command") or "").strip()
        criterion_id = str(command.get("criterion_id") or "").strip()
        if not command_text:
            continue
        window, exit_code = _recover_transcript_window(value, command_text, criterion_id)
        if not window or exit_code is None:
            continue
        results.append({
            "criterion_id": criterion_id,
            "command_run": command_text,
            "cwd": str(command.get("relative_cwd") or command.get("cwd") or "").strip(),
            "exit_code": exit_code,
            "output_excerpt": _output_excerpt_from_transcript(window, command_text),
            "notes": "Recovered from worker transcript without fenced DEV_VERIFICATION_RESULTS.",
        })
    if not results:
        return _empty_parse("missing", "Worker transcript did not contain recoverable command exit evidence.")
    return {
        "status": "warning",
        "warning": "Recovered verification results from worker transcript because DEV_VERIFICATION_RESULTS was missing or invalid.",
        "results": results,
    }


def _parse_worker_verification_output(
    text: str,
    executable_commands: Iterable[Dict[str, Any]],
) -> tuple[Dict[str, Any], list[str]]:
    warnings: list[str] = []
    parsed = parse_verification_results(text)
    if parsed.get("warning"):
        warnings.append(parsed["warning"])
    if parsed.get("results") and not _parsed_results_are_prompt_template(parsed):
        return parsed, warnings
    recovered = parse_transcript_verification_results(text, executable_commands)
    if recovered.get("results"):
        if recovered.get("warning"):
            warnings.append(recovered["warning"])
        return recovered, warnings
    if _parsed_results_are_prompt_template(parsed):
        parsed = _empty_parse("invalid", "Worker output only contained the prompt's DEV_VERIFICATION_RESULTS template.")
        warnings.append(parsed["warning"])
    elif recovered.get("warning"):
        warnings.append(recovered["warning"])
    return parsed, warnings


def _parsed_results_are_prompt_template(parsed: Dict[str, Any]) -> bool:
    for item in parsed.get("results") or []:
        excerpt = str(item.get("output_excerpt") or "").lower()
        if "include the real test/build summary line" in excerpt:
            return True
    return False


def reconcile_results(seed_results: list[Dict[str, Any]], worker_results: list[Dict[str, Any]]) -> tuple[list[Dict[str, Any]], list[str]]:
    by_id = {str(item.get("criterion_id") or ""): item for item in worker_results}
    warnings: list[str] = []
    reconciled: list[Dict[str, Any]] = []
    for seed in seed_results:
        status = seed.get("status")
        if status != "pending":
            reconciled.append(seed)
            continue
        worker = by_id.get(str(seed.get("criterion_id") or ""))
        if not worker:
            warning = f"{seed.get('criterion_id')}: worker did not return a result."
            warnings.append(warning)
            reconciled.append({**seed, "status": "unverifiable", "notes": warning, "warnings": [warning]})
            continue
        exit_code = worker.get("exit_code")
        output_excerpt = str(worker.get("output_excerpt") or "").strip()
        result_warnings: list[str] = []
        if exit_code is None:
            result_warnings.append("exit_code is missing or invalid.")
        if not output_excerpt:
            result_warnings.append("output_excerpt is required for executed verification.")
        elif not SUMMARY_RE.search(output_excerpt) and len(output_excerpt) < 24:
            result_warnings.append("output_excerpt does not contain a plausible command summary.")
        status_value = _status_for_worker_result(exit_code, output_excerpt)
        if status_value == "error":
            result_warnings.append("Verification command could not run; check verification_detail and working directory.")
        if result_warnings:
            warnings.extend(f"{seed.get('criterion_id')}: {item}" for item in result_warnings)
        reconciled.append({
            **seed,
            "status": status_value,
            "command_run": worker.get("command_run") or seed.get("command_run"),
            "cwd": worker.get("cwd") or seed.get("cwd"),
            "relative_cwd": worker.get("cwd") or seed.get("relative_cwd"),
            "exit_code": exit_code,
            "passed": exit_code == 0 if status_value in {"passed", "failed"} and exit_code is not None else False,
            "output_excerpt": output_excerpt,
            "notes": worker.get("notes") or ("; ".join(result_warnings) if result_warnings else ""),
            "warnings": result_warnings,
            "evidence_trust_boundary": "worker_reported_exit_code",
        })
    return reconciled, warnings


def launch_verification_run(
    *,
    execution_store: Any,
    verification_store: DevVerificationStore,
    plan_id: str,
    task_id: Optional[str] = None,
    bridge: Any = None,
    event_store: Any = None,
) -> Dict[str, Any]:
    plan = execution_store.get_plan(plan_id)
    if not plan:
        raise KeyError(f"Dev execution plan not found: {plan_id}")
    tasks = plan.get("tasks") or []
    if task_id:
        task = next((item for item in tasks if item.get("task_id") == task_id), None)
        if not task:
            raise KeyError(f"Dev execution plan task not found: {task_id}")
        return _launch_task_verification(
            execution_store=execution_store,
            verification_store=verification_store,
            plan=plan,
            task=task,
            bridge=bridge,
            event_store=event_store,
        )

    created: list[Dict[str, Any]] = []
    skipped: list[Dict[str, Any]] = []
    for task in tasks:
        try:
            run = _launch_task_verification(
                execution_store=execution_store,
                verification_store=verification_store,
                plan=plan,
                task=task,
                bridge=bridge,
                event_store=event_store,
            )
            if run.get("status") == "skipped":
                skipped.append({"task_id": task.get("task_id"), "reason": "; ".join(run.get("warnings") or [])})
            else:
                created.append(run)
        except ValueError as exc:
            skipped.append({"task_id": task.get("task_id"), "reason": str(exc)})
    return {
        "ok": bool(created),
        "object": "hermes.dev_verification_plan_launch",
        "plan_id": plan_id,
        "created": created,
        "skipped": skipped,
    }


def refresh_verification_run(
    *,
    verification_store: DevVerificationStore,
    verification_run_id: str,
    event_store: Any = None,
    bridge: Any = None,
) -> Dict[str, Any]:
    run = verification_store.get_run(verification_run_id)
    if not run:
        raise KeyError(f"Dev verification run not found: {verification_run_id}")
    if run.get("status") in TERMINAL_RUN_STATUSES:
        repaired = _repair_run_from_codex_final_message(
            verification_store=verification_store,
            run=run,
            bridge=bridge,
        )
        if repaired is not None:
            return repaired
        return run
    if _run_timed_out(run):
        return verification_store.update_run(verification_run_id, {
            "status": "needs_attention",
            "warnings": _unique([*(run.get("warnings") or []), "Verification worker did not complete before the launched-run timeout."]),
        })
    if not event_store:
        return run
    session_id = run.get("verification_session_id")
    if not session_id:
        return run
    latest_event = event_store.latest_event_for_ao_session(session_id)
    session = _verification_session(run, bridge=bridge)
    transcript = _verification_transcript(run, session=session, bridge=bridge)
    final_message = _verification_final_message(run, session=session)
    status = str((latest_event or {}).get("status") or "").lower()
    event_type = str((latest_event or {}).get("event") or "").lower()
    event_terminal = event_type == "subagent.complete" or status in TERMINAL_SESSION_STATUSES
    event_text = _event_text(latest_event or {})
    combined_text = "\n".join(part for part in (event_text, transcript, final_message) if str(part or "").strip())
    warnings = list(run.get("warnings") or [])
    parsed, parse_warnings = _parse_worker_verification_output(combined_text, run.get("executable_commands") or [])
    warnings.extend(parse_warnings)
    if not event_terminal and not _session_is_terminal(session) and not parsed.get("results"):
        return run
    if not parsed.get("results"):
        return verification_store.update_run(verification_run_id, {
            "status": "needs_attention",
            "warnings": _unique([*warnings, parsed.get("warning") or "Verification worker output could not be parsed."]),
        })
    results, reconcile_warnings = reconcile_results(run.get("results") or [], parsed.get("results") or [])
    return verification_store.update_run(verification_run_id, {
        "status": "completed",
        "results": results,
        "warnings": _unique([*warnings, *reconcile_warnings]),
    })


def list_verification_runs(
    *,
    verification_store: DevVerificationStore,
    plan_id: Optional[str] = None,
    task_id: Optional[str] = None,
    limit: int = 50,
    event_store: Any = None,
) -> Dict[str, Any]:
    runs = verification_store.list_runs(plan_id=plan_id, task_id=task_id, limit=limit)
    refreshed = [
        refresh_verification_run(
            verification_store=verification_store,
            verification_run_id=run["verification_run_id"],
            event_store=event_store,
        )
        for run in runs
    ]
    return {"object": "list", "data": refreshed, "total": len(refreshed)}


def annotate_plan_with_verification(plan: Dict[str, Any], verification_store: Optional[DevVerificationStore]) -> Dict[str, Any]:
    if verification_store is None:
        return plan
    annotated = dict(plan)
    tasks = []
    latest_runs = []
    for task in annotated.get("tasks") or []:
        task_copy = dict(task)
        latest = verification_store.latest_for_task(plan_id=annotated.get("plan_id"), task_id=task_copy.get("task_id"))
        if latest:
            summary = _compact_run_summary(latest)
            task_copy["latest_verification_run"] = latest
            task_copy["acceptance_verification"] = summary
            latest_runs.append(latest)
        tasks.append(task_copy)
    annotated["tasks"] = tasks
    if latest_runs:
        annotated["acceptance_verification"] = _aggregate_runs(latest_runs)
    return annotated


def _launch_task_verification(
    *,
    execution_store: Any,
    verification_store: DevVerificationStore,
    plan: Dict[str, Any],
    task: Dict[str, Any],
    bridge: Any,
    event_store: Any,
) -> Dict[str, Any]:
    derived_plan = _derived_plan_for_verification(execution_store, plan, bridge=bridge, event_store=event_store)
    derived_tasks = derived_plan.get("tasks") or []
    task_id = task.get("task_id")
    derived_task = next((item for item in derived_tasks if item.get("task_id") == task_id), task)
    derived_task = {**task, **derived_task}
    task_status = str(derived_task.get("status") or derived_task.get("derived_status") or task.get("status") or "").lower()
    if task_status not in {"completed", "needs_review"}:
        raise ValueError(f"Task {task.get('task_id')} is not idle/completed; current status is {task_status}.")
    busy_reason = _busy_worktree_reason(derived_task, derived_tasks)
    if busy_reason:
        raise ValueError(busy_reason)
    seed_results, executable, warnings = classify_acceptance_criteria(task.get("acceptance_criteria") or [])
    verified_against = _verified_against(derived_task)
    relative_cwd = _verification_relative_cwd(task, profile_project_id=task.get("project_id"))
    verified_against["verification_relative_cwd"] = relative_cwd
    seed_results = [_with_relative_cwd(item, relative_cwd) for item in seed_results]
    executable = [_with_relative_cwd(item, relative_cwd) for item in executable]
    if not executable:
        return verification_store.create_run(
            plan_id=plan["plan_id"],
            task_id=task.get("task_id"),
            target_type="task",
            status="skipped",
            results=seed_results,
            executable_commands=[],
            verified_against=verified_against,
            warnings=warnings or ["No executable machine-checkable criteria were available."],
        )

    from gateway.dev_execution import DEFAULT_RUNTIME, resolve_launch_defaults, select_execution_runtime
    from tools.ao_delegate_tool import build_ao_worker_prompt

    profile = resolve_launch_defaults(
        profile_id=VERIFY_PROFILE_ID,
        project_id=task.get("project_id"),
        permissions=VERIFY_PERMISSIONS,
        runtime_policy_evidence={},
    )
    runtime_selection = select_execution_runtime(
        goal=f"Verify {task.get('goal') or task.get('task_id')}",
        prompt="Run fixed verification commands.",
        profile_id=VERIFY_PROFILE_ID,
        project_id=profile.get("project_id"),
        permissions=VERIFY_PERMISSIONS,
        runtime_policy_evidence={},
    )
    profile["runtime"] = runtime_selection["selected_runtime"]
    relative_cwd = _verification_relative_cwd(task, profile_project_id=profile.get("project_id"))
    verified_against["verification_relative_cwd"] = relative_cwd
    seed_results = [_with_relative_cwd(item, relative_cwd) for item in seed_results]
    executable = [_with_relative_cwd(item, relative_cwd) for item in executable]
    prompt = _verification_prompt(plan=plan, task=task, commands=executable, verified_against=verified_against)
    launch_prompt = build_ao_worker_prompt(prompt, goal=f"Verify {task.get('goal') or task.get('task_id')}")
    router = _runtime_router(bridge)
    session = router.spawn(
        profile.get("runtime") or DEFAULT_RUNTIME,
        project_id=profile["project_id"],
        prompt=launch_prompt,
        issue_id=(task.get("payload") or {}).get("issue_id"),
        branch=(task.get("payload") or {}).get("branch"),
        agent=profile.get("agent"),
        model=profile.get("model"),
        reasoning_effort=profile.get("reasoning_effort"),
    )
    run = verification_store.create_run(
        plan_id=plan["plan_id"],
        task_id=task.get("task_id"),
        target_type="task",
        status="launched",
        results=seed_results,
        executable_commands=executable,
        verified_against=verified_against,
        warnings=warnings,
        verification_session_id=session.id,
        verification_runtime=profile.get("runtime"),
        worker_launch_profile_id=VERIFY_PROFILE_ID,
    )
    if event_store:
        _persist_verification_start_event(
            event_store=event_store,
            session=session,
            plan=plan,
            task=task,
            prompt=prompt,
            profile=profile,
            runtime_selection=runtime_selection,
            run=run,
        )
    return run


def _derived_plan_for_verification(execution_store: Any, plan: Dict[str, Any], *, bridge: Any, event_store: Any) -> Dict[str, Any]:
    try:
        from gateway.dev_execution import derive_execution_plan_status

        status_payload = derive_execution_plan_status(
            store=execution_store,
            plan_id=plan["plan_id"],
            bridge=bridge,
            event_store=event_store,
        )
        return status_payload.get("plan") or {**plan, "tasks": status_payload.get("tasks") or plan.get("tasks") or []}
    except Exception:
        return plan


def _busy_worktree_reason(target_task: Dict[str, Any], tasks: list[Dict[str, Any]]) -> Optional[str]:
    target_id = target_task.get("task_id")
    target_worktree = _task_workspace_path(target_task)
    for task in tasks:
        if task.get("task_id") == target_id:
            continue
        status = str(task.get("derived_status") or task.get("status") or "").lower()
        if status not in {"running", "launched", "spawned", "active", "in_progress"}:
            continue
        other_worktree = _task_workspace_path(task)
        if target_worktree and other_worktree and target_worktree == other_worktree:
            return (
                f"Worktree {target_worktree} is still in use by active task "
                f"{task.get('task_id')}; verification requires an idle worktree."
            )
        if not target_worktree:
            return (
                f"Task {task.get('task_id')} is still active in this plan; "
                "verification requires an idle target worktree."
            )
    return None


def _persist_verification_start_event(
    *,
    event_store: Any,
    session: Any,
    plan: Dict[str, Any],
    task: Dict[str, Any],
    prompt: str,
    profile: Dict[str, Any],
    runtime_selection: Dict[str, Any],
    run: Dict[str, Any],
) -> None:
    try:
        event_store.upsert_ao_prompt(
            ao_session_id=session.id,
            project_id=profile.get("project_id"),
            prompt=prompt,
            goal=f"Verify {task.get('goal') or task.get('task_id')}",
            issue_id=(task.get("payload") or {}).get("issue_id"),
            branch=(task.get("payload") or {}).get("branch"),
            agent=getattr(session, "agent", None) or profile.get("agent"),
            model=getattr(session, "model", None) or profile.get("model"),
            reasoning_effort=getattr(session, "reasoning_effort", None) or profile.get("reasoning_effort"),
            launch_profile_id=VERIFY_PROFILE_ID,
            launch_plan_id=plan.get("plan_id"),
            launch_task_id=f"{task.get('task_id')}:verification",
            permissions=VERIFY_PERMISSIONS,
            acceptance_criteria=task.get("acceptance_criteria") or [],
            runtime_selection=runtime_selection,
            selected_runtime=runtime_selection.get("selected_runtime"),
            runtime_selection_reason=runtime_selection.get("reason"),
            runtime_fallback_reason=runtime_selection.get("runtime_fallback_reason"),
        )
        payload = {
            "event": "subagent.start",
            "subagent_id": f"ao:{session.id}",
            "ao_session_id": session.id,
            "runtime": "ao",
            "runtime_session_id": session.id,
            "runtime_project_id": profile.get("project_id"),
            "ao_project_id": profile.get("project_id"),
            "goal": f"Verify {task.get('goal') or task.get('task_id')}",
            "preview": f"AO verification session {session.id} spawned for {task.get('task_id')}",
            "message": f"AO verification session {session.id} spawned for {task.get('task_id')}",
            "status": "running",
            "workspace_path": getattr(session, "workspace_path", None),
            "branch": getattr(session, "branch", None),
            "agent": getattr(session, "agent", None) or profile.get("agent"),
            "model": getattr(session, "model", None) or profile.get("model"),
            "reasoning_effort": getattr(session, "reasoning_effort", None) or profile.get("reasoning_effort"),
            "launch_profile_id": VERIFY_PROFILE_ID,
            "launch_plan_id": plan.get("plan_id"),
            "launch_task_id": f"{task.get('task_id')}:verification",
            "permissions": VERIFY_PERMISSIONS,
            "verification_run_id": run.get("verification_run_id"),
            "acceptance_criteria": task.get("acceptance_criteria") or [],
            "runtime_selection": runtime_selection,
            "selected_runtime": runtime_selection.get("selected_runtime"),
            "runtime_selection_reason": runtime_selection.get("reason"),
            "runtime_fallback_reason": runtime_selection.get("runtime_fallback_reason"),
            "timestamp": time.time(),
        }
        try:
            payload.update({key: value for key, value in session.event_fields().items() if value is not None})
        except Exception:
            pass
        event_store.append_event(payload)
    except Exception:
        return


def _verification_prompt(*, plan: Dict[str, Any], task: Dict[str, Any], commands: list[Dict[str, Any]], verified_against: Dict[str, Any]) -> str:
    relative_cwd = str(verified_against.get("verification_relative_cwd") or ".").strip() or "."
    cwd_instruction = (
        "Run every command from the AO worker's current worktree root."
        if relative_cwd == "."
        else f"Before running commands, change directory from the AO worker's current worktree root to: {relative_cwd}"
    )
    return "\n".join([
        "## Independent Dev Verification",
        "",
        "Run only the commands listed below, in order, from the bound repository/worktree.",
        cwd_instruction,
        "Do not use an absolute path to a different checkout; verification must run inside this worker's isolated worktree.",
        "Do not edit source files, create commits, create branches, open PRs, or run any command not listed here.",
        "Build/test artifacts may be produced by the listed commands; do not intentionally change tracked source files.",
        "Report raw exit codes and real output summary lines. Do not decide whether the feature is acceptable.",
        "",
        f"Plan: {plan.get('plan_id')} - {plan.get('title')}",
        f"Task: {task.get('task_id')} - {task.get('goal')}",
        f"Verified against: {json.dumps(verified_against, ensure_ascii=False)}",
        "",
        "Allowed commands:",
        *[
            f"- {item['criterion_id']}: {item['command']}\n  Working directory: {item.get('relative_cwd') or relative_cwd}\n  Statement: {item['statement']}"
            for item in commands
        ],
        "",
        "Final output is mandatory: return a short summary followed by this exact fenced JSON block as your final response.",
        "The fenced block must be valid JSON. Do not paste raw multi-line command output inside a JSON string.",
        "Keep output_excerpt to one short JSON string with the real summary line or command-output tail; escape any newline as \\n.",
        "Do not include terminal UI/progress text such as Working, esc to interrupt, or prompts in output_excerpt.",
        "Set exit_code to the real process exit code you observed. Do not leave the template exit_code or output_excerpt unchanged.",
        "```json DEV_VERIFICATION_RESULTS",
        json.dumps({
            "object": VERIFICATION_OBJECT,
            "results": [
                {
                    "criterion_id": item["criterion_id"],
                    "command_run": item["command"],
                    "cwd": item.get("relative_cwd") or relative_cwd,
                    "exit_code": 0,
                    "output_excerpt": "include the real test/build summary line",
                    "notes": "",
                }
                for item in commands
            ],
        }, indent=2),
        "```",
    ]).strip()


def _criterion_from_raw(raw: Any, index: int) -> Dict[str, Any]:
    if isinstance(raw, str):
        parsed = _criterion_from_string(raw, index)
        if parsed:
            return parsed
    normalized = normalize_acceptance_criteria([raw])
    if normalized:
        criterion = normalized[0]
    else:
        criterion = {
            "statement": str(raw or "").strip(),
            "verification_method": "manual",
            "verification_detail": "Review manually.",
            "machine_checkable": False,
        }
    return {
        "criterion_id": f"crit-{index}",
        "statement": criterion["statement"],
        "verification_method": criterion["verification_method"],
        "verification_detail": criterion["verification_detail"],
        "machine_checkable": bool(criterion["machine_checkable"]),
    }


def _criterion_from_string(value: str, index: int) -> Optional[Dict[str, Any]]:
    text = value.strip()
    match = re.match(r"^(?P<statement>.*?)\s+\(verify via (?P<method>test|command|probe|manual): (?P<detail>.*?); (?P<machine>machine-checkable|manual)\)$", text)
    if not match:
        return None
    return {
        "criterion_id": f"crit-{index}",
        "statement": match.group("statement").strip(),
        "verification_method": match.group("method"),
        "verification_detail": match.group("detail").strip(),
        "machine_checkable": match.group("machine") == "machine-checkable",
    }


def _find_verification_block(text: str) -> Optional[re.Match[str]]:
    matches = list(VERIFICATION_BLOCK_RE.finditer(text))
    if matches:
        preferred = [match for match in matches if "DEV_VERIFICATION_RESULTS" in match.group(0).upper()]
        return preferred[-1] if preferred else matches[-1]
    return None


def _parse_unfenced_results_object(text: str) -> Optional[Dict[str, Any]]:
    marker_index = text.rfind(f'"object": "{VERIFICATION_OBJECT}"')
    if marker_index < 0:
        marker_index = text.rfind(f'"object":"{VERIFICATION_OBJECT}"')
    if marker_index < 0:
        return None
    start = text.rfind("{", 0, marker_index)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for offset, char in enumerate(text[start:], start=start):
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return _parse_results_json(text[start : offset + 1])
    return None


def _empty_parse(status: str, warning: str) -> Dict[str, Any]:
    return {"status": status, "warning": warning, "results": []}


def _parse_results_json(raw_json: str) -> Dict[str, Any]:
    try:
        payload = json.loads(raw_json)
    except Exception as exc:
        parsed = _empty_parse("invalid", f"DEV_VERIFICATION_RESULTS is not valid JSON: {exc}")
        parsed["raw"] = raw_json
        return parsed
    if not isinstance(payload, dict):
        return _empty_parse("invalid", "DEV_VERIFICATION_RESULTS must be a JSON object.")
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        return _empty_parse("invalid", "DEV_VERIFICATION_RESULTS.results must be an array.")
    results: list[Dict[str, Any]] = []
    warnings: list[str] = []
    for item in raw_results:
        if not isinstance(item, dict):
            warnings.append("Ignoring non-object verification result.")
            continue
        criterion_id = str(item.get("criterion_id") or "").strip()
        command_run = str(item.get("command_run") or "").strip()
        output_excerpt = str(item.get("output_excerpt") or "").strip()
        exit_code: Optional[int]
        try:
            exit_code = int(item.get("exit_code"))
        except Exception:
            exit_code = None
            warnings.append(f"{criterion_id or command_run}: exit_code is missing or invalid.")
        results.append({
            "criterion_id": criterion_id,
            "command_run": command_run,
            "cwd": str(item.get("cwd") or item.get("relative_cwd") or "").strip(),
            "exit_code": exit_code,
            "output_excerpt": output_excerpt,
            "notes": str(item.get("notes") or "").strip(),
        })
    return {
        "status": "warning" if warnings else "ok",
        "warning": "; ".join(warnings) if warnings else None,
        "results": results,
        "raw": raw_json,
    }


def _transcript_window_for_command(text: str, command: str) -> str:
    window, _exit_code = _recover_transcript_window(text, command, "")
    return window


def _recover_transcript_window(text: str, command: str, criterion_id: str = "") -> tuple[str, Optional[int]]:
    for needle in (command, criterion_id):
        if not needle:
            continue
        for match in reversed(list(re.finditer(re.escape(needle), text))):
            forward_window = text[match.start() : min(len(text), match.start() + 2500)]
            if _is_prompt_template_transcript_window(forward_window):
                continue
            window = text[max(0, match.start() - 1000) : min(len(text), match.start() + 2500)]
            exit_code = _exit_code_from_transcript(window)
            if exit_code is not None:
                return window, exit_code
    return "", None


def _is_prompt_template_transcript_window(text: str) -> bool:
    lowered = str(text or "").lower()
    if "include the real test/build summary line" in lowered:
        return True
    if "set exit_code to the real process exit code" in lowered:
        return True
    return False


def _exit_code_from_transcript(text: str) -> Optional[int]:
    matches = list(TRANSCRIPT_EXIT_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    try:
        return int(match.group("code"))
    except Exception:
        return None


def _output_excerpt_from_transcript(text: str, command: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    useful = [
        line for line in lines
        if command not in line
        and not line.startswith("{")
        and not line.startswith("}")
        and (not line.startswith('"') or SUMMARY_RE.search(line))
        and "DEV_VERIFICATION_RESULTS" not in line
    ]
    if not useful:
        return text.strip()[:500]
    tail = useful[-4:]
    return "\n".join(tail)[-500:]


def _status_for_worker_result(exit_code: Optional[int], output_excerpt: str) -> str:
    if exit_code == 0:
        return "passed"
    if exit_code in {126, 127}:
        return "error"
    if _looks_unrunnable(output_excerpt):
        return "error"
    return "failed"


def _looks_unrunnable(output_excerpt: str) -> bool:
    text = str(output_excerpt or "")
    return any(pattern.search(text) for pattern in UNRUNNABLE_OUTPUT_RES)


def _verdict_payload(results: list[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {"passed": 0, "failed": 0, "error": 0, "unverifiable": 0, "manual_required": 0, "unchecked": 0}
    needs_review = False
    for result in results or []:
        status = str(result.get("status") or "").lower()
        if status == "passed":
            counts["passed"] += 1
        elif status == "failed":
            counts["failed"] += 1
        elif status == "error":
            counts["error"] += 1
            needs_review = True
        elif status == "unverifiable":
            counts["unverifiable"] += 1
        elif status == "manual_required":
            counts["manual_required"] += 1
        elif status == "unchecked":
            counts["unchecked"] += 1
        if result.get("warnings"):
            needs_review = True
    denominator = counts["passed"] + counts["failed"]
    score = round(counts["passed"] / denominator, 3) if denominator else None
    if needs_review:
        verdict = "needs_review"
    elif denominator and counts["failed"] == 0 and counts["unverifiable"] == 0:
        verdict = "verified"
    elif denominator and counts["passed"] > 0 and counts["failed"] > 0:
        verdict = "partial"
    elif denominator and counts["passed"] == 0:
        verdict = "failed"
    elif counts["unverifiable"] > 0:
        verdict = "unverifiable"
    elif counts["manual_required"] > 0:
        verdict = "manual_required"
    else:
        verdict = "unverifiable"
    return {
        "verdict": verdict,
        "acceptance_verification_score": score,
        "counts": counts,
    }


def _aggregate_runs(runs: list[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {"passed": 0, "failed": 0, "error": 0, "unverifiable": 0, "manual_required": 0, "unchecked": 0}
    for run in runs:
        for key in counts:
            counts[key] += int((run.get("counts") or {}).get(key) or 0)
    payload = _verdict_payload([
        {"status": key}
        for key, count in counts.items()
        for _ in range(count)
    ])
    payload["counts"] = counts
    payload["latest_run_ids"] = [run["verification_run_id"] for run in runs]
    return payload


def _compact_run_summary(run: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "verification_run_id": run.get("verification_run_id"),
        "status": run.get("status"),
        "verdict": run.get("verdict"),
        "acceptance_verification_score": run.get("acceptance_verification_score"),
        "counts": run.get("counts") or {},
        "verified_against": run.get("verified_against") or {},
        "updated_at": run.get("updated_at"),
    }


def _row_values(run: Dict[str, Any]) -> tuple[Any, ...]:
    return (
        run["verification_run_id"],
        run["plan_id"],
        run.get("task_id"),
        run["target_type"],
        run["status"],
        run.get("verdict"),
        run.get("acceptance_verification_score"),
        json.dumps(run.get("counts") or {}, ensure_ascii=False),
        json.dumps(run.get("results") or [], ensure_ascii=False),
        json.dumps(run.get("executable_commands") or [], ensure_ascii=False),
        json.dumps(run.get("verified_against") or {}, ensure_ascii=False),
        run.get("verification_session_id"),
        run.get("verification_runtime"),
        run.get("worker_launch_profile_id"),
        json.dumps(run.get("warnings") or [], ensure_ascii=False),
        run.get("created_at"),
        run.get("updated_at"),
        run.get("completed_at"),
    )


def _run_from_row(row: sqlite3.Row) -> Dict[str, Any]:
    run = dict(row)
    for key, default in (
        ("counts", {}),
        ("results", []),
        ("executable_commands", []),
        ("verified_against", {}),
        ("warnings", []),
    ):
        try:
            run[key] = json.loads(run.get(key) or json.dumps(default))
        except Exception:
            run[key] = default
    run["object"] = "hermes.dev_verification_run"
    return run


def _event_text(event: Dict[str, Any]) -> str:
    parts = [
        str(event.get("summary") or ""),
        str(event.get("message") or ""),
        str(event.get("preview") or ""),
    ]
    output_tail = event.get("output_tail")
    if isinstance(output_tail, list):
        for item in output_tail:
            if isinstance(item, dict):
                parts.append(str(item.get("preview") or item.get("text") or ""))
            else:
                parts.append(str(item))
    elif output_tail:
        parts.append(str(output_tail))
    return "\n".join(parts)


def _run_timed_out(run: Dict[str, Any]) -> bool:
    if run.get("status") != "launched":
        return False
    timeout = _env_int("HERMES_DEV_VERIFICATION_LAUNCHED_TIMEOUT_SECONDS", 1200)
    if timeout <= 0:
        return False
    created_at = float(run.get("created_at") or 0)
    return created_at > 0 and time.time() - created_at >= timeout


def _verification_session(run: Dict[str, Any], *, bridge: Any = None) -> Any:
    session_id = run.get("verification_session_id")
    if not session_id:
        return None
    try:
        router = _runtime_router(bridge)
        try:
            return router.status(run.get("verification_runtime"), session_id)
        except TypeError:
            return router.status(session_id)
    except Exception:
        return None


def _session_is_terminal(session: Any) -> bool:
    status = str(getattr(session, "status", None) or "").lower()
    return status in TERMINAL_SESSION_STATUSES


def _verification_transcript(run: Dict[str, Any], *, session: Any = None, bridge: Any = None) -> str:
    if session is None:
        session = _verification_session(run, bridge=bridge)
    if session is None:
        return ""
    try:
        router = _runtime_router(bridge)
        try:
            return str(router.capture_output(run.get("verification_runtime"), session, lines=120) or "")
        except TypeError:
            return str(router.capture_output(session, lines=120) or "")
    except Exception:
        return ""


def _repair_run_from_codex_final_message(
    *,
    verification_store: DevVerificationStore,
    run: Dict[str, Any],
    bridge: Any = None,
) -> Optional[Dict[str, Any]]:
    if not _has_transcript_recovery_warning(run.get("warnings") or []):
        return None
    session = _verification_session(run, bridge=bridge)
    final_message = _verification_final_message(run, session=session)
    if not final_message.strip():
        return None
    parsed = parse_verification_results(final_message)
    if not parsed.get("results") or _parsed_results_are_prompt_template(parsed):
        return None
    seed = _pending_seed_for_repair(run.get("results") or [], parsed.get("results") or [])
    results, reconcile_warnings = reconcile_results(seed, parsed.get("results") or [])
    warnings = [
        warning
        for warning in (run.get("warnings") or [])
        if not _is_transcript_recovery_warning(warning)
    ]
    return verification_store.update_run(str(run["verification_run_id"]), {
        "status": "completed",
        "results": results,
        "warnings": _unique([*warnings, *reconcile_warnings]),
    })


def _pending_seed_for_repair(existing_results: list[Dict[str, Any]], worker_results: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    worker_ids = {str(item.get("criterion_id") or "") for item in worker_results}
    seed: list[Dict[str, Any]] = []
    for item in existing_results:
        criterion_id = str(item.get("criterion_id") or "")
        if criterion_id in worker_ids:
            seed.append({
                **item,
                "status": "pending",
                "exit_code": None,
                "passed": None,
                "output_excerpt": "",
                "notes": "",
                "warnings": [],
            })
        else:
            seed.append(item)
    return seed


def _has_transcript_recovery_warning(warnings: Iterable[Any]) -> bool:
    return any(_is_transcript_recovery_warning(warning) for warning in warnings or [])


def _is_transcript_recovery_warning(warning: Any) -> bool:
    text = str(warning or "")
    return (
        "DEV_VERIFICATION_RESULTS is not valid JSON" in text
        or "Recovered verification results from worker transcript" in text
        or "DEV_VERIFICATION_RESULTS marker was present but no JSON object could be extracted" in text
    )


def _verification_final_message(run: Dict[str, Any], *, session: Any = None) -> str:
    """Return the worker's final assistant message when Codex session JSONL is available.

    Terminal captures can contain live UI redraw/progress text. The Codex JSONL
    task_complete payload carries the final answer as the model produced it, so it
    is the more authoritative source for fenced verification JSON.
    """

    started_at = _first_float(run.get("created_at"))
    completed_at = _first_float(run.get("completed_at")) or time.time()
    session_record: Optional[Dict[str, Any]] = None
    workspace_path = str(_session_field(session, "workspace_path") or "").strip()
    if workspace_path:
        session_record = _codex_session_final_message_for_workspace(
            workspace_path,
            started_at=started_at,
            completed_at=completed_at,
        )
    if not session_record:
        session_id = str(run.get("verification_session_id") or _session_field(session, "id") or "").strip()
        if session_id:
            session_record = _codex_session_final_message_for_session_id(
                session_id,
                started_at=started_at,
                completed_at=completed_at,
            )
    return str((session_record or {}).get("last_agent_message") or "")


def _session_field(session: Any, key: str) -> Any:
    if session is None:
        return None
    if isinstance(session, dict):
        return session.get(key)
    return getattr(session, key, None)


def _codex_session_final_message_for_workspace(
    workspace_path: str,
    *,
    started_at: Optional[float] = None,
    completed_at: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    try:
        workspace = str(Path(workspace_path).expanduser().resolve())
    except OSError:
        workspace = str(Path(workspace_path).expanduser())
    sessions_root = _codex_sessions_root()
    if not sessions_root.exists():
        return None
    lower_bound = float(started_at or 0) - 3600
    upper_bound = float(completed_at or time.time()) + 3600
    matches: list[Dict[str, Any]] = []
    for path in sessions_root.rglob("rollout-*.jsonl"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if started_at is not None and (mtime < lower_bound or mtime > upper_bound):
            continue
        parsed = _read_codex_final_message_file(path)
        if not parsed:
            continue
        cwd = str(parsed.get("cwd") or "").strip()
        if not cwd:
            continue
        try:
            resolved_cwd = str(Path(cwd).expanduser().resolve())
        except OSError:
            resolved_cwd = cwd
        if resolved_cwd == workspace and parsed.get("last_agent_message"):
            matches.append(parsed)
    if not matches:
        return None
    matches.sort(key=lambda item: float(item.get("updated_at") or 0), reverse=True)
    return matches[0]


def _codex_session_final_message_for_session_id(
    session_id: str,
    *,
    started_at: Optional[float] = None,
    completed_at: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    session_id = str(session_id or "").strip()
    if not session_id:
        return None
    sessions_root = _codex_sessions_root()
    if not sessions_root.exists():
        return None
    lower_bound = float(started_at or 0) - 3600
    upper_bound = float(completed_at or time.time()) + 3600
    matches: list[Dict[str, Any]] = []
    for path in sessions_root.rglob("rollout-*.jsonl"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if started_at is not None and (mtime < lower_bound or mtime > upper_bound):
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if session_id not in raw:
            continue
        parsed = _read_codex_final_message_file(path)
        if parsed and parsed.get("last_agent_message"):
            matches.append(parsed)
    if not matches:
        return None
    matches.sort(key=lambda item: float(item.get("updated_at") or 0), reverse=True)
    return matches[0]


def _codex_sessions_root() -> Path:
    configured = os.getenv("HERMES_DEV_VERIFICATION_CODEX_HOME") or os.getenv("HERMES_DEV_LAB_CODEX_HOME")
    if configured:
        return Path(configured).expanduser() / "sessions"
    lab_home = os.getenv("ORYN_LAB_HOME")
    if lab_home:
        return Path(lab_home).expanduser() / ".codex" / "sessions"
    return Path.home() / ".codex" / "sessions"


def _read_codex_final_message_file(path: Path) -> Optional[Dict[str, Any]]:
    meta: dict[str, Any] = {}
    last_agent_message = ""
    completed_at: Optional[float] = None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
                if entry.get("type") == "session_meta":
                    meta = payload
                elif payload.get("type") == "task_complete":
                    last_agent_message = str(payload.get("last_agent_message") or "")
                    completed_at = _first_float(payload.get("completed_at"))
    except OSError:
        return None
    if not meta:
        return None
    try:
        updated_at = path.stat().st_mtime
    except OSError:
        updated_at = completed_at or 0
    return {
        "source": "codex_session_jsonl",
        "session_id": meta.get("id"),
        "session_path": str(path),
        "cwd": meta.get("cwd"),
        "last_agent_message": last_agent_message,
        "completed_at": completed_at,
        "updated_at": updated_at,
    }


def _first_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _task_workspace_path(task: Dict[str, Any]) -> str:
    last_event = task.get("last_event") if isinstance(task.get("last_event"), dict) else {}
    return str(task.get("workspace_path") or last_event.get("workspace_path") or "").strip()


def _verified_against(task: Dict[str, Any]) -> Dict[str, Any]:
    workspace_path = _task_workspace_path(task)
    commit_sha = _read_head_commit(Path(workspace_path)) if workspace_path else None
    files_changed = task.get("files_changed") or task.get("files_written") or []
    tracked_diff = bool(files_changed)
    return {
        "commit_sha": commit_sha,
        "dirty": tracked_diff,
        "tracked_diff": tracked_diff,
        "workspace_path": workspace_path or None,
        "source": "git_head_file" if commit_sha else "task_event_metadata",
    }


def _with_relative_cwd(item: Dict[str, Any], relative_cwd: str) -> Dict[str, Any]:
    cwd = _safe_relative_cwd(relative_cwd)
    return {**item, "cwd": cwd, "relative_cwd": cwd}


def _verification_relative_cwd(task: Dict[str, Any], *, profile_project_id: Optional[str]) -> str:
    project_id = str(profile_project_id or task.get("project_id") or "").strip()
    overrides = _verification_workdir_overrides()
    if project_id in overrides:
        return _safe_relative_cwd(overrides[project_id])
    if project_id in DEFAULT_PROJECT_WORKDIRS:
        return _safe_relative_cwd(DEFAULT_PROJECT_WORKDIRS[project_id])
    return "."


def _verification_workdir_overrides() -> Dict[str, str]:
    raw = os.getenv("HERMES_DEV_VERIFICATION_WORKDIRS_JSON") or ""
    if not raw.strip():
        return {}
    try:
        decoded = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(decoded, dict):
        return {}
    result: Dict[str, str] = {}
    for key, value in decoded.items():
        project_id = str(key or "").strip()
        if project_id:
            result[project_id] = _safe_relative_cwd(str(value or "."))
    return result


def _safe_relative_cwd(value: str) -> str:
    text = str(value or ".").strip() or "."
    text = text.replace("\\", "/").rstrip("/") or "."
    if text == ".":
        return "."
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        return "."
    return text


def _read_head_commit(workspace: Path) -> Optional[str]:
    try:
        git_dir = workspace / ".git"
        if git_dir.is_file():
            content = git_dir.read_text(encoding="utf-8", errors="ignore").strip()
            if content.startswith("gitdir:"):
                git_dir = (workspace / content.split(":", 1)[1].strip()).resolve(strict=False)
        head = (git_dir / "HEAD").read_text(encoding="utf-8", errors="ignore").strip()
        if re.fullmatch(r"[0-9a-fA-F]{7,64}", head):
            return head
        if head.startswith("ref:"):
            ref = head.split(":", 1)[1].strip()
            value = (git_dir / ref).read_text(encoding="utf-8", errors="ignore").strip()
            if re.fullmatch(r"[0-9a-fA-F]{7,64}", value):
                return value
    except Exception:
        return None
    return None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _runtime_router(bridge: Any) -> Any:
    if bridge is not None:
        return bridge
    from gateway.dev_execution import _ensure_runtime_router

    return _ensure_runtime_router(None)


def _unique(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result
