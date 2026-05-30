"""Controlled runtime benchmarks for Dev harness runtime comparisons."""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from gateway.dev_execution import (
    DevExecutionStore,
    derive_execution_plan_status,
    launch_execution_plan,
)
from gateway.dev_control.worker_output_contract import (
    append_worker_output_contract,
    parse_worker_output_contract,
    worker_output_contract_score,
)
from gateway.dev_worker_runtimes import WorkerRuntimeRouter, list_worker_runtimes
from gateway.subagent_events import SubagentEventStore
from hermes_state import DEFAULT_DB_PATH, apply_wal_with_fallback


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS dev_harness_benchmark_runs (
    benchmark_run_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    live INTEGER NOT NULL,
    created_at REAL NOT NULL,
    completed_at REAL,
    payload TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dev_harness_benchmark_runs_created_at
    ON dev_harness_benchmark_runs(created_at DESC);
"""

DEFAULT_RUNTIMES = ("ao", "openhands")
DEFAULT_PROJECT_ID = "OrynWorkspace"
DEFAULT_TIMEOUT_SECONDS = 180
MAX_CASES_LIMIT = 10
MAX_LIVE_ITERATIONS = 3
MAX_FIXTURE_ITERATIONS = 10
TERMINAL_PLAN_STATUSES = {"completed", "failed", "needs_review"}

BUILTIN_BENCHMARK_CASES = [
    {
        "case_id": "agent_board_metadata",
        "title": "Agent Board metadata inspection",
        "marker": "BENCH_AGENT_BOARD_METADATA_DONE",
        "prompt": (
            "Read-only benchmark case. Inspect the Oryn Workspace Agent Board runtime metadata path. "
            "Report two concrete findings about runtime/session metadata display. "
            "At least one finding must cite WorkspaceAgentBoardView or WorkspaceSubagentActivity exactly. "
            "Do not edit files."
        ),
        "required_evidence_terms": ["WorkspaceAgentBoardView", "WorkspaceSubagentActivity"],
        "expected_evidence_terms": [
            "WorkspaceAgentBoardView",
            "WorkspaceSubagentActivity",
            "runtime_session_id",
            "agent",
            "model",
            "reasoning_effort",
        ],
    },
    {
        "case_id": "dev_plans_strip_metadata",
        "title": "Dev Plans strip metadata inspection",
        "marker": "BENCH_DEV_PLANS_STRIP_DONE",
        "prompt": (
            "Read-only benchmark case. Inspect the Oryn Workspace Dev Plans strip metadata path. "
            "Report two concrete findings about plan/runtime/review metadata display. "
            "At least one finding must cite devPlansStrip or WorkspaceDevExecutionPlan exactly. "
            "Do not edit files."
        ),
        "required_evidence_terms": ["devPlansStrip", "WorkspaceDevExecutionPlan"],
        "expected_evidence_terms": [
            "devPlansStrip",
            "WorkspaceDevExecutionPlan",
            "review_status",
            "policy_profile",
            "supervisor_status",
            "runtime_selection",
        ],
    },
    {
        "case_id": "runtime_selection_status",
        "title": "Runtime selection status inspection",
        "marker": "BENCH_RUNTIME_SELECTION_STATUS_DONE",
        "prompt": (
            "Read-only benchmark case. Inspect Hermes runtime selection status behavior. "
            "Report two concrete findings about selected runtime and fallback metadata. "
            "At least one finding must cite runtime_selection.py, select_worker_runtime, or selected_runtime exactly. "
            "Do not edit files."
        ),
        "required_evidence_terms": ["runtime_selection.py", "select_worker_runtime", "selected_runtime"],
        "expected_evidence_terms": [
            "runtime_selection.py",
            "select_worker_runtime",
            "selected_runtime",
            "runtime_fallback_reason",
            "runtime_selection",
            "runtime: auto",
            "OpenHands",
            "AO",
        ],
    },
]


@dataclass
class DevHarnessBenchmarkStore:
    """Persistence for benchmark run payloads."""

    db_path: Optional[Path] = None

    def __post_init__(self) -> None:
        self.db_path = self.db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        apply_wal_with_fallback(self._conn, db_label="state.db")
        with self._conn:
            self._conn.executescript(SCHEMA_SQL)

    def close(self) -> None:
        self._conn.close()

    def persist_run(self, payload: Dict[str, Any]) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO dev_harness_benchmark_runs (
                    benchmark_run_id, mode, live, created_at, completed_at, payload
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["benchmark_run_id"],
                    payload["mode"],
                    1 if payload.get("live") else 0,
                    float(payload["created_at"]),
                    payload.get("completed_at"),
                    json.dumps(payload, ensure_ascii=False),
                ),
            )

    def get_run(self, benchmark_run_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            """
            SELECT payload
            FROM dev_harness_benchmark_runs
            WHERE benchmark_run_id = ?
            """,
            (str(benchmark_run_id or "").strip(),),
        ).fetchone()
        if not row:
            return None
        return json.loads(row["payload"])

    def list_runs(self, *, limit: int = 50) -> list[Dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT benchmark_run_id, mode, live, created_at, completed_at, payload
            FROM dev_harness_benchmark_runs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (max(1, min(int(limit or 50), 200)),),
        ).fetchall()
        runs = []
        for row in rows:
            payload = json.loads(row["payload"])
            runs.append({
                "benchmark_run_id": row["benchmark_run_id"],
                "mode": row["mode"],
                "live": bool(row["live"]),
                "created_at": float(row["created_at"]),
                "completed_at": row["completed_at"],
                "summary": payload.get("summary") or {},
                "runtime_count": len(payload.get("runtime_results") or []),
                "case_count": len(payload.get("cases") or []),
            })
        return runs


def run_harness_benchmark(
    *,
    store: DevExecutionStore,
    event_store: SubagentEventStore,
    runtimes: Optional[list[str]] = None,
    cases: Optional[list[Dict[str, Any]]] = None,
    mode: Optional[str] = None,
    live: bool = False,
    project_id: str = DEFAULT_PROJECT_ID,
    max_cases: int = 3,
    iterations: int = 1,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    persist: bool = True,
    bridge: Any = None,
) -> Dict[str, Any]:
    """Run a safe-by-default benchmark. Real workers launch only when live is true."""

    created_at = time.time()
    live = bool(live)
    normalized_mode = "live" if live else _normalize_mode(mode)
    benchmark_run_id = f"devbench-{uuid.uuid4().hex[:10]}"
    selected_runtimes = _normalize_runtimes(runtimes)
    selected_cases = _normalize_cases(cases, max_cases=max_cases)
    normalized_iterations = _normalize_iterations(iterations, live=live)
    runtime_map = {runtime["id"]: runtime for runtime in list_worker_runtimes()}

    case_results = []
    for iteration in range(1, normalized_iterations + 1):
        for case in selected_cases:
            for runtime in selected_runtimes:
                result = _run_case_for_runtime(
                    store=store,
                    event_store=event_store,
                    runtime=runtime,
                    runtime_info=runtime_map.get(runtime),
                    case=case,
                    mode=normalized_mode,
                    live=live,
                    project_id=project_id,
                    timeout_seconds=timeout_seconds,
                    bridge=bridge,
                    iteration=iteration,
                    iterations=normalized_iterations,
                )
                case_results.append(result)

    runtime_results = _aggregate_runtime_results(case_results)
    completed_at = time.time()
    payload = {
        "ok": True,
        "object": "hermes.dev_harness_benchmark_run",
        "benchmark_run_id": benchmark_run_id,
        "mode": normalized_mode,
        "live": live,
        "created_at": created_at,
        "completed_at": completed_at,
        "timeout_seconds": timeout_seconds,
        "iterations": normalized_iterations,
        "project_id": project_id,
        "cases": selected_cases,
        "runtime_results": runtime_results,
        "case_results": case_results,
        "summary": _benchmark_summary(case_results, runtime_results),
        "recommended_interpretation": _recommended_interpretation(case_results, runtime_results, live=live),
        "non_goals": [
            "Benchmark results do not mutate runtime policy.",
            "Benchmark runs do not auto-supervise, retry, follow up, approve, or reassign workers.",
            "Dry-run and fixture modes do not launch AO, OpenHands, Codex, Cursor, or Claude workers.",
        ],
    }
    if persist:
        benchmark_store = DevHarnessBenchmarkStore(store.db_path)
        try:
            benchmark_store.persist_run(payload)
        finally:
            benchmark_store.close()
    return payload


def list_harness_benchmark_runs(*, store: DevExecutionStore, limit: int = 50) -> Dict[str, Any]:
    benchmark_store = DevHarnessBenchmarkStore(store.db_path)
    try:
        runs = benchmark_store.list_runs(limit=limit)
    finally:
        benchmark_store.close()
    return {"ok": True, "object": "list", "data": runs, "total": len(runs)}


def get_harness_benchmark_run(*, store: DevExecutionStore, benchmark_run_id: str) -> Dict[str, Any]:
    benchmark_store = DevHarnessBenchmarkStore(store.db_path)
    try:
        run = benchmark_store.get_run(benchmark_run_id)
    finally:
        benchmark_store.close()
    if not run:
        raise KeyError(f"Dev harness benchmark run not found: {benchmark_run_id}")
    return run


def _run_case_for_runtime(
    *,
    store: DevExecutionStore,
    event_store: SubagentEventStore,
    runtime: str,
    runtime_info: Optional[Dict[str, Any]],
    case: Dict[str, Any],
    mode: str,
    live: bool,
    project_id: str,
    timeout_seconds: int,
    bridge: Any,
    iteration: int = 1,
    iterations: int = 1,
) -> Dict[str, Any]:
    base = {
        "case_id": case["case_id"],
        "case_title": case["title"],
        "runtime": runtime,
        "iteration": iteration,
        "iterations": iterations,
        "marker": case["marker"],
        "expected_evidence_terms": case.get("expected_evidence_terms") or [],
        "required_evidence_terms": case.get("required_evidence_terms") or [],
        "mode": mode,
        "live": live,
        "plan_id": None,
        "task_id": None,
        "runtime_session_id": None,
        "status": "planned",
        "status_reason": None,
        "summary": None,
        "output_tail": None,
        "output_tail_captured": False,
        "marker_in_summary": False,
        "marker_in_output_tail": False,
        "runtime_delivery_failed": False,
        "delivery_status": "unknown",
        "delivery_failure_reason": None,
        "delivery_cleanup": None,
        "token_total": None,
        "cost_usd": None,
        "duration_seconds": 0.0,
        "skipped": False,
        "skip_reason": None,
    }
    if runtime_info is None:
        return _score_result({**base, "status": "skipped", "skipped": True, "skip_reason": f"Runtime {runtime} is not registered."})
    if mode == "fixture":
        return _score_result(_fixture_result(base, runtime=runtime, case=case))
    if not runtime_info.get("available") or not runtime_info.get("launch_supported"):
        return _score_result({
            **base,
            "status": "skipped",
            "skipped": True,
            "skip_reason": runtime_info.get("setup_warning") or f"Runtime {runtime} is unavailable.",
        })
    if mode == "dry_run":
        return _score_result({
            **base,
            "status": "validated",
            "status_reason": "Benchmark case and runtime are valid; live=false so no worker was launched.",
        })
    if mode != "live" or not live:
        return _score_result({
            **base,
            "status": "validated",
            "status_reason": "Live benchmark launch was not explicitly enabled.",
        })
    if runtime == "ao":
        return _run_ao_codex_exec_case(
            base=base,
            bridge=bridge,
            case=case,
            project_id=project_id,
            timeout_seconds=timeout_seconds,
        )

    started_at = time.time()
    profile_id = "workspace.openhands.inspect" if runtime == "openhands" else "workspace.inspect"
    plan = store.create_plan(
        title=f"Runtime benchmark: {runtime} / {case['case_id']} / {iteration}",
        vision_brief="Controlled read-only runtime benchmark generated by Hermes Dev harness.",
        tasks=[{
            "goal": f"Benchmark {runtime}: {case['title']}",
            "prompt": case["prompt"],
            "profile_id": profile_id,
            "runtime": runtime,
            "project_id": project_id,
            "permissions": "read_only",
            "dev_harness_benchmark": True,
            "minimal_worker_prompt": True,
            "acceptance_criteria": [
                f"Final line must be exactly: {_expected_marker_line(case['marker'])}",
        "Use the BENCHMARK_RESULT structured output contract.",
        "Include Worker Output Contract v2 DEV_WORKER_EVIDENCE JSON.",
        "Report concise file-backed findings.",
        "Do not edit files.",
            ],
        }],
    )
    task = plan["tasks"][0]
    launch = launch_execution_plan(
        store=store,
        plan_id=plan["plan_id"],
        bridge=bridge,
        event_store=event_store,
    )
    if not launch.get("launched"):
        return _score_result({
            **base,
            "plan_id": plan["plan_id"],
            "task_id": task["task_id"],
            "status": "failed",
            "status_reason": "; ".join(str(item.get("error")) for item in launch.get("failures") or []) or "Runtime launch failed.",
            "duration_seconds": time.time() - started_at,
        })

    deadline = time.time() + max(1, int(timeout_seconds or DEFAULT_TIMEOUT_SECONDS))
    status_payload = None
    while time.time() <= deadline:
        status_payload = derive_execution_plan_status(
            store=store,
            plan_id=plan["plan_id"],
            bridge=bridge,
            event_store=event_store,
        )
        if str(status_payload.get("status") or "").lower() in TERMINAL_PLAN_STATUSES:
            break
        time.sleep(2)

    status_payload = status_payload or derive_execution_plan_status(
        store=store,
        plan_id=plan["plan_id"],
        bridge=bridge,
        event_store=event_store,
    )
    task_status = (status_payload.get("tasks") or [{}])[0]
    timed_out = str(status_payload.get("status") or "").lower() not in TERMINAL_PLAN_STATUSES
    output_tail = _capture_benchmark_output_tail(
        runtime=runtime,
        session_id=task_status.get("runtime_session_id") or task_status.get("ao_session_id"),
        bridge=bridge,
    )
    runtime_session_id = task_status.get("runtime_session_id") or task_status.get("ao_session_id")
    delivery_failure_reason = _detect_prompt_delivery_failure(
        runtime=runtime,
        output_tail=output_tail,
        marker=case["marker"],
    )
    delivery_cleanup = None
    if delivery_failure_reason:
        delivery_cleanup = _cleanup_delivery_failed_session(
            runtime=runtime,
            session_id=runtime_session_id,
            bridge=bridge,
        )
    usage = _usage_from_task_status({**task_status, "output_tail": output_tail})
    return _score_result({
        **base,
        "plan_id": plan["plan_id"],
        "task_id": task["task_id"],
        "runtime_session_id": runtime_session_id,
        "status": (
            "runtime_delivery_failed"
            if delivery_failure_reason
            else "timeout" if timed_out else task_status.get("status")
        ),
        "status_reason": (
            delivery_failure_reason
            or ("Benchmark timed out before terminal state." if timed_out else task_status.get("status_reason"))
        ),
        "summary": task_status.get("summary"),
        "output_tail": output_tail,
        "output_tail_captured": bool(output_tail),
        "runtime_delivery_failed": bool(delivery_failure_reason),
        "delivery_status": "failed" if delivery_failure_reason else "delivered_or_terminal",
        "delivery_failure_reason": delivery_failure_reason,
        "delivery_cleanup": delivery_cleanup,
        "token_total": usage.get("token_total"),
        "cost_usd": usage.get("cost_usd"),
        "usage": usage or None,
        "duration_seconds": time.time() - started_at,
        "files_read_count": len(task_status.get("files_read") or []),
        "verification_evidence_count": len(task_status.get("verification_evidence") or []),
        "summary_warning": task_status.get("summary_warning"),
        "output_contract_status": task_status.get("output_contract_status"),
        "output_contract_warning": task_status.get("output_contract_warning"),
        "output_contract_score": task_status.get("output_contract_score"),
    })


def _run_ao_codex_exec_case(
    *,
    base: Dict[str, Any],
    bridge: Any,
    case: Dict[str, Any],
    project_id: str,
    timeout_seconds: int,
) -> Dict[str, Any]:
    started_at = time.time()
    if bridge is None:
        from tools.ao_bridge import AOBridge

        bridge = AOBridge()
    try:
        execution = bridge.run_codex_exec_benchmark(
            project_id=project_id,
            prompt=case["prompt"],
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        return _score_result({
            **base,
            "status": "failed",
            "status_reason": f"Codex exec benchmark failed: {exc}",
            "duration_seconds": time.time() - started_at,
            "runtime_delivery_failed": True,
            "delivery_status": "failed",
            "delivery_failure_reason": str(exc),
        })
    summary = execution.get("summary") or ""
    output_tail = execution.get("output_tail") or summary
    delivery_failure_reason = None
    if _marker_line_present("\n".join([summary, output_tail]), case["marker"]) and not _has_worker_benchmark_result(
        "\n".join([summary, output_tail]),
        case["marker"],
    ):
        delivery_failure_reason = (
            "AO benchmark output only contains the echoed prompt/placeholder contract; "
            "no worker benchmark result was delivered."
        )
    marker_present = _marker_line_present(summary, case["marker"]) or _marker_line_present(output_tail, case["marker"])
    raw_status = str(execution.get("status") or "").lower()
    status = (
        "runtime_delivery_failed"
        if delivery_failure_reason
        else "completed" if raw_status == "completed" and marker_present
        else "needs_review" if raw_status == "completed" else raw_status or "failed"
    )
    summary_warning = None
    if delivery_failure_reason:
        summary_warning = delivery_failure_reason
    elif raw_status == "completed" and not marker_present:
        summary_warning = f"Completed worker summary is missing required completion marker: {case['marker']}."
    return _score_result({
        **base,
        "runtime_session_id": execution.get("session_id"),
        "status": status,
        "status_reason": summary_warning or ("Codex exec benchmark completed." if status == "completed" else execution.get("status_reason")),
        "summary": summary,
        "output_tail": output_tail,
        "output_tail_captured": bool(output_tail),
        "runtime_delivery_failed": bool(delivery_failure_reason),
        "delivery_status": "failed" if delivery_failure_reason else "delivered_or_terminal",
        "delivery_failure_reason": delivery_failure_reason,
        "token_total": execution.get("token_total"),
        "cost_usd": None,
        "duration_seconds": execution.get("duration_seconds") or (time.time() - started_at),
        "summary_warning": summary_warning,
        "workspace_path": execution.get("workspace_path"),
        "agent": execution.get("agent") or "codex",
        "model": execution.get("model"),
        "reasoning_effort": execution.get("reasoning_effort"),
        "benchmark_execution_mode": "codex_exec",
    })


def _fixture_result(base: Dict[str, Any], *, runtime: str, case: Dict[str, Any]) -> Dict[str, Any]:
    if runtime == "ao":
        return {
            **base,
            "status": "completed",
            "status_reason": "Fixture AO benchmark completed.",
            "summary": (
                "Fixture AO found WorkspaceAgentBoardView renders runtime_session_id and "
                "WorkspaceSubagentActivity carries agent/model metadata.\n"
                f"{_expected_marker_line(case['marker'])}"
            ),
            "output_tail": (
                "Fixture AO transcript tail.\n"
                f"{_expected_marker_line(case['marker'])}"
            ),
            "output_tail_captured": True,
            "runtime_delivery_failed": False,
            "delivery_status": "delivered_or_terminal",
            "runtime_session_id": f"fixture-ao-{case['case_id']}",
            "duration_seconds": 42.0,
            "token_total": 3200,
            "cost_usd": 0.012,
            "files_read_count": 2,
            "verification_evidence_count": 1,
        }
    if runtime == "openhands":
        return {
            **base,
            "status": "needs_review",
            "status_reason": f"Completed worker summary is missing required completion marker: {case['marker']}.",
            "summary": "Fixture OpenHands completed the inspection: runtime metadata is displayed and session metadata is displayed.",
            "output_tail": "Fixture OpenHands transcript tail without the exact final marker.",
            "output_tail_captured": True,
            "runtime_delivery_failed": False,
            "delivery_status": "delivered_or_terminal",
            "runtime_session_id": f"fixture-openhands-{case['case_id']}",
            "duration_seconds": 31.0,
            "token_total": 1800,
            "cost_usd": 0.004,
            "files_read_count": 2,
            "verification_evidence_count": 1,
            "summary_warning": f"Completed worker summary is missing required completion marker: {case['marker']}.",
        }
    return {**base, "status": "skipped", "skipped": True, "skip_reason": f"No fixture benchmark for runtime {runtime}."}


def _score_result(result: Dict[str, Any]) -> Dict[str, Any]:
    status = str(result.get("status") or "").lower()
    summary = str(result.get("summary") or "")
    output_tail = str(result.get("output_tail") or "")
    marker = str(result.get("marker") or "")
    expected_terms = [str(term).strip() for term in (result.get("expected_evidence_terms") or []) if str(term).strip()]
    required_terms = [str(term).strip() for term in (result.get("required_evidence_terms") or []) if str(term).strip()]
    summary_warning = str(result.get("summary_warning") or result.get("status_reason") or "").lower()
    combined_output = "\n".join(part for part in (summary, output_tail) if part)
    output_contract = parse_worker_output_contract(combined_output)
    output_contract_score = result.get("output_contract_score")
    if output_contract_score is None:
        output_contract_score = worker_output_contract_score(output_contract, required_marker=marker)
    runtime_delivery_failed = bool(result.get("runtime_delivery_failed")) or status == "runtime_delivery_failed"
    status_score = 1.0 if status == "completed" else 0.5 if status == "needs_review" else 0.0
    marker_in_summary = False if runtime_delivery_failed else _marker_line_present(summary, marker)
    marker_in_output_tail = False if runtime_delivery_failed else _marker_line_present(output_tail, marker)
    marker_score = 1.0 if (marker_in_summary or marker_in_output_tail) else 0.0
    has_summary = bool(summary.strip())
    echo_like = "echo" in summary_warning or "contract" in summary_warning
    summary_score = 1.0 if has_summary and not echo_like and len(summary.strip()) >= 24 else 0.0
    evidence_count = int(result.get("files_read_count") or 0) + int(result.get("verification_evidence_count") or 0)
    evidence_score = 1.0 if evidence_count > 0 else 0.0
    structured_result_present = False if runtime_delivery_failed else _structured_result_present(combined_output)
    findings_count = _findings_count(combined_output)
    findings_score = 1.0 if findings_count >= 2 else 0.5 if findings_count == 1 else (0.5 if summary_score else 0.0)
    evidence_terms_matched = _matched_evidence_terms(combined_output, expected_terms)
    required_evidence_terms_matched = _matched_evidence_terms(combined_output, required_terms)
    strong_evidence_terms_matched = [term for term in evidence_terms_matched if _is_strong_evidence_term(term)]
    file_symbol_reference_count = _file_symbol_reference_count(combined_output)
    evidence_term_score = _weighted_evidence_term_score(evidence_terms_matched, expected_terms)
    generic_finding_penalty = _generic_finding_penalty(combined_output, findings_count=findings_count, evidence_terms_matched=evidence_terms_matched)
    specificity_score = max(0.0, round((evidence_term_score + findings_score) / 2.0 - generic_finding_penalty, 3))
    if expected_terms and not strong_evidence_terms_matched and file_symbol_reference_count <= 0:
        specificity_score = min(specificity_score, 0.55)
    required_evidence_score = 1.0 if not required_terms else 1.0 if required_evidence_terms_matched else 0.0
    if required_terms and not required_evidence_terms_matched:
        specificity_score = min(specificity_score, 0.5)
    delivery_score = 0.0 if runtime_delivery_failed else 1.0 if status not in {"validated", "skipped"} else None
    task_quality_score = None if status in {"validated", "skipped"} else round((summary_score + evidence_score + findings_score + specificity_score) / 4.0, 3)
    contract_compliance_score = None if status in {"validated", "skipped"} else round(
        (marker_score + (1.0 if structured_result_present else 0.0) + float(output_contract_score or 0.0)) / 3.0,
        3,
    )
    if status in {"validated", "skipped"}:
        overall = None
    else:
        quality = float(task_quality_score or 0.0)
        contract = float(contract_compliance_score or 0.0)
        delivery = float(delivery_score or 0.0)
        overall = round((delivery * 0.35) + (quality * 0.45) + (contract * 0.20), 3)
    result.update({
        "status_score": status_score,
        "marker_score": marker_score,
        "summary_score": summary_score,
        "evidence_score": evidence_score,
        "delivery_score": delivery_score,
        "task_quality_score": task_quality_score,
        "contract_compliance_score": contract_compliance_score,
        "output_contract_score": None if status in {"validated", "skipped"} else output_contract_score,
        "output_contract_status": result.get("output_contract_status") or output_contract.get("output_contract_status"),
        "output_contract_warning": result.get("output_contract_warning") or output_contract.get("output_contract_warning"),
        "findings_score": findings_score,
        "findings_count": findings_count,
        "expected_evidence_terms": expected_terms,
        "required_evidence_terms": required_terms,
        "evidence_terms_matched": evidence_terms_matched,
        "required_evidence_terms_matched": required_evidence_terms_matched,
        "required_evidence_score": required_evidence_score,
        "strong_evidence_terms_matched": strong_evidence_terms_matched,
        "file_symbol_reference_count": file_symbol_reference_count,
        "evidence_term_score": evidence_term_score,
        "specificity_score": specificity_score,
        "generic_finding_penalty": generic_finding_penalty,
        "structured_result_present": structured_result_present,
        "overall_score": overall,
        "marker_present": bool(marker_in_summary or marker_in_output_tail),
        "marker_in_summary": marker_in_summary,
        "marker_in_output_tail": marker_in_output_tail,
        "expected_marker_line": _expected_marker_line(marker) if marker else None,
    })
    return result


def _aggregate_runtime_results(case_results: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    grouped: Dict[str, list[Dict[str, Any]]] = {}
    for result in case_results:
        grouped.setdefault(str(result.get("runtime") or "unknown"), []).append(result)
    runtime_results = []
    for runtime, results in sorted(grouped.items()):
        scored = [float(item["overall_score"]) for item in results if item.get("overall_score") is not None]
        delivery_scored = [float(item["delivery_score"]) for item in results if item.get("delivery_score") is not None]
        quality_scored = [float(item["task_quality_score"]) for item in results if item.get("task_quality_score") is not None]
        contract_scored = [float(item["contract_compliance_score"]) for item in results if item.get("contract_compliance_score") is not None]
        output_contract_scored = [float(item["output_contract_score"]) for item in results if item.get("output_contract_score") is not None]
        marker_results = [item for item in results if item.get("overall_score") is not None]
        required_results = [item for item in results if item.get("overall_score") is not None and item.get("required_evidence_terms")]
        cost_results = [item for item in results if item.get("cost_usd") is not None]
        token_results = [item for item in results if item.get("token_total") is not None]
        runtime_results.append({
            "runtime": runtime,
            "case_count": len(results),
            "iteration_count": len({item.get("iteration") for item in results}),
            "completed": sum(1 for item in results if item.get("status") == "completed"),
            "needs_review": sum(1 for item in results if item.get("status") == "needs_review"),
            "failed": sum(1 for item in results if item.get("status") in {"failed", "timeout", "runtime_delivery_failed"}),
            "runtime_delivery_failed": sum(1 for item in results if item.get("runtime_delivery_failed")),
            "skipped": sum(1 for item in results if item.get("skipped")),
            "average_score": round(sum(scored) / len(scored), 3) if scored else None,
            "median_score": _median(scored),
            "average_delivery_score": round(sum(delivery_scored) / len(delivery_scored), 3) if delivery_scored else None,
            "median_delivery_score": _median(delivery_scored),
            "average_task_quality_score": round(sum(quality_scored) / len(quality_scored), 3) if quality_scored else None,
            "median_task_quality_score": _median(quality_scored),
            "average_contract_compliance_score": round(sum(contract_scored) / len(contract_scored), 3) if contract_scored else None,
            "median_contract_compliance_score": _median(contract_scored),
            "average_output_contract_score": round(sum(output_contract_scored) / len(output_contract_scored), 3) if output_contract_scored else None,
            "median_output_contract_score": _median(output_contract_scored),
            "marker_pass_count": sum(1 for item in results if item.get("marker_present")),
            "marker_pass_rate": _rate(sum(1 for item in marker_results if item.get("marker_present")), len(marker_results)),
            "required_evidence_pass_count": sum(1 for item in required_results if item.get("required_evidence_score") == 1.0),
            "required_evidence_pass_rate": _rate(
                sum(1 for item in required_results if item.get("required_evidence_score") == 1.0),
                len(required_results),
            ),
            "delivery_failure_rate": _rate(sum(1 for item in results if item.get("runtime_delivery_failed")), len(results)),
            "cost_sample_count": len(cost_results),
            "token_sample_count": len(token_results),
            "total_cost_usd": round(sum(float(item.get("cost_usd") or 0) for item in results), 6),
            "total_tokens": sum(int(item.get("token_total") or 0) for item in results),
            "total_duration_seconds": round(sum(float(item.get("duration_seconds") or 0) for item in results), 3),
            "average_duration_seconds": round(sum(float(item.get("duration_seconds") or 0) for item in results) / len(results), 3) if results else None,
        })
    return runtime_results


def _benchmark_summary(case_results: list[Dict[str, Any]], runtime_results: list[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [float(item["overall_score"]) for item in case_results if item.get("overall_score") is not None]
    delivery_scored = [float(item["delivery_score"]) for item in case_results if item.get("delivery_score") is not None]
    quality_scored = [float(item["task_quality_score"]) for item in case_results if item.get("task_quality_score") is not None]
    contract_scored = [float(item["contract_compliance_score"]) for item in case_results if item.get("contract_compliance_score") is not None]
    output_contract_scored = [float(item["output_contract_score"]) for item in case_results if item.get("output_contract_score") is not None]
    marker_results = [item for item in case_results if item.get("overall_score") is not None]
    required_results = [item for item in case_results if item.get("overall_score") is not None and item.get("required_evidence_terms")]
    return {
        "case_result_count": len(case_results),
        "runtime_count": len(runtime_results),
        "iteration_count": max((int(item.get("iterations") or 1) for item in case_results), default=1),
        "completed_count": sum(1 for item in case_results if item.get("status") == "completed"),
        "needs_review_count": sum(1 for item in case_results if item.get("status") == "needs_review"),
        "failed_count": sum(1 for item in case_results if item.get("status") in {"failed", "timeout", "runtime_delivery_failed"}),
        "runtime_delivery_failed_count": sum(1 for item in case_results if item.get("runtime_delivery_failed")),
        "skipped_count": sum(1 for item in case_results if item.get("skipped")),
        "average_score": round(sum(scored) / len(scored), 3) if scored else None,
        "median_score": _median(scored),
        "average_delivery_score": round(sum(delivery_scored) / len(delivery_scored), 3) if delivery_scored else None,
        "median_delivery_score": _median(delivery_scored),
        "average_task_quality_score": round(sum(quality_scored) / len(quality_scored), 3) if quality_scored else None,
        "median_task_quality_score": _median(quality_scored),
        "average_contract_compliance_score": round(sum(contract_scored) / len(contract_scored), 3) if contract_scored else None,
        "median_contract_compliance_score": _median(contract_scored),
        "average_output_contract_score": round(sum(output_contract_scored) / len(output_contract_scored), 3) if output_contract_scored else None,
        "median_output_contract_score": _median(output_contract_scored),
        "marker_pass_rate": _rate(sum(1 for item in marker_results if item.get("marker_present")), len(marker_results)),
        "required_evidence_pass_rate": _rate(
            sum(1 for item in required_results if item.get("required_evidence_score") == 1.0),
            len(required_results),
        ),
        "delivery_failure_rate": _rate(sum(1 for item in case_results if item.get("runtime_delivery_failed")), len(case_results)),
        "total_cost_usd": round(sum(float(item.get("cost_usd") or 0) for item in case_results), 6),
        "total_tokens": sum(int(item.get("token_total") or 0) for item in case_results),
    }


def _recommended_interpretation(
    case_results: list[Dict[str, Any]],
    runtime_results: list[Dict[str, Any]],
    *,
    live: bool,
) -> str:
    if not live and all(item.get("status") == "validated" for item in case_results):
        return "Dry-run only: runtimes and cases are valid, but no quality comparison was performed."
    if not live:
        return "Fixture benchmark: use this for harness validation, not production runtime policy decisions."
    scored = [item for item in runtime_results if item.get("average_score") is not None]
    if len(scored) < 2:
        return "Live benchmark completed with insufficient comparable runtime results for policy recommendations."
    best = max(scored, key=lambda item: float(item.get("average_score") or 0))
    return (
        f"Live benchmark suggests {best['runtime']} had the strongest score in this sample. "
        "Use repeated runs before changing runtime policy."
    )


def _normalize_mode(mode: Optional[str]) -> str:
    value = str(mode or "dry_run").strip().lower().replace("-", "_")
    return value if value in {"dry_run", "fixture"} else "dry_run"


def _normalize_runtimes(runtimes: Optional[list[str]]) -> list[str]:
    values = [str(item or "").strip().lower() for item in (runtimes or DEFAULT_RUNTIMES)]
    result = []
    for runtime in values:
        if runtime and runtime not in result:
            result.append(runtime)
    return result or list(DEFAULT_RUNTIMES)


def _normalize_iterations(iterations: int, *, live: bool) -> int:
    try:
        value = int(iterations or 1)
    except (TypeError, ValueError):
        value = 1
    upper = MAX_LIVE_ITERATIONS if live else MAX_FIXTURE_ITERATIONS
    return max(1, min(value, upper))


def _normalize_cases(cases: Optional[list[Dict[str, Any]]], *, max_cases: int) -> list[Dict[str, Any]]:
    raw_cases = cases or BUILTIN_BENCHMARK_CASES
    normalized = []
    for index, raw in enumerate(raw_cases[: max(1, min(int(max_cases or 3), MAX_CASES_LIMIT))], start=1):
        marker = str(raw.get("marker") or f"BENCH_CASE_{index}_DONE").strip()
        title = str(raw.get("title") or raw.get("case_id") or f"Benchmark case {index}").strip()
        prompt = str(raw.get("prompt") or "").strip()
        if not prompt:
            prompt = (
                f"Read-only benchmark case. Inspect the requested system area and report two concise findings. "
                "Do not edit files."
            )
        prompt = _with_marker_contract(prompt, marker)
        required_evidence_terms = [
            str(term).strip()
            for term in (raw.get("required_evidence_terms") or [])
            if str(term).strip()
        ]
        if required_evidence_terms:
            prompt = _with_required_evidence_contract(prompt, required_evidence_terms)
        normalized.append({
            "case_id": str(raw.get("case_id") or f"case_{index}").strip(),
            "title": title,
            "marker": marker,
            "prompt": prompt,
            "expected_evidence_terms": [
                str(term).strip()
                for term in (raw.get("expected_evidence_terms") or [])
                if str(term).strip()
            ],
            "required_evidence_terms": required_evidence_terms,
        })
    return normalized


def _expected_marker_line(marker: str) -> str:
    return f"FINAL_MARKER: {str(marker or '').strip()}"


def _median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[mid], 3)
    return round((ordered[mid - 1] + ordered[mid]) / 2.0, 3)


def _rate(count: int, total: int) -> Optional[float]:
    if total <= 0:
        return None
    return round(float(count) / float(total), 3)


def _with_marker_contract(prompt: str, marker: str) -> str:
    expected = _expected_marker_line(marker)
    if expected in str(prompt or "").splitlines():
        return str(prompt or "").strip()
    benchmark_prompt = (
        f"{str(prompt or '').strip()}\n\n"
        "Output contract:\n"
        "Include this benchmark result section:\n"
        "Return only this shape:\n"
        "BENCHMARK_RESULT\n"
        f"marker: {str(marker or '').strip()}\n"
        "finding_1: <one concrete finding>\n"
        "finding_2: <one concrete finding>\n"
        f"{expected}\n\n"
        "The final line must be exactly:\n"
        f"{expected}\n"
        "Do not translate, paraphrase, quote, explain, prefix, suffix, or omit the final marker line."
    )
    return append_worker_output_contract(benchmark_prompt)


def _with_required_evidence_contract(prompt: str, required_terms: list[str]) -> str:
    if not required_terms:
        return prompt
    required_line = ", ".join(required_terms)
    if "Required evidence terms:" in prompt:
        return prompt
    return (
        f"{prompt}\n\n"
        f"Required evidence terms: {required_line}\n"
        "At least one finding must include one of those exact terms. Do not substitute a generic phrase."
    )


def _marker_line_present(summary: str, marker: str) -> bool:
    if not marker:
        return False
    expected = _expected_marker_line(marker)
    return any(line.strip() == expected for line in str(summary or "").splitlines())


def _structured_result_present(text: str) -> bool:
    lowered = str(text or "").lower()
    return "benchmark_result" in lowered or ("finding_1:" in lowered and "finding_2:" in lowered)


def _findings_count(text: str) -> int:
    lowered = str(text or "").lower()
    count = 0
    for label in ("finding_1:", "finding_2:", "finding 1", "finding 2"):
        if label in lowered:
            count += 1
    if count:
        return min(count, 2)
    numbered = 0
    for line in str(text or "").splitlines():
        stripped = line.strip().lower()
        if stripped.startswith(("1.", "1)", "- 1.", "• 1.")):
            numbered += 1
        elif stripped.startswith(("2.", "2)", "- 2.", "• 2.")):
            numbered += 1
    return min(numbered, 2)


def _matched_evidence_terms(text: str, expected_terms: list[str]) -> list[str]:
    lowered = str(text or "").lower()
    matches = []
    for term in expected_terms:
        if term.lower() in lowered and term not in matches:
            matches.append(term)
    return matches


def _weighted_evidence_term_score(matched_terms: list[str], expected_terms: list[str]) -> float:
    if not expected_terms:
        return 1.0
    expected_weight = sum(_evidence_term_weight(term) for term in expected_terms)
    matched_weight = sum(_evidence_term_weight(term) for term in matched_terms)
    if expected_weight <= 0:
        return 0.0
    return round(min(1.0, matched_weight / min(expected_weight, 2.4)), 3)


def _evidence_term_weight(term: str) -> float:
    return 1.0 if _is_strong_evidence_term(term) else 0.35


def _is_strong_evidence_term(term: str) -> bool:
    text = str(term or "").strip()
    if not text:
        return False
    if "/" in text or "." in text:
        return True
    if text.endswith(("View", "Activity", "Plan", "Service", "Client", "Store", "Router", "Selection")):
        return True
    if any(char.isupper() for char in text[1:]) and len(text) >= 8:
        return True
    return False


def _file_symbol_reference_count(text: str) -> int:
    value = str(text or "")
    patterns = [
        r"\b[\w/.-]+\.(?:swift|py|ts|tsx|js|jsx|md)\b",
        r"\b[A-Z][A-Za-z0-9_]{7,}\b",
        r"\b[a-zA-Z_][a-zA-Z0-9_]*\(\)",
    ]
    refs = set()
    for pattern in patterns:
        refs.update(match.group(0) for match in re.finditer(pattern, value))
    return len(refs)


def _generic_finding_penalty(text: str, *, findings_count: int, evidence_terms_matched: list[str]) -> float:
    if not str(text or "").strip():
        return 0.0
    lowered = str(text or "").lower()
    generic_patterns = (
        "runtime metadata is displayed",
        "session metadata is displayed",
        "metadata is shown",
        "displays runtime metadata",
        "provides insight",
        "usage patterns",
        "performance",
        "current agent version",
    )
    generic_hits = sum(1 for pattern in generic_patterns if pattern in lowered)
    if generic_hits and not evidence_terms_matched:
        return 0.35
    if generic_hits and len(evidence_terms_matched) < 2:
        return 0.2
    return 0.0


def _capture_benchmark_output_tail(*, runtime: str, session_id: Optional[str], bridge: Any) -> str:
    if not session_id:
        return ""
    try:
        if runtime == "ao" and bridge is not None:
            router = WorkerRuntimeRouter(ao_bridge=bridge)
        else:
            router = WorkerRuntimeRouter()
        session = router.status(runtime, str(session_id))
        if session is None:
            return ""
        return (router.capture_output(runtime, session, lines=200) or "")[-12000:]
    except Exception:
        return ""


def _detect_prompt_delivery_failure(*, runtime: str, output_tail: str, marker: str) -> Optional[str]:
    if str(runtime or "").lower() != "ao":
        return None
    tail = str(output_tail or "")
    if not tail.strip():
        return "AO benchmark output tail was empty, so prompt delivery could not be verified."
    if _marker_line_present(tail, marker) or "BENCHMARK_RESULT" in tail:
        if not _has_worker_benchmark_result(tail, marker):
            return (
                "AO benchmark output only contains the echoed prompt/placeholder contract; "
                "no worker benchmark result was delivered."
            )
        return None
    lowered = tail.lower()
    idle_menu_patterns = (
        "› summarize recent commits",
        "> summarize recent commits",
        "summarize recent commits",
        "esc to interrupt",
        "ctrl-c to quit",
        "autocomplete",
        "what would you like",
        "press enter to",
    )
    if any(pattern in lowered for pattern in idle_menu_patterns):
        return (
            "AO/Codex appears to be idle in an interactive prompt/menu instead of executing "
            "the benchmark task; prompt delivery failed before model output."
        )
    return None


def _has_worker_benchmark_result(text: str, marker: str) -> bool:
    """Return true only when benchmark output has concrete findings, not just the prompt template."""

    value = str(text or "")
    if not _marker_line_present(value, marker) or "BENCHMARK_RESULT" not in value:
        return False
    finding_values: list[str] = []
    for line in value.splitlines():
        match = re.match(r"\s*finding_[12]\s*:\s*(.+?)\s*$", line, re.IGNORECASE)
        if not match:
            continue
        finding = match.group(1).strip()
        if not finding or finding == "<one concrete finding>" or finding.startswith("<"):
            continue
        finding_values.append(finding)
    return len(finding_values) >= 2


def _cleanup_delivery_failed_session(*, runtime: str, session_id: Optional[str], bridge: Any) -> Optional[str]:
    if str(runtime or "").lower() != "ao" or not session_id:
        return None
    try:
        if bridge is not None and hasattr(bridge, "kill"):
            bridge.kill(str(session_id))
            return "killed"
        router = WorkerRuntimeRouter()
        router.kill(runtime, str(session_id))
        return "killed"
    except Exception as exc:
        return f"kill_failed: {exc}"


def _token_total(task_status: Dict[str, Any]) -> Optional[int]:
    value = task_status.get("token_total") or task_status.get("total_tokens")
    if value is None:
        usage = _openhands_usage_from_text(
            "\n".join(
                str(task_status.get(key) or "")
                for key in ("summary", "output_tail", "status_reason")
            )
        )
        value = usage.get("token_total")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _usage_from_task_status(task_status: Dict[str, Any]) -> Dict[str, Any]:
    usage = _openhands_usage_from_text(
        "\n".join(
            str(task_status.get(key) or "")
            for key in ("summary", "output_tail", "status_reason")
        )
    )
    token_total = task_status.get("token_total") or task_status.get("total_tokens") or usage.get("token_total")
    cost_usd = task_status.get("cost_usd") or task_status.get("costUSD") or usage.get("cost_usd")
    payload = dict(usage)
    try:
        if token_total is not None:
            payload["token_total"] = int(token_total)
    except (TypeError, ValueError):
        pass
    try:
        if cost_usd is not None:
            payload["cost_usd"] = float(cost_usd)
    except (TypeError, ValueError):
        pass
    return payload


def _cost_usd(task_status: Dict[str, Any]) -> Optional[float]:
    value = task_status.get("cost_usd") or task_status.get("costUSD")
    if value is None:
        usage = _openhands_usage_from_text(
            "\n".join(
                str(task_status.get(key) or "")
                for key in ("summary", "output_tail", "status_reason")
            )
        )
        value = usage.get("cost_usd")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _openhands_usage_from_text(text: str) -> Dict[str, Any]:
    """Extract cumulative OpenHands usage/cost from transcript stats blocks."""

    value = str(text or "")
    if "usage_to_metrics" not in value and "accumulated_token_usage" not in value:
        return {}
    costs = [float(match) for match in re.findall(r"'accumulated_cost':\s*([0-9]+(?:\.[0-9]+)?)", value)]
    prompt_tokens = [int(match) for match in re.findall(r"'prompt_tokens':\s*([0-9]+)", value)]
    completion_tokens = [int(match) for match in re.findall(r"'completion_tokens':\s*([0-9]+)", value)]
    cache_read_tokens = [int(match) for match in re.findall(r"'cache_read_tokens':\s*([0-9]+)", value)]
    cache_write_tokens = [int(match) for match in re.findall(r"'cache_write_tokens':\s*([0-9]+)", value)]
    model_names = re.findall(r"'model_name':\s*'([^']+)'", value)
    latest_prompt = prompt_tokens[-1] if prompt_tokens else 0
    latest_completion = completion_tokens[-1] if completion_tokens else 0
    latest_cache_read = cache_read_tokens[-1] if cache_read_tokens else 0
    latest_cache_write = cache_write_tokens[-1] if cache_write_tokens else 0
    token_total = latest_prompt + latest_completion
    payload: Dict[str, Any] = {}
    if costs:
        payload["cost_usd"] = costs[-1]
    if token_total:
        payload["token_total"] = token_total
        payload["prompt_tokens"] = latest_prompt
        payload["completion_tokens"] = latest_completion
        payload["cache_read_tokens"] = latest_cache_read
        payload["cache_write_tokens"] = latest_cache_write
    if model_names:
        payload["usage_model"] = model_names[-1]
    return payload
