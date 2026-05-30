"""Recommendation-only feedback layer for Dev harness reports."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from gateway.dev_execution import DevExecutionStore
from gateway.dev_control.harness_observability import (
    DevHarnessObservabilityStore,
    generate_harness_report,
)
from gateway.dev_control.harness_benchmarks import DevHarnessBenchmarkStore
from gateway.subagent_events import SubagentEventStore
from hermes_state import DEFAULT_DB_PATH, apply_wal_with_fallback


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS dev_harness_recommendation_runs (
    recommendation_run_id TEXT PRIMARY KEY,
    report_id TEXT,
    created_at REAL NOT NULL,
    scope TEXT,
    component_hashes TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dev_harness_recommendation_runs_created_at
    ON dev_harness_recommendation_runs(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_dev_harness_recommendation_runs_report
    ON dev_harness_recommendation_runs(report_id, created_at DESC);
"""

MIN_RUNTIME_POLICY_SAMPLE = 5


@dataclass
class DevHarnessRecommendationStore:
    """Persistence for harness recommendation runs."""

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
                INSERT INTO dev_harness_recommendation_runs (
                    recommendation_run_id, report_id, created_at, scope, component_hashes, payload
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["recommendation_run_id"],
                    payload.get("report_id"),
                    float(payload["created_at"]),
                    _canonical_json(payload.get("scope") or {}),
                    _canonical_json(payload.get("component_hashes") or {}),
                    json.dumps(payload, ensure_ascii=False),
                ),
            )

    def get_run(self, recommendation_run_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            """
            SELECT payload
            FROM dev_harness_recommendation_runs
            WHERE recommendation_run_id = ?
            """,
            (str(recommendation_run_id or "").strip(),),
        ).fetchone()
        if not row:
            return None
        return json.loads(row["payload"])

    def list_runs(self, *, report_id: Optional[str] = None, limit: int = 50) -> list[Dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if report_id:
            clauses.append("report_id = ?")
            params.append(str(report_id).strip())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(int(limit or 50), 200)))
        rows = self._conn.execute(
            f"""
            SELECT recommendation_run_id, report_id, created_at, scope, component_hashes, payload
            FROM dev_harness_recommendation_runs
            {where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        runs = []
        for row in rows:
            payload = json.loads(row["payload"])
            runs.append({
                "recommendation_run_id": row["recommendation_run_id"],
                "report_id": row["report_id"],
                "created_at": float(row["created_at"]),
                "scope": payload.get("scope") or {},
                "component_hashes": payload.get("component_hashes") or {},
                "recommendation_count": len(payload.get("recommendations") or []),
                "summary": payload.get("summary") or {},
            })
        return runs


def generate_harness_recommendations(
    *,
    store: DevExecutionStore,
    event_store: SubagentEventStore,
    report_id: Optional[str] = None,
    report: Optional[Dict[str, Any]] = None,
    plan_ids: Optional[list[str]] = None,
    project_id: Optional[str] = None,
    limit: int = 25,
    since: Optional[float] = None,
    benchmark_run_id: Optional[str] = None,
    persist: bool = True,
) -> Dict[str, Any]:
    """Generate persistent, recommendation-only feedback from harness evidence."""

    report_store = DevHarnessObservabilityStore(store.db_path)
    try:
        if report is None and report_id:
            report = report_store.get_report(report_id)
            if report is None:
                raise KeyError(f"Dev harness report not found: {report_id}")
        if report is None:
            report = generate_harness_report(
                store=store,
                event_store=event_store,
                plan_ids=plan_ids,
                project_id=project_id,
                limit=limit,
                since=since,
                persist=True,
            )
    finally:
        report_store.close()

    benchmark = None
    if benchmark_run_id:
        benchmark_store = DevHarnessBenchmarkStore(store.db_path)
        try:
            benchmark = benchmark_store.get_run(benchmark_run_id)
        finally:
            benchmark_store.close()
        if benchmark is None:
            raise KeyError(f"Dev harness benchmark run not found: {benchmark_run_id}")

    created_at = time.time()
    run_id = f"devhrec-{uuid.uuid4().hex[:10]}"
    recommendations = _build_recommendations(report, benchmark=benchmark)
    result = {
        "ok": True,
        "object": "hermes.dev_harness_recommendation_run",
        "recommendation_run_id": run_id,
        "report_id": report.get("report_id"),
        "benchmark_run_id": (benchmark or {}).get("benchmark_run_id"),
        "created_at": created_at,
        "scope": report.get("scope") or {},
        "component_hashes": report.get("component_hashes") or {},
        "summary": {
            "recommendation_count": len(recommendations),
            "by_category": _count_by(recommendations, "category"),
            "by_priority": _count_by(recommendations, "priority"),
            "source_plan_count": (report.get("summary") or {}).get("plan_count", 0),
            "source_task_count": (report.get("summary") or {}).get("task_count", 0),
            "benchmark_run_id": (benchmark or {}).get("benchmark_run_id"),
        },
        "recommendations": recommendations,
        "report_snapshot": {
            "report_id": report.get("report_id"),
            "created_at": report.get("created_at"),
            "summary": report.get("summary") or {},
            "failure_patterns": report.get("failure_patterns") or [],
            "runtime_observations": report.get("runtime_observations") or [],
        },
        "benchmark_snapshot": _benchmark_snapshot(benchmark),
    }
    if persist:
        recommendation_store = DevHarnessRecommendationStore(store.db_path)
        try:
            recommendation_store.persist_run(result)
        finally:
            recommendation_store.close()
    return result


def list_harness_recommendation_runs(
    *,
    store: DevExecutionStore,
    report_id: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    recommendation_store = DevHarnessRecommendationStore(store.db_path)
    try:
        runs = recommendation_store.list_runs(report_id=report_id, limit=limit)
    finally:
        recommendation_store.close()
    return {"ok": True, "object": "list", "data": runs, "total": len(runs)}


def get_harness_recommendation_run(
    *,
    store: DevExecutionStore,
    recommendation_run_id: str,
) -> Dict[str, Any]:
    recommendation_store = DevHarnessRecommendationStore(store.db_path)
    try:
        run = recommendation_store.get_run(recommendation_run_id)
    finally:
        recommendation_store.close()
    if not run:
        raise KeyError(f"Dev harness recommendation run not found: {recommendation_run_id}")
    return run


def _build_recommendations(report: Dict[str, Any], *, benchmark: Optional[Dict[str, Any]] = None) -> list[Dict[str, Any]]:
    patterns = {item.get("pattern"): item for item in report.get("failure_patterns") or []}
    evidence_by_pattern: Dict[str, list[Dict[str, Any]]] = {}
    for evidence in report.get("evidence") or []:
        evidence_by_pattern.setdefault(str(evidence.get("pattern") or ""), []).append(evidence)

    recommendations: list[Dict[str, Any]] = []
    if prompt_echo := patterns.get("prompt_echo_summary"):
        recommendations.append(_recommend_prompt_echo(report, prompt_echo, evidence_by_pattern.get("prompt_echo_summary") or []))
    if missing_marker := patterns.get("missing_completion_marker"):
        recommendations.append(_recommend_missing_marker(report, missing_marker, evidence_by_pattern.get("missing_completion_marker") or []))
    if weak_summary := patterns.get("weak_or_missing_summary"):
        recommendations.append(_recommend_weak_summary(report, weak_summary, evidence_by_pattern.get("weak_or_missing_summary") or []))
    if runtime_fallback := patterns.get("runtime_fallback"):
        runtime_rec = _recommend_runtime_fallback(report, runtime_fallback, evidence_by_pattern.get("runtime_fallback") or [])
        if runtime_rec:
            recommendations.append(runtime_rec)
    runtime_warning_rec = _recommend_runtime_warning_rate(report, benchmark=benchmark)
    if runtime_warning_rec:
        recommendations.append(runtime_warning_rec)
    output_contract_rec = _recommend_output_contract_from_benchmark(report, benchmark)
    if output_contract_rec:
        recommendations.append(output_contract_rec)
    supervisor_rec = _recommend_supervisor_policy(report)
    if supervisor_rec:
        recommendations.append(supervisor_rec)

    unique: Dict[str, Dict[str, Any]] = {}
    for recommendation in recommendations:
        unique[recommendation["recommendation_id"]] = recommendation
    priority_order = {"high": 0, "medium": 1, "low": 2, "watch": 3}
    return sorted(
        unique.values(),
        key=lambda rec: (priority_order.get(str(rec.get("priority")), 9), str(rec.get("title") or "")),
    )


def _recommend_prompt_echo(
    report: Dict[str, Any],
    pattern: Dict[str, Any],
    evidence: list[Dict[str, Any]],
) -> Dict[str, Any]:
    count = int(pattern.get("count") or 0)
    repeated = count >= 2
    return _recommendation(
        report=report,
        category="prompt_template",
        title="Separate worker contract from task brief to reduce prompt echo summaries",
        priority="high" if repeated else "watch",
        confidence=0.86 if repeated else 0.62,
        impact="Improves final summaries by reducing cases where workers echo the delegation contract instead of reporting results.",
        risk="Low. Prompt framing changes can be tested against fixture and read-only plans before production use.",
        affected_components=["worker-contract-template", "summary-quality-classifier"],
        evidence_refs=_evidence_refs(evidence),
        reason=f"{count} task summary matched prompt/contract echo detection.",
        suggested_change="Move lengthy worker contract material out of the visible task brief where possible, and add an explicit final-answer section boundary.",
        implementation_brief=(
            "Update the AO/OpenHands worker prompt framing so the task brief is isolated under a clear heading, "
            "then add a regression test where the summary must contain findings rather than copied contract text."
        ),
        non_goals=["Do not disable any runtime based on prompt echo evidence alone.", "Do not auto-retry affected plans in this phase."],
    )


def _recommend_missing_marker(
    report: Dict[str, Any],
    pattern: Dict[str, Any],
    evidence: list[Dict[str, Any]],
) -> Dict[str, Any]:
    count = int(pattern.get("count") or 0)
    return _recommendation(
        report=report,
        category="prompt_template",
        title="Make completion marker requirements harder for workers to paraphrase",
        priority="medium" if count >= 3 else "low",
        confidence=0.78 if count >= 3 else 0.66,
        impact="Reduces false needs-review states when the worker completed the task but paraphrased the required marker.",
        risk="Low. This changes completion formatting guidance, not runtime routing or policy.",
        affected_components=["worker-contract-template", "summary-quality-classifier"],
        evidence_refs=_evidence_refs(evidence),
        reason=f"{count} task summary missed an explicitly required completion marker.",
        suggested_change="Use a final line template such as FINAL_MARKER: <literal marker>, and tell workers not to translate or paraphrase it.",
        implementation_brief=(
            "Adjust worker prompts for marker-sensitive tasks to reserve a final literal marker line. "
            "Add classifier tests for exact marker compliance and paraphrased marker failures."
        ),
        non_goals=["Do not loosen marker detection globally without tests.", "Do not auto-accept missing-marker summaries."],
    )


def _recommend_weak_summary(
    report: Dict[str, Any],
    pattern: Dict[str, Any],
    evidence: list[Dict[str, Any]],
) -> Dict[str, Any]:
    fixture_only = evidence and all(str(item.get("runtime") or "") == "fixture" for item in evidence)
    return _recommendation(
        report=report,
        category="summary_classifier",
        title="Track weak or missing summaries without changing production runtime policy",
        priority="watch" if fixture_only else "low",
        confidence=0.7 if fixture_only else 0.74,
        impact="Keeps summary-quality gates visible while avoiding overreaction to fixture-only or low-sample evidence.",
        risk="Low. This is classifier/runbook observation only.",
        affected_components=["summary-quality-classifier", "runbooks-policy-profiles"],
        evidence_refs=_evidence_refs(evidence),
        reason=f"{int(pattern.get('count') or 0)} weak or missing summary pattern(s) were found.",
        suggested_change="Keep the warning path active and require more production samples before changing runtime or runbook defaults.",
        implementation_brief=(
            "Add or maintain regression coverage for short summaries like 'unclear'. "
            "Only propose runbook limit changes when repeated production weak summaries exceed the current follow-up budget."
        ),
        non_goals=["Do not recommend disabling OpenHands or AO from weak fixture summaries.", "Do not increase follow-up limits from fixture-only evidence."],
    )


def _recommend_output_contract_from_benchmark(
    report: Dict[str, Any],
    benchmark: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not benchmark:
        return None
    summary = benchmark.get("summary") or {}
    score = _metric(summary, "median_output_contract_score")
    if score >= 0.8:
        return None
    runtime_results = benchmark.get("runtime_results") or []
    if not runtime_results:
        return None
    return _recommendation(
        report=report,
        category="prompt_template",
        title="Improve Worker Output Contract v2 compliance before increasing automation",
        priority="low" if score > 0 else "medium",
        confidence=0.72,
        impact="Structured evidence gives Hermes stronger review, supervisor, benchmark, and release-readiness inputs.",
        risk="Low. This recommends prompt/parser hardening only and does not change routing policy.",
        affected_components=["worker-contract-template", "summary-quality-classifier", "supervisor-review"],
        evidence_refs=[],
        reason=(
            f"Benchmark {benchmark.get('benchmark_run_id')} reported median output contract score {score}. "
            "Workers should return DEV_WORKER_EVIDENCE JSON consistently before Dev relies on higher automation."
        ),
        suggested_change="Tune worker prompts and parser tests until benchmark output_contract_score is consistently high.",
        implementation_brief=(
            "Run controlled AO/OpenHands read-only benchmarks and inspect malformed or missing DEV_WORKER_EVIDENCE blocks. "
            "Keep this as evidence for prompt improvements, not runtime disabling."
        ),
        non_goals=["Do not mutate runtime policy from output-contract evidence alone.", "Do not auto-retry affected plans in this phase."],
    )


def _recommend_runtime_fallback(
    report: Dict[str, Any],
    pattern: Dict[str, Any],
    evidence: list[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    task_count = int((report.get("summary") or {}).get("task_count") or 0)
    if task_count < MIN_RUNTIME_POLICY_SAMPLE:
        return None
    count = int(pattern.get("count") or 0)
    return _recommendation(
        report=report,
        category="runtime_policy",
        title="Review automatic runtime fallback causes before changing selection policy",
        priority="medium" if task_count >= 10 and count >= 3 else "low",
        confidence=0.76,
        impact="Improves routing stability by identifying why selected runtimes could not launch.",
        risk="Medium. Runtime policy changes can accidentally move write tasks to the wrong runtime.",
        affected_components=["runtime-selection-policy", "runtime-adapters"],
        evidence_refs=_evidence_refs(evidence),
        reason=f"{count} fallback event(s) occurred across {task_count} task(s).",
        suggested_change="Inspect fallback reasons by runtime and fix availability/launch checks before changing default selection preferences.",
        implementation_brief=(
            "Add tests around the observed fallback reason, then adjust runtime health checks or fallback messaging. "
            "Keep AO fallback for read-only auto-routing unless OpenHands reliability is proven with more samples."
        ),
        non_goals=["Do not disable a runtime from fallback evidence alone.", "Do not remove AO fallback in this phase."],
    )


def _recommend_runtime_warning_rate(
    report: Dict[str, Any],
    *,
    benchmark: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    benchmark_runtime_results = (benchmark or {}).get("runtime_results") or []
    if benchmark_runtime_results:
        scored = [item for item in benchmark_runtime_results if item.get("average_score") is not None]
        if scored:
            return _recommendation(
                report=report,
                category="runtime_policy",
                title="Use controlled benchmark evidence before changing runtime selection",
                priority="low",
                confidence=0.74,
                impact="Improves runtime policy decisions by favoring controlled AO/OpenHands comparisons over noisy historical samples.",
                risk="Medium. Benchmark samples are still limited and must not directly mutate routing defaults.",
                affected_components=["runtime-selection-policy", "launch-profiles"],
                evidence_refs=[],
                reason=_benchmark_runtime_reason(benchmark or {}, scored),
                suggested_change=_benchmark_runtime_suggested_change(scored),
                implementation_brief=_benchmark_runtime_implementation_brief(scored),
                non_goals=["Do not mutate runtime policy from one benchmark run.", "Do not change write-task routing from read-only benchmark evidence."],
            )
        return None

    candidates = []
    for runtime in report.get("runtime_observations") or []:
        tasks = int(runtime.get("tasks") or 0)
        warnings = int(runtime.get("warnings") or 0)
        if tasks >= MIN_RUNTIME_POLICY_SAMPLE and warnings / max(1, tasks) >= 0.4:
            candidates.append(runtime)
    if not candidates:
        return None
    runtime_labels = ", ".join(str(item.get("runtime")) for item in candidates)
    evidence_refs = [
        ref
        for pattern in report.get("failure_patterns") or []
        for ref in (pattern.get("evidence_refs") or [])
    ][:10]
    return _recommendation(
        report=report,
        category="runtime_policy",
        title="Investigate high warning rate for runtime output quality",
        priority="low",
        confidence=0.68,
        impact="Targets runtime-specific prompt/profile tuning without changing routing defaults prematurely.",
        risk="Medium. Warning-rate evidence may reflect test prompts rather than runtime quality.",
        affected_components=["runtime-selection-policy", "launch-profiles", "worker-contract-template"],
        evidence_refs=evidence_refs,
        reason=f"Runtime(s) with at least {MIN_RUNTIME_POLICY_SAMPLE} tasks crossed the warning-rate threshold: {runtime_labels}.",
        suggested_change="Compare prompts and models for warning-heavy runtimes before changing auto-routing preferences.",
        implementation_brief="Create a controlled read-only benchmark across AO and OpenHands before changing runtime preference rules.",
        non_goals=["Do not disable any runtime from warnings alone.", "Do not change write-task routing from this signal."],
    )


def _recommend_supervisor_policy(report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    summary = report.get("summary") or {}
    pending = int(summary.get("approval_pending_count") or 0)
    human_review = int(summary.get("human_review_count") or 0)
    if pending + human_review == 0:
        return None
    return _recommendation(
        report=report,
        category="supervisor_policy",
        title="Review unresolved supervisor decisions before increasing automation",
        priority="watch",
        confidence=0.64,
        impact="Keeps high-risk retry/repair decisions visible without bypassing approval gates.",
        risk="Low. Recommendation is advisory and preserves manual approval requirements.",
        affected_components=["supervisor-review", "runbooks-policy-profiles"],
        evidence_refs=[],
        reason=f"{pending} pending approval(s) and {human_review} human-review plan(s) were present in the report.",
        suggested_change="Clear or resolve pending approvals before tightening supervisor automation.",
        implementation_brief="Add a Dev prompt/report that lists stale approvals and human-review plans for manual triage.",
        non_goals=["Do not auto-apply retry, repair-retry, reassign, or human-review actions."],
    )


def _benchmark_runtime_reason(benchmark: Dict[str, Any], runtime_results: list[Dict[str, Any]]) -> str:
    metrics = []
    for item in runtime_results:
        cost_text = (
            f"${item.get('total_cost_usd')}"
            if int(item.get("cost_sample_count") or 0) > 0
            else "unavailable"
        )
        token_text = (
            str(item.get("total_tokens"))
            if int(item.get("token_sample_count") or 0) > 0
            else "unavailable"
        )
        metrics.append(
            f"{item.get('runtime')}: median score {item.get('median_score')}, "
            f"task quality {item.get('median_task_quality_score')}, "
            f"contract compliance {item.get('median_contract_compliance_score')}, "
            f"output contract {item.get('median_output_contract_score')}, "
            f"marker pass rate {item.get('marker_pass_rate')}, "
            f"required evidence pass rate {item.get('required_evidence_pass_rate')}, "
            f"delivery failure rate {item.get('delivery_failure_rate')}, "
            f"avg duration {item.get('average_duration_seconds')}s, "
            f"tokens {token_text}, "
            f"cost {cost_text}"
        )
    return (
        f"Benchmark {benchmark.get('benchmark_run_id')} provides controlled benchmark evidence: "
        + "; ".join(metrics)
        + ". Treat this as routing input, not an automatic policy change."
    )


def _benchmark_runtime_implementation_brief(runtime_results: list[Dict[str, Any]]) -> str:
    by_runtime = {str(item.get("runtime") or ""): item for item in runtime_results}
    observations = [
        "Compare median runtime score, task quality, contract compliance, marker pass rate, required evidence pass rate, delivery failure rate, duration, and cost across repeated live runs.",
    ]
    ao = by_runtime.get("ao")
    openhands = by_runtime.get("openhands")
    if ao and openhands:
        ao_duration = _metric(ao, "average_duration_seconds")
        openhands_duration = _metric(openhands, "average_duration_seconds")
        ao_score = _metric(ao, "median_score")
        openhands_score = _metric(openhands, "median_score")
        faster = "OpenHands" if openhands_duration and (not ao_duration or openhands_duration < ao_duration) else "AO"
        higher_score = "OpenHands" if openhands_score > ao_score else "AO" if ao_score > openhands_score else "neither runtime"
        observations.append(
            f"Track whether {faster} remains faster and whether {higher_score} keeps the stronger median quality score on read-only inspection prompts."
        )
    elif ao:
        observations.append("Track AO read-only benchmark latency, quality, and contract compliance across repeated runs.")
    elif openhands:
        observations.append("Track OpenHands read-only benchmark latency, quality, and contract compliance across repeated runs.")
    posture = _benchmark_runtime_policy_posture(runtime_results)
    if posture:
        observations.append(posture)
    observations.append("Require more benchmark samples before proposing runtime-selection default changes.")
    return " ".join(observations)


def _benchmark_runtime_suggested_change(runtime_results: list[Dict[str, Any]]) -> str:
    posture = _benchmark_runtime_policy_posture(runtime_results)
    if posture:
        return posture
    return "Repeat the benchmark across more read-only cases before changing runtime selection defaults."


def _benchmark_runtime_policy_posture(runtime_results: list[Dict[str, Any]]) -> str:
    by_runtime = {str(item.get("runtime") or ""): item for item in runtime_results}
    ao = by_runtime.get("ao")
    openhands = by_runtime.get("openhands")
    if not ao or not openhands:
        return ""

    ao_score = _metric(ao, "median_score")
    openhands_score = _metric(openhands, "median_score")
    ao_quality = _metric(ao, "median_task_quality_score")
    openhands_quality = _metric(openhands, "median_task_quality_score")
    ao_contract = _metric(ao, "median_contract_compliance_score")
    openhands_contract = _metric(openhands, "median_contract_compliance_score")
    ao_delivery_fail = _metric(ao, "delivery_failure_rate")
    openhands_delivery_fail = _metric(openhands, "delivery_failure_rate")
    ao_duration = _metric(ao, "average_duration_seconds")
    openhands_duration = _metric(openhands, "average_duration_seconds")

    if openhands_delivery_fail > 0:
        return "Keep AO as the read-only fallback and investigate OpenHands delivery failures before expanding auto-routing."
    if ao_delivery_fail > 0 and openhands_delivery_fail == 0:
        return "Preserve AO for write/recovery tasks, but investigate whether read-only inspection should prefer OpenHands when AO delivery fails."

    openhands_quality_advantage = openhands_quality >= ao_quality + 0.05
    openhands_contract_advantage = openhands_contract >= ao_contract + 0.25
    comparable_score = openhands_score >= ao_score - 0.05
    openhands_slower = openhands_duration > ao_duration * 1.5 if ao_duration > 0 else False

    if comparable_score and (openhands_quality_advantage or openhands_contract_advantage):
        speed_note = " despite higher latency" if openhands_slower else ""
        return (
            "If repeated live benchmarks confirm this pattern, propose keeping AO as the production default for write/retry/recovery "
            f"while preferring OpenHands for read-only inspection{speed_note}, with AO fallback unchanged."
        )
    if ao_score > openhands_score + 0.1 and ao_quality >= openhands_quality:
        return "Keep AO preferred for read-only inspection until OpenHands quality or contract compliance improves in repeated benchmarks."
    return "Keep current runtime-selection defaults and collect more comparable read-only benchmark samples."


def _metric(item: Dict[str, Any], key: str) -> float:
    try:
        return float(item.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _recommendation(
    *,
    report: Dict[str, Any],
    category: str,
    title: str,
    priority: str,
    confidence: float,
    impact: str,
    risk: str,
    affected_components: list[str],
    evidence_refs: list[str],
    reason: str,
    suggested_change: str,
    implementation_brief: str,
    non_goals: list[str],
) -> Dict[str, Any]:
    identity = {
        "report_id": report.get("report_id"),
        "category": category,
        "title": title,
        "evidence_refs": sorted(evidence_refs),
        "affected_components": sorted(affected_components),
    }
    return {
        "recommendation_id": f"devhrecitem-{_hash_value(identity)}",
        "category": category,
        "title": title,
        "status": "proposed",
        "priority": priority,
        "confidence": confidence,
        "impact": impact,
        "risk": risk,
        "affected_components": affected_components,
        "evidence_refs": evidence_refs,
        "reason": reason,
        "suggested_change": suggested_change,
        "implementation_brief": implementation_brief,
        "non_goals": non_goals,
    }


def _evidence_refs(evidence: list[Dict[str, Any]]) -> list[str]:
    return [str(item.get("evidence_id")) for item in evidence if item.get("evidence_id")]


def _count_by(items: list[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        value = str(item.get(key) or "unknown")
        counts[value] = int(counts.get(value) or 0) + 1
    return counts


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:12]


def _benchmark_snapshot(benchmark: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not benchmark:
        return None
    return {
        "benchmark_run_id": benchmark.get("benchmark_run_id"),
        "mode": benchmark.get("mode"),
        "live": benchmark.get("live"),
        "summary": benchmark.get("summary") or {},
        "runtime_results": benchmark.get("runtime_results") or [],
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
