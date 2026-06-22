"""Report dataclasses and renderers for Memory v2 evals."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from plugins.memory.memory_v2.retrieval import MemoryQueryRouter


@dataclass(frozen=True)
class EvalScoreRow:
    baseline: str
    query_id: str
    route: str
    source_recall: float
    text_contains: float
    suppression: float
    retrieved_count: int
    token_estimate: int
    latency_ms: float
    retrieved_source_refs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvalReport:
    dataset: str
    rows: list[EvalScoreRow]
    summary: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "dataset": self.dataset,
            "rows": [asdict(row) for row in self.rows],
            "summary": self.summary,
        }
        payload["acceptance"] = build_acceptance_scorecard(self)
        return payload


SOURCE_CORRECTNESS_MIN = 0.95
SUPPRESSION_MIN = 0.90
RAW_FTS_REGRESSION_TOLERANCE = 0.05


def build_acceptance_scorecard(report: EvalReport | dict[str, Any]) -> dict[str, Any]:
    """Build deterministic local acceptance checks for an eval report.

    The scorecard intentionally uses only metrics already present in the report;
    it performs no network or LLM calls. Per-query failures are returned under
    each check so regressions are visible in JSON output.
    """

    payload = _plain_report_payload(report)
    rows = [dict(row) for row in payload.get("rows", [])]
    summary = {str(key): dict(value) for key, value in payload.get("summary", {}).items()}
    target_baseline = "memory_v2" if "memory_v2" in summary else next(iter(summary), "")

    checks: list[dict[str, Any]] = []
    if target_baseline:
        target_rows = [row for row in rows if row.get("baseline") == target_baseline]
        target_summary = summary[target_baseline]
        checks.append(
            _threshold_check(
                name="source_correctness",
                baseline=target_baseline,
                metric="source_recall",
                actual=float(target_summary.get("source_recall_avg", 0.0)),
                threshold=SOURCE_CORRECTNESS_MIN,
                passed=False,
                failed_rows=_row_failures(
                    target_rows,
                    metric="source_recall",
                    threshold=SOURCE_CORRECTNESS_MIN,
                    comparator=">=",
                ),
                description="Average source recall should meet the local fixture acceptance floor.",
            )
        )
        checks.append(
            _threshold_check(
                name="irrelevant_suppression",
                baseline=target_baseline,
                metric="suppression",
                actual=float(target_summary.get("suppression_avg", 0.0)),
                threshold=SUPPRESSION_MIN,
                passed=False,
                failed_rows=_row_failures(
                    target_rows,
                    metric="suppression",
                    threshold=SUPPRESSION_MIN,
                    comparator=">=",
                ),
                description="Irrelevant-memory suppression should keep false positives under 10%.",
            )
        )
        for check in checks[-2:]:
            if check["name"] == "source_correctness":
                check["passed"] = float(target_summary.get("source_recall_avg", 0.0)) >= SOURCE_CORRECTNESS_MIN and not check["failed_rows"]
            elif check["name"] == "irrelevant_suppression":
                check["passed"] = float(target_summary.get("suppression_avg", 0.0)) >= SUPPRESSION_MIN and not check["failed_rows"]
        checks.append(_token_budget_check(target_rows, target_baseline))

    if "memory_v2" in summary and "raw_fts" in summary:
        memory_v2_source = float(summary["memory_v2"].get("source_recall_avg", 0.0))
        raw_fts_source = float(summary["raw_fts"].get("source_recall_avg", 0.0))
        floor = raw_fts_source - RAW_FTS_REGRESSION_TOLERANCE
        checks.append(
            _threshold_check(
                name="memory_v2_vs_raw_fts_source_recall",
                baseline="memory_v2",
                metric="source_recall_avg_delta_vs_raw_fts",
                actual=memory_v2_source - raw_fts_source,
                threshold=-RAW_FTS_REGRESSION_TOLERANCE,
                passed=memory_v2_source >= floor,
                failed_rows=[],
                description="Memory v2 should not trail raw FTS source recall by more than 5 percentage points.",
                details={"memory_v2_source_recall_avg": memory_v2_source, "raw_fts_source_recall_avg": raw_fts_source},
            )
        )

    return {
        "dataset": payload.get("dataset", ""),
        "target_baseline": target_baseline,
        "thresholds": {
            "source_correctness_min": SOURCE_CORRECTNESS_MIN,
            "suppression_min": SUPPRESSION_MIN,
            "raw_fts_regression_tolerance": RAW_FTS_REGRESSION_TOLERANCE,
        },
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def write_json_report(report: EvalReport | dict[str, Any], path: str | Path) -> None:
    """Write a stable, JSON-serializable eval report to ``path``."""

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_report_payload(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_markdown_report(report: EvalReport | dict[str, Any]) -> str:
    """Render a compact human-readable markdown scorecard."""

    payload = _report_payload(report)
    acceptance = payload.get("acceptance") or build_acceptance_scorecard(payload)
    lines = [f"# Memory v2 eval: {payload.get('dataset', '')}", "", "## Summary"]
    for baseline, values in sorted(payload.get("summary", {}).items()):
        lines.append(
            "- "
            f"{baseline}: source={values.get('source_recall_avg', 0):.3f}, "
            f"text={values.get('text_contains_avg', 0):.3f}, "
            f"suppression={values.get('suppression_avg', 0):.3f}, "
            f"tokens={values.get('token_estimate_total', 0)}"
        )
    lines.extend(["", f"## Acceptance: {'PASS' if acceptance.get('passed') else 'FAIL'}"])
    for check in acceptance.get("checks", []):
        status = "PASS" if check.get("passed") else "FAIL"
        lines.append(f"- {status} {check.get('name')}: actual={check.get('actual')} threshold={check.get('threshold')}")
        for failure in check.get("failed_rows", []):
            lines.append(f"  - {failure['baseline']} {failure['query_id']} {failure['metric']}={failure['actual']}")
    return "\n".join(lines) + "\n"


def _plain_report_payload(report: EvalReport | dict[str, Any]) -> dict[str, Any]:
    if isinstance(report, EvalReport):
        return {
            "dataset": report.dataset,
            "rows": [asdict(row) for row in report.rows],
            "summary": report.summary,
        }
    return dict(report)


def _report_payload(report: EvalReport | dict[str, Any]) -> dict[str, Any]:
    payload = _plain_report_payload(report)
    if "reports" in payload and "rows" not in payload:
        payload["reports"] = [_report_payload(item) for item in payload.get("reports", [])]
        return payload
    if "acceptance" not in payload:
        payload["acceptance"] = build_acceptance_scorecard(payload)
    return payload


def _threshold_check(
    *,
    name: str,
    baseline: str,
    metric: str,
    actual: float,
    threshold: float,
    passed: bool,
    failed_rows: list[dict[str, Any]],
    description: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "baseline": baseline,
        "metric": metric,
        "actual": actual,
        "threshold": threshold,
        "comparator": ">=",
        "passed": bool(passed),
        "description": description,
        "failed_rows": failed_rows,
        "details": details or {},
    }


def _row_failures(rows: list[dict[str, Any]], *, metric: str, threshold: float, comparator: str) -> list[dict[str, Any]]:
    failures = []
    for row in rows:
        actual = float(row.get(metric, 0.0))
        if actual < threshold:
            failures.append(
                {
                    "baseline": row.get("baseline", ""),
                    "query_id": row.get("query_id", ""),
                    "route": row.get("route", ""),
                    "metric": metric,
                    "actual": actual,
                    "threshold": threshold,
                    "comparator": comparator,
                    "retrieved_source_refs": list(row.get("retrieved_source_refs") or []),
                }
            )
    return failures


def _token_budget_check(rows: list[dict[str, Any]], baseline: str) -> dict[str, Any]:
    failures = []
    ratios = []
    for row in rows:
        route = str(row.get("route") or "")
        budget = MemoryQueryRouter._budget_and_limit(route)[0]
        actual = int(row.get("token_estimate") or 0)
        if budget > 0:
            ratios.append(actual / budget)
        if actual > budget:
            failures.append(
                {
                    "baseline": row.get("baseline", ""),
                    "query_id": row.get("query_id", ""),
                    "route": route,
                    "metric": "token_estimate",
                    "actual": actual,
                    "threshold": budget,
                    "comparator": "<=",
                }
            )
    return {
        "name": "token_budget",
        "baseline": baseline,
        "metric": "token_estimate",
        "actual": max(ratios) if ratios else 0.0,
        "threshold": 1.0,
        "comparator": "<=",
        "passed": not failures,
        "description": "Every Memory v2 eval row should stay within the router token budget for its route.",
        "failed_rows": failures,
        "details": {"route_budgets": {route: MemoryQueryRouter._budget_and_limit(route)[0] for route in sorted({str(row.get("route") or "") for row in rows})}},
    }
