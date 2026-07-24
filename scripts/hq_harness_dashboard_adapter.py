#!/usr/bin/env python3
"""Summarize HQ harness eval reports for Build/Review dispatch.

This adapter is intentionally dry-run/read-only: it reads an existing
`hq_harness_eval_latest.json` report and emits a compact deterministic status.
It does not mutate Gateway, cron, tool permissions, secrets, or live enforcement.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

STATUS_OK = "OK"
STATUS_REVIEW = "REVIEW"
STATUS_BLOCKED = "BLOCKED"
STATUS_RECOVER = "RECOVER"

REQUIRED_REPORT_FIELDS = (
    "run_id",
    "mode",
    "total_fixtures",
    "passed",
    "failed",
    "failures",
    "decisions",
    "safety_invariants",
)

REQUIRED_TRUE_INVARIANTS = (
    "synthetic_only",
    "no_live_tool_execution",
    "no_secret_access",
    "no_cron_mutation",
    "no_gateway_runtime_change",
)


@dataclass(frozen=True)
class DashboardSummary:
    status: str
    reason: str
    run_id: str
    mode: str
    passed: int
    total_fixtures: int
    failed: int
    failed_fixtures: list[str]
    false_invariants: list[str]
    safety_invariants: dict[str, bool]
    next_action: str


def default_report_path() -> Path:
    """Return the default latest harness report path for local Hermes homes."""

    windows_home = Path.home() / "AppData" / "Local" / "hermes"
    if windows_home.exists():
        return windows_home / "reports" / "hq_harness_eval_latest.json"
    return Path.home() / ".hermes" / "reports" / "hq_harness_eval_latest.json"


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _failed_fixture_ids(report: dict[str, Any]) -> list[str]:
    fixture_ids: list[str] = []
    failures = report.get("failures", [])
    if isinstance(failures, list):
        for failure in failures:
            if isinstance(failure, dict):
                fixture_ids.append(str(failure.get("fixture_id", "unknown")))
            else:
                fixture_ids.append("unknown")

    if fixture_ids:
        return fixture_ids

    decisions = report.get("decisions", [])
    if isinstance(decisions, list):
        for decision in decisions:
            if isinstance(decision, dict) and decision.get("passed") is False:
                fixture_ids.append(str(decision.get("fixture_id", "unknown")))
    return fixture_ids


def _summary(
    *,
    status: str,
    reason: str,
    report: dict[str, Any] | None = None,
    next_action: str,
    failed_fixtures: list[str] | None = None,
    false_invariants: list[str] | None = None,
) -> DashboardSummary:
    report = report or {}
    safety_invariants = report.get("safety_invariants", {})
    if not isinstance(safety_invariants, dict):
        safety_invariants = {}

    return DashboardSummary(
        status=status,
        reason=reason,
        run_id=str(report.get("run_id", "unknown")),
        mode=str(report.get("mode", "unknown")),
        passed=_int_value(report.get("passed")),
        total_fixtures=_int_value(report.get("total_fixtures")),
        failed=_int_value(report.get("failed")),
        failed_fixtures=failed_fixtures or [],
        false_invariants=false_invariants or [],
        safety_invariants={str(key): bool(value) for key, value in safety_invariants.items()},
        next_action=next_action,
    )


def summarize_report(report: dict[str, Any]) -> DashboardSummary:
    """Map an HQ harness eval report to OK/REVIEW/BLOCKED/RECOVER."""

    missing = [field for field in REQUIRED_REPORT_FIELDS if field not in report]
    if missing:
        return _summary(
            status=STATUS_REVIEW,
            reason="missing report fields: " + ", ".join(missing),
            report=report,
            next_action="inspect report producer before using dashboard status",
        )

    safety_invariants = report.get("safety_invariants")
    if not isinstance(safety_invariants, dict):
        return _summary(
            status=STATUS_REVIEW,
            reason="safety_invariants must be an object",
            report=report,
            next_action="inspect report schema before using dashboard status",
        )

    false_invariants = [
        invariant for invariant in REQUIRED_TRUE_INVARIANTS if safety_invariants.get(invariant) is not True
    ]
    if false_invariants:
        return _summary(
            status=STATUS_BLOCKED,
            reason="safety invariant false or missing: " + ", ".join(false_invariants),
            report=report,
            false_invariants=false_invariants,
            next_action="stop capability expansion and recover dry-run safety invariants",
        )

    total = _int_value(report.get("total_fixtures"))
    passed = _int_value(report.get("passed"))
    failed = _int_value(report.get("failed"))
    failed_fixtures = _failed_fixture_ids(report)
    decisions = report.get("decisions")
    mode = str(report.get("mode", ""))

    if mode != "dry_run_only":
        return _summary(
            status=STATUS_REVIEW,
            reason=f"unexpected harness mode: {mode or 'missing'}",
            report=report,
            next_action="inspect mode before treating the report as dry-run evidence",
        )

    if not isinstance(decisions, list) or not decisions:
        return _summary(
            status=STATUS_REVIEW,
            reason="decisions must be a non-empty list",
            report=report,
            next_action="inspect report decisions before using dashboard status",
        )

    if failed or failed_fixtures or passed != total:
        return _summary(
            status=STATUS_REVIEW,
            reason="one or more harness fixtures failed",
            report=report,
            failed_fixtures=failed_fixtures,
            next_action="inspect failed fixtures before broadening autonomy",
        )

    return _summary(
        status=STATUS_OK,
        reason="all dry-run fixtures passed and safety invariants are true",
        report=report,
        next_action="continue dry-run review; live enforcement still requires explicit approval",
    )


def summarize_path(path: Path) -> DashboardSummary:
    """Read a report from disk and summarize it without mutating anything."""

    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _summary(
            status=STATUS_RECOVER,
            reason=f"report missing: {path}",
            next_action="rerun the dry-run harness validator to regenerate the report",
        )
    except OSError as exc:
        return _summary(
            status=STATUS_RECOVER,
            reason=f"could not read report: {exc}",
            next_action="check local report path permissions and regenerate if needed",
        )

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        return _summary(
            status=STATUS_RECOVER,
            reason=f"could not parse report JSON: {exc.msg}",
            next_action="rerun the dry-run harness validator to replace malformed JSON",
        )

    if not isinstance(data, dict):
        return _summary(
            status=STATUS_REVIEW,
            reason="report JSON root must be an object",
            next_action="inspect report producer before using dashboard status",
        )
    return summarize_report(data)


def render_markdown(summary: DashboardSummary) -> str:
    failed = ", ".join(summary.failed_fixtures) if summary.failed_fixtures else "none"
    false_invariants = ", ".join(summary.false_invariants) if summary.false_invariants else "none"
    invariant_lines = [
        f"- {key}: `{value}`" for key, value in sorted(summary.safety_invariants.items())
    ] or ["- unavailable"]
    lines = [
        "# HQ Harness Dashboard Adapter",
        "",
        f"- Status: `{summary.status}`",
        f"- Reason: {summary.reason}",
        f"- Run ID: `{summary.run_id}`",
        f"- Mode: `{summary.mode}`",
        f"- Fixture pass: {summary.passed} / {summary.total_fixtures}",
        f"- Failed count: {summary.failed}",
        f"- Failed fixtures: {failed}",
        f"- False/missing invariants: {false_invariants}",
        f"- Next action: {summary.next_action}",
        "- Safety note: No live Gateway/cron/tool enforcement changed by this adapter.",
        "",
        "## Safety Invariants",
        "",
        *invariant_lines,
        "",
        "## Review Dispatch Input",
        "",
        f"Harness adapter status `{summary.status}` for `{summary.run_id}`: "
        f"{summary.passed}/{summary.total_fixtures} fixtures passed; "
        f"failed fixtures `{failed}`; false invariants `{false_invariants}`. "
        f"Next action: {summary.next_action}",
        "",
    ]
    return "\n".join(lines)


def render_json(summary: DashboardSummary) -> str:
    return json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=default_report_path(), help="Path to hq_harness_eval_latest.json")
    parser.add_argument("--json", action="store_true", help="Print compact JSON instead of Markdown")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarize_path(args.report)
    if args.json:
        print(render_json(summary), end="")
    else:
        print(render_markdown(summary), end="")
    return 0 if summary.status in {STATUS_OK, STATUS_REVIEW, STATUS_RECOVER} else 2


if __name__ == "__main__":
    raise SystemExit(main())
