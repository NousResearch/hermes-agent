#!/usr/bin/env python3
"""
braid_compare.py — Scorecard for the BRAID A/B prototype.

Compares the latest output of a baseline cron job vs a BRAID cron job for the
daily-market-briefing skill (arXiv:2512.15959). Runs a fixed set of format
compliance checks, computes a length-ratio proxy for token cost, and writes a
markdown scorecard. Designed to be invoked by Hermes's cron infrastructure via
the ``script`` field — when invoked that way, stdout is injected into the cron
LLM prompt so the agent can issue ``[SILENT]`` on parity and alert on regression.

Self-contained stdlib only. No imports from the hermes_agent package. Read-only
against ``~/.hermes/cron/output/{job_id}/`` and writes to a scorecard directory
under ``~/.hermes/cron/output/braid-compare/``.

Usage (CLI):
    braid_compare.py --baseline <job_id> --braid <job_id>
    braid_compare.py --baseline 5679360e714e --braid 1c01885c3562 --outdir /tmp/scorecards

Exit codes:
    0 — BRAID matches or exceeds baseline on every compliance check
    1 — BRAID regressed on at least one compliance check vs the baseline
    2 — Missing data (one or both jobs have no runs yet)
    3 — Unexpected error
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_ROOT = Path.home() / ".hermes" / "cron" / "output"
DEFAULT_SCORECARD_DIR = DEFAULT_OUTPUT_ROOT / "braid-compare"

# Format compliance patterns. These are calibrated against real baseline
# outputs of the daily-market-briefing skill (see memory
# reference_briefing_output_format.md).
COMPLIANCE_CHECKS: dict[str, dict] = {
    "emoji_risk_dashboard": {
        "pattern": r"(?:🟢|🟡|🔴)",
        "description": "Risk dashboard uses traffic-light emoji (🟢/🟡/🔴)",
        "min_count": 3,
    },
    "fii_dii_table": {
        "pattern": r"\b(?:FII|DII|FPI)\b[^\n]*\|[^\n]*\|",
        "description": "FII/DII/FPI flow row formatted as a pipe-delimited table",
        "min_count": 1,
    },
    "brent_crude": {
        "pattern": r"\bBrent\b",
        "description": "Brent crude oil referenced (key India macro driver)",
        "min_count": 1,
    },
    "section_headers": {
        "pattern": r"(?m)^##\s+",
        "description": "Structured with multiple `##` section headers",
        "min_count": 3,
    },
}

# Content that must NEVER appear — from the skill's strict exclusion policy.
EXCLUSION_PATTERNS = {
    "sports_entertainment": r"\b(?:cricket|IPL|Bollywood|celebrity|actress|actor)\b",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Single pattern check result for one job's output."""

    name: str
    description: str
    passed: bool
    count: int
    min_required: int


@dataclass
class JobReport:
    """All checks + metadata for one job's latest output."""

    job_id: str
    label: str
    output_path: Optional[str]
    run_at: Optional[str]
    response_chars: int
    checks: list[CheckResult]
    exclusion_hits: list[str]

    @property
    def passed_checks(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def total_checks(self) -> int:
        return len(self.checks)

    @property
    def all_passed(self) -> bool:
        return (
            self.passed_checks == self.total_checks and not self.exclusion_hits
        )


@dataclass
class Scorecard:
    """Full comparison outcome."""

    generated_at: str
    baseline: JobReport
    braid: JobReport
    braid_regressions: list[str]  # names of checks where baseline passed but BRAID failed
    braid_length_ratio: Optional[float]  # len(braid.response) / len(baseline.response)


# ---------------------------------------------------------------------------
# Output file discovery + parsing
# ---------------------------------------------------------------------------


def find_latest_output(job_dir: Path) -> Optional[Path]:
    """Return the most recent ``*.md`` output file, ordered by filename.

    The cron scheduler writes files named ``YYYY-MM-DD_HH-MM-SS.md`` so
    lexical sort is equivalent to chronological sort.
    """
    if not job_dir.is_dir():
        return None
    candidates = sorted(
        (p for p in job_dir.glob("*.md") if p.is_file()),
        reverse=True,
    )
    return candidates[0] if candidates else None


def extract_response_section(text: str) -> str:
    """Return the content of the ``## Response`` section, or empty string.

    Cron output files have a fixed structure: ``# Cron Job:`` header,
    metadata, ``## Prompt``, and ``## Response``. We only care about the
    response for compliance scoring.

    Matches the header at start-of-string or after a newline so fixture
    text that starts directly with ``## Response`` is handled too.
    """
    match = re.search(r"(?:^|\n)##\s+Response\s*\n", text)
    if match is None:
        return ""
    return text[match.end() :].strip()


def parse_run_at(path: Path) -> Optional[str]:
    """Extract the ISO timestamp from the filename ``YYYY-MM-DD_HH-MM-SS.md``."""
    stem = path.stem  # e.g. "2026-04-10_03-30-55"
    m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})", stem)
    if not m:
        return None
    date, hh, mm, ss = m.groups()
    return f"{date}T{hh}:{mm}:{ss}"


# ---------------------------------------------------------------------------
# Compliance scoring
# ---------------------------------------------------------------------------


def run_compliance_checks(response: str) -> list[CheckResult]:
    """Run every pattern in COMPLIANCE_CHECKS against the response text."""
    results: list[CheckResult] = []
    for name, spec in COMPLIANCE_CHECKS.items():
        pattern = spec["pattern"]
        min_count = int(spec["min_count"])
        matches = re.findall(pattern, response)
        count = len(matches)
        results.append(
            CheckResult(
                name=name,
                description=spec["description"],
                passed=count >= min_count,
                count=count,
                min_required=min_count,
            )
        )
    return results


def find_exclusion_hits(response: str) -> list[str]:
    """Return the list of exclusion-pattern names that matched the response."""
    hits: list[str] = []
    for name, pattern in EXCLUSION_PATTERNS.items():
        if re.search(pattern, response, re.IGNORECASE):
            hits.append(name)
    return hits


def build_job_report(
    job_id: str,
    label: str,
    output_root: Path,
) -> JobReport:
    """Assemble a JobReport from the latest output file for a job."""
    job_dir = output_root / job_id
    latest = find_latest_output(job_dir)

    if latest is None:
        return JobReport(
            job_id=job_id,
            label=label,
            output_path=None,
            run_at=None,
            response_chars=0,
            checks=[],
            exclusion_hits=[],
        )

    try:
        text = latest.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to read %s: %s", latest, exc)
        return JobReport(
            job_id=job_id,
            label=label,
            output_path=str(latest),
            run_at=parse_run_at(latest),
            response_chars=0,
            checks=[],
            exclusion_hits=[],
        )

    response = extract_response_section(text)
    return JobReport(
        job_id=job_id,
        label=label,
        output_path=str(latest),
        run_at=parse_run_at(latest),
        response_chars=len(response),
        checks=run_compliance_checks(response),
        exclusion_hits=find_exclusion_hits(response),
    )


def compare(baseline: JobReport, braid: JobReport) -> Scorecard:
    """Compute the BRAID vs baseline scorecard."""
    baseline_by_name = {c.name: c for c in baseline.checks}
    braid_by_name = {c.name: c for c in braid.checks}

    regressions: list[str] = []
    # A regression = a check the baseline passed but BRAID failed.
    for name, baseline_check in baseline_by_name.items():
        braid_check = braid_by_name.get(name)
        if baseline_check.passed and (braid_check is None or not braid_check.passed):
            regressions.append(name)

    # New exclusion hits in BRAID that weren't in baseline are also regressions.
    new_exclusions = set(braid.exclusion_hits) - set(baseline.exclusion_hits)
    for ex in sorted(new_exclusions):
        regressions.append(f"exclusion:{ex}")

    length_ratio: Optional[float] = None
    if baseline.response_chars > 0 and braid.response_chars > 0:
        length_ratio = braid.response_chars / baseline.response_chars

    return Scorecard(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        baseline=baseline,
        braid=braid,
        braid_regressions=regressions,
        braid_length_ratio=length_ratio,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_check_row(baseline: CheckResult, braid: Optional[CheckResult]) -> str:
    baseline_mark = "✅" if baseline.passed else "❌"
    baseline_cell = f"{baseline_mark} {baseline.count}/{baseline.min_required}"
    if braid is None:
        braid_cell = "— no data"
    else:
        braid_mark = "✅" if braid.passed else "❌"
        braid_cell = f"{braid_mark} {braid.count}/{braid.min_required}"
    return f"| `{baseline.name}` | {baseline_cell} | {braid_cell} | {baseline.description} |"


def render_scorecard_md(scorecard: Scorecard) -> str:
    lines: list[str] = []
    lines.append(f"# BRAID Scorecard — {scorecard.generated_at}")
    lines.append("")
    lines.append(
        "Automated compliance comparison for the daily-market-briefing "
        "BRAID A/B prototype (arXiv:2512.15959)."
    )
    lines.append("")

    # Status summary at top so [SILENT] gating is trivial
    braid = scorecard.braid
    baseline = scorecard.baseline
    if braid.output_path is None:
        lines.append("**Status: PENDING — BRAID job has no output yet.**")
    elif not scorecard.braid_regressions:
        lines.append(
            f"**Status: PARITY — BRAID passes {braid.passed_checks}/{braid.total_checks} checks, no regressions vs baseline.**"
        )
    else:
        joined = ", ".join(scorecard.braid_regressions)
        lines.append(f"**Status: REGRESSION — BRAID regressed on: {joined}**")
    lines.append("")

    # Metadata block
    lines.append("## Runs")
    lines.append("")
    lines.append("| Job | ID | Run At | Response Chars |")
    lines.append("|-----|----|--------|----------------|")
    lines.append(
        f"| baseline | `{baseline.job_id}` | {baseline.run_at or '—'} | {baseline.response_chars} |"
    )
    lines.append(
        f"| braid | `{braid.job_id}` | {braid.run_at or '—'} | {braid.response_chars} |"
    )
    if scorecard.braid_length_ratio is not None:
        pct = scorecard.braid_length_ratio * 100
        lines.append("")
        lines.append(
            f"**BRAID response length vs baseline:** {pct:.1f}% "
            f"({'shorter' if pct < 100 else 'longer'})"
        )
    lines.append("")

    # Compliance table
    lines.append("## Compliance Checks")
    lines.append("")
    lines.append("| Check | Baseline | BRAID | Description |")
    lines.append("|-------|----------|-------|-------------|")
    baseline_by_name = {c.name: c for c in baseline.checks}
    braid_by_name = {c.name: c for c in braid.checks}
    all_names = list(COMPLIANCE_CHECKS.keys())
    for name in all_names:
        b = baseline_by_name.get(name)
        if b is None:
            continue
        lines.append(_render_check_row(b, braid_by_name.get(name)))
    lines.append("")

    # Exclusion violations
    if baseline.exclusion_hits or braid.exclusion_hits:
        lines.append("## Exclusion Violations")
        lines.append("")
        lines.append(f"- baseline: {baseline.exclusion_hits or '(none)'}")
        lines.append(f"- braid: {braid.exclusion_hits or '(none)'}")
        lines.append("")

    # Explicit parity note
    lines.append("## Parity Analysis")
    lines.append("")
    if not scorecard.braid_regressions and braid.output_path is not None:
        lines.append(
            "BRAID matched or exceeded the baseline on every compliance "
            "check. If the BRAID job used a materially cheaper solver, "
            "this is evidence for the BRAID Parity Effect from the paper."
        )
    elif scorecard.braid_regressions:
        lines.append("BRAID failed the following checks that baseline passed:")
        lines.append("")
        for r in scorecard.braid_regressions:
            lines.append(f"- `{r}`")
    else:
        lines.append("No BRAID data yet — scorecard will populate after the next BRAID job run.")
    lines.append("")

    return "\n".join(lines)


def _jsonify_report(report: JobReport) -> dict:
    d = asdict(report)
    d["checks"] = [asdict(c) for c in report.checks]
    d["passed_checks"] = report.passed_checks
    d["total_checks"] = report.total_checks
    d["all_passed"] = report.all_passed
    return d


def render_scorecard_json(scorecard: Scorecard) -> str:
    payload = {
        "generated_at": scorecard.generated_at,
        "baseline": _jsonify_report(scorecard.baseline),
        "braid": _jsonify_report(scorecard.braid),
        "braid_regressions": scorecard.braid_regressions,
        "braid_length_ratio": scorecard.braid_length_ratio,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare BRAID vs baseline cron job output and emit a markdown "
            "scorecard. Exit 0 on parity, 1 on BRAID regression, 2 on missing data."
        )
    )
    parser.add_argument(
        "--baseline",
        default=os.environ.get("BRAID_BASELINE_JOB_ID", "5679360e714e"),
        help="Baseline job id (default: pre-market-briefing 5679360e714e)",
    )
    parser.add_argument(
        "--braid",
        default=os.environ.get("BRAID_BRAID_JOB_ID", "1c01885c3562"),
        help="BRAID job id (default: pre-market-briefing-braid 1c01885c3562)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root dir containing one subdir per job id",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_SCORECARD_DIR,
        help="Where to write the generated scorecard files",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Don't persist scorecard files; only emit to stdout",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON to stdout instead of markdown (scorecard file still markdown)",
    )
    return parser.parse_args(argv)


def run(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    baseline_report = build_job_report(args.baseline, "baseline", args.output_root)
    braid_report = build_job_report(args.braid, "braid", args.output_root)

    scorecard = compare(baseline_report, braid_report)
    md = render_scorecard_md(scorecard)
    js = render_scorecard_json(scorecard)

    if args.json:
        sys.stdout.write(js + "\n")
    else:
        sys.stdout.write(md + "\n")

    if not args.no_write:
        args.outdir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        md_path = args.outdir / f"scorecard-{stamp}.md"
        js_path = args.outdir / f"scorecard-{stamp}.json"
        md_path.write_text(md, encoding="utf-8")
        js_path.write_text(js, encoding="utf-8")
        sys.stderr.write(f"wrote {md_path}\n")
        sys.stderr.write(f"wrote {js_path}\n")

    # Exit codes reflect parity state — consumed by cron-invoked automation.
    if braid_report.output_path is None:
        return 2  # Missing data
    if scorecard.braid_regressions:
        return 1  # Regression
    return 0  # Parity


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception as exc:
        sys.stderr.write(f"braid_compare.py failed: {exc}\n")
        sys.exit(3)
