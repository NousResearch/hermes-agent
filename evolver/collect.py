"""Trace capture at the factory boundary (read-only).

The factory (~/workspace/hermes) already emits per-run records under
``fix_results/<issue>.json``. This collector ingests those records into the
evolver's :class:`evolver.archive.TraceArchive` without modifying the factory
or its records: it only reads.

Mapping (factory record -> failure record):
- task:        bugfix task keyed by the GitHub issue number
- trace:       one span per test-file outcome in the record
- failure_class: "regression" when the record proves red-on-base,
                 "test_failure" when tests are recorded but no red proof,
                 "unknown" otherwise
- outcome:     from the record's status field
"""
from __future__ import annotations

import json
from pathlib import Path

from .archive import make_record, make_span

_STATUS_TO_SPAN = {
    "passed": "ok",
    "failed": "error",
}


def _span_status(value: str) -> str:
    v = str(value).strip().lower()
    if v.startswith("pass"):
        return "ok"
    if v.startswith("fail"):
        return "error"
    return "unset"


def record_from_fix_result(data: dict, source_path: str = "") -> dict:
    """Convert one factory fix_results JSON dict into a failure record."""
    issue = data.get("issue", "?")
    tests = data.get("tests") or {}
    spans = [
        make_span(f"test:{name}", status=_span_status(result),
                  raw_result=result)
        for name, result in sorted(tests.items())
    ]
    if not spans:
        spans = [make_span("factory_run", status="unset",
                           note="no per-test outcomes recorded")]

    red_proof = str(data.get("red_on_base", ""))
    if red_proof:
        failure_class = "regression"
    elif tests:
        failure_class = "test_failure"
    else:
        failure_class = "unknown"

    status = str(data.get("status", "unknown")).lower()
    outcome_status = "failed" if "parked" in status or "fail" in status else (
        "passed" if "pr" in status or "merged" in status or "fixed" in status
        else "unknown")

    return make_record(
        task={
            "task_id": f"hermes-fix-{issue}",
            "kind": "bugfix",
            "summary": (data.get("note") or data.get("title") or "")[:280],
            "issue": issue,
        },
        trace_spans=spans,
        failure_class=failure_class,
        outcome={"status": outcome_status, "red_on_base": red_proof[:500]},
        source={"type": "factory", "path": source_path,
                "branch": data.get("branch", ""),
                "commit": data.get("commit", "")},
    )


def collect_fix_results(results_dir: str | Path):
    """Read every fix_results/*.json into failure records.

    Returns (records, errors): malformed files are reported, not fatal.
    """
    results_dir = Path(results_dir)
    records, errors = [], []
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as e:
            errors.append(f"{path.name}: {e}")
            continue
        try:
            records.append(record_from_fix_result(data, source_path=str(path)))
        except ValueError as e:
            errors.append(f"{path.name}: invalid record: {e}")
    return records, errors
