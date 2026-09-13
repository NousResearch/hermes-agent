#!/usr/bin/env python3
"""Convert OSV SARIF evidence into the unified CI review status."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _evidence_failure(summary: str) -> list[dict[str, Any]]:
    return [
        {
            "source": "osv scan",
            "results": [
                {
                    "kind": "action_required",
                    "title": "OSV scan evidence unavailable",
                    "summary": summary,
                    "how_to_fix": (
                        "Inspect the OSV-Scanner workflow and rerun it after the "
                        "scanner or artifact failure is resolved."
                    ),
                }
            ],
        }
    ]


def _sarif_findings(data: Any) -> list[tuple[str, str]]:
    if not isinstance(data, dict) or not isinstance(data.get("runs"), list):
        raise ValueError("the root must contain a runs array")
    findings: list[tuple[str, str]] = []
    for run in data["runs"]:
        if not isinstance(run, dict) or not isinstance(run.get("results"), list):
            raise ValueError("each run must contain a results array")
        for result in run.get("results", []):
            if not isinstance(result, dict):
                raise ValueError("each result must be an object")
            rule_id = result.get("ruleId", "unknown")
            locations = result.get("locations", [])
            if not isinstance(locations, list):
                raise ValueError("result locations must be an array")
            location = ""
            for index, item in enumerate(locations):
                uri = (
                    item
                    .get("physicalLocation", {})
                    .get("artifactLocation", {})
                    .get("uri", "")
                )
                if index == 0:
                    location = str(uri)
            findings.append((str(rule_id), location))
    return findings


def generate_review_status(
    scan_result: str, sarif_path: Path
) -> tuple[list[dict[str, Any]], bool]:
    """Return review status and whether complete, parseable evidence exists."""
    if scan_result != "success":
        summary = f"The OSV scan finished with result `{scan_result}`."
        return _evidence_failure(summary), False
    if not sarif_path.is_file():
        return _evidence_failure("The scan succeeded but its SARIF is missing."), False

    try:
        data = json.loads(
            sarif_path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (OSError, UnicodeError, ValueError) as exc:
        return _evidence_failure(f"The OSV SARIF could not be parsed: {exc}."), False

    try:
        findings = _sarif_findings(data)
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        return _evidence_failure(f"The OSV SARIF artifact is malformed: {exc}."), False

    if not findings:
        return [], True

    count = len(findings)
    noun = "vulnerability" if count == 1 else "vulnerabilities"
    detail = "\n".join(f"- {rule} in {loc}" for rule, loc in findings[:20])
    return [
        {
            "source": "osv scan",
            "results": [
                {
                    "kind": "warning",
                    "title": "OSV vulnerability scan",
                    "summary": f"{count} known {noun} found in pinned dependencies.",
                    "detail": detail,
                    "how_to_fix": (
                        "Review the findings in the [Security tab](../../security/code-scanning). "
                        "Update the affected dependencies if a patched version is available."
                    ),
                }
            ],
        }
    ], True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-result", required=True)
    parser.add_argument("--sarif", type=Path, required=True)
    parser.add_argument("--output", type=Path, action="append", required=True)
    args = parser.parse_args()

    status, evidence_ok = generate_review_status(args.scan_result, args.sarif)
    line = f"review_status={json.dumps(status, separators=(',', ':'))}\n"
    for index, output_path in enumerate(args.output):
        with output_path.open("a" if index == 0 else "w", encoding="utf-8") as output:
            output.write(line)
    return 0 if evidence_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
