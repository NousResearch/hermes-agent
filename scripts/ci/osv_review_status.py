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


def generate_review_status(
    scan_result: str, sarif_path: Path
) -> tuple[list[dict[str, Any]], bool]:
    """Return review status and whether complete, parseable evidence exists."""
    if scan_result != "success":
        return _evidence_failure(f"The OSV scan finished with result `{scan_result}`."), False
    if not sarif_path.is_file():
        return _evidence_failure("The scan succeeded but its SARIF artifact is missing."), False

    try:
        data = json.loads(sarif_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return _evidence_failure(f"The OSV SARIF artifact could not be parsed: {exc}."), False

    findings: list[tuple[str, str]] = []
    for run in data.get("runs", []):
        for result in run.get("results", []):
            rule_id = result.get("ruleId", "unknown")
            locations = result.get("locations") or [{}]
            location = (
                locations[0]
                .get("physicalLocation", {})
                .get("artifactLocation", {})
                .get("uri", "")
            )
            findings.append((str(rule_id), str(location)))

    if not findings:
        return [], True

    count = len(findings)
    noun = "vulnerability" if count == 1 else "vulnerabilities"
    detail = "\n".join(f"- {rule_id} in {location}" for rule_id, location in findings[:20])
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    status, evidence_ok = generate_review_status(args.scan_result, args.sarif)
    with args.output.open("a", encoding="utf-8") as output:
        output.write(f"review_status={json.dumps(status, separators=(',', ':'))}\n")
    return 0 if evidence_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
