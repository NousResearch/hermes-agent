#!/usr/bin/env python3
"""Strict CI aggregate gate (shared by ci.yaml and the stable orchestrator).

Behavior tests: tests/ci/test_required_results.py

Input: the JSON ``toJSON(needs)`` of the all-checks-pass job on stdin.
Any non-``success`` result fails the gate. ``skipped`` additionally fails in
release mode unless the job is in :data:`EXCLUDED_JOBS` (PR-only jobs that
cannot run on a tag event, plus the deferred Desktop E2E). The OSV scan is
advisory in its findings only — its execution is required.

    echo "$NEEDS" | python3 scripts/ci/required_results.py [--release]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

# Jobs that can never run on a release (push/tag) event, plus the deferred
# Desktop E2E. These are the only skips a strict run tolerates.
PR_ONLY_JOBS = ("history-check", "lockfile-diff", "supply-chain", "review-labels")
DEFERRED_JOBS = ("e2e-desktop",)
EXCLUDED_JOBS = frozenset((*PR_ONLY_JOBS, *DEFERRED_JOBS))

NEEDS_JSON_OUTPUT = "needs-json"


def evaluate_gate(
    needs: dict[str, dict[str, Any]] | None,
    release: bool = False,
) -> dict[str, Any]:
    """Verdict for a ``needs`` context; see the module docstring.

    Returns ``ok``, plus sorted ``failed`` (every non-success, non-allowed
    entry) and ``allowed_skips``.
    """
    failed: list[str] = []
    allowed_skips: list[str] = []
    entries = needs or {}
    if not entries:
        failed.append("<no-needs>")
    for name, info in entries.items():
        result = (info or {}).get("result")
        if result == "success":
            continue
        if result == "skipped" and (not release or name in EXCLUDED_JOBS):
            allowed_skips.append(name)
            continue
        failed.append(name)
    return {
        "ok": not failed,
        "failed": sorted(failed),
        "allowed_skips": sorted(allowed_skips),
    }


def compact_results(needs: dict[str, dict[str, Any]] | None) -> dict[str, str]:
    """{job_name: result} for every entry, for the PR comment assembler."""
    return {name: (info or {}).get("result", "") for name, info in (needs or {}).items()}


def render_report(needs: dict[str, dict[str, Any]] | None, verdict: dict[str, Any]) -> list[str]:
    """Per-job lines plus ::error:: annotations for the failed set."""
    lines: list[str] = []
    for name in sorted(needs or {}):
        result = (needs[name] or {}).get("result", "unknown")
        icon = {"success": "✅", "skipped": "⏭️"}.get(result, "❌")
        lines.append(f"{icon} {name}: {result}")
    if verdict["failed"]:
        failed = verdict["failed"]
        lines.append(f"::error::{len(failed)} job(s) failed: {', '.join(failed)}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--release",
        action="store_true",
        help="Strict mode: a skipped excluded job is the only tolerated skip.",
    )
    args = parser.parse_args(argv)

    needs = json.load(sys.stdin)
    verdict = evaluate_gate(needs, release=args.release)
    compact = compact_results(needs)

    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a", encoding="utf-8") as fh:
            fh.write(f"{NEEDS_JSON_OUTPUT}={json.dumps(compact)}\n")

    print(f"{NEEDS_JSON_OUTPUT}={json.dumps(compact)}")
    for line in render_report(needs, verdict):
        print(line)
    if verdict["ok"]:
        print("All checks passed" if args.release else "All checks passed (or were skipped)")
    return 0 if verdict["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
