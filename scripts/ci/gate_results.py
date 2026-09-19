#!/usr/bin/env python3
"""Aggregate ci.yaml's ``needs`` context into the single merge gate.

Branch protection requires only the ``all-checks-pass`` check, so this
script is the whole distance between a red job and ``main``.  It reads
``toJSON(needs)`` on stdin and exits non-zero unless every required job
reports a result in :data:`PASSING_RESULTS`.

``skipped`` passes on purpose: it is how a path-filtered lane reports when
a PR does not touch its area (``if: needs.detect.outputs.python == 'true'``
and friends).  Every other result fails, including ``cancelled`` — a job
GitHub cancelled ran none of its assertions, so it cannot stand in for a
green one.  Sub-workflows share a concurrency group per ref, so two merges
landing on ``main`` minutes apart used to cancel each other's lanes and
still record the commit as having passed a suite that never finished.

Also emits ``needs-json`` — a compact ``{job_name: result}`` dict — for the
live comment poller (``scripts/ci/assemble_review_comment.py``).
"""

from __future__ import annotations

import json
import os
import sys

#: Job results that satisfy the gate.  Everything else blocks the merge.
PASSING_RESULTS = frozenset({"success", "skipped"})


def compact_results(needs: dict[str, dict]) -> dict[str, str]:
    """Flatten the ``needs`` context to ``{job_name: result}``."""
    return {name: str(info.get("result", "")) for name, info in needs.items()}


def failing_jobs(results: dict[str, str]) -> list[str]:
    """Names of jobs whose result does not satisfy the gate, sorted.

    An unknown or missing result counts as failing: the gate never guesses
    that a job it cannot read was green. An empty ``results`` dict is
    handled by :func:`main`, which refuses to pass a gate that was handed
    no job to judge.
    """
    return sorted(name for name, result in results.items() if result not in PASSING_RESULTS)


def render_lines(results: dict[str, str]) -> list[str]:
    """One ``✅/❌ job: result`` line per job, sorted by job name."""
    return [
        f"{'✅' if result in PASSING_RESULTS else '❌'} {name}: {result}"
        for name, result in sorted(results.items())
    ]


def main() -> int:
    needs = json.load(sys.stdin)
    results = compact_results(needs)

    needs_json = f"needs-json={json.dumps(results)}"
    print(needs_json)
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as handle:
            handle.write(needs_json + "\n")

    for line in render_lines(results):
        print(line)

    if not results:
        print("::error::the gate received no job results at all")
        return 1

    failed = failing_jobs(results)
    if failed:
        print(f"::error::{len(failed)} job(s) did not pass: {', '.join(failed)}")
        return 1
    print("All checks passed (or were skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
