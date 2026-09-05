#!/usr/bin/env python3
"""Trigger one guarded skills-index rebuild after a failed freshness probe.

The freshness workflow runs more often than the builder and may observe the
same stale index while a rebuild is already queued. This helper keeps the
deduplication and fail-closed behavior testable without contacting GitHub:
active builder runs suppress a duplicate dispatch, while an API or dispatch
failure is reported to the workflow instead of causing issue reporting to be
skipped.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Iterable, Sequence


ACTIVE_STATUSES = frozenset({"queued", "in_progress", "waiting", "requested", "pending"})


def recovery_action(statuses: Iterable[object]) -> str:
    """Return the guarded action for the current builder run statuses."""
    if any(isinstance(status, str) and status in ACTIVE_STATUSES for status in statuses):
        return "already-running"
    return "dispatch"


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run a GitHub CLI command with captured output for deterministic handling."""
    return subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )


def trigger_recovery(repo: str, workflow: str) -> str:
    """Check for an active build and dispatch one when none is active."""
    list_result = _run([
        "gh", "run", "list", "--repo", repo, "--workflow", workflow,
        "--limit", "20", "--json", "status",
    ])
    if list_result.returncode != 0:
        print("Could not inspect active skills-index runs; recovery was not dispatched.", file=sys.stderr)
        return "check-failed"

    try:
        runs = json.loads(list_result.stdout)
        statuses = [run.get("status") for run in runs if isinstance(run, dict)]
    except (TypeError, json.JSONDecodeError):
        print("GitHub returned an invalid skills-index run list; recovery was not dispatched.", file=sys.stderr)
        return "check-failed"

    action = recovery_action(statuses)
    if action == "already-running":
        print("An active skills-index build already exists; recovery was not duplicated.", file=sys.stderr)
        return action

    dispatch_result = _run(["gh", "workflow", "run", workflow, "--repo", repo])
    if dispatch_result.returncode != 0:
        print("Could not dispatch the skills-index recovery build; issue reporting will continue.", file=sys.stderr)
        return "dispatch-failed"

    print("Dispatched one skills-index recovery build.", file=sys.stderr)
    return "dispatched"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--workflow", default="skills-index.yml")
    args = parser.parse_args(argv)
    print(f"recovery={trigger_recovery(args.repo, args.workflow)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
