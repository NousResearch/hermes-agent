#!/usr/bin/env python3
"""Strict CI aggregate gate (shared by ci.yaml and the stable orchestrator).

Behavior tests: tests/ci/test_required_results.py

Input: the JSON ``toJSON(needs)`` of the all-checks-pass job on stdin.
Failures always fail the gate. A skipped or missing update/E2E consumer also
fails whenever ``detect.outputs`` selects its lane. Other skips additionally
fail in release mode unless the job is in :data:`EXCLUDED_JOBS` (the deferred
Desktop E2E). The PR-only checks are steps of ``repo-guards``, a job that runs
on every event, so none of them reaches the gate as a skip. The OSV scan is
advisory in its findings only — its execution is required.

``--allow-superseded`` (pull_request runs only, never release): GitHub can
deliver the same ``pull_request`` event twice for one push (seen when the push
also dismisses a stale review). Both runs share the per-PR concurrency group,
so the newer cancels the older. The older run's gate then fails on
``cancelled`` jobs, and branch protection requires the gate of *every* check
suite on the head SHA, so the PR stays blocked even after the surviving run
passes. With the flag, a gate whose only non-successes are ``cancelled`` jobs
asks the Actions API whether a *newer* run (strictly greater run id) of the
same workflow, event and head SHA is still live or succeeded. If so, that run's
own gate carries the required verdict, and this one passes. Delegation only
points forward, so it is acyclic: the newest cancelled run can never pass, and
two cancelled duplicates cannot vouch for each other. Only a run's first
attempt may delegate; a rerun (``run_attempt`` > 1) of a cancelled run fails as
before. Any failure, missing job, API error or lack of such a run keeps the
gate failing.

    echo "$NEEDS" | python3 scripts/ci/required_results.py [--release | --allow-superseded]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from typing import Any, Callable

# The deferred Desktop E2E: the only skip a strict run tolerates.
DEFERRED_JOBS = ("e2e-desktop",)
EXCLUDED_JOBS = frozenset(DEFERRED_JOBS)

NEEDS_JSON_OUTPUT = "needs-json"

# Reusable-workflow parents of the update/E2E consumers, and of the real
# remote-backend and catalog lanes. The actual detect outputs already arrive in
# toJSON(needs); a selected consumer cannot skip merely because this is a PR
# rather than a release.
_UPDATE_CONSUMER_LANES = {
    "tests": ("python", "e2e", "e2e_upgrade"),
    "tests-os": ("python", "desktop_updater", "e2e", "e2e_upgrade"),
    "e2e-desktop-update": ("e2e_desktop_update",),
    "remote-backends": ("remote_backends",),
    "catalog-tests": ("catalog",),
}


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
    lanes = (entries.get("detect") or {}).get("outputs") or {}
    required = {job for job, keys in _UPDATE_CONSUMER_LANES.items()
                if any(lanes.get(key) in ("true", True) for key in keys)}
    if not entries:
        failed.append("<no-needs>")
    failed.extend(required - entries.keys())
    for name, info in entries.items():
        result = (info or {}).get("result")
        if result == "success":
            continue
        if result == "skipped" and name not in required and (not release or name in EXCLUDED_JOBS):
            allowed_skips.append(name)
            continue
        failed.append(name)
    return {
        "ok": not failed,
        "failed": sorted(failed),
        "allowed_skips": sorted(allowed_skips),
    }


_LIVE_STATUSES = frozenset({"queued", "in_progress", "waiting", "pending", "requested"})


def only_cancellations(needs: dict[str, dict[str, Any]] | None, verdict: dict[str, Any]) -> bool:
    """True when the gate failed solely because jobs were ``cancelled``."""
    failed = verdict["failed"]
    entries = needs or {}
    return bool(failed) and all(
        name in entries and (entries[name] or {}).get("result") == "cancelled" for name in failed
    )


def find_superseding_run(own: dict[str, Any], runs: list[dict[str, Any]]) -> int | None:
    """Id of a newer live or successful run of ``own``'s workflow on its head SHA.

    Only a strictly newer run id qualifies, so delegation is acyclic and the
    newest cancelled run always fails. A rerun of ``own`` (``run_attempt`` other
    than 1) never delegates: the newer twin may itself have been cancelled.
    """
    own_id = own.get("id")
    if not isinstance(own_id, int) or own.get("run_attempt") != 1:
        return None
    for run in runs:
        run_id = run.get("id")
        if not isinstance(run_id, int) or run_id <= own_id:
            continue
        if any(run.get(key) != own.get(key) for key in ("workflow_id", "head_sha", "event")):
            continue
        if run.get("status") in _LIVE_STATUSES or (
            run.get("status") == "completed" and run.get("conclusion") == "success"
        ):
            return run_id
    return None


def _github_get(path: str) -> Any:
    api = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    request = urllib.request.Request(f"{api}/{path.lstrip('/')}")
    request.add_header("Accept", "application/vnd.github+json")
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        request.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def superseding_run_id(fetch: Callable[[str], Any] = _github_get) -> int | None:
    """Look up this run (GITHUB_REPOSITORY/GITHUB_RUN_ID) and its duplicates."""
    repo = os.environ["GITHUB_REPOSITORY"]
    own = fetch(f"repos/{repo}/actions/runs/{int(os.environ['GITHUB_RUN_ID'])}")
    listing = fetch(
        f"repos/{repo}/actions/workflows/{int(own['workflow_id'])}/runs"
        f"?head_sha={own['head_sha']}&per_page=100"
    )
    return find_superseding_run(own, listing.get("workflow_runs") or [])


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


def main(argv: list[str] | None = None, fetch: Callable[[str], Any] = _github_get) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--release",
        action="store_true",
        help="Strict mode: a skipped excluded job is the only tolerated skip.",
    )
    parser.add_argument(
        "--allow-superseded",
        action="store_true",
        help="Pass a gate failed only by cancelled jobs when a newer duplicate run of the same head SHA is live.",
    )
    args = parser.parse_args(argv)
    if args.release and args.allow_superseded:
        parser.error("--allow-superseded never applies to release runs")

    needs = json.load(sys.stdin)
    verdict = evaluate_gate(needs, release=args.release)
    compact = compact_results(needs)

    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a", encoding="utf-8") as fh:
            fh.write(f"{NEEDS_JSON_OUTPUT}={json.dumps(compact)}\n")

    print(f"{NEEDS_JSON_OUTPUT}={json.dumps(compact)}")
    duplicate = None
    if args.allow_superseded and only_cancellations(needs, verdict):
        try:
            duplicate = superseding_run_id(fetch)
        except (OSError, ValueError, KeyError, TypeError) as exc:  # fail closed
            print(f"Superseding-run lookup failed, keeping the failure: {exc}")
    for line in render_report(needs, verdict):
        if duplicate is None or not line.startswith("::error::"):
            print(line)
    if duplicate is not None:
        print(f"::notice::Cancelled duplicate run: run {duplicate} on the same head SHA "
              "carries the required verdict")
        return 0
    if verdict["ok"]:
        print("All checks passed" if args.release else "All checks passed (or were skipped)")
    return 0 if verdict["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
