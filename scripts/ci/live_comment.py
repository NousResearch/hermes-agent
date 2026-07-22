#!/usr/bin/env python3
"""Live-updating CI review comment.

Polls the GitHub Actions API for job statuses in the current run, assembles
the review comment from whatever results are available, and upserts it as a
PR comment. Repeats every ``--interval`` seconds until all jobs are
completed (or ``--timeout`` is reached), so the comment updates in real time
as each job finishes.

The comment is identified by the ``<!-- hermes-ci-review-bot -->`` marker
— the same one ``assemble_review_comment.py`` uses — so it replaces any
previous comment from an earlier run.

Architecture:

  - :func:`classify_jobs` (pure, testable) — takes a list of raw API job
    dicts and returns ``(completed, pending, job_urls)`` where ``completed``
    is a ``{name: result}`` dict (for :func:`assemble_review_comment.assemble`)
    and ``pending`` is a list of job names still running.

  - :func:`find_comment_id` / :func:`upsert_comment` — thin API wrappers.

  - :func:`_fetch_timings_statuses` — downloads the ci-timings artifact
    (if available) and parses the ``review_status=`` line from it, merging
    the status objects into the review statuses array.

  - :func:`run` — the polling loop. Calls the API, classifies, assembles,
    upserts, sleeps, repeats. Exits when all jobs are completed.

The orchestrator job names (detect, all-checks-pass, comment-live, etc.)
are excluded from the comment — they're infrastructure, not review signal.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API_BASE = "https://api.github.com"

# Job names that are infrastructure (this script, the gate, the detector)
# and should never appear in the review comment.
_INFRA_JOBS = frozenset({
    "detect",
    "all-checks-pass",
    "comment-pending",
    "comment-results",
    "comment-live",
    "CI review comment (pending)",
    "CI review comment (results)",
    "CI review comment (live)",
    "All required checks pass",
    "Detect affected areas",
})

# Map GitHub API conclusion values to our result strings.
_CONCLUSION_MAP = {
    "success": "success",
    "failure": "failure",
    "skipped": "skipped",
    "cancelled": "skipped",
    "neutral": "skipped",
    "timed_out": "failure",
    "action_required": "skipped",
}


def classify_jobs(api_jobs: list[dict]) -> tuple[dict[str, str], list[str], dict[str, str]]:
    """Classify raw API job dicts into completed + pending + job_urls.

    Returns ``(completed, pending, job_urls)``:

    - ``completed``: ``{job_name: result}`` where result is
      ``"success"`` / ``"failure"`` / ``"skipped"``. Only non-infra jobs
      that have finished.
    - ``pending``: list of job names still running (in_progress / queued
      / waiting). Excludes infra jobs.
    - ``job_urls``: ``{job_name: html_url}`` — direct links to each
      job's logs page, for the assembler to use in ❌ Error links.

    The API returns orchestrator-level jobs and sub-workflow jobs
    (workflow_call) in separate runs — :func:`collect_run_jobs` merges
    them. Each sub-workflow job has a ``_workflow_name`` prefix so the
    display name is ``"Workflow / job"``.
    """
    completed: dict[str, str] = {}
    pending: list[str] = []
    job_urls: dict[str, str] = {}

    for job in api_jobs:
        name = job.get("name", "unknown")
        if job.get("_workflow_name"):
            name = f"{job['_workflow_name']} / {name}"
        if name in _INFRA_JOBS:
            continue
        status = job.get("status", "")
        conclusion = job.get("conclusion", "")
        html_url = job.get("html_url", "")

        if html_url:
            job_urls[name] = html_url

        if status in ("in_progress", "queued", "waiting"):
            pending.append(name)
        elif status == "completed":
            result = _CONCLUSION_MAP.get(conclusion, "skipped")
            completed[name] = result
        # else: unknown status → skip

    return completed, pending, job_urls


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _api_request(url: str, token: str) -> dict:
    """Authenticated GitHub API GET (single page)."""
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "ci-live-comment",
    })
    with urllib.request.urlopen(req) as resp:
        data: dict = json.loads(resp.read())
        return data


def _api_get_paginated(url: str, token: str, list_key: str | None = None) -> list:
    """Authenticated GitHub API GET with pagination."""
    results: list = []
    while url:
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "ci-live-comment",
        })
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
            link_header = resp.headers.get("Link", "")

        if list_key:
            results.extend(data.get(list_key, []))
        elif isinstance(data, list):
            results.extend(data)
        else:
            return data

        next_url = None
        for part in link_header.split(","):
            part = part.strip()
            if 'rel="next"' in part:
                next_url = part[part.find("<") + 1:part.find(">")]
                break
        url = next_url

    return results


def collect_run_jobs(token: str, repo: str, run_id: str) -> list[dict]:
    """Collect all jobs in the orchestrator run + sub-workflow runs.

    Returns a flat list of job dicts (same shape as the API returns, plus
    ``_workflow_name`` on sub-workflow jobs).
    """
    owner, repo_name = repo.split("/")
    run_info = _api_request(f"{API_BASE}/repos/{owner}/{repo_name}/actions/runs/{run_id}", token)
    created_at = run_info.get("created_at", "")
    head_sha = run_info.get("head_sha", "")

    # Orchestrator jobs
    orch_jobs = _api_get_paginated(
        f"{API_BASE}/repos/{owner}/{repo_name}/actions/runs/{run_id}/jobs",
        token, list_key="jobs",
    )

    # Sub-workflow runs (workflow_call)
    sub_runs = _api_get_paginated(
        f"{API_BASE}/repos/{owner}/{repo_name}/actions/runs?head_sha={head_sha}&event=workflow_call&per_page=100",
        token, list_key="workflow_runs",
    )
    sub_runs = [r for r in sub_runs if r.get("created_at", "") >= created_at]

    all_jobs: list[dict] = []
    # Orchestrator jobs: skip workflow-call placeholder steps (they're
    # sub-workflow triggers, not review signal), but KEEP in_progress /
    # queued jobs so the poller knows they're still running.
    for job in orch_jobs:
        steps = job.get("steps") or []
        if any(s.get("name", "").startswith("Run ./.github/") for s in steps):
            continue
        all_jobs.append(job)

    # Sub-workflow jobs (workflow_call).
    # These runs may not exist yet on the first few polls — that's fine,
    # classify_jobs() will just show 0 pending for them.
    for sr in sub_runs:
        sr_id = sr["id"]
        sr_name = sr.get("name", "")
        sr_jobs = _api_get_paginated(
            f"{API_BASE}/repos/{owner}/{repo_name}/actions/runs/{sr_id}/jobs",
            token, list_key="jobs",
        )
        for j in sr_jobs:
            j["_workflow_name"] = sr_name
            all_jobs.append(j)

    return all_jobs


def find_comment_id(token: str, repo: str, pr_number: str) -> int | None:
    """Find our existing review comment by marker prefix."""
    owner, repo_name = repo.split("/")
    comments = _api_get_paginated(
        f"{API_BASE}/repos/{owner}/{repo_name}/issues/{pr_number}/comments",
        token,
    )
    for c in comments:
        body = c.get("body", "") if isinstance(c, dict) else ""
        if body.startswith("<!-- hermes-ci-review-bot -->"):
            return c.get("id") if isinstance(c, dict) else None
    return None


def upsert_comment(
    token: str, repo: str, pr_number: str, body: str, comment_id: int | None = None
) -> int | None:
    """Create or update the review comment. Returns the comment ID."""
    owner, repo_name = repo.split("/")
    if comment_id is None:
        comment_id = find_comment_id(token, repo, pr_number)

    if comment_id:
        url = f"{API_BASE}/repos/{owner}/{repo_name}/issues/comments/{comment_id}"
        method = "PATCH"
    else:
        url = f"{API_BASE}/repos/{owner}/{repo_name}/issues/{pr_number}/comments"
        method = "POST"

    data = json.dumps({"body": body}).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
        "User-Agent": "ci-live-comment",
    })
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
            return result.get("id")
    except urllib.error.HTTPError as e:
        print(f"  API error {e.code}: {e.reason}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Artifact fetching (ci-timings review_status)
# ---------------------------------------------------------------------------


def _fetch_artifact_statuses(
    token: str, repo: str, run_id: str, artifact_name: str,
) -> list[dict]:
    """Download a workflow artifact and extract review_status entries.

    The ci-timings job writes a ``review-status.json`` file containing
    ``review_status=<json>`` (GITHUB_OUTPUT format) into its artifact.
    This function downloads the artifact, parses the line, and returns
    the parsed status array. Returns ``[]`` if the artifact doesn't exist
    yet or can't be parsed.
    """
    try:
        result = subprocess.run(
            ["gh", "run", "download", run_id, "--repo", repo,
             "--name", artifact_name, "--dir", "/tmp/artifact-dl"],
            capture_output=True, timeout=30,
        )
        if result.returncode != 0:
            return []
    except Exception:
        return []

    status_file = Path("/tmp/artifact-dl/review-status.json")
    if not status_file.exists():
        return []

    try:
        content = status_file.read_text(encoding="utf-8").strip()
        # GITHUB_OUTPUT format: review_status=<json>
        if content.startswith("review_status="):
            content = content[len("review_status="):]
        statuses = json.loads(content)
        if isinstance(statuses, list):
            return statuses
    except (json.JSONDecodeError, OSError):
        pass

    return []


# ---------------------------------------------------------------------------
# Comment assembly
# ---------------------------------------------------------------------------


def _import_assembler():
    """Import assemble_review_comment.py from the same directory."""
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    import assemble_review_comment as asm
    return asm


def build_comment_body(
    asm_mod,
    completed: dict[str, str],
    pending: list[str],
    run_url: str,
    job_urls: dict[str, str],
    review_statuses_json: str,
    commit_info: str = "",
) -> str:
    """Assemble the comment body from current job states + static inputs."""
    needs_json = json.dumps(completed) if completed else ""

    return asm_mod.assemble(
        needs_json=needs_json,
        run_url=run_url,
        job_urls=job_urls,
        review_statuses_json=review_statuses_json,
        pending_jobs=pending if pending else None,
        commit_info=commit_info,
    )


def _merge_statuses(
    base_statuses: list[dict], extra_statuses: list[dict]
) -> str:
    """Merge two status arrays into one JSON string."""
    merged = list(base_statuses) + list(extra_statuses)
    return json.dumps(merged) if merged else ""


def _commit_info_for_state(commit_info: str, pending: list[str]) -> str:
    """Use past tense in the final comment after every CI job completes."""
    if pending:
        return commit_info
    return commit_info.replace("<sub>running on ", "<sub>ran on ", 1)


# ---------------------------------------------------------------------------
# Polling loop
# ---------------------------------------------------------------------------


def run(
    token: str,
    repo: str,
    run_id: str,
    pr_number: str,
    run_url: str,
    review_statuses_json: str = "",
    commit_info: str = "",
    interval: int = 15,
    timeout: int = 1800,
    dry_run: bool = False,
) -> int:
    """Poll for job statuses and update the PR comment until all done.

    Returns 0 always — comment posting is best-effort.
    """
    asm = _import_assembler()
    start = time.time()
    last_body = ""

    # Parse the base statuses once (from review-labels, lockfile-diff, etc.)
    try:
        base_statuses = json.loads(review_statuses_json) if review_statuses_json else []
    except (json.JSONDecodeError, TypeError):
        base_statuses = []
    print(f"  Loaded {len(base_statuses)} base review status entries")

    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            print(f"Timeout ({timeout}s) reached — stopping poll.", file=sys.stderr)
            break

        try:
            jobs = collect_run_jobs(token, repo, run_id)
        except Exception as e:
            print(f"  API error collecting jobs: {e}", file=sys.stderr)
            time.sleep(interval)
            continue

        completed, pending, job_urls = classify_jobs(jobs)
        total = len(completed) + len(pending)
        print(f"  [{elapsed:.0f}s] {len(completed)} completed, {len(pending)} pending "
              f"({total} total jobs)")

        # Try to fetch ci-timings artifact statuses (may not exist yet).
        artifact_statuses = _fetch_artifact_statuses(
            token, repo, run_id, "ci-timings-report",
        )
        if artifact_statuses:
            print(f"  Found ci-timings artifact with {len(artifact_statuses)} status entries")

        merged_json = _merge_statuses(base_statuses, artifact_statuses)
        current_commit_info = _commit_info_for_state(commit_info, pending)

        body = build_comment_body(
            asm, completed, pending, run_url, job_urls,
            merged_json,
            current_commit_info,
        )

        if body != last_body:
            if dry_run:
                print("--- DRY RUN — comment body ---")
                print(body)
                print("--- END ---")
            else:
                cid = upsert_comment(token, repo, pr_number, body)
                if cid:
                    print(f"  Updated comment {cid}")
                else:
                    print("  Failed to update comment (will retry)", file=sys.stderr)
            last_body = body
        else:
            print("  No change since last poll.")

        if not pending:
            # Check if any dependency failed. If so, exit non-zero so the
            # run shows as failed — this lets ``gh run rerun --failed``
            # (e.g. from label-rerun.yml) pick up and rerun the failed jobs.
            failed_deps = [name for name, result in completed.items() if result == "failure"]
            if failed_deps:
                print(f"  All jobs done, but {len(failed_deps)} failed: {', '.join(failed_deps)}")
                print("  Exiting with error so the run can be rerun via --failed.")
                return 1
            print("  All jobs completed — done.")
            break

        time.sleep(interval)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=15,
                        help="Seconds between polls (default: 15).")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Max seconds to poll before giving up (default: 1800).")
    parser.add_argument("--review-statuses-file", type=Path, default=None,
                        help="Path to a JSON file with merged review statuses from workflow_call jobs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print comment body instead of posting to PR.")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    pr_number = os.environ.get("PR_NUMBER", "")
    run_url = os.environ.get("RUN_URL", "")

    if not args.dry_run:
        if not token:
            print("GITHUB_TOKEN is required", file=sys.stderr)
            return 1
        if not repo:
            print("GITHUB_REPOSITORY is required", file=sys.stderr)
            return 1
        if not run_id:
            print("GITHUB_RUN_ID is required", file=sys.stderr)
            return 1
        if not pr_number:
            print("PR_NUMBER is required", file=sys.stderr)
            return 1

    # Read merged review statuses from file (prepared by the ci.yml step).
    review_statuses_json = ""
    if args.review_statuses_file:
        try:
            review_statuses_json = args.review_statuses_file.read_text(encoding="utf-8")
        except OSError as e:
            print(f"Warning: could not read review statuses file: {e}", file=sys.stderr)

    # Build commit info line from env vars (set by ci.yml).
    commit_sha = os.environ.get("COMMIT_SHA", "")
    commit_msg = os.environ.get("COMMIT_MESSAGE", "")
    commit_url = os.environ.get("COMMIT_URL", "")
    commit_info = ""
    if commit_sha:
        short_sha = commit_sha[:7]
        if commit_msg:
            # Truncate commit message to first line, max 60 chars.
            first_line = commit_msg.split("\n")[0][:60]
            if commit_url:
                commit_info = f"<sub>running on [{short_sha}]({commit_url}) — {first_line}</sub>"
            else:
                commit_info = f"<sub>running on {short_sha} — {first_line}</sub>"
        elif commit_url:
            commit_info = f"<sub>running on [{short_sha}]({commit_url})</sub>"
        else:
            commit_info = f"<sub>running on {short_sha}</sub>"

    return run(
        token=token,
        repo=repo,
        run_id=run_id,
        pr_number=pr_number,
        run_url=run_url,
        review_statuses_json=review_statuses_json,
        commit_info=commit_info,
        interval=args.interval,
        timeout=args.timeout,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
