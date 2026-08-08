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

  - :func:`fetch_all_review_statuses` — lists all ``review-status-*``
    artifacts on the orchestrator run (GitHub attaches reusable-workflow
    artifacts to the caller run), downloads each, parses the
    ``review_status=`` line from ``review-status.json``, and merges into
    one array. Recomputed from source every poll cycle, so statuses
    appear as soon as each job uploads its artifact.

  - :func:`run` — the polling loop. Calls the API, classifies,
    fetches artifacts, assembles, upserts, sleeps, repeats. Before
    its final exit, it gives downstream jobs a short grace period
    to appear.

The orchestrator job names (detect, all-checks-pass, comment-live, etc.)
are excluded from the comment — they're infrastructure, not review signal.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
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
        if any(s.get("name", "").startswith("Run ./.github/workflows/") for s in steps):
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
# Artifact fetching (dynamic review-status artifacts)
# ---------------------------------------------------------------------------

# Prefix for all review-status artifacts uploaded by status-producing jobs.
# Each job uploads a ``review-status-<name>`` artifact containing a
# ``review-status.json`` file in GITHUB_OUTPUT format:
#   review_status=<json array of {source, results: [...]} objects>
_REVIEW_STATUS_ARTIFACT_PREFIX = "review-status-"


def _list_artifacts(token: str, repo: str, run_id: str) -> list[dict]:
    """List artifacts for a given run (paginated)."""
    owner, repo_name = repo.split("/")
    return _api_get_paginated(
        f"{API_BASE}/repos/{owner}/{repo_name}/actions/runs/{run_id}/artifacts",
        token, list_key="artifacts",
    )


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Redirect handler that never follows — used to capture the Location."""

    def redirect_request(self, *args, **kwargs):
        return None


def _download_artifact(
    token: str, repo: str, artifact: dict, dest_dir: Path,
) -> Path | None:
    """Download a single artifact zip via the API and extract it.

    Returns the path to ``review-status.json`` inside the extracted dir,
    or ``None`` if the download or extraction failed.
    """
    owner, repo_name = repo.split("/")
    archive_download_url = artifact.get("archive_download_url", "")
    if not archive_download_url:
        return None

    # The archive_download_url is an API URL that 302s to a signed blob
    # URL. Hop 1 authenticates to the API; hop 2 follows the redirect
    # WITHOUT the Authorization header — the blob rejects a request that
    # carries both a SAS token and an Authorization header (401).
    opener = urllib.request.build_opener(_NoRedirectHandler)
    location = ""
    try:
        opener.open(urllib.request.Request(archive_download_url, headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "ci-live-comment",
        }), timeout=30)
    except urllib.error.HTTPError as e:
        location = e.headers.get("Location", "") if e.code == 302 else ""
    except Exception:
        location = ""
    if not location:
        return None

    zip_path = dest_dir / f"{artifact['name']}.zip"
    try:
        # No auth headers here; further redirects are safe to follow.
        with urllib.request.urlopen(
            urllib.request.Request(location, headers={"User-Agent": "ci-live-comment"}),
            timeout=60,
        ) as resp:
            zip_path.write_bytes(resp.read())
    except Exception:
        return None

    extract_dir = dest_dir / artifact["name"]
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            if any(".." in name or name.startswith("/") for name in zf.namelist()):
                return None
            zf.extractall(extract_dir)
    except Exception:
        return None

    status_file = extract_dir / "review-status.json"
    return status_file if status_file.exists() else None


def _parse_status_file(status_file: Path) -> list[dict]:
    """Parse a review-status.json file in GITHUB_OUTPUT format."""
    try:
        content = status_file.read_text(encoding="utf-8").strip()
        if content.startswith("review_status="):
            content = content[len("review_status="):]
        statuses = json.loads(content)
        if isinstance(statuses, list):
            return statuses
    except (json.JSONDecodeError, OSError):
        pass
    return []


def fetch_all_review_statuses(
    token: str, repo: str, run_id: str,
) -> list[dict]:
    """Fetch and merge all review-status artifacts from the run.

    Lists artifacts with the ``review-status-`` prefix on the orchestrator
    run, downloads each, parses the ``review-status.json`` inside, and
    merges into a single flat array. GitHub attaches artifacts uploaded by
    reusable workflow jobs to the caller run, so one listing covers every
    status-producing job.

    Returns the merged list of ``{source, results: [...]}`` objects.
    Artifacts that don't exist yet or fail to parse are silently skipped.
    """
    all_statuses: list[dict] = []
    temp_base = Path("/tmp/review-status-artifacts")

    try:
        artifacts = _list_artifacts(token, repo, run_id)
    except Exception:
        return all_statuses

    rs_artifacts = [
        a for a in artifacts
        if a.get("name", "").startswith(_REVIEW_STATUS_ARTIFACT_PREFIX)
    ]
    if not rs_artifacts:
        return all_statuses

    # Clean temp dir for this run's artifacts.
    run_dl_dir = temp_base / str(run_id)
    if run_dl_dir.exists():
        shutil.rmtree(run_dl_dir)
    run_dl_dir.mkdir(parents=True, exist_ok=True)

    for artifact in rs_artifacts:
        status_file = _download_artifact(token, repo, artifact, run_dl_dir)
        if status_file is None:
            continue
        statuses = _parse_status_file(status_file)
        all_statuses.extend(statuses)

    # A re-run can leave several non-expired artifacts with the same name,
    # each carrying the same source — dedupe by source so the comment
    # doesn't render duplicate sections.
    seen: set[str] = set()
    deduped: list[dict] = []
    for status in all_statuses:
        src = status.get("source", "")
        if src in seen:
            continue
        if src:
            seen.add(src)
        deduped.append(status)
    return deduped


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
    commit_info: str = "",
    interval: int = 15,
    timeout: int = 1800,
    dry_run: bool = False,
) -> int:
    """Poll for job statuses and update the PR comment until all done.

    Returns 0 on success; 1 when all jobs completed but a dependency
    failed (so ``gh run rerun --failed`` can pick it up). Comment posting
    is best-effort.
    """
    asm = _import_assembler()
    start = time.time()
    last_body = ""
    quiet_grace_used = False
    prev_completed: dict[str, str] = {}
    prev_pending: list[str] = []
    prev_artifact_count = 0

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
        infra_count = len(jobs) - total
        print(f"  [{elapsed:.0f}s] fetched {len(jobs)} jobs from API "
              f"({infra_count} infra filtered) → {len(completed)} completed, "
              f"{len(pending)} pending ({total} review jobs)")

        # Log transitions since last poll.
        new_completed = {k: v for k, v in completed.items() if k not in prev_completed}
        new_pending = [j for j in pending if j not in prev_pending]
        gone_pending = [j for j in prev_pending if j not in pending and j not in completed]
        if new_completed:
            parts = [f"{name}={result}" for name, result in new_completed.items()]
            print(f"  → {len(new_completed)} job(s) newly completed: {', '.join(parts)}")
        if new_pending:
            print(f"  → {len(new_pending)} job(s) newly appeared: {', '.join(new_pending)}")
        if gone_pending:
            print(f"  → {len(gone_pending)} job(s) disappeared from pending: {', '.join(gone_pending)}")

        # Dynamically fetch all review-status artifacts from the run.
        artifact_statuses = fetch_all_review_statuses(token, repo, run_id)
        artifact_count_changed = len(artifact_statuses) != prev_artifact_count
        if artifact_count_changed:
            print(f"  Found {len(artifact_statuses)} review status entries from artifacts "
                  f"(was {prev_artifact_count} last poll)")
        prev_artifact_count = len(artifact_statuses)

        merged_json = json.dumps(artifact_statuses) if artifact_statuses else ""
        current_commit_info = _commit_info_for_state(commit_info, pending)

        body = build_comment_body(
            asm, completed, pending, run_url, job_urls,
            merged_json,
            current_commit_info,
        )

        if body != last_body:
            change_reasons = []
            if new_completed:
                change_reasons.append(f"{len(new_completed)} new completion(s)")
            if new_pending:
                change_reasons.append(f"{len(new_pending)} new pending job(s)")
            if gone_pending:
                change_reasons.append(f"{len(gone_pending)} job(s) left pending")
            if artifact_count_changed:
                change_reasons.append("artifact statuses updated")
            if not change_reasons:
                change_reasons.append("initial post")
            reason = "; ".join(change_reasons)

            if dry_run:
                print(f"  Comment body changed ({reason}) — DRY RUN:")
                print("--- DRY RUN — comment body ---")
                print(body)
                print("--- END ---")
            else:
                cid = upsert_comment(token, repo, pr_number, body)
                if cid:
                    print(f"  Updated comment {cid} ({reason})")
                else:
                    print(f"  Failed to update comment ({reason}, will retry)", file=sys.stderr)
            last_body = body
        else:
            if pending:
                print(f"  No change since last poll. Still waiting on: {', '.join(pending)}")
            else:
                print("  No change since last poll.")

        prev_completed = completed
        prev_pending = pending

        if not pending and not quiet_grace_used:
            quiet_grace_used = True
            print("  No visible jobs pending — waiting 10s for downstream jobs to appear.")
            time.sleep(10)
            continue

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

        quiet_grace_used = False
        time.sleep(interval)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=15,
                        help="Seconds between polls (default: 15).")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Max seconds to poll before giving up (default: 1800).")
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
        commit_info=commit_info,
        interval=args.interval,
        timeout=args.timeout,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
