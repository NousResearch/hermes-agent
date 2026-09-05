#!/usr/bin/env python3
"""Live-updating CI review comment.

Polls the GitHub Actions API for job statuses in the CI run, assembles
the review comment from whatever results are available, and upserts it as a
PR comment. Repeats every ``--interval`` seconds until all jobs are
completed (or ``--timeout`` is reached), so the comment updates in real time
as each job finishes.

The comment is identified by the ``<!-- hermes-ci-review-bot -->`` marker
— the same one ``assemble_review_comment.py`` uses — so it replaces any
previous comment from an earlier run.

This runs from ``.github/workflows/ci-review-comment.yml``, a separate
``workflow_run`` workflow. Thus ``CI_RUN_ID`` names the CI run to report
on, not the run that contains this script. (The variable cannot be
called ``GITHUB_RUN_ID``: the Actions runner sets the ``GITHUB_*``
defaults itself and ignores an ``env:`` override, so that name would
silently resolve to the poller's own run — which stays ``in_progress``
for as long as the poller runs, deadlocking it against itself.)
The poller reports on runs that
it does not belong to. This is also how it covers a workflow that CI does
not contain: ``WATCH_WORKFLOWS`` names sibling workflows that the same
commit triggered (the Docker image build). Their jobs join the comment.

Architecture:

  - :func:`classify_jobs` (pure, testable) — takes a list of raw API job
    dicts and returns ``(completed, pending, job_urls)`` where ``completed``
    is a ``{name: result}`` dict (for :func:`assemble_review_comment.assemble`)
    and ``pending`` is a list of job names still running.

  - :func:`select_watched_runs` (pure, testable) — picks the sibling runs
    to merge in, newest attempt per workflow.

  - :func:`find_comment_id` / :func:`upsert_comment` — thin API wrappers.

  - :func:`fetch_all_review_statuses` — lists all ``review-status-*``
    artifacts on the CI run (GitHub attaches reusable-workflow
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

# The only conclusions that count as a pass. This mirrors the merge gate's
# ``gate_results.PASSING_RESULTS`` on purpose: the comment and the gate must
# agree on what "passed" means, or the comment renders a green ✅ next to the
# very lane ``all-checks-pass`` is going red on. The same allowlist applies to
# a *workflow run*'s own conclusion — a run and its jobs are separate
# settlement coordinates.
_PASSING_CONCLUSIONS = frozenset({"success", "skipped"})


def _display_name(job: dict) -> str:
    """Comment-facing job name — ``"Workflow / job"`` for a watched run's job."""
    name = str(job.get("name", "unknown"))
    workflow = job.get("_workflow_name")
    return f"{workflow} / {name}" if workflow else name


def _project_conclusion(conclusion: str) -> str:
    """Project an API conclusion onto a comment result, failing closed.

    Only ``success`` and ``skipped`` pass. Every other conclusion keeps its
    own name — ``cancelled``, ``action_required``, ``neutral``, ``timed_out``,
    or whatever GitHub adds next — so the comment reports what actually
    happened rather than laundering it into ``skipped``. A completed job that
    carries no conclusion at all is reported too, never assumed green: an
    absence of execution must not read as a positive result.
    """
    if conclusion in _PASSING_CONCLUSIONS:
        return conclusion
    return conclusion or "no conclusion"

def classify_jobs(api_jobs: list[dict]) -> tuple[dict[str, str], list[str], dict[str, str]]:
    """Classify raw API job dicts into completed + pending + job_urls.

    Returns ``(completed, pending, job_urls)``:

    - ``completed``: ``{job_name: result}`` where result is the job's own
      conclusion, projected by :func:`_project_conclusion` so that only
      ``"success"`` and ``"skipped"`` pass. Only non-infra jobs that have
      finished.
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
        name = _display_name(job)
        if name in _INFRA_JOBS:
            continue
        status = job.get("status", "")
        html_url = job.get("html_url", "")

        if html_url:
            job_urls[name] = html_url

        if status in ("in_progress", "queued", "waiting"):
            pending.append(name)
        elif status == "completed":
            completed[name] = _project_conclusion(str(job.get("conclusion") or ""))
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


def select_watched_runs(
    runs: list[dict], watch_names: list[str], exclude_run_id: str = "",
) -> list[dict]:
    """Pick the sibling runs whose jobs belong in the comment.

    ``runs`` is the API's run list for one commit. ``watch_names`` holds
    workflow names from ``WATCH_WORKFLOWS``. One commit can have more than
    one run of the same workflow, after a rerun or a new push. Thus this
    keeps only the newest run for each workflow name. An older attempt
    reports results that a rerun replaced.

    ``exclude_run_id`` removes the CI run itself when its name is also in
    ``watch_names``.
    """
    newest: dict[str, dict] = {}
    wanted = {n.strip() for n in watch_names if n.strip()}

    for candidate in runs:
        name = str(candidate.get("name", ""))
        if name not in wanted:
            continue
        if exclude_run_id and str(candidate.get("id", "")) == str(exclude_run_id):
            continue
        current = newest.get(name)
        if current is None or str(candidate.get("created_at", "")) > str(current.get("created_at", "")):
            newest[name] = candidate

    return list(newest.values())


def runs_all_completed(runs: list[dict]) -> bool:
    """True only when every run in the list reports ``status: completed``.

    The job list alone cannot answer "is CI done": a run that GitHub just
    created has no jobs yet, and a mid-run poll can catch the moment where
    every visible job finished but a downstream sub-workflow has not
    spawned its jobs. Both look identical to "all done" at the job level.
    The run's own ``status`` is the authoritative signal, so the poller
    must not exit while any relevant run is still ``queued`` or
    ``in_progress``. An empty list is not done — it means the poller has
    no run information at all.
    """
    return bool(runs) and all(str(r.get("status", "")) == "completed" for r in runs)


def classify_runs(
    runs: list[dict], api_jobs: list[dict],
) -> tuple[dict[str, str], dict[str, str]]:
    """Blocking entries for workflow runs that finished without passing.

    ``status: completed`` says a run stopped, not that it passed. A run
    concludes ``cancelled``, ``action_required`` or ``startup_failure``
    while its jobs endpoint returns ``total_count: 0`` — no job ever
    materialised. Classified from jobs alone that is indistinguishable from
    "everything green", so the comment would have no blocking item and
    render ``all good!`` for a commit that ran no assertions at all.

    Returns ``({run_name: conclusion}, {run_name: html_url})``. The names
    join the ``{name: result}`` dict the assembler already blocks on, so a
    run failure surfaces the same way a job failure does.

    A run whose own comment-visible jobs already report a non-passing or
    still-pending result is left out: its failure is on the comment
    already, and repeating it at run level would double every ordinary
    red build.
    """
    explained: set[str] = set()
    for job in api_jobs:
        if _display_name(job) in _INFRA_JOBS:
            continue
        status = str(job.get("status", ""))
        if status in ("in_progress", "queued", "waiting"):
            explained.add(str(job.get("_run_id", "")))
        elif status == "completed":
            if _project_conclusion(str(job.get("conclusion") or "")) not in _PASSING_CONCLUSIONS:
                explained.add(str(job.get("_run_id", "")))

    blocking: dict[str, str] = {}
    run_urls: dict[str, str] = {}
    for run_info in runs:
        if str(run_info.get("status", "")) != "completed":
            continue
        conclusion = str(run_info.get("conclusion") or "")
        if conclusion in _PASSING_CONCLUSIONS:
            continue
        run_id = str(run_info.get("id", ""))
        if run_id in explained:
            continue
        name = str(run_info.get("name") or "") or f"run {run_id}"
        blocking[name] = conclusion or "no conclusion"
        html_url = str(run_info.get("html_url") or "")
        if html_url:
            run_urls[name] = html_url

    return blocking, run_urls


def collect_run_jobs(
    token: str, repo: str, run_id: str, watch_workflows: list[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Collect all jobs in the CI run + any watched sibling runs.

    Returns ``(jobs, runs)``: a flat list of job dicts (same shape as the
    API returns, plus ``_workflow_name`` on jobs from a watched run and
    ``_run_id`` on every job), and the run objects those jobs came from.
    The run objects travel with the jobs because a run carries settlement
    information its jobs cannot: whether it is still queued
    (:func:`runs_all_completed`) and what it actually concluded
    (:func:`classify_runs`).

    Reusable-workflow (``workflow_call``) jobs need no special handling:
    GitHub flattens them into the caller run's job list, already named
    ``\"Workflow / job\"``. Watched runs are separate top-level runs
    (the Docker image build), so their jobs are fetched per run and
    prefixed here.
    """
    owner, repo_name = repo.split("/")
    run_info = _api_request(f"{API_BASE}/repos/{owner}/{repo_name}/actions/runs/{run_id}", token)
    head_sha = run_info.get("head_sha", "")

    # CI run jobs (includes every reusable-workflow job).
    all_jobs: list[dict] = []
    orch_jobs = _api_get_paginated(
        f"{API_BASE}/repos/{owner}/{repo_name}/actions/runs/{run_id}/jobs",
        token, list_key="jobs",
    )

    # Skip workflow-call placeholder steps (they're sub-workflow triggers,
    # not review signal), but KEEP in_progress / queued jobs so the poller
    # knows they're still running.
    for job in orch_jobs:
        steps = job.get("steps") or []
        if any(s.get("name", "").startswith("Run ./.github/workflows/") for s in steps):
            continue
        job["_run_id"] = str(run_info.get("id", run_id))
        all_jobs.append(job)

    if not watch_workflows or not head_sha:
        return all_jobs, [run_info]

    # Watched sibling runs for the same commit. A run can be absent on the
    # first polls. Then classify_jobs() shows nothing for it.
    sibling_runs = _api_get_paginated(
        f"{API_BASE}/repos/{owner}/{repo_name}/actions/runs?head_sha={head_sha}&per_page=100",
        token, list_key="workflow_runs",
    )
    relevant_runs = [run_info]
    for watched in select_watched_runs(sibling_runs, watch_workflows, exclude_run_id=run_id):
        relevant_runs.append(watched)
        watched_jobs = _api_get_paginated(
            f"{API_BASE}/repos/{owner}/{repo_name}/actions/runs/{watched['id']}/jobs",
            token, list_key="jobs",
        )
        for job in watched_jobs:
            job["_workflow_name"] = watched.get("name", "")
            job["_run_id"] = str(watched.get("id", ""))
            all_jobs.append(job)

    return all_jobs, relevant_runs


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
    waiting: bool = False,
    failed_runs: dict[str, str] | None = None,
    run_urls: dict[str, str] | None = None,
) -> str:
    """Assemble the comment body from current job states + static inputs.

    ``failed_runs`` is :func:`classify_runs`' ``{run_name: conclusion}`` —
    workflow runs that finished without an accepted conclusion and whose
    jobs did not already report the failure.
    """
    needs_json = json.dumps(completed) if completed else ""
    runs_json = json.dumps(failed_runs) if failed_runs else ""

    return asm_mod.assemble(
        needs_json=needs_json,
        runs_json=runs_json,
        run_url=run_url,
        job_urls=job_urls,
        run_urls=run_urls,
        review_statuses_json=review_statuses_json,
        pending_jobs=pending if pending else None,
        commit_info=commit_info,
        waiting=waiting,
    )


def _commit_info_for_state(commit_info: str, pending: bool) -> str:
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
    watch_workflows: list[str] | None = None,
) -> int:
    """Poll for job statuses and update the PR comment until all done.

    Always returns 0. The poller reports on the CI run from a different run.
    Thus a failed CI job is not a failure of this job. The CI run has its
    own gate, which reports that. Comment posting is best-effort.
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
            jobs, runs = collect_run_jobs(token, repo, run_id, watch_workflows)
        except Exception as e:
            print(f"  API error collecting jobs: {e}", file=sys.stderr)
            time.sleep(interval)
            continue

        runs_completed = runs_all_completed(runs)
        completed, pending, job_urls = classify_jobs(jobs)
        failed_runs, run_urls = classify_runs(runs, jobs)
        if failed_runs:
            parts = [f"{name}={conclusion}" for name, conclusion in sorted(failed_runs.items())]
            print(f"  → {len(failed_runs)} workflow run(s) finished without passing: "
                  f"{', '.join(parts)}")
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
        # The run status is authoritative for "done": an empty job list on
        # a run that is still queued/in_progress means GitHub has not
        # spawned the jobs yet, not that everything passed.
        all_done = not pending and runs_completed
        current_commit_info = _commit_info_for_state(commit_info, pending=not all_done)

        body = build_comment_body(
            asm, completed, pending, run_url, job_urls,
            merged_json,
            current_commit_info,
            waiting=not runs_completed,
            failed_runs=failed_runs,
            run_urls=run_urls,
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

        if all_done and not quiet_grace_used:
            quiet_grace_used = True
            print("  No jobs pending and runs report completed — "
                  "waiting 10s for downstream jobs to appear.")
            time.sleep(10)
            continue

        if all_done:
            failed = [
                name for name, result in completed.items()
                if result not in _PASSING_CONCLUSIONS
            ]
            failed += [f"{name} (workflow run)" for name in sorted(failed_runs)]
            if failed:
                print(f"  All jobs done, {len(failed)} did not pass: {', '.join(failed)}")
            else:
                print("  All jobs completed — done.")
            break

        if not pending:
            print("  No visible jobs pending, but a run is still queued or "
                  "in progress — waiting for its jobs to appear.")

        quiet_grace_used = False
        time.sleep(interval)

    return 0


def parse_watch_workflows(raw: str) -> list[str]:
    """Parse the ``WATCH_WORKFLOWS`` value into workflow names.

    One name per line. Not comma-separated: a workflow name can contain a
    comma ("Docker Build, Test, and Publish").
    """
    return [name.strip() for name in raw.splitlines() if name.strip()]


def resolve_pr_number(token: str, repo: str, head_sha: str) -> str:
    """Find the PR number for a commit when the event payload has none.

    ``workflow_run.pull_requests`` is empty for some runs. The poller has no
    comment to post without a number.
    """
    if not head_sha:
        return ""
    owner, repo_name = repo.split("/")
    try:
        results = _api_get_paginated(
            f"{API_BASE}/repos/{owner}/{repo_name}/commits/{head_sha}/pulls",
            token,
        )
    except Exception as e:
        print(f"  API error resolving PR number: {e}", file=sys.stderr)
        return ""
    for item in results:
        if isinstance(item, dict) and item.get("state") == "open":
            return str(item.get("number", ""))
    return ""


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
    run_id = os.environ.get("CI_RUN_ID", "")
    pr_number = os.environ.get("PR_NUMBER", "")
    run_url = os.environ.get("RUN_URL", "")

    # Sibling workflows to merge into the comment, one name per line. Their
    # runs are separate from the CI run, so the poller resolves them by name.
    watch_workflows = parse_watch_workflows(os.environ.get("WATCH_WORKFLOWS", ""))

    if not args.dry_run:
        if not token:
            print("GITHUB_TOKEN is required", file=sys.stderr)
            return 1
        if not repo:
            print("GITHUB_REPOSITORY is required", file=sys.stderr)
            return 1
        if not run_id:
            print("CI_RUN_ID is required", file=sys.stderr)
            return 1

    # Build commit info line from env vars (set by ci-review-comment.yml).
    commit_sha = os.environ.get("COMMIT_SHA", "")
    commit_msg = os.environ.get("COMMIT_MESSAGE", "")

    if not pr_number and not args.dry_run:
        pr_number = resolve_pr_number(token, repo, commit_sha)
        if not pr_number:
            print("No PR number found — nothing to comment on.", file=sys.stderr)
            return 0
        print(f"Resolved PR #{pr_number} from commit {commit_sha[:7]}")

    commit_url = os.environ.get("COMMIT_URL", "")
    if not commit_url and commit_sha and pr_number:
        server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
        commit_url = f"{server}/{repo}/pull/{pr_number}/commits/{commit_sha}"

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
        watch_workflows=watch_workflows,
    )


if __name__ == "__main__":
    sys.exit(main())
