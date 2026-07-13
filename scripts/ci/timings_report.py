#!/usr/bin/env python3
"""Collect CI job/step timings from the GitHub API and generate an HTML diff report.

In CI, the script reads GITHUB_TOKEN, GITHUB_REPOSITORY, GITHUB_RUN_ID, and
GITHUB_SHA from the environment to collect timings via the REST API.

If a baseline JSON file (ci-timings-baseline.json by default) exists, the
report includes a diff with per-job and per-step deltas, plus a gantt chart
overlaying current vs baseline bars.

Usage:
    # Collect from API (CI mode):
    python scripts/ci/timings_report.py

    # Regenerate HTML from saved JSON (testing):
    python scripts/ci/timings_report.py --from-json ci-timings.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from html import escape

API_BASE = "https://api.github.com"

# Retry policy for GitHub API calls. The repo-scoped GITHUB_TOKEN shares a
# rate-limit budget across every concurrent workflow run; when several PRs
# run CI at once, this report job (which makes dozens of paginated calls)
# regularly hits 403 rate-limit responses. Those are transient — retry with
# backoff, honoring Retry-After / X-RateLimit-Reset when present.
_RETRY_STATUSES = {403, 429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 5
_MAX_RETRY_WAIT_S = 120.0


class TimingsUnavailable(Exception):
    """GitHub API data could not be collected (rate limit, outage, ...).

    This is a REPORT job — never a reason to fail the PR's checks. main()
    catches this and exits 0 with a degraded summary.
    """


def _retry_wait_s(headers, attempt: int) -> float:
    """Seconds to wait before the next attempt, honoring server hints."""
    retry_after = (headers.get("Retry-After") or "").strip() if headers else ""
    if retry_after.isdigit():
        return min(float(retry_after), _MAX_RETRY_WAIT_S)
    reset = (headers.get("X-RateLimit-Reset") or "").strip() if headers else ""
    remaining = (headers.get("X-RateLimit-Remaining") or "").strip() if headers else ""
    if remaining == "0" and reset.isdigit():
        return min(max(float(reset) - time.time(), 1.0), _MAX_RETRY_WAIT_S)
    return min(2.0 ** attempt * 2.0, _MAX_RETRY_WAIT_S)  # 4s, 8s, 16s, 32s


def _urlopen_with_retry(req: urllib.request.Request):
    """urlopen with backoff on rate-limit/transient statuses.

    Returns (parsed_json, link_header). Raises TimingsUnavailable when
    attempts are exhausted — callers treat that as "no report this run",
    not a job failure.
    """
    last_err: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read()), resp.headers.get("Link", "")
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code not in _RETRY_STATUSES or attempt == _MAX_ATTEMPTS:
                break
            wait = _retry_wait_s(e.headers, attempt)
            print(f"GitHub API {e.code} on {req.full_url} — "
                  f"retry {attempt}/{_MAX_ATTEMPTS - 1} in {wait:.0f}s",
                  file=sys.stderr)
            time.sleep(wait)
        except urllib.error.URLError as e:
            last_err = e
            if attempt == _MAX_ATTEMPTS:
                break
            wait = _retry_wait_s(None, attempt)
            print(f"GitHub API connection error on {req.full_url} ({e.reason}) — "
                  f"retry {attempt}/{_MAX_ATTEMPTS - 1} in {wait:.0f}s",
                  file=sys.stderr)
            time.sleep(wait)
    raise TimingsUnavailable(
        f"GitHub API unavailable after {_MAX_ATTEMPTS} attempts: {last_err}"
    )


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

def api_get(path: str, token: str, params: dict | None = None,
            list_key: str | None = None) -> list | dict:
    """Authenticated GitHub API GET with automatic pagination.

    For list endpoints, pass list_key to extract items from the paginated
    wrapper response (e.g. list_key='jobs' for {'total_count': N, 'jobs': [...]}).
    When list_key is omitted, a non-list response is returned as-is (single object).

    Transient failures (403 rate limit, 429, 5xx, connection errors) are
    retried with backoff; exhausted retries raise TimingsUnavailable.
    """
    url = f"{API_BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)

    results: list = []
    while url:
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "ci-timings-report",
        })
        data, link_header = _urlopen_with_retry(req)

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


def parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def dur_s(started: str | None, completed: str | None) -> float | None:
    s = parse_ts(started)
    e = parse_ts(completed)
    if not s or not e:
        return None
    return (e - s).total_seconds()


def is_skipped(job: dict) -> bool:
    """A job is 'skipped' when GitHub didn't actually run it.

    Skipped jobs have conclusion == 'skipped' and typically have null or
    equal started_at/completed_at timestamps, yielding duration_s of None
    or 0. They should be excluded from delta comparisons, gantt bars,
    regression tables, and aggregate stats (wall/compute).
    """
    return job.get("conclusion") == "skipped"


# ---------------------------------------------------------------------------
# Timings collection
# ---------------------------------------------------------------------------

def _normalize_job(raw: dict) -> dict:
    steps = []
    for step in (raw.get("steps") or []):
        steps.append({
            "name": step.get("name", ""),
            "number": step.get("number", 0),
            "status": step.get("status", ""),
            "conclusion": step.get("conclusion", ""),
            "started_at": step.get("started_at"),
            "completed_at": step.get("completed_at"),
            "duration_s": dur_s(step.get("started_at"), step.get("completed_at")),
        })
    return {
        "name": raw.get("name", "unknown"),
        "workflow_name": raw.get("_workflow_name", ""),
        "job_id": raw.get("id"),
        "status": raw.get("status", ""),
        "conclusion": raw.get("conclusion", ""),
        "started_at": raw.get("started_at"),
        "completed_at": raw.get("completed_at"),
        "duration_s": dur_s(raw.get("started_at"), raw.get("completed_at")),
        "html_url": raw.get("html_url", ""),
        "steps": steps,
    }


def collect_timings(token: str, repo: str, run_id: str, head_sha: str) -> dict:
    """Collect job/step timings from the GitHub API.

    1. Get orchestrator run's direct jobs (detect, all-checks-pass, etc.).
       Skip workflow-call placeholder jobs (step name starts with "Run ./.github/").
    2. Find sub-workflow runs via head_sha + event=workflow_call.
    3. Get each sub-workflow run's jobs with full step timing.
    """
    owner, repo_name = repo.split("/")

    # Orchestrator run info
    run_info = api_get(f"/repos/{owner}/{repo_name}/actions/runs/{run_id}", token)
    created_at = run_info.get("created_at", "")

    # Orchestrator direct jobs
    orch_jobs = api_get(f"/repos/{owner}/{repo_name}/actions/runs/{run_id}/jobs",
                        token, list_key="jobs")

    direct = []
    for job in orch_jobs:
        steps = job.get("steps") or []
        if any(s.get("name", "").startswith("Run ./.github/") for s in steps):
            continue  # workflow-call placeholder
        if job.get("status") in ("in_progress", "queued"):
            continue  # skip self / unfinished
        direct.append(job)

    # Sub-workflow runs
    sub_runs = api_get(f"/repos/{owner}/{repo_name}/actions/runs", token, params={
        "head_sha": head_sha,
        "event": "workflow_call",
        "per_page": 100,
    }, list_key="workflow_runs")
    sub_runs = [r for r in sub_runs if r.get("created_at", "") >= created_at]

    sub_jobs_raw = []
    for sr in sub_runs:
        sr_id = sr["id"]
        sr_name = sr.get("name", "")
        sr_jobs = api_get(f"/repos/{owner}/{repo_name}/actions/runs/{sr_id}/jobs",
                          token, list_key="jobs")
        for j in sr_jobs:
            j["_workflow_name"] = sr_name
            j["_workflow_run_id"] = sr_id
            sub_jobs_raw.append(j)

    # Normalize + sort
    all_jobs = [_normalize_job(j) for j in direct + sub_jobs_raw]
    all_jobs = [j for j in all_jobs if j["status"] not in ("in_progress", "queued")]
    all_jobs.sort(key=lambda j: j.get("started_at") or "")

    return {
        "run_id": run_id,
        "head_sha": head_sha,
        "created_at": created_at,
        "jobs": all_jobs,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_dur(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m = int(seconds // 60)
    s = seconds % 60
    if s == 0:
        return f"{m}m"
    return f"{m}m{s:.0f}s"


def fmt_delta(current: float | None, baseline: float | None) -> tuple[str, str]:
    """Return (text, css_class) for a delta."""
    if current is None or baseline is None:
        return ("—", "neutral")
    delta = current - baseline
    if baseline == 0:
        pct_str = "new" if delta > 0 else "0%"
    else:
        pct = (delta / baseline) * 100
        pct_str = f"{pct:+.1f}%"
    if abs(delta) < 1.0:
        cls = "neutral"
    elif delta > 0:
        cls = "slower"
    else:
        cls = "faster"
    sign = "+" if delta >= 0 else ""
    return (f"{sign}{delta:.1f}s ({pct_str})", cls)


def nice_ticks(max_seconds: float, num_ticks: int = 8) -> list[int]:
    if max_seconds <= 0:
        return [0]
    raw = max_seconds / num_ticks
    for nice in [5, 10, 15, 30, 60, 120, 180, 300, 600, 900, 1800, 3600, 7200]:
        if nice >= raw:
            step = nice
            break
    else:
        step = max(int(raw), 3600)
    return list(range(0, int(max_seconds) + step + 1, step))


def fmt_tick(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if s == 0:
        return f"{m}m"
    return f"{m}m{s}s"


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------

def compute_stats(timings: dict, baseline: dict | None = None) -> dict:
    jobs_all = timings.get("jobs", [])
    jobs = [j for j in jobs_all if not is_skipped(j)]
    bl_jobs_all = (baseline or {}).get("jobs", [])
    bl_jobs = [j for j in bl_jobs_all if not is_skipped(j)]
    bl_map = {j["name"]: j for j in bl_jobs}

    # Wall time (skipped jobs have no real timestamps)
    starts = [s for s in (parse_ts(j.get("started_at")) for j in jobs) if s is not None]
    ends = [e for e in (parse_ts(j.get("completed_at")) for j in jobs) if e is not None]
    wall = (max(ends) - min(starts)).total_seconds() if starts and ends else 0
    compute = sum(j.get("duration_s") or 0 for j in jobs)

    # Baseline wall/compute
    bl_wall = None
    bl_compute = None
    if baseline:
        bl_starts = [s for s in (parse_ts(j.get("started_at")) for j in bl_jobs) if s is not None]
        bl_ends = [e for e in (parse_ts(j.get("completed_at")) for j in bl_jobs) if e is not None]
        if bl_starts and bl_ends:
            bl_wall = (max(bl_ends) - min(bl_starts)).total_seconds()
        bl_compute = sum(j.get("duration_s") or 0 for j in bl_jobs)

    # Per-job deltas (skipped excluded)
    faster = 0
    slower = 0
    unchanged = 0
    no_baseline = 0
    for j in jobs:
        bl = bl_map.get(j["name"])
        if not bl:
            no_baseline += 1
            continue
        cur_d = j.get("duration_s") or 0
        bl_d = bl.get("duration_s") or 0
        if abs(cur_d - bl_d) < 1.0:
            unchanged += 1
        elif cur_d > bl_d:
            slower += 1
        else:
            faster += 1

    skipped = sum(1 for j in jobs_all if is_skipped(j))
    bl_skipped = sum(1 for j in bl_jobs_all if is_skipped(j))

    return {
        "wall": wall,
        "compute": compute,
        "bl_wall": bl_wall,
        "bl_compute": bl_compute,
        "faster": faster,
        "slower": slower,
        "unchanged": unchanged,
        "no_baseline": no_baseline,
        "skipped": skipped,
        "bl_skipped": bl_skipped,
        "total_jobs": len(jobs_all),
    }


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  background: #0d1117; color: #e6edf3; line-height: 1.5; padding: 24px;
}
h1 { font-size: 24px; border-bottom: 1px solid #30363d; padding-bottom: 12px; margin-bottom: 8px; }
.meta { color: #8b949e; font-size: 13px; margin-bottom: 24px; }
h2 { font-size: 18px; margin: 32px 0 12px; }

/* Stats cards */
.stats { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 24px; }
.stat-card {
  background: #161b22; border: 1px solid #30363d; border-radius: 8px;
  padding: 14px 18px; min-width: 140px;
}
.stat-label { font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }
.stat-value { font-size: 22px; font-weight: 600; margin: 4px 0; }
.stat-delta { font-size: 13px; }
.faster { color: #3fb950; }
.slower { color: #f85149; }
.neutral { color: #8b949e; }

/* Gantt */
.gantt-wrap { overflow-x: auto; }
.gantt { min-width: 700px; }
.gantt-row { display: flex; align-items: center; height: 28px; }
.gantt-label {
  width: 220px; padding-right: 12px; text-align: right;
  font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.gantt-track { flex: 1; position: relative; height: 100%; border-left: 1px solid #21262d; }
.gantt-bar {
  position: absolute; height: 18px; border-radius: 3px;
  display: flex; align-items: center; justify-content: center;
  font-size: 10px; color: transparent; overflow: hidden;
  transition: color 0.15s;
}
.gantt-bar:hover { color: #fff; z-index: 10; }
.gantt-bar.current { background: #1f6feb; top: 5px; z-index: 2; }
.gantt-bar.baseline {
  background: transparent; border: 1px dashed #8b949e; top: 2px; height: 24px; z-index: 1;
}
.gantt-axis { display: flex; height: 20px; position: relative; border-top: 1px solid #30363d; margin-top: 4px; }
.gantt-tick { position: absolute; font-size: 10px; color: #8b949e; transform: translateX(-50%); top: 4px; }
.gantt-tick::before { content: ''; position: absolute; top: -4px; left: 50%; width: 1px; height: 4px; background: #30363d; }
.legend { display: flex; gap: 16px; margin-top: 8px; font-size: 12px; color: #8b949e; }
.legend-swatch { display: inline-block; width: 16px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }

/* Tables */
table { border-collapse: collapse; width: 100%; font-size: 13px; margin-bottom: 16px; }
th, td { border: 1px solid #30363d; padding: 6px 10px; text-align: left; }
th { background: #161b22; font-weight: 600; position: sticky; top: 0; }
tr:hover td { background: #161b22; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.job-name { font-weight: 500; }

/* Step details */
details { margin-bottom: 8px; background: #161b22; border: 1px solid #30363d; border-radius: 6px; }
summary { padding: 8px 12px; cursor: pointer; font-weight: 500; font-size: 14px; user-select: none; }
summary:hover { background: #21262d; }
details[open] summary { border-bottom: 1px solid #30363d; }
details table { border: none; margin: 0; }
details td, details th { font-size: 12px; }

/* Worst regressions */
.regressions { margin-bottom: 24px; }
.regressions table { font-size: 13px; }
.tag {
  display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 11px; font-weight: 500;
}
.tag.slow { background: rgba(248,81,73,0.15); color: #f85149; }
.tag.fast { background: rgba(63,185,80,0.15); color: #3fb950; }
"""


def _gantt_bars(timings: dict, baseline: dict | None) -> str:
    """Render the gantt chart HTML.

    Both current and baseline timelines are normalized to start at t=0
    (relative to each run's earliest job start). The axis scale spans
    0..max_end across both runs so bars are directly comparable.
    """
    jobs = [j for j in timings.get("jobs", [])
            if j.get("started_at") and j.get("completed_at") and not is_skipped(j)]
    bl_map = {j["name"]: j for j in (baseline or {}).get("jobs", [])}

    # Current run: relative offsets from earliest start
    cur_starts = [s for s in (parse_ts(j.get("started_at")) for j in jobs) if s is not None]
    cur_ends = [e for e in (parse_ts(j.get("completed_at")) for j in jobs) if e is not None]
    if not cur_starts or not cur_ends:
        return '<p style="color:#8b949e">No timing data available.</p>'
    cur_t0 = min(cur_starts)
    cur_max = (max(cur_ends) - cur_t0).total_seconds()

    # Baseline run: relative offsets from its earliest start
    bl_t0 = None
    bl_max = 0.0
    bl_jobs_timed = []
    for bl_j in bl_map.values():
        if is_skipped(bl_j):
            continue
        s = parse_ts(bl_j.get("started_at"))
        e = parse_ts(bl_j.get("completed_at"))
        if s is not None and e is not None:
            bl_jobs_timed.append((bl_j, s, e))
            if bl_t0 is None or s < bl_t0:
                bl_t0 = s
            rel_end = (e - s).total_seconds() + (s - (bl_t0 or s)).total_seconds()
    if bl_t0 is not None:
        bl_max = max((e - bl_t0).total_seconds() for _, _, e in bl_jobs_timed) if bl_jobs_timed else 0

    total_s = max(cur_max, bl_max)
    if total_s <= 0:
        total_s = 1

    rows = []
    for j in jobs:
        s = parse_ts(j.get("started_at"))
        e = parse_ts(j.get("completed_at"))
        if s is None or e is None:
            continue
        left = (s - cur_t0).total_seconds() / total_s * 100
        width = max((e - s).total_seconds() / total_s * 100, 0.5)  # min 0.5% for visibility
        dur = j.get("duration_s") or 0

        bl = bl_map.get(j["name"])
        bl_bar = ""
        if bl and not is_skipped(bl) and bl_t0 is not None:
            bl_s = parse_ts(bl.get("started_at"))
            bl_e = parse_ts(bl.get("completed_at"))
            if bl_s is not None and bl_e is not None:
                bl_left = (bl_s - bl_t0).total_seconds() / total_s * 100
                bl_width = max((bl_e - bl_s).total_seconds() / total_s * 100, 0.5)
                bl_dur = bl.get("duration_s") or 0
                bl_bar = (
                    f'<div class="gantt-bar baseline" '
                    f'style="left:{bl_left:.2f}%;width:{bl_width:.2f}%" '
                    f'title="baseline: {fmt_dur(bl_dur)}"></div>'
                )

        name_display = escape(j["name"])
        if j.get("workflow_name"):
            name_display = f'{escape(j["workflow_name"])} / {escape(j["name"])}'

        delta_info = ""
        if bl and not is_skipped(bl) and bl.get("duration_s") is not None:
            d_text, d_cls = fmt_delta(dur, bl.get("duration_s"))
            delta_info = f' — {d_text}'

        rows.append(
            f'<div class="gantt-row">'
            f'<div class="gantt-label" title="{escape(j["name"])}">{name_display}</div>'
            f'<div class="gantt-track">'
            f'{bl_bar}'
            f'<div class="gantt-bar current" '
            f'style="left:{left:.2f}%;width:{width:.2f}%" '
            f'title="{escape(j["name"])}: {fmt_dur(dur)}{delta_info}"></div>'
            f'</div></div>'
        )

    # Axis
    ticks = nice_ticks(total_s)
    tick_html = "".join(
        f'<span class="gantt-tick" style="left:{(t / total_s * 100):.1f}%">{fmt_tick(t)}</span>'
        for t in ticks
    )
    axis = f'<div class="gantt-axis"><div class="gantt-track">{tick_html}</div></div>'

    legend = (
        '<div class="legend">'
        '<span><span class="legend-swatch" style="background:#1f6feb"></span>Current</span>'
    )
    if baseline:
        legend += '<span><span class="legend-swatch" style="border:1px dashed #8b949e"></span>Baseline (main)</span>'
    legend += '</div>'

    return f'<div class="gantt-wrap"><div class="gantt">{"".join(rows)}{axis}</div></div>{legend}'


def _stats_cards(stats: dict) -> str:
    wall_text = fmt_dur(stats["wall"])
    wall_delta = ""
    if stats["bl_wall"] is not None:
        d, cls = fmt_delta(stats["wall"], stats["bl_wall"])
        wall_delta = f'<span class="stat-delta {cls}">{d}</span>'

    compute_text = fmt_dur(stats["compute"])
    compute_delta = ""
    if stats["bl_compute"] is not None:
        d, cls = fmt_delta(stats["compute"], stats["bl_compute"])
        compute_delta = f'<span class="stat-delta {cls}">{d}</span>'

    cards = [
        f'<div class="stat-card"><span class="stat-label">Wall Time</span>'
        f'<div class="stat-value">{wall_text}</div>{wall_delta}</div>',
        f'<div class="stat-card"><span class="stat-label">Total Compute</span>'
        f'<div class="stat-value">{compute_text}</div>{compute_delta}</div>',
        f'<div class="stat-card"><span class="stat-label">Jobs Faster</span>'
        f'<div class="stat-value faster">{stats["faster"]}</div></div>',
        f'<div class="stat-card"><span class="stat-label">Jobs Slower</span>'
        f'<div class="stat-value slower">{stats["slower"]}</div></div>',
        f'<div class="stat-card"><span class="stat-label">Unchanged</span>'
        f'<div class="stat-value neutral">{stats["unchanged"]}</div></div>',
        f'<div class="stat-card"><span class="stat-label">No Baseline</span>'
        f'<div class="stat-value neutral">{stats["no_baseline"]}</div></div>',
    ]
    return f'<div class="stats">{"".join(cards)}</div>'


def _job_table(timings: dict, baseline: dict | None) -> str:
    bl_map = {j["name"]: j for j in (baseline or {}).get("jobs", [])}
    rows = []
    for j in timings.get("jobs", []):
        name = escape(j["name"])
        if j.get("workflow_name"):
            name = f'{escape(j["workflow_name"])} / {escape(j["name"])}'

        concl = j.get("conclusion", "")
        concl_icon = {"success": "✓", "failure": "✗", "skipped": "⊘"}.get(concl, "?")
        concl_cls = {"success": "faster", "failure": "slower", "skipped": "neutral"}.get(concl, "neutral")

        if is_skipped(j):
            rows.append(
                f'<tr>'
                f'<td class="job-name">{name}</td>'
                f'<td class="num neutral">skipped</td>'
                f'<td class="num neutral">—</td>'
                f'<td class="num neutral">—</td>'
                f'<td class="{concl_cls}" style="text-align:center">{concl_icon}</td>'
                f'</tr>'
            )
            continue

        dur = j.get("duration_s")
        bl = bl_map.get(j["name"])
        bl_dur = bl.get("duration_s") if bl and not is_skipped(bl) else None
        delta_text, delta_cls = fmt_delta(dur, bl_dur)

        rows.append(
            f'<tr>'
            f'<td class="job-name">{name}</td>'
            f'<td class="num">{fmt_dur(dur)}</td>'
            f'<td class="num">{fmt_dur(bl_dur)}</td>'
            f'<td class="num {delta_cls}">{delta_text}</td>'
            f'<td class="{concl_cls}" style="text-align:center">{concl_icon}</td>'
            f'</tr>'
        )

    return (
        '<table><thead><tr>'
        '<th>Job</th><th class="num">Current</th><th class="num">Baseline</th>'
        '<th class="num">Delta</th><th>Status</th>'
        '</tr></thead><tbody>' + "".join(rows) + '</tbody></table>'
    )


def _step_details(timings: dict, baseline: dict | None) -> str:
    bl_map = {j["name"]: j for j in (baseline or {}).get("jobs", [])}
    blocks = []
    for j in timings.get("jobs", []):
        if not j.get("steps"):
            continue
        if is_skipped(j):
            continue
        bl = bl_map.get(j["name"], {})
        bl_steps = {s["name"]: s for s in bl.get("steps", [])}

        dur = j.get("duration_s") or 0
        bl_dur = bl.get("duration_s") if bl else None
        delta_text, delta_cls = fmt_delta(dur, bl_dur)

        summary_text = f'{escape(j["name"])} — {fmt_dur(dur)}'
        if bl_dur is not None:
            summary_text += f' <span class="{delta_cls}">({delta_text})</span>'

        step_rows = []
        for s in j["steps"]:
            s_dur = s.get("duration_s")
            bl_s = bl_steps.get(s["name"])
            bl_s_dur = bl_s.get("duration_s") if bl_s else None
            s_delta, s_cls = fmt_delta(s_dur, bl_s_dur)

            step_rows.append(
                f'<tr>'
                f'<td>{escape(s["name"])}</td>'
                f'<td class="num">{fmt_dur(s_dur)}</td>'
                f'<td class="num">{fmt_dur(bl_s_dur)}</td>'
                f'<td class="num {s_cls}">{s_delta}</td>'
                f'</tr>'
            )

        blocks.append(
            f'<details><summary>{summary_text}</summary>'
            f'<table><thead><tr>'
            '<th>Step</th><th class="num">Current</th><th class="num">Baseline</th>'
            '<th class="num">Delta</th>'
            f'</tr></thead><tbody>{"".join(step_rows)}</tbody></table>'
            f'</details>'
        )

    return "".join(blocks) if blocks else '<p style="color:#8b949e">No step data available.</p>'


def _regressions(timings: dict, baseline: dict | None) -> str:
    """Show top 10 biggest absolute regressions/improvements across all steps."""
    if not baseline:
        return ""
    bl_map = {j["name"]: j for j in baseline.get("jobs", [])}

    deltas = []  # (abs_delta, job_name, step_name, current, baseline, is_slower)
    for j in timings.get("jobs", []):
        if is_skipped(j):
            continue
        bl = bl_map.get(j["name"])
        if not bl or is_skipped(bl):
            continue
        bl_steps = {s["name"]: s for s in bl.get("steps", [])}
        for s in j.get("steps", []):
            bl_s = bl_steps.get(s["name"])
            if not bl_s:
                continue
            cur = s.get("duration_s") or 0
            bl_d = bl_s.get("duration_s") or 0
            diff = cur - bl_d
            if abs(diff) < 1.0:
                continue
            deltas.append((abs(diff), diff, j["name"], s["name"], cur, bl_d))

    deltas.sort(key=lambda x: x[0], reverse=True)
    top = deltas[:10]
    if not top:
        return ""

    rows = []
    for _, diff, job, step, cur, bl_d in top:
        cls = "slower" if diff > 0 else "faster"
        tag = f'<span class="tag {"slow" if diff > 0 else "fast"}">{"+" if diff > 0 else ""}{diff:.1f}s</span>'
        rows.append(
            f'<tr>'
            f'<td class="job-name">{escape(job)}</td>'
            f'<td>{escape(step)}</td>'
            f'<td class="num">{fmt_dur(cur)}</td>'
            f'<td class="num">{fmt_dur(bl_d)}</td>'
            f'<td>{tag}</td>'
            f'</tr>'
        )

    return (
        '<div class="regressions">'
        '<table><thead><tr>'
        '<th>Job</th><th>Step</th><th class="num">Current</th><th class="num">Baseline</th>'
        '<th>Delta</th>'
        '</tr></thead><tbody>' + "".join(rows) + '</tbody></table>'
        '</div>'
    )


def generate_html(timings: dict, baseline: dict | None = None) -> str:
    stats = compute_stats(timings, baseline)

    sha_short = (timings.get("head_sha") or "")[:7]
    run_id = timings.get("run_id", "?")
    created = timings.get("created_at", "")

    bl_info = ""
    if baseline:
        bl_sha = (baseline.get("head_sha") or "")[:7]
        bl_info = f' | Baseline: <code>{bl_sha}</code> (main)'

    html = (
        f'<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        f'<meta charset="utf-8">\n'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<title>CI Timing Report — {sha_short}</title>\n'
        f'<style>{CSS}</style>\n'
        f'</head>\n<body>\n'
        f'<h1>CI Timing Report</h1>\n'
        f'<div class="meta">Run <code>{escape(run_id)}</code> | SHA <code>{sha_short}</code>'
        f' | Generated {escape(created)}{bl_info}</div>\n'
    )

    html += '<h2>Global Stats</h2>\n'
    html += _stats_cards(stats)

    if baseline:
        html += '<h2>Top Regressions & Improvements</h2>\n'
        html += _regressions(timings, baseline)

    html += '<h2>Gantt Chart</h2>\n'
    html += _gantt_bars(timings, baseline)

    html += '<h2>Per-Job Comparison</h2>\n'
    html += _job_table(timings, baseline)

    html += '<h2>Step Details</h2>\n'
    html += _step_details(timings, baseline)

    html += '</body>\n</html>\n'
    return html


# ---------------------------------------------------------------------------
# Markdown summary for $GITHUB_STEP_SUMMARY
# ---------------------------------------------------------------------------

def generate_summary(timings: dict, baseline: dict | None = None) -> str:
    stats = compute_stats(timings, baseline)
    bl_map = {j["name"]: j for j in (baseline or {}).get("jobs", [])}

    lines = ["## CI Timing Summary\n"]

    # Global stats table
    lines.append("| Metric | Current | Baseline | Delta |")
    lines.append("|--------|---------|----------|-------|")

    wall_d = ""
    if stats["bl_wall"] is not None:
        d, _ = fmt_delta(stats["wall"], stats["bl_wall"])
        wall_d = d
    lines.append(f"| Wall time | {fmt_dur(stats['wall'])} | {fmt_dur(stats['bl_wall'])} | {wall_d} |")

    compute_d = ""
    if stats["bl_compute"] is not None:
        d, _ = fmt_delta(stats["compute"], stats["bl_compute"])
        compute_d = d
    lines.append(f"| Total compute | {fmt_dur(stats['compute'])} | {fmt_dur(stats['bl_compute'])} | {compute_d} |")

    lines.append(f"| Jobs faster | {stats['faster']} | — | — |")
    lines.append(f"| Jobs slower | {stats['slower']} | — | — |")
    lines.append(f"| Jobs unchanged | {stats['unchanged']} | — | — |")
    lines.append(f"| Jobs without baseline | {stats['no_baseline']} | — | — |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def expect_env(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise ValueError(f"missing environment variable {var}")
    return val

def main():
    parser = argparse.ArgumentParser(description="Collect CI timings and generate HTML report")
    parser.add_argument("--from-json", help="Read timings from JSON instead of API")
    parser.add_argument("--baseline", default="ci-timings-baseline.json",
                        help="Baseline JSON path (default: ci-timings-baseline.json)")
    parser.add_argument("--output", default="ci-timings-report.html",
                        help="HTML output path (default: ci-timings-report.html)")
    parser.add_argument("--json-out", default="ci-timings.json",
                        help="JSON output path (default: ci-timings.json)")
    parser.add_argument("--summary-out", default="ci-timings-summary.md",
                        help="Markdown summary output path (default: ci-timings-summary.md)")
    args = parser.parse_args()

    # Collect or load timings
    if args.from_json:
        with open(args.from_json, encoding="utf-8") as f:
            timings = json.load(f)
    else:
        token = expect_env("GITHUB_TOKEN")
        repo = expect_env("GITHUB_REPOSITORY")
        run_id = expect_env("GITHUB_RUN_ID")
        head_sha = expect_env("GITHUB_SHA")
        try:
            timings = collect_timings(token, repo, run_id, head_sha)
        except TimingsUnavailable as e:
            # Observability job: a missing report must never redden the PR.
            # Emit a degraded summary + placeholder artifact and exit 0.
            msg = f"CI timing data unavailable this run: {e}"
            print(msg, file=sys.stderr)
            with open(args.summary_out, "a", encoding="utf-8") as f:
                f.write(f"\n> ⚠️ {msg}\n")
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(f"<html><body><p>{escape(msg)}</p></body></html>\n")
            # No JSON on purpose: an empty timings file must never be cached
            # as the main baseline.
            sys.exit(0)

    # Save JSON
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(timings, f, indent=2)
    print(f"Saved timings to {args.json_out} ({len(timings.get('jobs', []))} jobs)")

    # Load baseline
    baseline = None
    if os.path.exists(args.baseline):
        with open(args.baseline, encoding="utf-8") as f:
            baseline = json.load(f)
        print(f"Loaded baseline from {args.baseline}")
    else:
        print(f"No baseline file at {args.baseline} — generating current-only report")

    # Generate HTML
    html = generate_html(timings, baseline)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Generated HTML report: {args.output}")

    # Write summary
    summary = generate_summary(timings, baseline)
    with open(args.summary_out, "a", encoding="utf-8") as f:
        f.write(summary)
        print(f"Wrote summary to {args.summary_out}")


if __name__ == "__main__":
    main()
