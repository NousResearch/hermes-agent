#!/usr/bin/env python3
"""Assemble the unified CI review comment for a pull request.

Every CI job that wants to appear in the review comment emits a
``review_status`` output: a JSON array of objects, each with a ``source``
(the workflow name, used for dedup) and a ``results`` array of typed
result objects::

    [
      {
        "source": "review-label-gate",
        "results": [
          {"kind": "action_required", "title": "...", "summary": "...",
           "how_to_fix": "..."},
          {"kind": "info", "title": "...", "summary": "..."}
        ]
      },
      {
        "source": "ci-timings",
        "results": [
          {"kind": "warning", "title": "CI timings", "summary": "...",
           "detail": "...", "link": "..."}
        ]
      }
    ]

Each result object has:

    kind:        "error" | "action_required" | "warning" | "info" | "debug"
    title:       section heading
    summary:     one-line description
    detail:      markdown detail (optional)
    how_to_fix:  markdown checklist (optional)
    link:        URL (optional)
    link_label:  label for the link (optional, default "View logs")

The assembler flattens all results into a flat list of ReviewItems,
grouped by severity in the comment. Jobs that failed (from the
``needs`` context) but didn't emit any status get synthesized ❌ Error
items. Jobs that DID emit a status are excluded from the synthesized
error list — their own output is the authority for their classification.

Exits 0 always — comment posting is best-effort (fork PRs are read-only).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Hidden marker the comment system uses to find-and-edit its
# previous comment instead of stacking new ones on each run.
MARKER = "<!-- hermes-ci-review-bot -->"

# Severity ordering for display.
_SEVERITY_ORDER = ["error", "action_required", "warning", "info", "debug"]

# Severities that trigger the "blocking issues" layout (vs. the
# "looks good!" banner).
_BLOCKING_SEVERITIES = ("error", "action_required", "warning")

_SEVERITY_GROUP_HEADER = {
    "error": "## ❌ Job failures",
    "action_required": "## ⚠️ Action required",
    "warning": "## ⚠️ Warnings",
    "info": "## ℹ️ Details",
}


@dataclass
class ReviewItem:
    """A single piece of review information with a severity tag."""

    severity: str  # "error" | "action_required" | "warning" | "info" | "debug"
    title: str  # short section title, e.g. "package-lock.json"
    summary: str  # one-line summary
    detail: str = ""  # optional markdown detail (tables, bullet lists, etc.)
    link: str = ""  # optional URL emitted by the job (e.g. report URL)
    link_label: str = "View report"  # label for the emitted link
    how_to_fix: str = ""  # optional markdown checklist for action_required items
    source: str = ""  # workflow that declared this status (for dedup)
    job_url: str = ""  # auto-attached per-job log link (from the live poller)


# ---------------------------------------------------------------------------
# Collectors — each returns a list of ReviewItems (possibly empty)
# ---------------------------------------------------------------------------


def collect_from_statuses(review_statuses_json: str) -> tuple[list[ReviewItem], set[str]]:
    """Parse the nested review_status JSON into flat ReviewItems.

    The input is a JSON array of ``{source, results: [...]}`` objects.
    Each entry in ``results`` becomes one ReviewItem, tagged with the
    parent's ``source``.

    Returns ``(items, sources)`` where ``sources`` is the set of source
    values — used by :func:`collect_failed_jobs` to exclude jobs that
    already declared their own status (so a failing job that emitted an
    ``action_required`` status doesn't also show as a synthesized ❌ Error).
    """
    if not review_statuses_json:
        return [], set()
    try:
        data = json.loads(review_statuses_json)
    except (json.JSONDecodeError, TypeError):
        return [], set()
    if not isinstance(data, list):
        return [], set()

    items: list[ReviewItem] = []
    sources: set[str] = set()

    for entry in data:
        if not isinstance(entry, dict):
            continue
        source = entry.get("source", "")
        if source:
            sources.add(source)
        for r in entry.get("results", []):
            if not isinstance(r, dict):
                continue
            kind = r.get("kind", "info")
            if kind not in _SEVERITY_ORDER:
                kind = "info"
            items.append(ReviewItem(
                severity=kind,
                title=r.get("title", "Unknown"),
                summary=r.get("summary", ""),
                detail=r.get("detail", ""),
                link=r.get("link", ""),
                link_label=r.get("link_label", "View logs"),
                how_to_fix=r.get("how_to_fix", ""),
                source=source,
            ))

    return items, sources


def collect_failed_jobs(
    needs_json: str,
    run_url: str,
    exclude_sources: set[str] | None = None,
    job_urls: dict[str, str] | None = None,
) -> list[ReviewItem]:
    """Build error items for failed CI jobs from the ``needs`` context.

    ``needs_json`` is the JSON string emitted by ``all-checks-pass`` — a
    ``{job_name: result}`` dict where result is ``success`` / ``failure``
    / ``skipped``. Only ``failure`` entries become error items.

    ``exclude_sources`` is a set of ``source`` values from status objects
    declared by workflow_call jobs. Job names containing any of these
    source strings are excluded — their failure is already covered by their
    own status output.

    ``job_urls`` is an optional ``{job_name: html_url}`` dict from the
    live poller. When a job's name is in this dict, the ❌ Error link
    points directly to that job's logs page instead of the whole run.
    Falls back to ``run_url`` when no per-job URL is available.
    """
    if not needs_json:
        return []
    try:
        needs = json.loads(needs_json)
    except (json.JSONDecodeError, TypeError):
        return []

    # Pre-normalize exclude sources once: lowercase + hyphens→spaces, so
    # "review-label-gate" matches "Review label gate / Review label gate".
    norm_sources = {
        src.lower().replace("-", " ") for src in (exclude_sources or set())
    }

    items: list[ReviewItem] = []
    for name, result in sorted(needs.items()):
        if result != "failure":
            continue
        if norm_sources:
            norm = name.lower().replace("-", " ")
            if any(src in norm for src in norm_sources):
                continue
        job_url = (job_urls or {}).get(name, run_url)
        items.append(ReviewItem(
            severity="error",
            title=name,
            summary=f"Job **{name}** failed.",
            job_url=job_url,
        ))
    return items


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_item(item: ReviewItem) -> str:
    """Render a single ReviewItem as a markdown block.

    The group header (``## ❌ Job failures`` etc.) carries the severity
    emoji, so items don't repeat it. Links are shown inline next to the
    title. Layout per item::

        ### {title} · [View report](url) · [View job](url)

        {summary}

        {detail}

        **How to fix:**

        {how_to_fix}
    """
    title = f"### {item.title}"
    # Build inline links next to the title.
    links: list[str] = []
    if item.link:
        links.append(f"[{item.link_label}]({item.link})")
    if item.job_url:
        links.append(f"[View job]({item.job_url})")
    if links:
        title += " · " + " · ".join(links)

    parts = [title, "", item.summary]

    if item.detail:
        parts += ["", item.detail]
    if item.how_to_fix:
        parts += ["", "**How to fix:**", "", item.how_to_fix]

    return "\n".join(parts)


def _render_group(header: str, items: list[ReviewItem]) -> str:
    """Render a severity group: ``##`` header + items separated by ``---``."""
    blocks = [_render_item(i) for i in items]
    return f"{header}\n\n" + "\n\n---\n\n".join(blocks)


def _render_debug_details(items: list[ReviewItem]) -> str:
    """Render each debug item as its own collapsible ``<details>`` block."""
    blocks = []
    for item in items:
        inner = _render_item(item)
        blocks.append(
            f"<details>\n<summary>{item.title}</summary>\n\n{inner}\n\n</details>"
        )
    return "### debug info\n\n" + "\n\n".join(blocks)


def _render_pending_items(pending_jobs: list[str]) -> str:
    """Render the dimmed ``<sub>`` items for jobs still running."""
    job_list = ", ".join(f"`{j}`" for j in sorted(pending_jobs))
    return f"\n\n---\n\n<sub>Still running {len(pending_jobs)} job{'s' if len(pending_jobs) != 1 else ''}: {job_list}</sub>\n"


def render_comment(items: list[ReviewItem], pending_jobs: list[str] | None = None, commit_info: str = "") -> str:
    """Render the full comment body from a list of review items.

    Items are grouped by severity under ``##`` group headers, separated
    by ``---``. Errors and action_required items are always visible.
    Warnings are shown only when present. Info items are visible; debug items
    are in a collapsible ``<details>`` block. If ``pending_jobs`` is non-empty, a dimmed
    ``<sub>`` footer is appended listing jobs still running.

    When there are no errors, action_required, or warnings, an "all good!"
    banner is shown at the top. Info items remain visible and debug items
    follow in collapsible ``<details>`` blocks.
    """
    pending = pending_jobs or []

    # Group by severity
    by_severity: dict[str, list[ReviewItem]] = {s: [] for s in _SEVERITY_ORDER}
    for item in items:
        by_severity.setdefault(item.severity, []).append(item)

    info = by_severity.get("info", [])
    debug = by_severity.get("debug", [])
    has_blocking = any(by_severity.get(s) for s in _BLOCKING_SEVERITIES)

    body = f"{MARKER}\n# ૮ >ﻌ< ა ci review\n\n"

    if commit_info:
        body += f"{commit_info}\n\n"

    if not items and not pending:
        return f"{body}all good!"

    sections: list[str] = []

    for sev in _BLOCKING_SEVERITIES:
        group = by_severity.get(sev, [])
        if group:
            sections.append(_render_group(_SEVERITY_GROUP_HEADER[sev], group))

    if info:
        sections.append(_render_group("## ℹ️ Info", info))

    # Debug: collapsible <details>
    if debug:
        sections.append(_render_debug_details(debug))

    if pending:
        body += _render_pending_items(pending)

    if sections:
        body += "\n\n---\n\n".join(sections)

    return body


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _attach_job_urls(items: list[ReviewItem], job_urls: dict[str, str], run_url: str) -> None:
    """Fill in per-job log links for all items.

    Uses the same case-insensitive, hyphen-normalized matching as
    :func:`collect_failed_jobs`: the item's ``source`` is matched against
    job names in ``job_urls``. Sets ``job_url`` on the item — this is
    separate from ``link`` (the job-emitted URL, e.g. a report artifact),
    so both can appear in the rendered comment.
    """
    if not job_urls and not run_url:
        return
    # Pre-normalize job_url keys once.
    norm_urls: dict[str, str] = {}
    for name, url in job_urls.items():
        norm_urls[name.lower().replace("-", " ")] = url

    for item in items:
        if item.job_url:
            continue
        src = item.source.lower().replace("-", " ")
        # Try exact match first, then substring match.
        if src in norm_urls:
            item.job_url = norm_urls[src]
            continue
        for norm_name, url in norm_urls.items():
            if src and src in norm_name:
                item.job_url = url
                break
        # If no per-job URL found, fall back to run_url for items with a source.
        if not item.job_url and item.source and run_url:
            item.job_url = run_url


def assemble(
    needs_json: str = "",
    run_url: str = "",
    job_urls: dict[str, str] | None = None,
    review_statuses_json: str = "",
    pending_jobs: list[str] | None = None,
    commit_info: str = "",
) -> str:
    """Assemble the full comment body from all available inputs."""
    items: list[ReviewItem] = []

    # 1. Structured statuses from workflow_call jobs (review-labels, etc.)
    status_items, sources = collect_from_statuses(review_statuses_json)
    items.extend(status_items)

    # 2. Synthesized error items for failed jobs not covered by statuses
    items.extend(collect_failed_jobs(needs_json, run_url, exclude_sources=sources, job_urls=job_urls))

    # 3. Attach per-job log links to all items (not just synthesized errors)
    _attach_job_urls(items, job_urls or {}, run_url)

    return render_comment(items, pending_jobs, commit_info)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--needs-json",
        default="",
        help="JSON string of {job_name: result} from the all-checks-pass job.",
    )
    parser.add_argument(
        "--run-url",
        default="",
        help="URL to the CI run summary page (for failed job links).",
    )
    parser.add_argument(
        "--review-statuses-json",
        default="",
        help="JSON array of {source, results: [...]} objects from workflow_call jobs.",
    )
    parser.add_argument(
        "--pending-jobs",
        default="",
        help="Comma-separated list of job names still running (shown in a dimmed footer).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file for the assembled comment body.",
    )
    args = parser.parse_args()

    pending = [j.strip() for j in args.pending_jobs.split(",") if j.strip()] if args.pending_jobs else None

    body = assemble(
        needs_json=args.needs_json,
        run_url=args.run_url,
        review_statuses_json=args.review_statuses_json,
        pending_jobs=pending,
    )

    args.output.write_text(body)
    print(f"Wrote {len(body)} chars to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
