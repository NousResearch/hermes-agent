"""Tests for scripts/ci/assemble_review_comment.py.

The assembler collects status from every CI sub-workflow into ReviewItems
classified by severity (error / action_required / warning / info / debug), then
renders them into a single PR comment body.

Status data comes from two sources:
  1. --review-statuses-json: JSON array of {source, results: [...]} objects
     from workflow_call jobs. Each result has kind/title/summary/detail/
     how_to_fix/link. The assembler flattens all results into ReviewItems.
  2. --needs-json: {job_name: result} from all-checks-pass. Failed jobs not
     claimed by any status become synthesized ❌ Error items.

Layout rules tested here:
  - group headers: ## ❌ Job failures, ## ⚠️ Action required, ## ⚠️ Warnings
  - each item is a ### section under its group header
  - errors + action_required always visible
  - warnings shown only when present
  - info above the fold; debug in a collapsible <details> block
  - sections separated by ---
  - how_to_fix rendered at bottom of action_required items
  - empty → clean banner
  - jobs with declared statuses excluded from failed-jobs list
  - per-job URLs used for failed job links when available
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "assemble_review_comment.py"
_spec = importlib.util.spec_from_file_location("assemble_review_comment", _PATH)
if _spec is None or _spec.loader is None:
    raise ImportError("Failed to load assemble_review_comment.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["assemble_review_comment"] = _mod
_spec.loader.exec_module(_mod)

MARKER = _mod.MARKER
ReviewItem = _mod.ReviewItem


def _status(source: str, results: list[dict]) -> str:
    """Helper: build a review_statuses JSON string with one source entry."""
    return json.dumps([{"source": source, "results": results}])


# ─── collect_from_statuses ──────────────────────────────────────────


def test_statuses_empty_json():
    items, sources = _mod.collect_from_statuses("")
    assert items == []
    assert sources == set()


def test_statuses_bad_json():
    items, sources = _mod.collect_from_statuses("not json")
    assert items == []
    assert sources == set()


def test_statuses_info():
    statuses = _status("review-label-gate", [{
        "kind": "info",
        "title": "CI-sensitive file review",
        "summary": "Label present.",
    }])
    items, sources = _mod.collect_from_statuses(statuses)
    assert len(items) == 1
    assert items[0].severity == "info"
    assert sources == {"review-label-gate"}


# ─── collect_failed_jobs ─────────────────────────────────────────────


def test_failed_jobs_empty_needs():
    assert _mod.collect_failed_jobs("", "https://run") == []


def test_failed_jobs_excluded_by_source():
    """Jobs whose name contains a declared source are excluded."""
    needs = json.dumps({
        "Review label gate / Review label gate": "failure",
        "tests": "failure",
    })
    items = _mod.collect_failed_jobs(needs, "https://run", exclude_sources={"review-label-gate"})
    assert len(items) == 1
    assert items[0].title == "tests"


# ─── render_comment ───────────────────────────────────────────────────


# ─── render_comment (pending jobs) ────────────────────────────────────


# ─── render_comment (waiting for jobs to start) ───────────────────────


def test_waiting_with_no_items_shows_waiting_not_all_good():
    """A run with no jobs yet must not render the final 'all good!' banner."""
    body = _mod.render_comment([], waiting=True)
    assert "all good" not in body
    assert "waiting for jobs to start" in body


def test_waiting_with_items_but_no_pending_keeps_a_live_footer():
    """Between job waves: results exist, nothing pending, run not done."""
    items = [ReviewItem(severity="info", title="lockfile", summary="No changes.")]
    body = _mod.render_comment(items, waiting=True)
    assert "waiting for more jobs to start" in body
    assert "### lockfile" in body


def test_not_waiting_and_no_items_still_renders_all_good():
    body = _mod.render_comment([])
    assert "all good!" in body


# ─── assemble (integration) ──────────────────────────────────────────


def test_assemble_with_timings_status():
    """Timings status from the nested format renders as debug or warning."""
    statuses = _status("ci-timings", [{
        "kind": "debug",
        "title": "CI timings",
        "summary": "Wall time 3m (no baseline yet).",
        "detail": "",
        "link": "https://report",
    }])
    body = _mod.assemble(review_statuses_json=statuses)
    assert "<details>" in body
    assert "### CI timings" in body
    assert "Wall time 3m" in body
    assert "## ❌" not in body
    assert "## ⚠️" not in body


# ─── _attach_job_urls ────────────────────────────────────────────────


def test_attach_job_urls_fills_missing_links():
    """Items without a link get one from job_urls via source matching."""
    items = [
        ReviewItem(severity="info", title="Supply chain scan",
                   summary="No risks.", source="supply chain"),
        ReviewItem(severity="warning", title="CI timings",
                   summary="Slower.", source="ci timings",
                   link="https://report"),  # already has a link
    ]
    job_urls = {
        "Supply Chain Audit / Scan PR for critical supply chain risks": "https://run/1/job/2",
    }
    _mod._attach_job_urls(items, job_urls, "https://fallback")
    # First item gets the per-job URL as job_url (link untouched)
    assert items[0].job_url == "https://run/1/job/2"
    assert items[0].link == ""  # no emitted link
    # Second item keeps its existing link, job_url is set separately
    assert items[1].link == "https://report"
    assert items[1].job_url == "https://fallback"  # fell back to run_url
