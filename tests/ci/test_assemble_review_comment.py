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

import pytest

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


def test_failed_jobs_include_cancelled_not_just_failure():
    """A cancelled lane blocks the merge, so the comment must name it.

    ``all-checks-pass`` fails on any result outside success/skipped. When
    this list only matched ``"failure"``, the gate went red while the
    review comment showed nothing to explain it.
    """
    needs = json.dumps({
        "tests": "cancelled",
        "lint": "success",
        "docker-lint": "skipped",
    })
    items = _mod.collect_failed_jobs(needs, "https://run")
    assert [item.title for item in items] == ["tests"]
    assert items[0].severity == "error"
    assert "cancelled" in items[0].summary


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








def test_render_group_header_for_errors():
    """Errors appear under a '## ❌ Job failures' group header."""
    items = [
        ReviewItem(severity="error", title="tests", summary="Job **tests** failed.", link="https://run"),
        ReviewItem(severity="error", title="lint", summary="Job **lint** failed.", link="https://run"),
    ]
    body = _mod.render_comment(items)
    assert "## ❌ Job failures" in body
    assert "### tests" in body
    assert "### lint" in body
    assert body.index("## ❌ Job failures") < body.index("### tests")


















# ─── render_comment (pending jobs) ────────────────────────────────────


def test_render_pending_only_shows_header_with_clock():
    """Pending jobs only — header has 'still waiting', footer lists jobs, no sections."""
    body = _mod.render_comment([], pending_jobs=["ci-timings"])
    assert body.startswith(MARKER)
    assert "૮ >ﻌ< ა" in body
    assert "Still running" in body
    assert "`ci-timings`" in body
    assert "##" not in body


def test_render_pending_notif():
    items = [ReviewItem(severity="info", title="lockfile", summary="No changes.")]
    body = _mod.render_comment(items, pending_jobs=["ci-timings"])
    assert "૮ >ﻌ< ა" in body
    assert "<sub>Still running 1 job: `ci-timings`</sub>" in body


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


def test_assemble_passes_waiting_through():
    body = _mod.assemble(waiting=True)
    assert "waiting for jobs to start" in body
    assert "all good" not in body






# ─── assemble (integration) ──────────────────────────────────────────








def test_assemble_review_status_detail_renders_sensitive_file_links():
    statuses = _status("review-label-gate", [{
        "kind": "action_required",
        "title": "CI-sensitive file review",
        "summary": "Changes detected.",
        "detail": "**Sensitive files:**\n- [`ci.yml`](https://example.test/ci.yml)",
    }])
    body = _mod.assemble(review_statuses_json=statuses)
    assert "**Sensitive files:**" in body
    assert "[`ci.yml`](https://example.test/ci.yml)" in body


def test_assemble_info_keeps_screenshot_details_visible_below_its_summary():
    statuses = _status("playwright e2e", [{
        "kind": "info",
        "title": "Desktop E2E screenshots",
        "summary": "1 screenshot captured; 0 visual diffs.",
        "detail": "<details>\n<summary>1 captured screenshot</summary>\n\n- [`proof.png`](https://example.test/artifact)\n\n</details>",
    }])
    body = _mod.assemble(review_statuses_json=statuses)
    assert "## ℹ️ Info" in body
    assert "1 screenshot captured; 0 visual diffs." in body
    assert "<summary>1 captured screenshot</summary>" in body
    assert "[`proof.png`](https://example.test/artifact)" in body






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


def test_assemble_with_lockfile_status():
    """Lockfile no-changes status renders as visible info."""
    statuses = _status("lockfile-diff", [{
        "kind": "info",
        "title": "package-lock.json",
        "summary": "No lockfile changes — locked versions match the target branch.",
    }])
    body = _mod.assemble(review_statuses_json=statuses)
    assert "## ℹ️ Info" in body
    assert "### package-lock.json" in body
    assert "No lockfile changes" in body




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








def test_render_commit_info_below_header():
    """Commit info is rendered below the header, above the content."""
    body = _mod.render_comment(
        [ReviewItem(severity="error", title="tests", summary="failed.")],
        commit_info="<sub>running on [abc1234](https://commit-url) — fix: thing</sub>",
    )
    assert "# ૮ >ﻌ< ა ci review" in body
    assert "running on [abc1234](https://commit-url)" in body
    assert "fix: thing" in body
    # Commit info appears before the content
    assert body.index("abc1234") < body.index("## ❌")




def test_assemble_passes_commit_info():
    """assemble() passes commit_info through to render_comment."""
    body = _mod.assemble(commit_info="<sub>running on abc1234</sub>")
    assert "running on abc1234" in body
    assert "all good!" in body


def test_render_both_emitted_link_and_job_url():
    """An item with both an emitted link and a job_url shows both."""
    item = ReviewItem(
        severity="warning",
        title="CI timings",
        summary="Slower.",
        link="https://artifact/report.html",
        link_label="View report",
        source="ci timings",
        job_url="https://github.com/run/1/job/5",
    )
    body = _mod.render_comment([item])
    assert "[View report](https://artifact/report.html)" in body
    assert "[View job](https://github.com/run/1/job/5)" in body
    # Both links on the same line, separated by ·
    assert " · " in body




# ─── workflow-run conclusions ─────────────────────────────────────────


def test_a_run_that_stopped_without_passing_becomes_a_blocking_item():
    """A run with zero jobs leaves ``needs_json`` empty; the run itself blocks."""
    body = _mod.assemble(
        needs_json="",
        runs_json=json.dumps({"CI": "action_required"}),
        run_urls={"CI": "https://example/runs/1"},
    )
    assert "all good!" not in body
    assert "## ❌ Job failures" in body
    assert "CI (workflow run)" in body
    assert "action_required" in body
    assert "[View run](https://example/runs/1)" in body


def test_no_runs_json_leaves_the_banner_alone():
    assert "all good!" in _mod.assemble(needs_json="", runs_json="")


def test_malformed_runs_json_is_ignored_rather_than_crashing():
    assert "all good!" in _mod.assemble(runs_json="{not json")
    assert "all good!" in _mod.assemble(runs_json='["CI"]')
