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


def test_statuses_action_required():
    statuses = _status("review-label-gate", [{
        "kind": "action_required",
        "title": "CI-sensitive file review",
        "summary": "Changes detected.",
        "how_to_fix": "Add the label.",
    }])
    items, sources = _mod.collect_from_statuses(statuses)
    assert len(items) == 1
    assert items[0].severity == "action_required"
    assert items[0].title == "CI-sensitive file review"
    assert items[0].how_to_fix == "Add the label."
    assert items[0].source == "review-label-gate"
    assert sources == {"review-label-gate"}


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


def test_statuses_debug():
    statuses = _status("ci-timings", [{
        "kind": "debug",
        "title": "CI timings",
        "summary": "No regression.",
    }])
    items, _ = _mod.collect_from_statuses(statuses)
    assert items[0].severity == "debug"


def test_statuses_multiple_results_same_source():
    """One source can emit multiple results of different kinds."""
    statuses = _status("review-label-gate", [
        {"kind": "action_required", "title": "CI review", "summary": "Missing label."},
        {"kind": "action_required", "title": "MCP review", "summary": "Missing label."},
    ])
    items, sources = _mod.collect_from_statuses(statuses)
    assert len(items) == 2
    assert sources == {"review-label-gate"}


def test_statuses_mixed_kinds_same_source():
    """One source can emit both a warning and an info."""
    statuses = _status("ci-timings", [
        {"kind": "warning", "title": "CI timings", "summary": "Slower."},
        {"kind": "info", "title": "Baseline", "summary": "OK."},
    ])
    items, sources = _mod.collect_from_statuses(statuses)
    assert len(items) == 2
    assert items[0].severity == "warning"
    assert items[1].severity == "info"
    assert sources == {"ci-timings"}


def test_statuses_multiple_sources():
    statuses = json.dumps([
        {"source": "review-label-gate", "results": [
            {"kind": "action_required", "title": "CI review", "summary": "Missing."},
        ]},
        {"source": "lockfile-diff", "results": [
            {"kind": "info", "title": "package-lock.json", "summary": "No changes."},
        ]},
    ])
    items, sources = _mod.collect_from_statuses(statuses)
    assert len(items) == 2
    assert sources == {"review-label-gate", "lockfile-diff"}


def test_statuses_unknown_kind_becomes_info():
    statuses = _status("some-job", [{
        "kind": "bogus",
        "title": "X",
        "summary": "Y",
    }])
    items, _ = _mod.collect_from_statuses(statuses)
    assert items[0].severity == "info"


def test_statuses_no_source():
    """Status without a source — still rendered, just not excluded from errors."""
    statuses = json.dumps([{
        "results": [{"kind": "info", "title": "X", "summary": "Y"}],
    }])
    items, sources = _mod.collect_from_statuses(statuses)
    assert len(items) == 1
    assert sources == set()


def test_statuses_passes_through_optional_fields():
    statuses = _status("ci-timings", [{
        "kind": "warning",
        "title": "CI timings",
        "summary": "Slower.",
        "detail": "- job: +5s",
        "link": "https://report",
        "link_label": "View report",
        "how_to_fix": "Optimize.",
    }])
    items, _ = _mod.collect_from_statuses(statuses)
    assert items[0].detail == "- job: +5s"
    assert items[0].link == "https://report"
    assert items[0].link_label == "View report"
    assert items[0].how_to_fix == "Optimize."


# ─── collect_failed_jobs ─────────────────────────────────────────────


def test_failed_jobs_empty_needs():
    assert _mod.collect_failed_jobs("", "https://run") == []


def test_failed_jobs_no_failures():
    needs = json.dumps({"tests": "success", "lint": "skipped"})
    assert _mod.collect_failed_jobs(needs, "https://run") == []


def test_failed_jobs_collects_only_failures():
    needs = json.dumps({"tests": "success", "lint": "failure", "js-tests": "failure"})
    items = _mod.collect_failed_jobs(needs, "https://run/123")
    assert len(items) == 2
    assert all(i.severity == "error" for i in items)
    # sorted by name
    names = [i.title for i in items]
    assert names == ["js-tests", "lint"]
    assert all(i.job_url == "https://run/123" for i in items)


def test_failed_jobs_bad_json():
    assert _mod.collect_failed_jobs("not json", "https://run") == []


def test_failed_jobs_excluded_by_source():
    """Jobs whose name contains a declared source are excluded."""
    needs = json.dumps({
        "Review label gate / Review label gate": "failure",
        "tests": "failure",
    })
    items = _mod.collect_failed_jobs(needs, "https://run", exclude_sources={"review-label-gate"})
    assert len(items) == 1
    assert items[0].title == "tests"


def test_failed_jobs_no_exclusion_without_sources():
    """Without exclude_sources, all failures are shown."""
    needs = json.dumps({"review-label-gate": "failure", "tests": "failure"})
    items = _mod.collect_failed_jobs(needs, "https://run")
    assert len(items) == 2


def test_failed_jobs_per_job_url():
    """When job_urls is provided, the link points to the specific job."""
    needs = json.dumps({"tests": "failure", "lint": "failure"})
    job_urls = {"tests": "https://run/1/job/2", "lint": "https://run/1/job/3"}
    items = _mod.collect_failed_jobs(needs, "https://fallback", job_urls=job_urls)
    assert len(items) == 2
    urls = {i.title: i.job_url for i in items}
    assert urls["tests"] == "https://run/1/job/2"
    assert urls["lint"] == "https://run/1/job/3"


def test_failed_jobs_fallback_to_run_url():
    """Jobs not in job_urls fall back to run_url."""
    needs = json.dumps({"tests": "failure", "lint": "failure"})
    job_urls = {"tests": "https://run/1/job/2"}
    items = _mod.collect_failed_jobs(needs, "https://fallback", job_urls=job_urls)
    urls = {i.title: i.job_url for i in items}
    assert urls["tests"] == "https://run/1/job/2"
    assert urls["lint"] == "https://fallback"


# ─── render_comment ───────────────────────────────────────────────────


def test_render_empty_shows_clean_banner():
    """Completely clean — dog kaomoji + 'all good' banner, no sections."""
    body = _mod.render_comment([])
    assert body.startswith(MARKER)
    assert "૮ >ﻌ< ა" in body
    assert "all good!" in body
    assert "##" not in body  # no section headers


def test_render_info_only_is_visible_above_the_fold():
    """Info items are visible rather than hidden in debug details."""
    items = [
        ReviewItem(severity="info", title="lockfile", summary="No changes."),
        ReviewItem(severity="info", title="timings", summary="OK."),
    ]
    body = _mod.render_comment(items)
    assert "૮ >ﻌ< ა" in body
    assert "## ℹ️ Info" in body
    assert "<details>" not in body
    assert "No changes." in body
    assert "OK." in body
    # No blocking sections
    assert "## ❌" not in body
    assert "## ⚠️" not in body


def test_render_info_only_with_pending_shows_info_plus_footer():
    items = [ReviewItem(severity="info", title="lockfile", summary="No changes.")]
    body = _mod.render_comment(items, pending_jobs=["ci-timings"])
    assert "૮ >ﻌ< ა" in body
    assert "Still running" in body
    assert "## ℹ️ Info" in body
    assert "Still running" in body
    assert "`ci-timings`" in body


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


def test_render_group_header_for_action_required():
    items = [
        ReviewItem(severity="action_required", title="CI review", summary="Need label."),
    ]
    body = _mod.render_comment(items)
    assert "## ⚠️ Action required" in body
    assert "### CI review" in body


def test_render_group_header_for_warnings():
    items = [
        ReviewItem(severity="warning", title="CI timings", summary="Slower."),
    ]
    body = _mod.render_comment(items)
    assert "## ⚠️ Warnings" in body
    assert "### CI timings" in body
    assert "<details>" not in body

    items2 = [ReviewItem(severity="info", title="x", summary="y")]
    body2 = _mod.render_comment(items2)
    assert "## ⚠️ Warnings" not in body2


def test_render_no_duplicated_severity_in_item_body():
    """Items don't repeat the severity label — the group header carries it."""
    items = [ReviewItem(severity="error", title="tests", summary="Job failed.", link="https://run")]
    body = _mod.render_comment(items)
    assert "### tests" in body
    assert "Job failed." in body
    assert "**❌ Error**" not in body


def test_render_how_to_fix_at_bottom():
    items = [
        ReviewItem(severity="action_required", title="CI review", summary="Need label.",
                   how_to_fix="Add the `ci-reviewed` label."),
    ]
    body = _mod.render_comment(items)
    assert "**How to fix:**" in body
    assert "Add the `ci-reviewed` label." in body
    assert body.index("Need label.") < body.index("How to fix")


def test_render_sections_separated_by_hr():
    items = [
        ReviewItem(severity="error", title="tests", summary="failed."),
        ReviewItem(severity="action_required", title="CI review", summary="need label."),
    ]
    body = _mod.render_comment(items)
    assert "\n\n---\n\n" in body


def test_render_errors_always_visible():
    items = [
        ReviewItem(severity="error", title="tests", summary="Job **tests** failed.", job_url="https://run"),
        ReviewItem(severity="info", title="lockfile", summary="No changes."),
    ]
    body = _mod.render_comment(items)
    assert "## ❌ Job failures" in body
    assert "### tests" in body
    assert "Job **tests** failed." in body
    assert "[View job](https://run)" in body
    assert "## ℹ️ Info" in body
    assert "No changes." in body


def test_render_debug_in_collapsible_details():
    """Debug items have a small label and each has its own <details> block."""
    items = [
        ReviewItem(severity="debug", title="lockfile", summary="No changes."),
        ReviewItem(severity="debug", title="timings", summary="OK."),
    ]
    body = _mod.render_comment(items)
    assert "### debug info" in body
    assert body.index("### debug info") < body.index("<details>")
    assert body.count("<details>") == 2
    assert body.count("</details>") == 2
    assert "<summary>lockfile</summary>" in body
    assert "<summary>timings</summary>" in body
    assert "No changes." in body
    assert "OK." in body


def test_render_order_errors_then_action_then_warn_then_info_then_debug():
    items = [
        ReviewItem(severity="info", title="i", summary="info"),
        ReviewItem(severity="debug", title="d", summary="debug"),
        ReviewItem(severity="warning", title="w", summary="warn"),
        ReviewItem(severity="action_required", title="a", summary="action"),
        ReviewItem(severity="error", title="e", summary="error"),
    ]
    body = _mod.render_comment(items)
    error_pos = body.index("## ❌ Job failures")
    action_pos = body.index("## ⚠️ Action required")
    warn_pos = body.index("## ⚠️ Warnings")
    info_pos = body.index("## ℹ️ Info")
    debug_pos = body.index("<details>")
    assert error_pos < action_pos < warn_pos < info_pos < debug_pos


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


def test_render_pending_multiple_jobs_sorted():
    body = _mod.render_comment([], pending_jobs=["docker", "ci-timings"])
    assert "`ci-timings`" in body
    assert "`docker`" in body
    assert body.index("`ci-timings`") < body.index("`docker`")


def test_render_no_pending_no_footer():
    items = [ReviewItem(severity="info", title="x", summary="y")]
    body = _mod.render_comment(items)
    assert "Still running" not in body


# ─── assemble (integration) ──────────────────────────────────────────


def test_assemble_all_skipped_clean_banner():
    body = _mod.assemble()
    assert body.startswith(MARKER)
    assert "૮ >ﻌ< ა" in body
    assert "all good!" in body
    assert "##" not in body


def test_assemble_failed_job_shown():
    needs = json.dumps({"tests": "failure", "lint": "success"})
    body = _mod.assemble(needs_json=needs, run_url="https://run/1")
    assert "## ❌ Job failures" in body
    assert "### tests" in body
    assert "[View job](https://run/1)" in body


def test_assemble_with_review_statuses():
    """Statuses from review-labels render directly + exclude gate from errors."""
    statuses = _status("review-label-gate", [{
        "kind": "action_required",
        "title": "CI-sensitive file review",
        "summary": "Changes detected.",
        "how_to_fix": "Add the label.",
    }])
    needs = json.dumps({
        "Review label gate / Review label gate": "failure",
        "tests": "success",
    })
    body = _mod.assemble(
        needs_json=needs,
        run_url="https://run",
        review_statuses_json=statuses,
    )
    assert "## ⚠️ Action required" in body
    assert "### CI-sensitive file review" in body
    assert "Add the label." in body
    assert "## ❌ Job failures" not in body


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


def test_assemble_pending_jobs():
    body = _mod.assemble(pending_jobs=["ci-timings"])
    assert "Still running" in body
    assert "`ci-timings`" in body


def test_assemble_with_items_and_pending():
    needs = json.dumps({"tests": "failure"})
    body = _mod.assemble(needs_json=needs, run_url="https://run", pending_jobs=["ci-timings"])
    assert "## ❌ Job failures" in body
    assert "### tests" in body
    assert "Still running" in body
    assert "`ci-timings`" in body


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


def test_assemble_with_lockfile_changed_status():
    """Lockfile changed status renders as action_required with ci-reviewed how_to_fix."""
    statuses = _status("lockfile-diff", [{
        "kind": "action_required",
        "title": "package-lock.json",
        "summary": "Locked npm dependency versions changed.",
        "detail": "#### `package-lock.json`\n\n| col | | |",
        "how_to_fix": "Add the `ci-reviewed` label after verifying the version changes are expected.",
    }])
    body = _mod.assemble(review_statuses_json=statuses)
    assert "## ⚠️ Action required" in body
    assert "### package-lock.json" in body
    assert "Locked npm dependency versions changed." in body
    assert "Add the `ci-reviewed` label" in body
    assert "<details>" not in body  # action_required is not in the collapsible block


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


def test_attach_job_urls_fallback_to_run_url():
    """Items with a source but no matching job URL fall back to run_url."""
    items = [
        ReviewItem(severity="info", title="X", summary="Y", source="some-job"),
    ]
    _mod._attach_job_urls(items, {}, "https://fallback")
    assert items[0].job_url == "https://fallback"


def test_attach_job_urls_no_source_no_link():
    """Items without a source don't get a link."""
    items = [
        ReviewItem(severity="info", title="X", summary="Y"),  # no source
    ]
    _mod._attach_job_urls(items, {"job": "https://run/1"}, "https://fallback")
    assert items[0].job_url == ""


def test_assemble_attaches_links_to_all_items():
    """Integration: assemble() attaches job URLs to status items, not just errors."""
    statuses = _status("supply chain", [{
        "kind": "info",
        "title": "Supply chain scan",
        "summary": "No risks.",
    }])
    job_urls = {
        "Supply Chain Audit / Scan PR for critical supply chain risks": "https://run/1/job/2",
    }
    body = _mod.assemble(
        review_statuses_json=statuses,
        job_urls=job_urls,
        run_url="https://fallback",
    )
    assert "[View job](https://run/1/job/2)" in body


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


def test_render_no_commit_info_when_empty():
    """No commit_info → no extra line below header."""
    body = _mod.render_comment([])
    assert "running on" not in body


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


def test_assemble_both_links_for_ci_timings():
    """Integration: ci-timings has a report URL (link) AND gets a job_url."""
    statuses = _status("ci timings", [{
        "kind": "warning",
        "title": "CI timings",
        "summary": "Wall time 5m vs 3m (+66%).",
        "detail": "- tests: +120s",
        "link": "https://artifact/report.html",
        "link_label": "View report",
    }])
    job_urls = {"CI timings": "https://github.com/run/1/job/5"}
    body = _mod.assemble(
        review_statuses_json=statuses,
        job_urls=job_urls,
        run_url="https://fallback",
    )
    assert "[View report](https://artifact/report.html)" in body
    assert "[View job](https://github.com/run/1/job/5)" in body
