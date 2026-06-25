"""Tests for the deception-harvest loop (agent/autopilot/harvest.py)."""

from __future__ import annotations

from agent.autopilot import harvest


def _write_adr(tmp_path, body):
    f = tmp_path / "AUTOPILOT-demo-20260623.md"
    f.write_text(body, encoding="utf-8")
    return tmp_path


def test_harvest_empty_when_no_files(tmp_path):
    rep = harvest.harvest(tmp_path / "nope")
    assert rep["files_scanned"] == 0
    assert rep["category_counts"] == {}


def test_harvest_counts_categories(tmp_path):
    root = _write_adr(tmp_path, (
        "# Autopilot decision log\n\n"
        "## 2026-06-23 10:00:00Z — deception\n"
        "- reviewer: deception-detector\n"
        "- gap found / why not passing: caught deception: await_user, effort_excuse\n"
        "- rationale: You tried to hand off; you used effort as an excuse.\n\n"
        "## 2026-06-23 10:05:00Z — deception\n"
        "- gap found / why not passing: caught deception: await_user\n"
        "- rationale: handoff again\n\n"
        "## 2026-06-23 10:10:00Z — completion\n"
        "- verdict: allow\n"
    ))
    rep = harvest.harvest(root)
    assert rep["files_scanned"] == 1
    assert rep["category_counts"]["await_user"] == 2
    assert rep["category_counts"]["effort_excuse"] == 1
    # the 'completion' section is ignored
    assert "allow" not in rep["category_counts"]


def test_harvest_surfaces_novel_phrase(tmp_path):
    root = _write_adr(tmp_path, (
        "## 2026-06-23 10:00:00Z — deception\n"
        "- gap found / why not passing: caught deception: await_user\n"
        "- rationale: the candidate said quote i shall return the baton to my liege quote\n"
    ))
    rep = harvest.harvest(root)
    # a clause that no existing pattern matches should appear as a candidate
    assert any("baton" in p for p in rep["novel_phrases"])


def test_format_report_runs(tmp_path):
    root = _write_adr(tmp_path, (
        "## 2026-06-23 10:00:00Z — deception\n"
        "- gap found / why not passing: caught deception: scope_shrink\n"
        "- rationale: focused on the core\n"
    ))
    out = harvest.format_report(harvest.harvest(root))
    assert "Deception harvest" in out
    assert "scope_shrink" in out
