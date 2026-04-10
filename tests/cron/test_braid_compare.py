"""Tests for scripts/braid_compare.py — BRAID A/B scorecard generator.

The script is self-contained stdlib and lives under scripts/ (not a Python
package), so we load it via importlib.util.spec_from_file_location. Tests
use tmp_path fixtures to stage fake cron output directories and invoke
``run()`` with explicit --output-root / --outdir.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load scripts/braid_compare.py as a module via importlib
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "braid_compare.py"

_spec = importlib.util.spec_from_file_location("braid_compare", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None, f"could not load {_SCRIPT_PATH}"
braid_compare = importlib.util.module_from_spec(_spec)
sys.modules["braid_compare"] = braid_compare
_spec.loader.exec_module(braid_compare)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_cron_output(
    output_root: Path,
    job_id: str,
    stamp: str,
    response: str,
) -> Path:
    """Create a fake cron output file that matches the real layout."""
    job_dir = output_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / f"{stamp}.md"
    path.write_text(
        f"# Cron Job: fake\n\n## Prompt\n\nfake prompt\n\n## Response\n\n{response}\n",
        encoding="utf-8",
    )
    return path


# A compliant baseline response — risk emoji, FII/DII table, Brent,
# ≥3 section headers, no exclusion strings.
_COMPLIANT_RESPONSE = """## Global Overnight
| Market | Close |
|--------|-------|
| S&P 500 | 6500 |

## India Indicators
| Indicator | Level |
|-----------|-------|
| Brent Crude | $82 |
| INR/USD | 83.40 |

## Risk Dashboard
| Factor | Status |
|--------|--------|
| INR/USD | 🟢 |
| Crude | 🟡 |
| VIX | 🔴 |

## FII/DII Flows
| Category | Buy | Sell | Net |
|----------|-----|------|-----|
| FII | 5000 | 3000 | +2000 |
| DII | 4000 | 2000 | +2000 |
"""

# A non-compliant response — no emoji, no FII/DII table, no Brent.
_BROKEN_RESPONSE = """The market did something today. Probably.

The end.
"""

# A tainted response — contains a sports exclusion hit.
_TAINTED_RESPONSE = _COMPLIANT_RESPONSE + "\n## Bonus\nToday's cricket highlights: Mumbai Indians won.\n"


# ---------------------------------------------------------------------------
# Unit: find_latest_output
# ---------------------------------------------------------------------------


class TestFindLatestOutput:
    def test_returns_none_for_missing_dir(self, tmp_path: Path):
        assert braid_compare.find_latest_output(tmp_path / "nope") is None

    def test_returns_none_for_empty_dir(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert braid_compare.find_latest_output(empty) is None

    def test_picks_lexically_latest(self, tmp_path: Path):
        job = tmp_path / "job"
        job.mkdir()
        (job / "2026-04-04_00-00-00.md").write_text("old")
        (job / "2026-04-10_03-30-55.md").write_text("new")
        (job / "2026-04-05_12-00-00.md").write_text("mid")
        latest = braid_compare.find_latest_output(job)
        assert latest is not None
        assert latest.name == "2026-04-10_03-30-55.md"

    def test_ignores_non_md_files(self, tmp_path: Path):
        job = tmp_path / "job"
        job.mkdir()
        (job / "2026-04-10_03-30-55.md").write_text("yes")
        (job / "2026-04-20_00-00-00.json").write_text("no")
        latest = braid_compare.find_latest_output(job)
        assert latest is not None
        assert latest.name == "2026-04-10_03-30-55.md"


# ---------------------------------------------------------------------------
# Unit: extract_response_section
# ---------------------------------------------------------------------------


class TestExtractResponseSection:
    def test_extracts_response_body(self):
        text = "# Header\n\n## Prompt\n\nsome prompt\n\n## Response\n\nthe answer\n"
        assert braid_compare.extract_response_section(text).strip() == "the answer"

    def test_returns_empty_when_no_response_section(self):
        assert braid_compare.extract_response_section("## Prompt\n\nno response\n") == ""

    def test_handles_multiline_response(self):
        text = "## Response\n\nline one\nline two\nline three\n"
        body = braid_compare.extract_response_section(text)
        assert "line one" in body
        assert "line three" in body


# ---------------------------------------------------------------------------
# Unit: parse_run_at
# ---------------------------------------------------------------------------


class TestParseRunAt:
    def test_parses_standard_filename(self, tmp_path: Path):
        p = tmp_path / "2026-04-10_03-30-55.md"
        assert braid_compare.parse_run_at(p) == "2026-04-10T03:30:55"

    def test_returns_none_for_nonconforming(self, tmp_path: Path):
        p = tmp_path / "random-name.md"
        assert braid_compare.parse_run_at(p) is None


# ---------------------------------------------------------------------------
# Unit: run_compliance_checks
# ---------------------------------------------------------------------------


class TestRunComplianceChecks:
    def test_compliant_response_passes_all(self):
        results = braid_compare.run_compliance_checks(_COMPLIANT_RESPONSE)
        failures = [r for r in results if not r.passed]
        assert failures == [], f"expected all checks to pass, got failures: {failures}"

    def test_broken_response_fails_most(self):
        results = braid_compare.run_compliance_checks(_BROKEN_RESPONSE)
        passed = [r for r in results if r.passed]
        assert len(passed) == 0, f"expected all checks to fail, passed: {passed}"

    def test_specific_checks_are_named(self):
        results = braid_compare.run_compliance_checks(_COMPLIANT_RESPONSE)
        names = {r.name for r in results}
        assert "emoji_risk_dashboard" in names
        assert "fii_dii_table" in names
        assert "brent_crude" in names
        assert "section_headers" in names


# ---------------------------------------------------------------------------
# Unit: find_exclusion_hits
# ---------------------------------------------------------------------------


class TestFindExclusionHits:
    def test_clean_response_no_hits(self):
        assert braid_compare.find_exclusion_hits(_COMPLIANT_RESPONSE) == []

    def test_cricket_mention_is_hit(self):
        hits = braid_compare.find_exclusion_hits(_TAINTED_RESPONSE)
        assert "sports_entertainment" in hits

    def test_case_insensitive(self):
        hits = braid_compare.find_exclusion_hits("Watch the CRICKET tonight")
        assert "sports_entertainment" in hits


# ---------------------------------------------------------------------------
# Integration: build_job_report
# ---------------------------------------------------------------------------


class TestBuildJobReport:
    def test_handles_missing_output_dir(self, tmp_path: Path):
        report = braid_compare.build_job_report("ghost", "baseline", tmp_path)
        assert report.job_id == "ghost"
        assert report.output_path is None
        assert report.run_at is None
        assert report.response_chars == 0
        assert report.checks == []

    def test_builds_report_for_compliant_run(self, tmp_path: Path):
        _write_cron_output(tmp_path, "job-a", "2026-04-10_03-30-55", _COMPLIANT_RESPONSE)
        report = braid_compare.build_job_report("job-a", "baseline", tmp_path)
        assert report.output_path is not None
        assert report.run_at == "2026-04-10T03:30:55"
        assert report.response_chars > 0
        assert report.all_passed is True
        assert report.exclusion_hits == []

    def test_picks_latest_of_multiple_runs(self, tmp_path: Path):
        _write_cron_output(tmp_path, "job-a", "2026-04-04_00-00-00", _BROKEN_RESPONSE)
        _write_cron_output(tmp_path, "job-a", "2026-04-10_03-30-55", _COMPLIANT_RESPONSE)
        report = braid_compare.build_job_report("job-a", "baseline", tmp_path)
        assert report.all_passed is True  # latest wins


# ---------------------------------------------------------------------------
# Integration: compare (scorecard)
# ---------------------------------------------------------------------------


class TestCompare:
    def test_parity_when_both_pass(self, tmp_path: Path):
        _write_cron_output(tmp_path, "base", "2026-04-10_03-30-55", _COMPLIANT_RESPONSE)
        _write_cron_output(tmp_path, "braid", "2026-04-10_03-40-55", _COMPLIANT_RESPONSE)
        base = braid_compare.build_job_report("base", "baseline", tmp_path)
        braid = braid_compare.build_job_report("braid", "braid", tmp_path)
        sc = braid_compare.compare(base, braid)
        assert sc.braid_regressions == []
        assert sc.braid_length_ratio == pytest.approx(1.0, rel=0.01)

    def test_braid_regression_flagged(self, tmp_path: Path):
        _write_cron_output(tmp_path, "base", "2026-04-10_03-30-55", _COMPLIANT_RESPONSE)
        _write_cron_output(tmp_path, "braid", "2026-04-10_03-40-55", _BROKEN_RESPONSE)
        base = braid_compare.build_job_report("base", "baseline", tmp_path)
        braid = braid_compare.build_job_report("braid", "braid", tmp_path)
        sc = braid_compare.compare(base, braid)
        assert len(sc.braid_regressions) >= 1
        assert "emoji_risk_dashboard" in sc.braid_regressions

    def test_new_braid_exclusion_is_a_regression(self, tmp_path: Path):
        _write_cron_output(tmp_path, "base", "2026-04-10_03-30-55", _COMPLIANT_RESPONSE)
        _write_cron_output(tmp_path, "braid", "2026-04-10_03-40-55", _TAINTED_RESPONSE)
        base = braid_compare.build_job_report("base", "baseline", tmp_path)
        braid = braid_compare.build_job_report("braid", "braid", tmp_path)
        sc = braid_compare.compare(base, braid)
        assert any(r.startswith("exclusion:") for r in sc.braid_regressions)

    def test_missing_braid_data_does_not_crash(self, tmp_path: Path):
        _write_cron_output(tmp_path, "base", "2026-04-10_03-30-55", _COMPLIANT_RESPONSE)
        # No BRAID output
        base = braid_compare.build_job_report("base", "baseline", tmp_path)
        braid = braid_compare.build_job_report("braid", "braid", tmp_path)
        sc = braid_compare.compare(base, braid)
        assert braid.output_path is None
        assert sc.braid_length_ratio is None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestRenderScorecardMd:
    def test_parity_status_line(self, tmp_path: Path):
        _write_cron_output(tmp_path, "base", "2026-04-10_03-30-55", _COMPLIANT_RESPONSE)
        _write_cron_output(tmp_path, "braid", "2026-04-10_03-40-55", _COMPLIANT_RESPONSE)
        base = braid_compare.build_job_report("base", "baseline", tmp_path)
        braid = braid_compare.build_job_report("braid", "braid", tmp_path)
        md = braid_compare.render_scorecard_md(braid_compare.compare(base, braid))
        assert "# BRAID Scorecard" in md
        assert "Status: PARITY" in md
        assert "## Compliance Checks" in md

    def test_regression_status_line(self, tmp_path: Path):
        _write_cron_output(tmp_path, "base", "2026-04-10_03-30-55", _COMPLIANT_RESPONSE)
        _write_cron_output(tmp_path, "braid", "2026-04-10_03-40-55", _BROKEN_RESPONSE)
        base = braid_compare.build_job_report("base", "baseline", tmp_path)
        braid = braid_compare.build_job_report("braid", "braid", tmp_path)
        md = braid_compare.render_scorecard_md(braid_compare.compare(base, braid))
        assert "Status: REGRESSION" in md
        assert "emoji_risk_dashboard" in md

    def test_pending_status_when_no_braid_run(self, tmp_path: Path):
        _write_cron_output(tmp_path, "base", "2026-04-10_03-30-55", _COMPLIANT_RESPONSE)
        base = braid_compare.build_job_report("base", "baseline", tmp_path)
        braid = braid_compare.build_job_report("braid", "braid", tmp_path)
        md = braid_compare.render_scorecard_md(braid_compare.compare(base, braid))
        assert "Status: PENDING" in md


class TestRenderScorecardJson:
    def test_json_round_trip(self, tmp_path: Path):
        _write_cron_output(tmp_path, "base", "2026-04-10_03-30-55", _COMPLIANT_RESPONSE)
        _write_cron_output(tmp_path, "braid", "2026-04-10_03-40-55", _COMPLIANT_RESPONSE)
        base = braid_compare.build_job_report("base", "baseline", tmp_path)
        braid = braid_compare.build_job_report("braid", "braid", tmp_path)
        js = braid_compare.render_scorecard_json(braid_compare.compare(base, braid))
        parsed = json.loads(js)
        assert parsed["baseline"]["all_passed"] is True
        assert parsed["braid"]["all_passed"] is True
        assert parsed["braid_regressions"] == []
        assert "generated_at" in parsed


# ---------------------------------------------------------------------------
# CLI: run() entry point and exit codes
# ---------------------------------------------------------------------------


class TestRunCli:
    def test_exit_0_on_parity(self, tmp_path: Path, capsys):
        _write_cron_output(tmp_path, "base", "2026-04-10_03-30-55", _COMPLIANT_RESPONSE)
        _write_cron_output(tmp_path, "braid", "2026-04-10_03-40-55", _COMPLIANT_RESPONSE)
        outdir = tmp_path / "scorecards"
        rc = braid_compare.run(
            [
                "--baseline", "base",
                "--braid", "braid",
                "--output-root", str(tmp_path),
                "--outdir", str(outdir),
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "Status: PARITY" in out
        # Scorecard file was written
        assert any(outdir.glob("scorecard-*.md"))
        assert any(outdir.glob("scorecard-*.json"))

    def test_exit_1_on_braid_regression(self, tmp_path: Path):
        _write_cron_output(tmp_path, "base", "2026-04-10_03-30-55", _COMPLIANT_RESPONSE)
        _write_cron_output(tmp_path, "braid", "2026-04-10_03-40-55", _BROKEN_RESPONSE)
        rc = braid_compare.run(
            [
                "--baseline", "base",
                "--braid", "braid",
                "--output-root", str(tmp_path),
                "--outdir", str(tmp_path / "sc"),
                "--no-write",
            ]
        )
        assert rc == 1

    def test_exit_2_on_missing_braid_data(self, tmp_path: Path):
        _write_cron_output(tmp_path, "base", "2026-04-10_03-30-55", _COMPLIANT_RESPONSE)
        rc = braid_compare.run(
            [
                "--baseline", "base",
                "--braid", "braid-ghost",
                "--output-root", str(tmp_path),
                "--outdir", str(tmp_path / "sc"),
                "--no-write",
            ]
        )
        assert rc == 2

    def test_json_mode_emits_json(self, tmp_path: Path, capsys):
        _write_cron_output(tmp_path, "base", "2026-04-10_03-30-55", _COMPLIANT_RESPONSE)
        _write_cron_output(tmp_path, "braid", "2026-04-10_03-40-55", _COMPLIANT_RESPONSE)
        rc = braid_compare.run(
            [
                "--baseline", "base",
                "--braid", "braid",
                "--output-root", str(tmp_path),
                "--outdir", str(tmp_path / "sc"),
                "--no-write",
                "--json",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        # Parseable JSON
        parsed = json.loads(out)
        assert parsed["baseline"]["job_id"] == "base"
        assert parsed["braid"]["job_id"] == "braid"
