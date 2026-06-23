"""Unit tests for ``hermes_cli.skills_drift`` — the skills drift detector.

The drift detector walks every per-profile skills tree under the Hermes
root, MD5-hashes each SKILL.md, and reports any skill whose content
differs across copies.  These tests build tmp ``profiles/`` trees with
known drift layouts and assert the scanner's structured report.

The audit motivating this module found ``agent-handoff`` distributed
across 13 profile copies with 3 distinct content versions; the test
fixture mirrors that layout so the test exercises the exact case the
detector is meant to catch.
"""

from __future__ import annotations

import io
import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path

from rich.console import Console

from hermes_cli.skills_drift import (
    DRIFT_ROOT_ENV,
    format_human_report,
    format_json_report,
    run_drift_check,
    scan_skill_drift,
    warn_on_skill_drift,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_profile_tree(root: Path) -> dict:
    """Build a pre-collapse-shaped profiles tree under ``root``.

    Mirrors the audit's agent-handoff finding:
      * 13 profiles carrying agent-handoff SKILL.md
      * 3 distinct content versions (1 + 11 + 1 by profile count)
      * 1 additional clean skill carried identically across 3 profiles
    Returns the raw content-version map for the caller to assert against.
    """
    root.mkdir(parents=True, exist_ok=True)

    content_v1 = b"version-one-content " * 50   # code-craftsman copy
    content_v2 = b"version-two-content " * 50   # 11-copy canonical
    content_v3 = b"version-three-content " * 50 # wags copy

    mapping = {
        "code-craftsman": content_v1,
        "content-curator": content_v2,
        "fleet-coach": content_v2,
        "pm-interface": content_v2,
        "spec-writer": content_v2,
        "totum-build": content_v2,
        "totum-interface": content_v2,
        "totum-operator": content_v2,
        "totum-orchestrator": content_v2,
        "totum-runner": content_v2,
        "totum-t": content_v2,
        "vultr-bridge": content_v2,
        "wags": content_v3,
    }

    for profile, body in mapping.items():
        pdir = root / "profiles" / profile / "skills" / "agent-handoff"
        pdir.mkdir(parents=True)
        (pdir / "SKILL.md").write_bytes(body)

    # Clean skill — same content everywhere, MUST NOT be flagged.
    clean_body = b"clean-skill-content " * 30
    for profile in ("code-craftsman", "wags", "totum-operator"):
        cdir = root / "profiles" / profile / "skills" / "clean-skill"
        cdir.mkdir(parents=True)
        (cdir / "SKILL.md").write_bytes(clean_body)

    return {
        "mapping": mapping,
        "v1_profile": "code-craftsman",
        "v3_profile": "wags",
        "clean_body": clean_body,
    }


@contextmanager
def _scoped_root(tmp_path: Path):
    """Point HERMES_DRIFT_ROOT at ``tmp_path`` for the duration of a test.

    The scanner reads HERMES_DRIFT_ROOT at call time (not at import time)
    so monkeypatching it here is the right hook for isolation — no need
    to override ``get_default_hermes_root`` or patch internal state.
    """
    old = os.environ.get(DRIFT_ROOT_ENV)
    os.environ[DRIFT_ROOT_ENV] = str(tmp_path)
    try:
        yield tmp_path
    finally:
        if old is None:
            os.environ.pop(DRIFT_ROOT_ENV, None)
        else:
            os.environ[DRIFT_ROOT_ENV] = old


# ---------------------------------------------------------------------------
# Acceptance: pre-collapse layout detected with correct 3-version breakdown
# ---------------------------------------------------------------------------


def test_pre_collapse_layout_detected_with_3_md5_breakdown(tmp_path):
    """The exact audit case: 13 copies, 3 distinct content versions."""
    root = tmp_path / ".hermes"
    seeded = _make_profile_tree(root)

    with _scoped_root(root):
        report = scan_skill_drift(skill_filter="agent-handoff")

    assert report["ok"] is False
    assert len(report["findings"]) == 1
    finding = report["findings"][0]
    assert finding["name"] == "agent-handoff"
    assert finding["distinct_versions"] == 3

    md5_to_profiles = {v["md5"]: v["profiles"] for v in finding["versions"]}
    expected_v2 = sorted(
        p for p, c in seeded["mapping"].items() if c == b"version-two-content " * 50
    )
    # The version with 11 profiles is the v2 canonical one.
    v2_profiles = max(md5_to_profiles.values(), key=len)
    assert sorted(v2_profiles) == expected_v2
    assert len(v2_profiles) == 11
    # v1 = code-craftsman (single profile), v3 = wags (single profile).
    assert [seeded["v1_profile"]] in md5_to_profiles.values()
    assert [seeded["v3_profile"]] in md5_to_profiles.values()
    # The canonical MD5s for v1/v2/v3 should all differ from each other.
    assert len(set(md5_to_profiles)) == 3


def test_clean_skill_is_not_flagged_in_same_tree(tmp_path):
    """A skill with identical content across copies MUST NOT appear in findings."""
    root = tmp_path / ".hermes"
    _make_profile_tree(root)

    with _scoped_root(root):
        report = scan_skill_drift(skill_filter="clean-skill")

    assert report["ok"] is True
    assert report["skills_scanned"] == 1
    assert report["findings"] == []
    assert report["scanned_skills"][0]["profiles"] == [
        "code-craftsman", "totum-operator", "wags",
    ]


# ---------------------------------------------------------------------------
# --skill filter scoping
# ---------------------------------------------------------------------------


def test_skill_filter_scopes_to_one_skill(tmp_path):
    """``--skill <name>`` MUST scope the scan to a single skill name."""
    root = tmp_path / ".hermes"
    _make_profile_tree(root)

    with _scoped_root(root):
        full = scan_skill_drift()
        scoped = scan_skill_drift(skill_filter="agent-handoff")
        empty = scan_skill_drift(skill_filter="nonexistent-skill")

    # Full scan sees both skills, finds drift only in agent-handoff.
    assert full["skills_scanned"] == 2
    assert [f["name"] for f in full["findings"]] == ["agent-handoff"]
    # Scoped scan sees only the one we asked about.
    assert scoped["skills_scanned"] == 1
    assert scoped["findings"][0]["name"] == "agent-handoff"
    # Missing skill → no scan output, no findings.
    assert empty["skills_scanned"] == 0
    assert empty["ok"] is True


# ---------------------------------------------------------------------------
# Layouts: post-collapse + minimal default-only install
# ---------------------------------------------------------------------------


def test_post_collapse_state_reports_clean(tmp_path):
    """A single canonical copy in ``<root>/skills`` MUST report clean.

    This is the post-collapse shape the agent-handoff sibling task
    produces: drift_check would have flagged the broken pre-collapse
    tree; once collapse lands it goes silent.
    """
    root = tmp_path / ".hermes"
    canonical = b"the one canonical content " * 100
    (root / "skills" / "agent-handoff").mkdir(parents=True)
    (root / "skills" / "agent-handoff" / "SKILL.md").write_bytes(canonical)

    with _scoped_root(root):
        report = scan_skill_drift(skill_filter="agent-handoff")

    assert report["ok"] is True
    assert report["skills_scanned"] == 1
    assert report["scanned_skills"][0]["profiles"] == ["default"]


def test_default_only_install_reports_clean(tmp_path):
    """A single profile (the implicit default) with one skill is clean."""
    root = tmp_path / ".hermes"
    (root / "skills" / "lonely").mkdir(parents=True)
    (root / "skills" / "lonely" / "SKILL.md").write_bytes(b"only copy on disk")

    with _scoped_root(root):
        report = scan_skill_drift()

    assert report["ok"] is True
    assert report["profiles_scanned"] == 1
    assert report["skills_scanned"] == 1


def test_default_root_skills_are_scanned_alongside_named_profiles(tmp_path):
    """``<root>/skills/`` is the default profile and MUST be included.

    Without the default-profile branch, a skill shipped only at the
    root would silently disappear from the report.
    """
    root = tmp_path / ".hermes"
    (root / "skills" / "everywhere").mkdir(parents=True)
    (root / "skills" / "everywhere" / "SKILL.md").write_bytes(b"same everywhere")
    (root / "profiles" / "p1" / "skills" / "everywhere").mkdir(parents=True)
    (root / "profiles" / "p1" / "skills" / "everywhere" / "SKILL.md").write_bytes(b"same everywhere")

    with _scoped_root(root):
        report = scan_skill_drift(skill_filter="everywhere")

    assert report["ok"] is True
    assert report["profiles_scanned"] == 2  # default + p1
    assert report["scanned_skills"][0]["profiles"] == ["default", "p1"]


# ---------------------------------------------------------------------------
# Output renderers + CLI exit codes
# ---------------------------------------------------------------------------


def test_human_report_clean_shows_green_message():
    report = {
        "ok": True,
        "root": "/fake",
        "profiles_scanned": 1,
        "skills_scanned": 1,
        "findings": [],
        "scanned_skills": [],
    }
    out = format_human_report(report)
    assert "No skill drift detected" in out
    assert "1 profile" in out
    assert "1 skill" in out


def test_human_report_drifted_groups_by_skill_name():
    report = {
        "ok": False,
        "root": "/fake",
        "profiles_scanned": 2,
        "skills_scanned": 1,
        "findings": [
            {
                "name": "demo",
                "distinct_versions": 2,
                "versions": [
                    {"md5": "aaa", "profiles": ["p1"], "first_path": "/p1"},
                    {"md5": "bbb", "profiles": ["p2"], "first_path": "/p2"},
                ],
            }
        ],
        "scanned_skills": [],
    }
    out = format_human_report(report)
    assert "demo" in out
    assert "aaa" in out and "bbb" in out
    assert "p1" in out and "p2" in out


def test_json_report_is_machine_readable_json():
    report = {
        "ok": False,
        "root": "/fake",
        "profiles_scanned": 1,
        "skills_scanned": 1,
        "findings": [
            {
                "name": "x",
                "distinct_versions": 2,
                "versions": [
                    {"md5": "aaa", "profiles": ["p1"], "first_path": "/p1"},
                ],
            }
        ],
        "scanned_skills": [],
    }
    out = format_json_report(report)
    # Round-trip: must be valid JSON with stable shape.
    parsed = json.loads(out)
    assert parsed["ok"] is False
    assert parsed["findings"][0]["name"] == "x"
    assert parsed["findings"][0]["versions"][0]["md5"] == "aaa"


def test_run_drift_check_exits_zero_on_clean_tree(tmp_path):
    root = tmp_path / ".hermes"
    (root / "skills" / "one").mkdir(parents=True)
    (root / "skills" / "one" / "SKILL.md").write_bytes(b"only copy")

    with _scoped_root(root):
        exit_code = run_drift_check(console=Console(file=io.StringIO()))
    assert exit_code == 0


def test_run_drift_check_exits_nonzero_on_drift(tmp_path):
    root = tmp_path / ".hermes"
    _make_profile_tree(root)

    with _scoped_root(root):
        exit_code = run_drift_check(console=Console(file=io.StringIO()))
    assert exit_code == 1


def test_run_drift_check_json_flag_emits_valid_json(tmp_path):
    root = tmp_path / ".hermes"
    _make_profile_tree(root)

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, color_system=None, width=200)
    with _scoped_root(root):
        exit_code = run_drift_check(console=console, as_json=True)
    assert exit_code == 1
    parsed = json.loads(buf.getvalue())
    assert parsed["ok"] is False
    assert parsed["findings"][0]["name"] == "agent-handoff"


# ---------------------------------------------------------------------------
# warn_on_skill_drift — startup-time hook
# ---------------------------------------------------------------------------


def test_warn_on_skill_drift_emits_warning_on_drift(tmp_path, caplog):
    """``warn_on_skill_drift`` MUST log a warning when drift is present."""
    root = tmp_path / ".hermes"
    _make_profile_tree(root)

    with _scoped_root(root):
        with caplog.at_level(logging.WARNING, logger="hermes_cli.skills_drift"):
            report = warn_on_skill_drift()

    assert report is not None
    assert report["ok"] is False
    # One WARNING line with the skill names enumerated.
    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and r.name == "hermes_cli.skills_drift"
    ]
    assert len(warnings) == 1
    assert "agent-handoff" in warnings[0].getMessage()
    assert "hermes skills drift-check" in warnings[0].getMessage()


def test_warn_on_skill_drift_silent_on_clean_tree(tmp_path, caplog):
    """A clean install MUST NOT emit a drift warning at startup."""
    root = tmp_path / ".hermes"
    (root / "skills" / "one").mkdir(parents=True)
    (root / "skills" / "one" / "SKILL.md").write_bytes(b"only copy")

    with _scoped_root(root):
        with caplog.at_level(logging.WARNING, logger="hermes_cli.skills_drift"):
            report = warn_on_skill_drift()

    assert report is not None
    assert report["ok"] is True
    drift_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and r.name == "hermes_cli.skills_drift"
        and "drift" in r.getMessage().lower()
    ]
    assert drift_warnings == []
