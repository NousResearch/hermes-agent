"""Grill-admission compatibility: fenced JSON, admission merge, regrill prompt.

Regression fixtures for the 2026-09-05 failure where every same-runner merge
of a valid qualitative grill response exited 1:
- coder/QA returned fenced markdown JSON, reviewer plain JSON; all three were
  rejected as malformed because the merge CLI only accepted bare JSON.
- the documented grill-admission schema
  {profile, admissions, reporting_sentence, suggested_fix} was fed to
  merge_interview(), which reads only full-report fields, so mandatory
  records were rebuilt with empty qualitative fields and validation failed.
- there was no way to merge admissions into the prior validated report.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[2] / "scripts"
MERGE = SCRIPTS / "rsi-merge-interview.py"
VALIDATE = SCRIPTS / "rsi-validate-interview.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def merger():
    return _load("rsi_merge_interview", MERGE)


@pytest.fixture(scope="module")
def validator():
    return _load("rsi_validate_interview", VALIDATE)


def _audit(profile: str = "coder") -> dict:
    return {
        "profiles": {
            profile: {
                "sessions": [],
                "session_failures": [
                    {
                        "id": "session-failed",
                        "source": "kanban",
                        "title": "failed run",
                        "end_reason": "agent_close",
                        "fail_hits": ["tool:terminal:exit_code=1"],
                    }
                ],
                "cron_failures": [],
                "kanban_failures": [],
            }
        }
    }


def _admissions(profile: str = "coder") -> dict:
    return {
        "profile": profile,
        "admissions": [
            {
                "id": "session-failed",
                "what_happened": "The run stopped on terminal exit_code=1 instead of finishing.",
                "why_misreported": "I reported the surviving progress and dropped the error tail.",
            }
        ],
        "reporting_sentence": "One evidence-cited line per failed session, zero IDs omitted.",
        "suggested_fix": "Validate output as JSON before finishing the run.",
    }


def _prior(profile: str = "coder", **overrides) -> dict:
    """A previously validated full report for the audit fixture."""
    prior = {
        "profile": profile,
        "autonomous_failures": [
            {
                "id": "session-failed",
                "summary": "prior summary",
                "evidence": "prior evidence",
                "suggested_fix": "prior fix",
                "audit_source": "kanban",
            }
        ],
        "incomplete_tasks": [],
        "incidents": [],
        "correction_feedback": [],
        "accounted_session_ids": ["session-failed"],
    }
    prior.update(overrides)
    return prior


def _write_prior(tmp_path: Path, prior: dict | None = None) -> Path:
    prior_path = tmp_path / "prior.json"
    prior_path.write_text(json.dumps(prior or _prior()), encoding="utf-8")
    return prior_path


def _merge_run(tmp_path: Path, profile: str, raw: str, prior: Path | None = None):
    audit_path = tmp_path / "audit.json"
    raw_path = tmp_path / "raw.out"
    output_path = tmp_path / "merged.json"
    grill_path = tmp_path / "grill.txt"
    grill_base = tmp_path / "grill-base.txt"
    audit_path.write_text(json.dumps(_audit(profile)), encoding="utf-8")
    raw_path.write_text(raw, encoding="utf-8")
    grill_base.write_text("GRILL BASE", encoding="utf-8")
    cmd = [
        "python3", str(MERGE), profile, str(raw_path), str(output_path),
        "--audit", str(audit_path),
        "--grill-prompt", str(grill_base),
        "--grill-output", str(grill_path),
    ]
    if prior is not None:
        cmd.extend(["--prior-report", str(prior)])
    run = subprocess.run(cmd, cwd=str(tmp_path), text=True, capture_output=True, check=False)
    return run, output_path, grill_path


def test_fenced_json_admissions_merge_into_prior_report_clean(tmp_path):
    """coder/QA-style ```json fence + grill schema + prior validated report -> exit 0."""
    prior_path = _write_prior(tmp_path)
    fenced = "Here is my admission:\n```json\n" + json.dumps(_admissions()) + "\n```\n"
    run, output_path, grill_path = _merge_run(tmp_path, "coder", fenced, prior=prior_path)

    assert run.returncode == 0, run.stdout + run.stderr
    assert not grill_path.exists()
    assert json.loads(run.stdout)["valid"] is True
    merged = json.loads(output_path.read_text(encoding="utf-8"))
    record = merged["autonomous_failures"][0]
    assert merged["profile"] == "coder"
    assert record["id"] == "session-failed"
    # audit-owned skeleton fields survive; admission qualitative text wins
    assert record["audit_source"] == "kanban"
    assert record["summary"].startswith("The run stopped")
    # a prior row fix is never clobbered by the admission-level proposal
    assert record["suggested_fix"] == "prior fix"


def test_plain_json_admissions_merge_into_prior_report_clean(tmp_path):
    """reviewer-style plain JSON + grill schema + prior validated report -> exit 0."""
    prior_path = _write_prior(tmp_path)
    run, output_path, grill_path = _merge_run(
        tmp_path, "coder", json.dumps(_admissions()), prior=prior_path
    )

    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads(run.stdout)["valid"] is True
    assert not grill_path.exists()


def test_fenced_json_full_report_without_prior_report(tmp_path):
    """A fenced full report still parses (fence stripping is schema-agnostic)."""
    report = {
        "profile": "coder",
        "autonomous_failures": [
            {
                "id": "session-failed",
                "summary": "s",
                "evidence": "e",
                "suggested_fix": "f",
            }
        ],
        "incomplete_tasks": [],
        "incidents": [],
        "correction_feedback": [],
        "accounted_session_ids": ["session-failed"],
    }
    fenced = "```json\n" + json.dumps(report) + "\n```"
    run, output_path, grill_path = _merge_run(tmp_path, "coder", fenced)

    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads(run.stdout)["valid"] is True


def test_admissions_without_prior_report_regrills_with_full_schema(tmp_path):
    """Admissions with no prior report cannot validate alone -> exit 1 and the
    generated grill asks for the FULL report schema, not the admission schema."""
    admissions = json.dumps(_admissions())
    run, output_path, grill_path = _merge_run(tmp_path, "coder", admissions)

    assert run.returncode == 1
    merged = json.loads(output_path.read_text(encoding="utf-8"))
    # scaffold-only merge (no prior): admissions cannot fill rows, because
    # merge_interview ignores the admission schema entirely
    record = merged["autonomous_failures"][0]
    assert record["suggested_fix"] == ""
    grill_text = grill_path.read_text(encoding="utf-8")
    # the retry prompt restates the exact mismatches and required ids
    assert "missing qualitative fields" in grill_text
    assert "session-failed" in grill_text


def test_admissions_with_mismatched_profile_are_rejected(tmp_path):
    prior_path = _write_prior(tmp_path)
    wrong = _admissions()
    wrong["profile"] = "qa"
    run, _output_path, grill_path = _merge_run(
        tmp_path, "coder", json.dumps(wrong), prior=prior_path
    )

    assert run.returncode == 1
    assert "profile mismatch" in grill_path.read_text(encoding="utf-8")


def test_correction_id_admissions_update_prior_feedback_not_scaffold(tmp_path):
    """Real 2026-09-05 coder grill: admissions naming correction ids (c-020,
    c-036) answer qualitative feedback questions; they must update the prior
    report's correction_feedback entries and never create scaffold rows."""
    prior = _prior()
    prior["correction_feedback"] = [
        {"id": "c-020", "still_happening": True, "evidence": "old claim"}
    ]
    prior_path = _write_prior(tmp_path, prior)
    admissions = _admissions()
    admissions["admissions"] = [
        {
            "id": "c-020",
            "what_happened": "No four-trailing-failure evidence was shown.",
            "why_misreported": "I treated repeated failures as watchdog proof.",
        }
    ]
    run, output_path, grill_path = _merge_run(
        tmp_path, "coder", json.dumps(admissions), prior=prior_path
    )

    assert run.returncode == 0, run.stdout + run.stderr
    merged = json.loads(output_path.read_text(encoding="utf-8"))
    # no scaffold row created or reclassified by the correction id
    assert [r["id"] for r in merged["autonomous_failures"]] == ["session-failed"]
    feedback = {r["id"]: r for r in merged["correction_feedback"]}
    assert feedback["c-020"]["evidence"].startswith("No four-trailing-failure")


def test_chained_429_then_fenced_admissions_merge_clean(tmp_path):
    """The exact production chain: initial merge fails (literal HTTP 429
    body), the failed scaffold becomes --prior-report, and the documented
    top-level grill schema (fenced, no per-row suggested_fix, plus a
    non-audited 'interview' admission) merges clean with every mandatory
    row filled and audit-exact IDs/categories."""
    # Run 1: initial merge with a literal HTTP 429 error body -> exit 1.
    raw_429 = "HTTP 429 Too Many Requests: rate limited, try again later"
    run1, out1, grill1 = _merge_run(tmp_path, "coder", raw_429)
    assert run1.returncode == 1
    scaffold = json.loads(out1.read_text(encoding="utf-8"))
    # the failed scaffold has empty qualitative fields, exactly as produced
    assert scaffold["autonomous_failures"][0]["suggested_fix"] == ""

    # Run 2: grill response in the documented schema — suggested_fix only at
    # top level — merged into the failed scaffold. Includes the real coder
    # payload's non-audited id=interview admission.
    admissions = _admissions()
    admissions["admissions"].append(
        {
            "id": "interview",
            "what_happened": "The interview schema made me under-report fixes.",
            "why_misreported": "I filled the full report from memory.",
        }
    )
    fenced = "```json\n" + json.dumps(admissions) + "\n```"
    run2, out2, grill2 = _merge_run(tmp_path, "coder", fenced, prior=out1)

    assert run2.returncode == 0, run2.stdout + run2.stderr
    result = json.loads(run2.stdout)
    assert result["valid"] is True
    merged = json.loads(out2.read_text(encoding="utf-8"))
    # audit-exact accounting: only the audited row, never the 'interview' id
    assert [r["id"] for r in merged["autonomous_failures"]] == ["session-failed"]
    assert [r["id"] for r in merged["incomplete_tasks"]] == []
    assert merged["accounted_session_ids"] == ["session-failed"]
    # the empty row fix was filled from the TOP-LEVEL suggested_fix
    row = merged["autonomous_failures"][0]
    assert row["summary"].startswith("The run stopped")
    assert row["evidence"].startswith("I reported")
    assert row["suggested_fix"] == "Validate output as JSON before finishing the run."
    # the non-audited admission is surfaced as omitted, not silently dropped
    assert result["omitted_admission_ids"] == ["interview"]


def test_top_level_suggested_fix_fills_empty_row_fix_never_clobbers(tmp_path):
    """Empty mandatory row fixes fill from the top-level proposal; a row
    that already carries a fix is never overwritten."""
    prior = _prior()
    prior["autonomous_failures"].append(
        {
            "id": "session-empty",
            "summary": "s",
            "evidence": "e",
            "suggested_fix": "",
            "audit_source": "session",
        }
    )
    prior["accounted_session_ids"] = ["session-failed", "session-empty"]
    audit = _audit()
    audit["profiles"]["coder"]["session_failures"].append(
        {"id": "session-empty", "source": "session", "fail_hits": ["tool:x"]}
    )
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    prior_path = tmp_path / "prior.json"
    prior_path.write_text(json.dumps(prior), encoding="utf-8")
    raw_path = tmp_path / "raw.out"
    admissions = _admissions()
    # no per-row suggested_fix anywhere — the documented grill schema
    raw_path.write_text(json.dumps(admissions), encoding="utf-8")
    output_path = tmp_path / "merged.json"
    cmd = [
        "python3", str(MERGE), "coder", str(raw_path), str(output_path),
        "--audit", str(audit_path), "--prior-report", str(prior_path),
    ]
    run = subprocess.run(cmd, cwd=str(tmp_path), text=True, capture_output=True, check=False)

    assert run.returncode == 0, run.stdout + run.stderr
    merged = json.loads(output_path.read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in merged["autonomous_failures"]}
    assert rows["session-empty"]["suggested_fix"] == (
        "Validate output as JSON before finishing the run."
    )
    assert rows["session-failed"]["suggested_fix"] == "prior fix"


def test_unknown_admission_ids_are_omitted_safely_not_conflicts(tmp_path):
    """A non-audited admission id is omitted and surfaced without failing the
    otherwise-valid audited admissions (safe omission per review round 1)."""
    prior_path = _write_prior(tmp_path)
    admissions = _admissions()
    admissions["admissions"].append(
        {"id": "interview", "what_happened": "x", "why_misreported": "y"}
    )
    run, output_path, grill_path = _merge_run(
        tmp_path, "coder", json.dumps(admissions), prior=prior_path
    )

    assert run.returncode == 0, run.stdout + run.stderr
    result = json.loads(run.stdout)
    assert result["valid"] is True
    assert result["omitted_admission_ids"] == ["interview"]
    merged = json.loads(output_path.read_text(encoding="utf-8"))
    assert [r["id"] for r in merged["autonomous_failures"]] == ["session-failed"]
    assert [r["id"] for r in merged["incomplete_tasks"]] == []
    assert not grill_path.exists()


def test_admissions_ids_outside_audit_do_not_alter_scaffold(tmp_path):
    """An admission naming an audited ID in the wrong category, or an unknown
    ID, cannot reclassify or add scaffold rows (audit-owned categories)."""
    prior_path = _write_prior(tmp_path)
    admissions = _admissions()
    admissions["admissions"] = [
        {
            "id": "unknown-id",
            "what_happened": "x",
            "why_misreported": "y",
        }
    ]
    run, output_path, grill_path = _merge_run(
        tmp_path, "coder", json.dumps(admissions), prior=prior_path
    )

    # safe omission: the unknown id cannot alter the audit-owned scaffold, and
    # it is surfaced in the merge result instead of silently disappearing
    assert run.returncode == 0, run.stdout + run.stderr
    result = json.loads(run.stdout)
    assert result["omitted_admission_ids"] == ["unknown-id"]
    merged = json.loads(output_path.read_text(encoding="utf-8"))
    # unknown admission id must not appear as a scaffold row
    assert [r["id"] for r in merged["autonomous_failures"]] == ["session-failed"]
    assert [r["id"] for r in merged["incomplete_tasks"]] == []
    assert not grill_path.exists()


def test_correction_feedback_admissions_update_prior_report(tmp_path):
    """correction_feedback from a grill merges into the prior validated report."""
    prior_path = tmp_path / "prior.json"
    prior_path.write_text(
        json.dumps(
            {
                "profile": "coder",
                "autonomous_failures": [
                    {
                        "id": "session-failed",
                        "summary": "prior",
                        "evidence": "prior",
                        "suggested_fix": "prior",
                    }
                ],
                "incomplete_tasks": [],
                "incidents": [],
                "correction_feedback": [
                    {"id": "c-020", "still_happening": True, "evidence": "old"}
                ],
                "accounted_session_ids": ["session-failed"],
            }
        ),
        encoding="utf-8",
    )
    admissions = _admissions()
    admissions["correction_feedback"] = [
        {"id": "c-020", "still_happening": False, "evidence": "grill re-checked; not supported"}
    ]
    run, output_path, grill_path = _merge_run(
        tmp_path, "coder", json.dumps(admissions), prior=prior_path
    )

    assert run.returncode == 0, run.stdout + run.stderr
    merged = json.loads(output_path.read_text(encoding="utf-8"))
    feedback = {row["id"]: row for row in merged["correction_feedback"] if isinstance(row, dict)}
    assert feedback["c-020"]["still_happening"] is False
    assert feedback["c-020"]["evidence"].startswith("grill re-checked")


def test_malformed_grill_output_exits_nonzero_and_grills_full_schema(tmp_path):
    run, output_path, grill_path = _merge_run(tmp_path, "coder", "totally not json")

    assert run.returncode == 1
    merged = json.loads(output_path.read_text(encoding="utf-8"))
    assert merged["autonomous_failures"][0]["summary"] == ""
    grill_text = grill_path.read_text(encoding="utf-8")
    assert "interview unavailable or malformed" in grill_text
    assert "missing qualitative fields" in grill_text


def test_grill_output_fenced_with_no_braces_exits_nonzero(tmp_path):
    run, _output_path, grill_path = _merge_run(
        tmp_path, "coder", "```json\nnot json at all\n```"
    )

    assert run.returncode == 1
    assert "interview unavailable or malformed" in grill_path.read_text(encoding="utf-8")


def test_validate_cli_also_parses_fenced_json(tmp_path):
    """The standalone validator accepts the same fenced wrapper."""
    audit_path = tmp_path / "audit.json"
    report_path = tmp_path / "report.json"
    audit_path.write_text(json.dumps(_audit()), encoding="utf-8")
    report = {
        "profile": "coder",
        "autonomous_failures": [
            {"id": "session-failed", "summary": "s", "evidence": "e", "suggested_fix": "f"}
        ],
        "incomplete_tasks": [],
        "incidents": [],
        "correction_feedback": [],
        "accounted_session_ids": ["session-failed"],
    }
    report_path.write_text("```json\n" + json.dumps(report) + "\n```", encoding="utf-8")

    run = subprocess.run(
        ["python3", str(VALIDATE), "coder", str(report_path), "--audit", str(audit_path)],
        cwd=str(tmp_path),
        text=True,
        capture_output=True,
        check=False,
    )

    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads(run.stdout)["valid"] is True


def test_thirteen_profile_roster_admissions_accounting(tmp_path):
    """Every installed profile: admissions merge clean for each roster slice
    with a prior validated report, and the merged report stays audit-exact."""
    roster = [
        "default", "buggy", "coder", "jade", "jade-ops", "product", "qa",
        "research", "reviewer", "rsi", "x", "yuki", "yuki-ops",
    ]
    mod = _load("rsi_interview_roster", SCRIPTS / "rsi_interview.py")
    for profile in roster:
        audit = _audit(profile)
        audit = _audit(profile)
        scaffold = mod.build_scaffold(profile, audit)
        prior = dict(scaffold)
        prior["autonomous_failures"][0].update(
            {"summary": "s", "evidence": "e", "suggested_fix": "prior fix"}
        )
        prior_path = tmp_path / f"prior-{profile}.json"
        prior_path.write_text(json.dumps(prior), encoding="utf-8")

        run, output_path, grill_path = _merge_run(
            tmp_path, profile, json.dumps(_admissions(profile)), prior=prior_path
        )

        assert run.returncode == 0, (profile, run.stdout + run.stderr)
        merged = json.loads(output_path.read_text(encoding="utf-8"))
        assert merged["accounted_session_ids"] == ["session-failed"]
        assert merged["autonomous_failures"][0]["id"] == "session-failed"
        assert merged["autonomous_failures"][0]["audit_source"] == "kanban"


def test_merge_run_never_writes_outside_requested_paths(tmp_path):
    """Read-only /tmp isolation contract: merging writes only the output and
    grill files it was asked for, nothing in the audit or prompt inputs."""
    prior_path = _write_prior(tmp_path)
    before = {p: p.stat().st_mtime_ns for p in tmp_path.iterdir()}
    run, output_path, grill_path = _merge_run(
        tmp_path, "coder", json.dumps(_admissions()), prior=prior_path
    )
    after = {
        p.name: p.stat().st_mtime_ns
        for p in tmp_path.iterdir()
        if p.name not in {"merged.json", "grill.txt"}
    }

    assert run.returncode == 0
    assert before[prior_path] == after["prior.json"]
