#!/usr/bin/env python3
"""Tests for scripts/render_report.py — the deterministic report renderer.

Covers: valid island injection, refusal on malformed JSON, refusal on the
placeholder subject, the --md archival rendering, and that both shipped
shells carry a parseable placeholder island.
Run with: python3 -m pytest test_render_report.py
(or plain `python3 test_render_report.py` for a lightweight self-check).
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parents[3]
SCRIPT = SKILLS_DIR / "bmad-workflow-builder" / "scripts" / "render_report.py"
SHELLS = [
    SKILLS_DIR / "bmad-workflow-builder" / "assets" / "report-shell.html",
    SKILLS_DIR / "bmad-agent-builder" / "assets" / "report-shell.html",
]
ISLAND_RE = re.compile(
    r'<script[^>]*\bid="report-data"[^>]*>(.*?)</script>', re.DOTALL
)

VALID_DATA = {
    "schema_version": 2,
    "subject": "skills/example-skill",
    "generated": "2026-06-10",
    "verdict": "One ceremony section; otherwise sound.",
    "grade": "good",
    "summary": "Solid structure and clean wiring. The main opportunity is one over-scripted reference.",
    "standards": {
        "canon": "/abs/skills/bmad-workflow-builder/references/prompt-quality-canon.md",
        "principles": "/abs/skills/bmad-workflow-builder/references/skill-quality-principles.md",
        "scripts": "/abs/skills/bmad-workflow-builder/references/script-standards.md",
    },
    "themes": [
        {
            "title": "Scripted sequences where goals suffice",
            "root_cause": "Steps are numbered without true ordering dependencies.",
            "finding_ids": ["leanness-1"],
            "action": "Replace ordered lists with goal sentences.",
        }
    ],
    "strengths": ["Frontmatter and routing map are exemplary."],
    "recommendations": [
        {"rank": 1, "action": "De-script the finalize section.", "resolves": ["leanness-1"]}
    ],
    "findings": [
        {
            "id": "leanness-1",
            "lens": "leanness",
            "severity": "high",
            "title": "Numbered finalize steps are decoration",
            "location": "references/build-process.md:finalize",
            "evidence": "No step depends on a prior step's output.",
            "recommendation": "Replace with a single goal sentence.",
        }
    ],
}


def run_render(args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *[str(a) for a in args]],
        capture_output=True,
        text=True,
    )


def test_valid_island_injection():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        findings = tmp / "findings.json"
        out = tmp / "report.html"
        findings.write_text(json.dumps(VALID_DATA), encoding="utf-8")

        result = run_render([findings, "--shell", SHELLS[0], "-o", out])
        assert result.returncode == 0, result.stderr
        html = out.read_text(encoding="utf-8")

        match = ISLAND_RE.search(html)
        assert match, "rendered HTML has no report-data island"
        island = json.loads(match.group(1))
        assert island["subject"] == "skills/example-skill"
        assert island["findings"][0]["id"] == "leanness-1"
        assert island["standards"]["canon"].endswith("prompt-quality-canon.md")
        assert "__PLACEHOLDER__" not in match.group(1)

        stdout = json.loads(result.stdout)
        assert stdout["counts"] == {"critical": 0, "high": 1, "medium": 0, "low": 0}
        assert stdout["grade"] == "good"


def test_refuses_bad_json():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        findings = tmp / "findings.json"
        out = tmp / "report.html"
        findings.write_text("{ this is not json", encoding="utf-8")

        result = run_render([findings, "--shell", SHELLS[0], "-o", out])
        assert result.returncode != 0
        assert "not valid JSON" in result.stderr
        assert not out.exists(), "refused render must not write output"


def test_refuses_placeholder_subject():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        findings = tmp / "findings.json"
        out = tmp / "report.html"
        data = dict(VALID_DATA, subject="__PLACEHOLDER__")
        findings.write_text(json.dumps(data), encoding="utf-8")

        result = run_render([findings, "--shell", SHELLS[0], "-o", out])
        assert result.returncode != 0
        assert "placeholder" in result.stderr.lower()
        assert not out.exists(), "refused render must not write output"


def test_md_output():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        findings = tmp / "findings.json"
        out = tmp / "report.html"
        md = tmp / "report.md"
        findings.write_text(json.dumps(VALID_DATA), encoding="utf-8")

        result = run_render([findings, "--shell", SHELLS[0], "-o", out, "--md", md])
        assert result.returncode == 0, result.stderr
        text = md.read_text(encoding="utf-8")
        assert "# Analysis Report: skills/example-skill" in text
        assert "**Grade: Good**" in text
        assert "## Themes" in text
        assert "Scripted sequences where goals suffice" in text
        assert "## Strengths" in text
        assert "## Recommendations" in text
        assert "### High (1)" in text
        assert "leanness-1" in text


def test_shipped_shells_carry_placeholder_island():
    for shell in SHELLS:
        match = ISLAND_RE.search(shell.read_text(encoding="utf-8"))
        assert match, f"{shell} has no report-data island"
        island = json.loads(match.group(1))
        assert island["subject"] == "__PLACEHOLDER__", (
            f"{shell} ships a non-placeholder island; a failed injection "
            "would show its contents as real findings"
        )
        assert island["findings"] == []


def test_render_script_copies_identical():
    other = SKILLS_DIR / "bmad-agent-builder" / "scripts" / "render_report.py"
    assert SCRIPT.read_bytes() == other.read_bytes(), (
        "render_report.py copies have drifted between the two builder skills"
    )


if __name__ == "__main__":
    test_valid_island_injection()
    test_refuses_bad_json()
    test_refuses_placeholder_subject()
    test_md_output()
    test_shipped_shells_carry_placeholder_island()
    test_render_script_copies_identical()
    print("ok: render_report tests passed")
