from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "skills" / "productivity" / "desired-state"
SCRIPTS = SKILL / "scripts"


def run_ds(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "ds.py", *args],
        cwd=SCRIPTS,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )


def test_desired_state_cli_tracks_and_reports_gap(tmp_path: Path):
    run_ds(
        tmp_path,
        "define",
        "finance",
        "Hit 15% savings rate",
        "--direction",
        "increase",
        "--baseline",
        "9",
        "--current",
        "9",
        "--target",
        "15",
        "--unit",
        "%",
        "--start-date",
        "2026-01-01",
        "--target-date",
        "2026-12-31",
    )
    run_ds(tmp_path, "track", "finance", "hit-15-savings-rate", "12.4")

    gap = run_ds(tmp_path, "--json", "gap", "finance", "hit-15-savings-rate")
    payload = json.loads(gap.stdout)

    assert payload["domain"] == "finance"
    assert payload["slug"] == "hit-15-savings-rate"
    assert payload["gap"]["progress"] == 0.5667
    assert payload["gap"]["direction"] == "increase"
    assert payload["gap"]["pace"] in {"ahead", "on_track", "behind", "met", "unknown"}


def test_desired_state_skill_metadata_matches_contribution_shape():
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")

    assert "description: Track durable goals and current-vs-target gaps." in text
    assert "# Desired-State Skill" in text
    for heading in (
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ):
        assert heading in text
