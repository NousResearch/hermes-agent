"""Readiness tests for optional kanban-video-orchestrator skill."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "creative"
    / "kanban-video-orchestrator"
)
BOOTSTRAP_PATH = SKILL_DIR / "scripts" / "bootstrap_pipeline.py"
VALIDATOR_PATH = SKILL_DIR / "scripts" / "validate_fixture.py"
FIXTURE_PLAN = SKILL_DIR / "fixtures" / "sample-plan.json"


def load_bootstrap_module():
    spec = importlib.util.spec_from_file_location("bootstrap_pipeline", BOOTSTRAP_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixture_plan_is_valid_and_renders_expected_contract():
    bootstrap = load_bootstrap_module()
    plan = json.loads(FIXTURE_PLAN.read_text())

    assert bootstrap.validate_plan(plan) == []

    setup = bootstrap.render_setup_sh(
        plan,
        bootstrap.render_brief(plan),
        bootstrap.render_team_md(plan),
    )
    assert "hermes kanban create \"Direct production of Fixture Product Teaser\"" in setup
    assert "--workspace dir:\"$WORKSPACE\"" in setup
    assert "--tenant \"$TENANT\"" in setup
    assert "cfg[\"approvals\"] =" not in setup
    assert "approvals.mode" not in setup
    assert "--yolo" not in setup


def test_fixture_validator_generates_deterministic_sample_artifacts(tmp_path):
    out_dir = tmp_path / "fixture-run"
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR_PATH),
            "--check-determinism",
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "PASS kanban-video-orchestrator fixture validation" in result.stdout
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert sorted(manifest["artifacts"]) == [
        "TEAM.md",
        "brief.md",
        "setup.sh",
        "validation.log",
    ]
    assert (out_dir / "setup.sh").stat().st_mode & 0o777 == 0o755
    assert "workspace_kind=\"dir\"" in (out_dir / "TEAM.md").read_text()
