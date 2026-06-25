"""Tests for AI-Scientist vendor sync and fork templates."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
OVERLAY_ROOT = REPO_ROOT / "scripts" / "merge_tools" / "overlays" / "ai-scientist"
SYNC_SCRIPT = REPO_ROOT / "scripts" / "sync_ai_scientist_vendor.py"
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "verify_ai_scientist_templates.py"
FORK_TEMPLATES = ("nc_kan", "nc_kan_proof", "hermes_self_evolve")


@pytest.mark.parametrize("name", FORK_TEMPLATES)
def test_fork_template_has_baseline(name: str) -> None:
    template_dir = OVERLAY_ROOT / "templates" / name
    assert template_dir.is_dir(), f"missing overlay template {name}"
    for fname in ("experiment.py", "plot.py", "prompt.json", "seed_ideas.json"):
        assert (template_dir / fname).is_file()
    baseline = template_dir / "run_0" / "final_info.json"
    assert baseline.is_file(), f"{name}: run bootstrap required"
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    assert isinstance(payload, dict) and payload


def test_verify_script_ok() -> None:
    proc = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_sync_dry_run_reports_overlay_source() -> None:
    proc = subprocess.run(
        [sys.executable, str(SYNC_SCRIPT), "--dry-run"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "overlay_source apply:" in proc.stdout
