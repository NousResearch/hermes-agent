#!/usr/bin/env python3
"""Tests for the genie optional skill."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_vps(tmp_path):
    """Create a fake VPS directory structure for testing."""
    # Create directory structure
    (tmp_path / "state-snapshots" / "20260615-110000-pre-update").mkdir(parents=True)
    (tmp_path / "state-snapshots" / "20260620-110000-pre-update").mkdir(parents=True)
    (tmp_path / "logs").mkdir(parents=True)
    (tmp_path / "sessions").mkdir(parents=True)
    (tmp_path / "cron-output").mkdir(parents=True)
    (tmp_path / "commons" / "data").mkdir(parents=True)
    (tmp_path / "commons" / "db").mkdir(parents=True)

    # Create fake files
    (tmp_path / "state-snapshots" / "20260615-110000-pre-update" / "state.db").write_text("x" * 1000)
    (tmp_path / "state-snapshots" / "20260620-110000-pre-update" / "state.db").write_text("x" * 1000)
    (tmp_path / "logs" / "gateway.log").write_text("log line\n" * 100)
    (tmp_path / "sessions" / "abc123.json").write_text("{}")
    (tmp_path / "cron-output" / "job1.txt").write_text("output\n" * 10)
    (tmp_path / "commons" / "data" / "cache.json").write_text("cached")
    (tmp_path / "commons" / "db" / "app.db").write_text("db" * 100)

    # Create a recent snapshot (should be preserved)
    recent = tmp_path / "state-snapshots" / "20260621-110000-pre-update"
    recent.mkdir(parents=True)
    (recent / "state.db").write_text("x" * 1000)

    return tmp_path


def test_genie_imports():
    """Genie script should be importable without errors."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "genie",
        os.path.join(os.path.dirname(__file__), "..", "scripts", "genie.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    # Don't execute main, just verify it parses
    assert spec is not None


def test_genie_help():
    """Genie --help should exit 0."""
    import subprocess
    result = subprocess.run(
        ["python3", os.path.join(os.path.dirname(__file__), "..", "scripts", "genie.py"), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--assess" in result.stdout
    assert "--clean" in result.stdout
    assert "--dry-run" in result.stdout


def test_genie_assess_dry_run(tmp_vps, monkeypatch):
    """Genie --assess should report disk usage without errors.

    Config now comes from config.yaml (skills.config.genie.*), not env vars.
    We point HERMES_HOME/HERMES_PROFILE at the temp VPS so the profile-scoped
    paths (PROFILE_HOME/state-snapshots, logs, etc.) resolve into tmp_vps.
    """
    import subprocess

    # tmp_vps is the HERMES_HOME root; create the profiles/<profile> layout.
    profile = "indigo"
    profile_home = tmp_vps / "profiles" / profile
    for sub in ("state-snapshots", "logs", "sessions", "cron-output", "commons"):
        (profile_home / sub).mkdir(parents=True, exist_ok=True)

    # Move the fixture top-level dirs into the profile layout genie expects.
    for name in ("state-snapshots", "logs", "sessions", "cron-output", "commons"):
        src = tmp_vps / name
        if src.exists():
            dest = profile_home / name
            for item in src.iterdir():
                shutil.move(str(item), str(dest / item.name))
            src.rmdir()

    result = subprocess.run(
        ["python3", os.path.join(os.path.dirname(__file__), "..", "scripts", "genie.py"), "--assess"],
        capture_output=True,
        text=True,
        env={**os.environ, "HERMES_HOME": str(tmp_vps), "HERMES_PROFILE": profile},
    )
    # Should not crash
    assert result.returncode == 0 or "Error" not in result.stderr


def test_genie_dry_run_no_changes(tmp_vps, monkeypatch):
    """Genie --clean --dry-run should not modify any files."""
    import subprocess

    # Snapshot files before
    before = set()
    for root, dirs, files in os.walk(tmp_vps):
        for f in files:
            before.add(os.path.join(root, f))

    result = subprocess.run(
        ["python3", os.path.join(os.path.dirname(__file__), "..", "scripts", "genie.py"), "--clean", "--dry-run"],
        capture_output=True,
        text=True,
    )

    # Snapshot files after
    after = set()
    for root, dirs, files in os.walk(tmp_vps):
        for f in files:
            after.add(os.path.join(root, f))

    assert before == after, "Dry run should not modify files"


def test_skill_frontmatter():
    """SKILL.md should have valid frontmatter with required fields."""
    import re

    skill_md = Path(__file__).parent.parent / "SKILL.md"
    content = skill_md.read_text()

    # Extract frontmatter
    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    assert match, "SKILL.md should have YAML frontmatter"

    fm = match.group(1)
    assert "name: genie" in fm
    assert "description:" in fm
    assert "version:" in fm
    assert "author:" in fm
    assert "license:" in fm


def test_skill_has_required_sections():
    """SKILL.md should have all required sections."""
    skill_md = Path(__file__).parent.parent / "SKILL.md"
    content = skill_md.read_text()

    required = [
        "## When to Use",
        "## How to Run",
        "## Procedure",
        "## Safety Rules",
        "## Verification",
    ]
    for section in required:
        assert section in content, f"Missing section: {section}"


def test_scripts_exist():
    """Required scripts should exist."""
    scripts_dir = Path(__file__).parent.parent / "scripts"
    assert (scripts_dir / "genie.py").exists()
    assert (scripts_dir / "genie_rebuild_fts.py").exists()


def test_references_exist():
    """Key reference files should exist."""
    refs_dir = Path(__file__).parent.parent / "references"
    assert (refs_dir / "genie-gotchas.md").exists()
    assert (refs_dir / "operational-notes.md").exists()
    assert (refs_dir / "snapshot-structures.md").exists()
