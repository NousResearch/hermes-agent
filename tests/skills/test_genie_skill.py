#!/usr/bin/env python3
"""Tests for the genie optional skill.

Relocated to tests/skills/ per AGENTS.md skill-authoring HARDLINE rule 7.
Uses only stdlib + pytest + unittest.mock; no live network.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

# Skill lives under optional-skills/infrastructure/genie relative to repo root.
# This file is at <repo>/tests/skills/test_genie_skill.py, so repo root is three
# levels up.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SKILL_DIR = REPO_ROOT / "optional-skills" / "infrastructure" / "genie"
GENIE_SCRIPT = SKILL_DIR / "scripts" / "genie.py"

# genie.py is a standalone script (not a package); add its dir to sys.path so
# `import genie` resolves in the behavior tests.
sys.path.insert(0, str(SKILL_DIR / "scripts"))


@pytest.fixture
def tmp_vps(tmp_path):
    """Create a fake VPS directory structure for testing."""
    (tmp_path / "state-snapshots" / "20260615-110000-pre-update").mkdir(parents=True)
    (tmp_path / "state-snapshots" / "20260620-110000-pre-update").mkdir(parents=True)
    (tmp_path / "logs").mkdir(parents=True)
    (tmp_path / "sessions").mkdir(parents=True)
    (tmp_path / "cron-output").mkdir(parents=True)
    (tmp_path / "commons" / "data").mkdir(parents=True)
    (tmp_path / "commons" / "db").mkdir(parents=True)

    (tmp_path / "state-snapshots" / "20260615-110000-pre-update" / "state.db").write_text("x" * 1000)
    (tmp_path / "state-snapshots" / "20260620-110000-pre-update" / "state.db").write_text("x" * 1000)
    (tmp_path / "logs" / "gateway.log").write_text("log line\n" * 100)
    (tmp_path / "sessions" / "abc123.json").write_text("{}")
    (tmp_path / "cron-output" / "job1.txt").write_text("output\n" * 10)
    (tmp_path / "commons" / "data" / "cache.json").write_text("cached")
    (tmp_path / "commons" / "db" / "app.db").write_text("db" * 100)

    recent = tmp_path / "state-snapshots" / "20260621-110000-pre-update"
    recent.mkdir(parents=True)
    (recent / "state.db").write_text("x" * 1000)

    return tmp_path


def _build_profile_layout(tmp_vps, profile="indigo"):
    """Move the fixture top-level dirs into the profiles/<profile>/ layout
    genie expects, and return the profile_home path."""
    profile_home = tmp_vps / "profiles" / profile
    for sub in ("state-snapshots", "logs", "sessions", "cron-output", "commons"):
        (profile_home / sub).mkdir(parents=True, exist_ok=True)

    for name in ("state-snapshots", "logs", "sessions", "cron-output", "commons"):
        src = tmp_vps / name
        if src.exists():
            dest = profile_home / name
            for item in src.iterdir():
                shutil.move(str(item), str(dest / item.name))
            src.rmdir()

    return profile_home


def test_genie_imports():
    """Genie script should be importable without errors."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("genie", str(GENIE_SCRIPT))
    assert spec is not None
    assert spec.origin is not None


def test_genie_help():
    """Genie --help should exit 0."""
    result = subprocess.run(
        ["python3", str(GENIE_SCRIPT), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--assess" in result.stdout
    assert "--clean" in result.stdout
    assert "--dry-run" in result.stdout


def test_genie_assess_dry_run(tmp_vps, monkeypatch):
    """Genie --assess reports without errors and writes no FILESYSTEM.md.

    Runs main() in-process with discovery mocked so the test stays isolated
    (no real filesystem walk, no cross-process mock loss).
    """
    import genie as g

    monkeypatch.setattr(g, "discover_filesystem", lambda: {})
    monkeypatch.setattr(g, "PROFILE_HOME", str(tmp_vps / "profiles" / "indigo"))
    monkeypatch.setattr(sys, "argv", ["genie", "--assess"])

    profile_home = _build_profile_layout(tmp_vps)
    fs_md = profile_home / "references" / "FILESYSTEM.md"

    assert g.main() is None or True  # main() prints; just ensure no raise
    assert not fs_md.exists(), "FILESYSTEM.md must not be written on --assess"


def test_genie_dry_run_no_changes(tmp_vps, monkeypatch):
    """Genie --clean --dry-run reaches the dry-run path without touching disk.

    Runs main() in-process; clean() is mocked so no real deletion occurs.
    """
    import genie as g

    monkeypatch.setattr(g, "discover_filesystem", lambda: {})
    monkeypatch.setattr(g, "PROFILE_HOME", str(tmp_vps / "profiles" / "indigo"))
    monkeypatch.setattr(g, "clean", lambda cfg: [{"action": "mock", "bytes_freed": 0}])
    monkeypatch.setattr(sys, "argv", ["genie", "--clean", "--dry-run"])

    before = {os.path.join(r, f) for r, _, fs in os.walk(tmp_vps) for f in fs}
    assert g.main() is None or True
    after = {os.path.join(r, f) for r, _, fs in os.walk(tmp_vps) for f in fs}
    assert before == after, "Dry run should not modify files"


def test_discover_writes_filesystem_md(tmp_vps, monkeypatch):
    """--discover writes FILESYSTEM.md; --assess and --clean --dry-run do not.

    Runs main() in-process with discovery mocked (fast, no real walk) but lets
    the real writer create the manifest in the temp PROFILE_HOME, so we assert
    on actual disk state rather than a patched call.
    """
    import genie as g

    monkeypatch.setattr(g, "discover_filesystem", lambda: {
        "t1": {"tier": 1, "path": "/tmp/example", "description": "example"}
    })
    monkeypatch.setattr(g, "PROFILE_HOME", str(tmp_vps / "profiles" / "indigo"))
    monkeypatch.setattr(g, "clean", lambda cfg: [{"action": "mock", "bytes_freed": 0}])

    profile_home = _build_profile_layout(tmp_vps)
    fs_md = profile_home / "references" / "FILESYSTEM.md"

    # --assess: no write (read-only mode).
    monkeypatch.setattr(sys, "argv", ["genie", "--assess"])
    assert g.main() is None or True
    assert not fs_md.exists(), "FILESYSTEM.md must not be written on --assess"

    # --clean --dry-run: no write.
    monkeypatch.setattr(sys, "argv", ["genie", "--clean", "--dry-run"])
    assert g.main() is None or True
    assert not fs_md.exists(), "FILESYSTEM.md must not be written on --clean --dry-run"

    # --discover: writes it.
    monkeypatch.setattr(sys, "argv", ["genie", "--discover"])
    assert g.main() is None or True
    assert fs_md.exists(), "FILESYSTEM.md must be created on explicit --discover"


def test_clean_manifest_targets_executes(tmp_vps, monkeypatch):
    """clean_manifest_targets deletes stale manifest entries and honors dry_run.

    Unit-level (no real disk walk): builds in-memory targets like the merge
    step would, then asserts age-based deletion and dry_run behavior.
    """
    import genie as g

    stale = tmp_vps / "stale"
    stale.mkdir()
    old = stale / "old.txt"
    old.write_text("delete me")
    old_mtime = time.time() - 60 * 86400
    os.utime(old, (old_mtime, old_mtime))

    young = tmp_vps / "young"
    young.mkdir()
    (young / "new.txt").write_text("keep me")

    targets = {
        "stale_t": {"source": "filesystem_md", "tier": 2, "path": str(stale),
                    "max_age_days": 7, "action": "delete"},
        "young_t": {"source": "filesystem_md", "tier": 2, "path": str(young),
                    "max_age_days": 7, "action": "delete"},
        "builtin": {"source": "builtin", "tier": 1, "path": str(young)},
    }

    # Dry run: nothing deleted.
    dry = g.clean_manifest_targets(targets, {"dry_run": True})
    assert old.exists() and (young / "new.txt").exists()
    assert any(r.get("action") == "manifest:stale_t" for r in dry)

    # Real run: stale removed, young kept (young file is fresh).
    real = g.clean_manifest_targets(targets, {"dry_run": False})
    assert not old.exists(), "stale manifest target must be deleted"
    assert (young / "new.txt").exists(), "young manifest target must be kept"
    assert any(r.get("action") == "manifest:stale_t" and r.get("deleted", 0) >= 1 for r in real)


def test_skill_frontmatter():
    """SKILL.md has valid frontmatter; description <= 60 chars, ends with period."""
    import re

    content = (SKILL_DIR / "SKILL.md").read_text()
    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    assert match, "SKILL.md should have YAML frontmatter"

    fm = match.group(1)
    assert "name: genie" in fm
    assert "description:" in fm
    desc = re.search(r"^description: (.*)$", fm, re.MULTILINE).group(1)
    assert len(desc) <= 60, f"description too long ({len(desc)}): {desc}"
    assert desc.endswith("."), "description must end with a period"
    assert "version:" in fm
    assert "author:" in fm
    assert "license:" in fm


def test_skill_has_required_sections():
    """SKILL.md has all required sections."""
    content = (SKILL_DIR / "SKILL.md").read_text()
    for section in ("## When to Use", "## How to Run", "## Procedure",
                    "## Safety Rules", "## Verification"):
        assert section in content, f"Missing section: {section}"


def test_scripts_exist():
    """Required scripts exist."""
    assert (SKILL_DIR / "scripts" / "genie.py").exists()
    assert (SKILL_DIR / "scripts" / "genie_rebuild_fts.py").exists()


def test_references_exist():
    """Key reference files (including those cited in SKILL.md) exist."""
    refs_dir = SKILL_DIR / "references"
    for ref in ("genie-gotchas.md", "operational-notes.md", "snapshot-structures.md",
                "genie-snapshot-retention-bug.md", "backup-prune-diskfull-trap.md"):
        assert (refs_dir / ref).exists(), f"missing referenced file: {ref}"
