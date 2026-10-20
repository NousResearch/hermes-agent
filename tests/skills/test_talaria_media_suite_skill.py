"""Invariant tests for the optional talaria-media-suite skill.

Covers optional-skills/media/talaria-media-suite — the self-hosted
*arr + Jellyfin stack driver. Contracts only, no snapshots: frontmatter,
stdlib-only import surface, LAN-only network posture, and the deterministic
offline CLI behavior (missing env names its var, exit 1, zero requests).
No network access.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SKILL = REPO / "optional-skills" / "media" / "talaria-media-suite"


def _frontmatter() -> dict:
    content = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert content.startswith("---")
    m = re.search(r"\n---\s*\n", content[3:])
    assert m, "unclosed frontmatter"
    import hermes_yaml as yaml

    return yaml.safe_load(content[3 : m.start() + 3])


def test_frontmatter_contract():
    fm = _frontmatter()
    assert fm["name"] == "talaria-media-suite"
    desc = fm["description"]
    assert len(desc) <= 60 and desc.endswith(".")
    assert fm["author"].startswith("Nhut Duong")
    assert fm["license"] == "MIT"
    assert set(fm["platforms"]) == {"linux", "macos"}
    hermes = fm.get("metadata", {}).get("hermes", {})
    assert hermes.get("tags")
    assert "schedule" in hermes.get("blueprint", {})
    env_names = {e["name"] for e in fm["required_environment_variables"]}
    assert env_names == {
        "TALARIA_HOST",
        "SONARR_KEY",
        "RADARR_KEY",
        "LIDARR_KEY",
        "PROWLARR_KEY",
        "BAZARR_KEY",
        "OVERSEERR_KEY",
        "JELLYFIN_KEY",
    }


def test_stdlib_only_import_surface():
    # The package must import clean on a bare stdlib interpreter.
    proc = subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, '.'); import talaria; print(talaria.__version__)"],
        cwd=SKILL, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert re.match(r"\d+\.\d+\.\d+$", proc.stdout.strip())


def test_no_third_party_imports():
    allowed = {
        "argparse", "dataclasses", "json", "socket", "urllib", "datetime",
        "html", "re", "sys", "os", "pathlib", "time", "typing",
    }
    for py in (SKILL / "talaria").glob("*.py"):
        for m in re.finditer(r"^(?:import|from)\s+([a-zA-Z_][\w.]*)", py.read_text(encoding="utf-8"), re.M):
            root = m.group(1).split(".")[0]
            assert root in allowed or root == "talaria", f"{py.name}: imports {root}"


def test_lan_only_no_external_hosts():
    # Every request targets TALARIA_HOST — no hardcoded external hosts
    # (the only http:// literals are host templates and the SVG xmlns).
    for py in (SKILL / "talaria").glob("*.py"):
        content = py.read_text(encoding="utf-8")
        assert not re.search(
            r"https?://(?!(?:127\.0\.0\.1|localhost|\{|\%|\$|www\.w3\.org))(?=\S)", content
        ), f"{py.name}: hardcoded host literal"


def test_cli_missing_env_is_loud_and_offline():
    """No env at all: a service command exits 1 with one 'talaria:' stderr
    line naming the missing host, and zero requests (deterministic without
    any server). `status` is the documented exception — it degrades to a
    badge table with exit 0."""
    proc = subprocess.run(
        [sys.executable, "-m", "talaria", "sonarr", "queue"],
        cwd=SKILL, capture_output=True, text=True, timeout=60,
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
    )
    assert proc.returncode == 1
    assert proc.stderr.startswith("talaria:")
    assert "host not set" in proc.stderr
    degrade = subprocess.run(
        [sys.executable, "-m", "talaria", "status"],
        cwd=SKILL, capture_output=True, text=True, timeout=60,
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
    )
    assert degrade.returncode == 0
    assert "no key" in degrade.stdout


def test_cli_version_contract():
    proc = subprocess.run(
        [sys.executable, "-m", "talaria", "version"],
        cwd=SKILL, capture_output=True, text=True, timeout=60,
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
    )
    assert proc.returncode == 0
    assert re.search(r"\d+\.\d+\.\d+", proc.stdout)
