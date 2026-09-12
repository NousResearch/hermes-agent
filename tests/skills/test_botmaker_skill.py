"""Botmaker checks the selected installation and its certified roster."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


SKILL = Path(__file__).resolve().parents[2] / "optional-skills/autonomous-ai-agents/botmaker"


def _install(root):
    skill = root / "skills/autonomous-ai-agents/botmaker"
    shutil.copytree(SKILL, skill)
    return skill / "scripts/drift_check.py"


def _check(script, root, home, guide, active_home):
    env = {
        **os.environ,
        "HOME": str(home),
        "USERPROFILE": str(home),
        "HERMES_HOME": str(active_home),
        "BOTMAKER_VAULT_GUIDE": str(guide),
    }
    return subprocess.run(
        [sys.executable, str(script), "--hermes-root", str(root)],
        env=env, capture_output=True, text=True, check=False,
    )


@pytest.mark.parametrize("custom", [False, True])
def test_drift_checks_selected_root_from_a_named_profile(tmp_path, custom):
    home = tmp_path / "user"
    root = home / ("custom hermes" if custom else ".hermes")
    script = _install(root)
    (root / "profiles/researcher").mkdir(parents=True)
    guide = tmp_path / "making-bots.md"
    guide.write_text("| `@researcher` | `researcher` | hosted | research |\n", encoding="utf-8")
    result = _check(script, root, home, guide, root / "profiles/coordinator")
    assert result.returncode == 0, result.stdout + result.stderr

    # An existing roster entry must be checked against this root, not another install.
    (root / "profiles/researcher").rmdir()
    result = _check(script, root, home, guide, root / "profiles/coordinator")
    assert result.returncode == 1
    assert "roster row names a missing profile: researcher" in result.stdout


def test_roster_allows_non_specialists_but_rejects_missing_members(tmp_path):
    home = tmp_path / "user"
    root = home / ".hermes"
    script = _install(root)
    for name in ("researcher", "persona", "daily-driver", "awaiting-certification"):
        (root / "profiles" / name).mkdir(parents=True)
    guide = tmp_path / "making-bots.md"
    guide.write_text("| `@researcher` | `researcher` | hosted | research |\n", encoding="utf-8")
    result = _check(script, root, home, guide, root)
    assert result.returncode == 0, result.stdout + result.stderr

    guide.write_text("| `@missing` | `missing` | hosted | research |\n", encoding="utf-8")
    result = _check(script, root, home, guide, root)
    assert result.returncode == 1
    assert "roster row names a missing profile: missing" in result.stdout
