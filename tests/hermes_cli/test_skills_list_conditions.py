"""Malformed activation metadata must not prevent listing installed skills."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


def _write_skill(home: Path, name: str, requires_toolsets: str) -> None:
    skill = home / "skills" / name / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        f"---\nname: {name}\ndescription: Local listing fixture.\n"
        f"metadata:\n  hermes:\n    requires_toolsets: {requires_toolsets}\n---\n"
        "Describe the local fixture.\n",
        encoding="utf-8",
    )


def _run_skills_list(home: Path) -> subprocess.CompletedProcess:
    """Real CLI child: argparse → skills dispatcher → do_list, nothing mocked."""
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ, HERMES_HOME=str(home), HOME=str(home), USERPROFILE=str(home),
               LOCALAPPDATA=str(home / "local"), APPDATA=str(home / "roaming"),
               HERMES_DISABLE_LAZY_INSTALLS="1", PYTHONPATH=str(repo),
               PYTHONUTF8="1", PYTHONIOENCODING="utf-8", NO_COLOR="1", COLUMNS="180")
    env.pop("HERMES_HEAD_HOME", None)
    return subprocess.run(
        [sys.executable, "-c", "from hermes_cli.main import main; main()", "skills", "list"],
        cwd=home.parent, env=env, stdin=subprocess.DEVNULL,
        capture_output=True, encoding="utf-8", timeout=30,
    )


@pytest.mark.parametrize("requirement", ["false", "42"])
def test_list_survives_scalar_toolset_condition(tmp_path, requirement):
    home = tmp_path / "home"
    for name, value in (("bad-condition", requirement), ("valid-skill", "[terminal]"),
                        ("unknown-skill", "[unregistered-fixture-toolset]")):
        _write_skill(home, name, value)
    proc = _run_skills_list(home)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "bad-condition" in proc.stdout
    assert "valid-skill" in proc.stdout
    assert "unknown-skill" in proc.stdout
    assert "gated: unknown toolset" in proc.stdout
    assert "unregistered-fixture-toolset" in proc.stdout


def test_list_annotates_unresolvable_toolset_through_real_cli(tmp_path):
    """The #99877 shape end-to-end: a near-miss plural in real frontmatter must
    surface the gated annotation through the actual CLI entry point. This pins
    parse → conditions extraction → discovery → annotation as one chain, not a
    mocked seam; the correctly-spelled twin is the in-scene control."""
    home = tmp_path / "home"
    _write_skill(home, "typo-skill", "[terminal, files]")
    _write_skill(home, "ok-skill", "[terminal, file]")
    proc = _run_skills_list(home)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "typo-skill" in proc.stdout
    assert "ok-skill" in proc.stdout
    # Exactly one annotation, carrying the offending name; the twin stays clean.
    assert proc.stdout.count("gated: unknown toolset") == 1
    assert "'files'" in proc.stdout
