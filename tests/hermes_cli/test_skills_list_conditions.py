"""Malformed activation metadata must not prevent listing installed skills."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("requirement", ["false", "42"])
def test_list_survives_scalar_toolset_condition(tmp_path, requirement):
    home = tmp_path / "home"
    for name, value in (("bad-condition", requirement), ("valid-skill", "[terminal]"),
                        ("unknown-skill", "[unregistered-fixture-toolset]")):
        skill = home / "skills" / name / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(
            f"---\nname: {name}\ndescription: Local listing fixture.\n"
            f"metadata:\n  hermes:\n    requires_toolsets: {value}\n---\n"
            "Describe the local fixture.\n",
            encoding="utf-8",
        )

    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ, HERMES_HOME=str(home), HOME=str(home), USERPROFILE=str(home),
               LOCALAPPDATA=str(home / "local"), APPDATA=str(home / "roaming"),
               HERMES_DISABLE_LAZY_INSTALLS="1", PYTHONPATH=str(repo),
               PYTHONUTF8="1", PYTHONIOENCODING="utf-8", NO_COLOR="1", COLUMNS="180")
    env.pop("HERMES_HEAD_HOME", None)
    proc = subprocess.run(
        [sys.executable, "-c", "from hermes_cli.main import main; main()", "skills", "list"],
        cwd=tmp_path, env=env, stdin=subprocess.DEVNULL,
        capture_output=True, encoding="utf-8", timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "bad-condition" in proc.stdout
    assert "valid-skill" in proc.stdout
    assert "unknown-skill" in proc.stdout
    assert "gated: unknown toolset" in proc.stdout
    assert "unregistered-fixture-toolset" in proc.stdout
