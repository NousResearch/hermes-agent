"""Artifact-level regression for the plugin-local A2A skill."""

from __future__ import annotations

import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_SUFFIX = "plugins/platforms/a2a/skills/a2a-peer/SKILL.md"


@pytest.mark.integration
def test_built_wheel_and_sdist_include_a2a_peer_skill(tmp_path):
    build = subprocess.run(
        ["uv", "build", "--wheel", "--sdist", "--out-dir", str(tmp_path), "."],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert build.returncode == 0, f"uv build failed:\n{build.stderr}"

    wheel = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        assert any(name.endswith(SKILL_SUFFIX) for name in archive.namelist())

    sdist = next(tmp_path.glob("*.tar.gz"))
    with tarfile.open(sdist) as archive:
        assert any(name.endswith(SKILL_SUFFIX) for name in archive.getnames())
