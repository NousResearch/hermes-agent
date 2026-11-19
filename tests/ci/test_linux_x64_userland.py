"""Preparation must not fall through to Jammy's unpatched system Python."""
from pathlib import Path
import subprocess

import pytest


@pytest.mark.platforms("linux")
def test_build_requires_a_real_bootstrap_before_launching_docker(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/ci/linux_x64_userland.sh"
    launcher = tmp_path / "docker"
    launcher.write_text("#!/bin/sh\nexit 0\n")
    launcher.chmod(0o755)
    step = tmp_path / "prepare.sh"
    step.touch()
    result = subprocess.run(["bash", script, "build", step], capture_output=True, text=True, check=False,
                            env={"GITHUB_WORKSPACE": str(tmp_path), "GITHUB_PATH": str(tmp_path / "paths"),
                                 "GITHUB_ENV": str(tmp_path / "env"), "PATH": f"{tmp_path}:/usr/bin:/bin",
                                 "X64_FLOOR_IMAGE": "floor", "X64_BUILDER_IMAGE": "builder",
                                 "X64_BOOTSTRAP_PYTHON": str(tmp_path / "missing" / "python3")})
    assert result.returncode != 0
    assert "bootstrap" in result.stderr.lower()
