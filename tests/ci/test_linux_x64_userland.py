"""Preparation must not fall through to Jammy's unpatched system Python."""
import os
from pathlib import Path
import shlex
import subprocess

import pytest

from tests.ci.test_desktop_release_tag_admission import _workflow


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("target", ["linux-x64", "linux-arm64"])
def test_linux_workflow_shell_selects_the_right_userland(tmp_path, target):
    job = _workflow()["jobs"][f"build-{target}"]
    # GitHub validates step shells before a matrix row exists. The runner passes
    # the step body as a file; it does not expand matrix expressions here.
    for step in job["steps"]:
        assert "${{" not in step.get("shell", ""), step.get("name")
    shell = next(step["shell"] for step in job["steps"] if step.get("name") == "Build and package")
    step_file = tmp_path / "step with spaces.sh"
    step_file.write_text("printf 'native arm step\\n'\nfalse | true\n", encoding="utf-8")
    row, = job["strategy"]["matrix"]["target"]
    env = {k: v.replace("${{ matrix.target.label }}", row["label"]) for k, v in job["env"].items()}
    result = subprocess.run([arg.replace("{0}", str(step_file)) for arg in shlex.split(shell)],
                            env={**os.environ, **env, "GITHUB_WORKSPACE": str(tmp_path),
                                 "GITHUB_ENV": str(tmp_path / "env"), "GITHUB_PATH": str(tmp_path / "paths"),
                                 "HERMES_NODE": "", "HERMES_PYTHON": "", "X64_BOOTSTRAP_PYTHON": ""},
                            capture_output=True, text=True, check=False)
    assert result.returncode != 0
    if target == "linux-arm64":
        assert result.stdout.strip() == "native arm step"
    else:
        assert result.stdout == "" and "Missing PM x64 bootstrap Python" in result.stderr


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
