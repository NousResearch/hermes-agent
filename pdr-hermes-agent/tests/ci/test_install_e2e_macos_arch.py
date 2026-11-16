"""install-e2e's macOS packaged arm refuses an Intel bundle pair before any setup.

The arm runs on an Apple Silicon runner and no Intel macOS bundle ships, so an
x64 pair would otherwise get as far as app launch (or Rosetta) before failing.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from hermes_agent import hermes_yaml
import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "install-e2e-macos-run.yml"


def _bundled_job() -> dict:
    jobs = hermes_yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    (job,) = [job for job in jobs.values()
              if "packaged-app" in str(job.get("if", "")) and "open-app-update" in str(job.get("if", ""))]
    return job


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("arch,refused", [("arm64", False), ("x64", True), ("", True)])
def test_the_packaged_arm_refuses_a_non_arm64_bundle_before_setup(arch, refused):
    steps = _bundled_job()["steps"]
    first = steps[0]
    # The gate runs before checkout, toolchain setup and the installer.
    assert "uses" not in first
    assert first["env"]["BUNDLE_ARCH"] == "${{ inputs.bundle-arch }}"
    ran = subprocess.run(["bash", "-e", "-o", "pipefail", "-c", first["run"]], env={"BUNDLE_ARCH": arch},
                         capture_output=True, text=True, encoding="utf-8", check=False)
    assert (ran.returncode != 0) is refused, ran.stdout + ran.stderr
    if refused:
        assert "unsupported macOS bundle architecture" in ran.stdout
