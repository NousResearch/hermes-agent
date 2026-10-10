"""Behavioral provenance tests for the protected bootstrap-installer build."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "bootstrap-installer-build.yml"

pytestmark = pytest.mark.platforms("posix")


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _seed_repo(root: Path) -> Path:
    origin = root / "origin"
    origin.mkdir()
    _git(origin, "init", "-b", "main")
    _git(origin, "config", "user.email", "ci@example.com")
    _git(origin, "config", "user.name", "ci")
    (origin / "README.md").write_text("trusted\n", encoding="utf-8")
    _git(origin, "add", "README.md")
    _git(origin, "commit", "-m", "trusted")

    clone = root / "clone"
    _git(root, "clone", str(origin), str(clone))
    _git(clone, "config", "user.email", "ci@example.com")
    _git(clone, "config", "user.name", "ci")
    return clone


def test_signed_build_admits_only_a_full_sha_on_main_and_propagates_it(tmp_path: Path):
    yaml = pytest.importorskip("hermes_yaml")
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8-sig"))
    triggers = workflow.get("on", workflow.get(True))
    dispatch_inputs = triggers["workflow_dispatch"]["inputs"]
    assert dispatch_inputs["commit"]["required"] is True
    assert not ({"ref", "pin_branch", "pin_commit"} & set(dispatch_inputs))

    jobs = workflow["jobs"]
    admission = jobs["admit"]
    assert admission["outputs"]["commit"] == "${{ steps.admission.outputs.commit }}"
    controller_checkout = next(
        step for step in admission["steps"] if "actions/checkout@" in step.get("uses", "")
    )
    assert controller_checkout["with"]["ref"] == "${{ github.sha }}"
    assert controller_checkout["with"]["persist-credentials"] is False
    admission_step = next(step for step in admission["steps"] if step.get("id") == "admission")

    for name in ("windows-x64", "macos-arm64"):
        job = jobs[name]
        assert job["needs"] == "admit"
        assert job["environment"] == "release-signing"
        assert job["env"]["HERMES_BUILD_PIN_COMMIT"] == "${{ needs.admit.outputs.commit }}"
        assert "HERMES_BUILD_PIN_BRANCH" not in job["env"]
        checkout = next(step for step in job["steps"] if "actions/checkout@" in step.get("uses", ""))
        assert checkout["with"]["ref"] == "${{ needs.admit.outputs.commit }}"

    clone = _seed_repo(tmp_path)
    trusted = _git(clone, "rev-parse", "HEAD")
    output = clone / "github-output"
    base_env = {
        **os.environ,
        "DEFAULT_BRANCH": "main",
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_REPOSITORY": "NousResearch/hermes-agent",
        "GITHUB_WORKFLOW_REF": (
            "NousResearch/hermes-agent/.github/workflows/"
            "bootstrap-installer-build.yml@refs/heads/main"
        ),
        "GITHUB_OUTPUT": str(output),
    }

    def run_admission(commit: str) -> subprocess.CompletedProcess[str]:
        output.write_text("", encoding="utf-8")
        return subprocess.run(
            [shutil.which("bash") or "bash", "-c", admission_step["run"]],
            cwd=clone,
            env={**base_env, "BUILD_COMMIT": commit},
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )

    accepted = run_admission(trusted)
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    assert output.read_text(encoding="utf-8-sig") == f"commit={trusted}\n"

    (clone / "unreviewed.txt").write_text("unreviewed\n", encoding="utf-8")
    _git(clone, "add", "unreviewed.txt")
    _git(clone, "commit", "-m", "unreviewed")
    unreviewed = _git(clone, "rev-parse", "HEAD")

    for refused in ("main", trusted[:12], unreviewed):
        rejected = run_admission(refused)
        assert rejected.returncode != 0, f"admitted untrusted or mutable input {refused!r}"
        assert output.read_text(encoding="utf-8-sig") == ""
