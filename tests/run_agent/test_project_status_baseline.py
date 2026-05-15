import shutil
import subprocess

import pytest

from run_agent import _build_project_status_baseline, _looks_like_project_status_request


pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is required")


def test_project_status_baseline_includes_current_git_and_status_docs(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "STATUS.md").write_text("Phase A is complete.\nPhase B is pending.\n", encoding="utf-8")
    (tmp_path / "tracked.txt").write_text("work", encoding="utf-8")

    baseline = _build_project_status_baseline("what is the project status?", cwd=tmp_path)

    assert "<project-status-baseline>" in baseline
    assert f"git_root: {tmp_path.resolve()}" in baseline
    assert "git_status:" in baseline
    assert "STATUS.md" in baseline
    assert "Phase A is complete." in baseline
    assert "Current disk/git evidence is higher authority" in baseline


def test_project_status_baseline_skips_unrelated_turns(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)

    assert _build_project_status_baseline("hello there", cwd=tmp_path) == ""


def test_project_status_request_detector_handles_chinese_continue():
    assert _looks_like_project_status_request("继续推进这个项目")
