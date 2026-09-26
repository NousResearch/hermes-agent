"""The bootstrap's separately launched stages must each use PM's pinned Git."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


INSTALLER = Path(__file__).resolve().parents[3] / "scripts" / "install.ps1"


@pytest.mark.platforms("windows")
def test_stage_processes_restore_pinned_git_and_never_fall_back(tmp_path):
    powershell = shutil.which("powershell.exe")
    system_git = shutil.which("git")
    assert powershell and system_git

    origin = tmp_path / "origin"
    origin.mkdir()
    subprocess.run([system_git, "-C", str(origin), "init", "-q", "-b", "main"], check=True)
    (origin / "README").write_text("fixture", encoding="utf-8")
    subprocess.run([system_git, "-C", str(origin), "add", "README"], check=True)
    subprocess.run([system_git, "-C", str(origin), "-c", "user.name=Smoke",
                    "-c", "user.email=smoke@example.invalid", "commit", "-qm", "initial"], check=True)
    expected = subprocess.run([system_git, "-C", str(origin), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()

    home = tmp_path / "home"
    store = tmp_path / "tools"
    env = dict(os.environ, HERMES_HOME=str(home), HERMES_RUNTIME_DIR=str(store),
               HERMES_REPO_URL=str(origin))

    def stage(name):
        result = subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                                 "-File", str(INSTALLER), "-Stage", name, "-Json"],
                                env=env, capture_output=True, text=True, timeout=240)
        frames = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
        assert result.returncode == 0 and len(frames) == 1 and frames[0]["ok"], result.stdout + result.stderr

    stage("prerequisites")
    staged = list(store.glob("git-*/cmd/git.exe"))
    assert len(staged) == 1
    # A wrong Git earlier on the parent's PATH must not win in either new stage.
    poison = tmp_path / "poison"
    poison.mkdir()
    (poison / "git.cmd").write_text("@echo unpinned git invoked 1>&2 & exit /b 73\r\n", encoding="utf-8")
    env["PATH"] = str(poison) + os.pathsep + env["PATH"]
    stage("repository")  # new process; prerequisites' PATH cannot propagate
    stage("complete")  # the marker's bare git call is also a new process
    checkout = home / "hermes-agent"
    actual = subprocess.run([str(staged[0]), "-C", str(checkout), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    marker = json.loads((checkout / ".hermes-bootstrap-complete").read_text(encoding="utf-8-sig"))
    assert actual == expected == marker["pinnedCommit"]

    # Even when system Git is on PATH, an unsupported pin must fail closed.
    probe = (f". '{INSTALLER}'; "
             "$script:GitPinFiles.Remove(('win32-' + (Get-WindowsArch))); "
             "if (Ensure-Git) { exit 1 } else { exit 0 }")
    refused = subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                              "-Command", probe], env=env, capture_output=True, text=True, timeout=30)
    assert refused.returncode == 0, refused.stdout + refused.stderr
