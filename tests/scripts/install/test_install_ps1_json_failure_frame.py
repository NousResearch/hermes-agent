"""A fatal stage under -Stage -Json yields exactly one failure frame carrying the reason."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.platforms("windows")
INSTALLER = Path(__file__).resolve().parents[3] / "scripts" / "install.ps1"


def test_fail_inside_a_stage_emits_one_json_frame_with_the_reason(tmp_path):
    powershell = shutil.which("powershell")
    assert powershell
    # A nonempty non-checkout refuses before any clone retries, through Fail.
    install_dir = tmp_path / "install"
    install_dir.mkdir()
    (install_dir / "user-file").write_text("preserve me", encoding="utf-8")
    tools = tmp_path / "empty-tools"
    env = dict(os.environ, HERMES_HOME=str(tmp_path / "home"),
               HERMES_RUNTIME_DIR=str(tools))
    result = subprocess.run(
        [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(INSTALLER),
         "-Stage", "repository", "-Json", "-InstallDir", str(install_dir)],
        env=env, capture_output=True, text=True, timeout=30)
    frames = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert result.returncode == 1
    assert len(frames) == 1, result.stdout
    assert frames[0]["ok"] is False and frames[0]["stage"] == "repository"
    assert "exists and is not a Hermes git checkout" in frames[0]["reason"]
    assert (install_dir / "user-file").read_text(encoding="utf-8") == "preserve me"
    assert not tools.exists()
