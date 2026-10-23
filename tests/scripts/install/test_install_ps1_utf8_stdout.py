"""install.ps1 must emit UTF-8 on the stage-protocol stdout pipe.

apps/desktop spawns this script through ``windowsHide: true``
(CREATE_NO_WINDOW) and reads each stage's stdout as UTF-8. A process with no
console falls back to the machine ANSI code page (936 here), so a non-ASCII
profile path in the JSON frame is written as GBK bytes and the driver reports
a stdout read error instead of a stage result.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("windows")
INSTALLER = Path(__file__).resolve().parents[3] / "scripts" / "install.ps1"


def test_show_resolved_paths_stdout_is_utf8_without_a_console(tmp_path):
    powershell = shutil.which("powershell")
    assert powershell, "Windows PowerShell 5.1 is required for this regression"

    home = tmp_path / "主机用户" / "hermes"
    env = dict(os.environ, HERMES_HOME=str(home))
    result = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(INSTALLER),
            "-ShowResolvedPaths",
        ],
        env=env,
        capture_output=True,
        timeout=60,
        # windowsHide: true -- the console-less spawn the desktop uses.
        creationflags=subprocess.CREATE_NO_WINDOW,
    )

    stderr = result.stderr.decode("utf-8", errors="replace")
    assert result.returncode == 0, stderr
    # Strict decode: GBK-encoded CJK bytes raise here, which is the failure the
    # desktop driver surfaces as a stage that produced no JSON result frame.
    stdout = result.stdout.decode("utf-8")
    report = json.loads(stdout)
    assert "主机用户" in json.dumps(report, ensure_ascii=False)
