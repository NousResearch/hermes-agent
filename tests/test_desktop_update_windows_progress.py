"""Behavioral coverage for quiet Windows Desktop updater progress."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from urllib.request import urlopen

import pytest


pytestmark = pytest.mark.windows_only

REPO_ROOT = Path(__file__).resolve().parent.parent
WINDOWS_UPDATE_PS1 = REPO_ROOT / "scripts" / "desktop-update" / "windows.ps1"


def _read_progress(url: str) -> dict[str, object]:
    with urlopen(f"{url}progress", timeout=2) as response:
        return json.loads(response.read().decode("utf-8"))


def test_progress_advances_while_update_child_is_silent(tmp_path: Path) -> None:
    powershell = shutil.which("powershell.exe")
    assert powershell, "Windows updater tests require Windows PowerShell."

    output_path = tmp_path / "self-test-output.log"
    env = os.environ.copy()
    env["TEMP"] = str(tmp_path)
    env["TMP"] = str(tmp_path)
    env["HERMES_SELFTEST_HOLD_SECONDS"] = "3"
    env["HERMES_SELFTEST_SILENT_CHILD"] = "1"
    env["HERMES_SELFTEST_PYTHON"] = sys.executable

    with output_path.open("wb") as output:
        process = subprocess.Popen(
            [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(WINDOWS_UPDATE_PS1),
                "-SelfTestUi",
                "-NoUi",
            ],
            stdout=output,
            stderr=subprocess.STDOUT,
            env=env,
        )

    try:
        deadline = time.monotonic() + 10
        shim_url = None
        while time.monotonic() < deadline:
            text = output_path.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"SELF-TEST: shim at (http://127\.0\.0\.1:\d+/)", text)
            if match:
                shim_url = match.group(1)
                break
            if process.poll() is not None:
                break
            time.sleep(0.1)

        assert shim_url, output_path.read_text(encoding="utf-8", errors="replace")
        first = _read_progress(shim_url)
        time.sleep(1.2)
        second = _read_progress(shim_url)

        assert first["status"] == "running"
        assert first["message"] == "Testing quiet update"
        assert second["message"] == first["message"]
        assert int(second["elapsed_seconds"]) > int(first["elapsed_seconds"])

        assert process.wait(timeout=10) == 0
        final_output = output_path.read_text(encoding="utf-8", errors="replace")
        assert "SELF-TEST: silent child exit code: 0" in final_output
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
