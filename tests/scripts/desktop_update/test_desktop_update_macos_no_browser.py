"""The macOS hand-off never launches a second browser instance (#96374).

The progress shim used to start the user's Google Chrome/Chromium binary
directly with its own `--user-data-dir`. That is a second instance of the
same bundle, and the Dock records each one as another recent-app tile that it
never merges with the pinned browser, so every update added one more
duplicate icon. On macOS the outcome goes through the notification +
next-boot result dialog instead. This drives the real `posix.sh`
`--self-test-ui` path with Chrome set as the default browser and watches the
process table for a browser pointed at the shim.
"""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SHIM = REPO_ROOT / "scripts" / "desktop-update" / "posix.sh"


def _shim_browser_processes(tmp_path: Path) -> list[str]:
    ps = subprocess.run(
        ["ps", "-axww", "-o", "command="], capture_output=True, text=True, check=False
    )
    return [
        line
        for line in ps.stdout.splitlines()
        if "--app=" in line and f"--user-data-dir={tmp_path}" in line
    ]


@pytest.mark.platforms("macos")
def test_macos_handoff_opens_no_browser_window_when_chrome_is_default(tmp_path):
    # Chrome is the system default https handler: the case the
    # default-browser gate let through.
    prefs = tmp_path / "Library" / "Preferences" / "com.apple.LaunchServices"
    prefs.mkdir(parents=True)
    with open(prefs / "com.apple.launchservices.secure.plist", "wb") as f:
        plistlib.dump(
            {"LSHandlers": [{"LSHandlerURLScheme": "https", "LSHandlerRoleAll": "com.google.chrome"}]},
            f,
        )
    install = tmp_path / "hermes-agent"
    install.mkdir()
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "TMPDIR": str(tmp_path),
        "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
        "HERMES_SELFTEST_HOLD_SECONDS": "3",
    }
    env.pop("HERMES_SELFTEST_FAIL", None)

    proc = subprocess.Popen(
        ["bash", str(SHIM), "--install-root", str(install), "--self-test-ui"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    seen: list[str] = []
    try:
        while proc.poll() is None:
            seen += _shim_browser_processes(tmp_path)
            time.sleep(0.2)
        stdout, stderr = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)

    assert proc.returncode == 0, stdout + stderr
    assert seen == []
