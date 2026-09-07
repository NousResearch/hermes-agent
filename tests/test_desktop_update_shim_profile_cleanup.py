"""The shim's throwaway Chromium profile must not outlive the hand-off.

`start_ui` launches the shim window on a fresh `--user-data-dir` under TMPDIR
so the user's real browser profile is never touched. Nothing removed it, so
every Desktop update parked another full Chromium profile (~117MB measured on
macOS) in the temp directory until the OS cleared it (#104350). These drive
the real `posix.sh` `--self-test-ui` path — the one that exercises
`start_ui` → exit → `finish` → `stop_ui` without running an update — against
a fake Chromium that materializes the profile dir exactly where the real
one would.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SHIM = REPO_ROOT / "scripts" / "desktop-update" / "posix.sh"

# posix.sh finds the browser via PATH on Linux (`command -v google-chrome …`)
# but via hardcoded /Applications paths on macOS, so only Linux can inject a
# stand-in. CI's Python tests run on Linux; the hand-off itself stays covered
# there end to end.
requires_linux_browser_probe = pytest.mark.skipif(
    sys.platform != "linux",
    reason="find_browser is PATH-based only on Linux",
)

# A stand-in Chromium: claims the profile dir the real browser would create
# (so the leak is observable), records it for the assertion, then idles until
# stop_ui's TERM. `trap … TERM` + 1s sleeps instead of one long `sleep` so the
# signal is acted on within a second of arriving.
FAKE_BROWSER = """#!/bin/bash
trap 'exit 0' TERM INT
for arg in "$@"; do
  case "$arg" in --user-data-dir=*) dir="${arg#--user-data-dir=}" ;; esac
done
mkdir -p "$dir"
echo profile > "$dir/Sentinel"
printf '%s\\n' "$dir" > "$HERMES_TEST_PROFILE_SENTINEL"
while :; do sleep 1; done
"""


def _run_selftest_ui(tmp_path: Path, extra_env: dict[str, str] | None = None) -> Path:
    """Run the real hand-off in self-test UI mode; return the profile dir
    the fake browser claimed (fail loudly if the UI path was skipped)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    browser = bin_dir / "google-chrome"
    browser.write_text(FAKE_BROWSER)
    browser.chmod(0o755)
    # posix.sh only opens the window when the system default browser is
    # Chromium-family; vouch for our fake one so the UI path always runs.
    xdg = bin_dir / "xdg-settings"
    xdg.write_text('#!/bin/sh\ncase "$1" in get) echo google-chrome.desktop ;; esac\n')
    xdg.chmod(0o755)

    install_root = tmp_path / "hermes-agent"
    install_root.mkdir()
    sentinel = tmp_path / "profile-dir"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "TMPDIR": str(tmp_path),
        "HERMES_SELFTEST_HOLD_SECONDS": "1",
        "HERMES_TEST_PROFILE_SENTINEL": str(sentinel),
        **(extra_env or {}),
    }
    subprocess.run(
        ["/bin/bash", str(SHIM), "--install-root", str(install_root), "--self-test-ui"],
        env=env,
        timeout=60,
        check=False,
    )

    assert sentinel.exists(), (
        "the shim never launched the browser window; the UI path was skipped "
        "and this test proved nothing"
    )
    return Path(sentinel.read_text().strip())


@requires_linux_browser_probe
def test_selftest_ui_leaves_no_profile_dir_behind(tmp_path):
    """The clean self-test exit runs `stop_ui` with no grace window; the
    throwaway profile it created must be gone when the hand-off exits."""
    profile_dir = _run_selftest_ui(tmp_path)

    assert not profile_dir.exists(), (
        f"throwaway profile {profile_dir} outlived the hand-off"
    )
    assert list(tmp_path.glob("hermes-update-ui-*")) == []


@requires_linux_browser_probe
def test_error_exit_leaves_no_profile_dir_behind(tmp_path):
    """`HERMES_SELFTEST_FAIL` forces the error outcome, which tears the UI
    down via `stop_ui leave-window` — the same path a failed update takes
    after showing its message. The profile must be cleaned there too."""
    profile_dir = _run_selftest_ui(
        tmp_path,
        {
            "HERMES_SELFTEST_FAIL": "1",
            "HERMES_UPDATE_SHIM_GRACE_SECONDS": "1",
        },
    )

    assert not profile_dir.exists(), (
        f"throwaway profile {profile_dir} outlived the failed hand-off"
    )
    assert list(tmp_path.glob("hermes-update-ui-*")) == []
