"""The ``--no-notify`` hand-off flag (#108150).

``repro.sh`` drives the real orchestrator against fake homes; without a
suppression flag those harness runs fired REAL desktop notifications from the
staged fixture (osascript on macOS, notify-send/zenity/kdialog on Linux).
``--no-notify`` makes ``notify_fallback`` return early — the same shape as
``--no-ui`` skipping the shim window — while the result file stays the
durable channel. These drive the real ``posix.sh`` argument parser and the
real ``notify_fallback`` body (extracted verbatim into a stub harness with the
OS notifier stubbed, the same pattern ``test_desktop_update_shim_progress.py``
uses); no mocks of the flow, no source-text assertions.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SHIM_DIR = REPO_ROOT / "scripts" / "desktop-update"
POSIX_SH = SHIM_DIR / "posix.sh"

requires_bash = pytest.mark.skipif(
    not os.path.exists("/bin/bash"), reason="posix.sh needs /bin/bash"
)


@requires_bash
def test_no_notify_flag_is_parsed_by_the_real_handoff(tmp_path):
    """Red on origin/main: ``--no-notify`` used to be an unknown arg (exit 64),
    so repro.sh could not pass it and every fake-home run notified the user."""
    install = tmp_path / "install"
    unpacked = install / "apps" / "desktop" / "release" / "linux-unpacked"
    unpacked.mkdir(parents=True)
    (unpacked / "hermes").touch()

    proc = subprocess.run(
        [
            "/bin/bash", str(POSIX_SH), "--self-test-gate", "--no-notify",
            "--install-root", str(install),
            "--relaunch-target", str(unpacked / "hermes"),
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, f"--no-notify was rejected: {proc.stderr}"
    # The gate decision still prints: the flag suppresses notifications only.
    assert proc.stdout.strip().split(":", 1)[0] in {"skew", "relaunch", "manual"}


def _extract_notify_fallback(src: str) -> str:
    start = src.index("notify_fallback() {")
    end = src.index("\nwrite_status()", start)
    return src[start:end].rstrip() + "\n"


@requires_bash
def test_no_notify_suppresses_the_os_notifier(tmp_path):
    """The flag makes notify_fallback return before reaching any notifier rung."""
    src = _extract_notify_fallback(POSIX_SH.read_text(encoding="utf-8"))
    record = tmp_path / "osascript.log"
    stub = tmp_path / "osascript"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" >> '{record}'\n"
        "exit 0\n",
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    src = src.replace("/usr/bin/osascript", str(stub))
    harness = tmp_path / "harness.sh"
    harness.write_text(
        "#!/usr/bin/env bash\n"
        "set -u\n"
        "log() { :; }\n"
        "uname() { echo Darwin; }\n"
        "NO_NOTIFY=0\n"
        f"{src}\n"
        "NO_NOTIFY=1\n"
        "notify_fallback done 'would-have-notified'\n"
        "notify_fallback manual 'would-have-notified'\n"
        "NO_NOTIFY=0\n"
        "notify_fallback error 'must-still-notify'\n",
        encoding="utf-8",
    )
    harness.chmod(harness.stat().st_mode | stat.S_IEXEC)
    subprocess.run(["/bin/bash", str(harness)], check=True, timeout=10)
    logged = record.read_text(encoding="utf-8")
    # Suppressed when the flag is set — a fake-home repro run stays silent.
    assert "would-have-notified" not in logged
    # Un-set flag keeps the renderer-less recovery surface (#103058).
    assert "must-still-notify" in logged
