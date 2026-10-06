"""The hand-off log rolls over after a clean run instead of growing forever.

Every hand-off appends the whole `hermes update` output to
logs/desktop-update-handoff.log. Pinned against the real shims: after a clean
run an oversized log moves to `.1`; a failed run leaves it in place, because
that file is the one Desktop's "Open log" button reveals for the failure.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SHIM_DIR = REPO_ROOT / "scripts" / "desktop-update"
OVERSIZED = "old hand-off output\n" * 280_000  # past the 5 MiB ceiling

FAKE_HERMES = """#!/usr/bin/env bash
case "$*" in *--help*) echo "--keep-stash"; exit 0 ;; esac
exit {code}
"""

posix_only = pytest.mark.skipif(
    not (os.path.exists("/bin/bash") and os.path.exists("/usr/bin/python3")),
    reason="posix.sh detaches through /bin/bash and /usr/bin/python3",
)


def _run_posix_handoff(tmp_path: Path, previous: str, update_exit: int) -> Path:
    install_root = tmp_path / "hermes-agent"
    (install_root / "venv" / "bin").mkdir(parents=True)
    hermes = install_root / "venv" / "bin" / "hermes"
    hermes.write_text(FAKE_HERMES.format(code=update_exit))
    hermes.chmod(0o755)
    log = tmp_path / "logs" / "desktop-update-handoff.log"
    log.parent.mkdir()
    log.write_text(previous)

    env = {**os.environ, "TMPDIR": str(tmp_path), "HERMES_HOME": str(tmp_path)}
    subprocess.run(["/bin/bash", str(SHIM_DIR / "posix.sh"), "--install-root", str(install_root), "--no-ui"],
                   env=env, timeout=60, check=True)
    result = tmp_path / ".hermes-update-result.json"
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline and not result.exists():
        time.sleep(0.1)
    assert result.exists(), "hand-off never wrote its result file"
    return log


def _wait_for(predicate, timeout: float) -> bool:
    """finish() rolls the log over just after it writes the result; give it a bounded moment."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return predicate()


@posix_only
def test_oversized_handoff_log_rolls_over_after_a_clean_run(tmp_path):
    log = _run_posix_handoff(tmp_path, OVERSIZED, update_exit=0)
    rolled_path = tmp_path / "logs" / "desktop-update-handoff.log.1"
    assert _wait_for(rolled_path.exists, 15), "clean run did not roll the log over"

    rolled = rolled_path.read_text()
    assert rolled.startswith(OVERSIZED)
    assert "hand-off start: root=" in rolled[len(OVERSIZED):]
    assert not log.exists() or "old hand-off output" not in log.read_text()


@posix_only
def test_failed_run_keeps_its_output_in_the_log_desktop_reveals(tmp_path):
    log = _run_posix_handoff(tmp_path, OVERSIZED, update_exit=1)

    assert not _wait_for((tmp_path / "logs" / "desktop-update-handoff.log.1").exists, 3)
    text = log.read_text()
    assert text.startswith(OVERSIZED)
    assert "hand-off start: root=" in text[len(OVERSIZED):]


@posix_only
def test_log_under_the_ceiling_is_not_rolled_over(tmp_path):
    log = _run_posix_handoff(tmp_path, "small\n", update_exit=0)

    assert not _wait_for((tmp_path / "logs" / "desktop-update-handoff.log.1").exists, 3)
    assert log.read_text().startswith("small\n")


@pytest.mark.platforms("windows")
def test_windows_rolls_over_after_a_clean_run(tmp_path, monkeypatch):
    shell = shutil.which("powershell.exe")
    assert shell, "native Windows acceptance requires PowerShell"
    script = SHIM_DIR / "windows.ps1"
    log = tmp_path / "logs" / "desktop-update-handoff.log"
    log.parent.mkdir()
    log.write_text(OVERSIZED, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_UPDATE_STARTED_AT", raising=False)
    # -SelfTestMarker runs the real start, claim and finally/result publication
    # as a clean run, without updating a checkout or launching anything.
    command = "& $env:HERMES_ROLLOVER_TEST_SCRIPT -InstallRoot $env:HERMES_HOME -NoUi -NoMarkerCleanup -SelfTestMarker"
    result = subprocess.run([shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
                            env={**os.environ, "HERMES_ROLLOVER_TEST_SCRIPT": str(script)},
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr

    rolled = (tmp_path / "logs" / "desktop-update-handoff.log.1").read_text(encoding="utf-8-sig")
    assert rolled.startswith(OVERSIZED)
    assert "hand-off start: root=" in rolled[len(OVERSIZED):]
    assert "could not roll over" not in result.stdout
