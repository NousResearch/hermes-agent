"""Exercise the real handoff's publication barrier with owned inert processes."""
from __future__ import annotations

import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

import pytest

fcntl = pytest.importorskip("fcntl")
pytestmark = pytest.mark.platforms("linux")

from tests.scripts.desktop_update.test_desktop_update_posix_handoff_protocol import _env
from tests.scripts.desktop_update.test_desktop_update_posix_marker import POSIX, _calls, _custodian, _install


@pytest.mark.parametrize("claim", ["missing", "foreign"])
def test_delegate_publication_refuses_without_our_claim(tmp_path, claim):
    marker = tmp_path / "marker"
    child = subprocess.Popen(["sleep", "60"])
    try:
        if claim == "foreign":
            marker.write_text(f"{child.pid}\n{int(time.time())}\n")
        original = marker.read_bytes() if marker.exists() else None
        prelude = (f"log() {{ :; }}; MARKER={shlex.quote(str(marker))}; "
                   "DESKTOP_PID=0 HANDOFF_RUN='' MARKER_CLAIMED=1; "
                   f". {shlex.quote(str(POSIX.with_name('marker.sh')))}; "
                   f"marker_add_delegate {child.pid}")
        result = subprocess.run(["bash", "-c", prelude], cwd=tmp_path, timeout=20, check=False)
        assert result.returncode != 0, "a missing/foreign claim is not a published delegate"
        assert (marker.read_bytes() if marker.exists() else None) == original
    finally:
        child.terminate()
        child.wait(timeout=10)


def test_sidecar_timeout_never_releases_the_parked_update(tmp_path):
    # The installer fixture is an inert executable witness, not a real Hermes
    # update. Production bash, locks, child creation and barrier all run unchanged.
    home, install = _install(tmp_path)
    marker = home / ".hermes-update-in-progress"
    log = home / "logs" / "desktop-update-handoff.log"
    script = subprocess.Popen(
        ["bash", str(POSIX), "--daemonized", "--no-ui", "--install-root", str(install)],
        env=_env(tmp_path, home), cwd=tmp_path, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, start_new_session=True)
    lock_fd = None
    try:
        deadline = time.monotonic() + 30
        while not (marker.exists() and _custodian(home) and marker.read_text().splitlines()[0] == _custodian(home)):
            assert script.poll() is None and time.monotonic() < deadline
            time.sleep(0.005)
        lock_fd = os.open(str(marker) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        while "update marker lock is busy" not in (log.read_text() if log.exists() else ""):
            assert script.poll() is None and time.monotonic() < deadline
            time.sleep(0.01)
        # Lock timeout is the actual production 10-second deadline, not a
        # shortened/mocked predicate. Let the caller process the refused write.
        deadline = time.monotonic() + 5
        while script.poll() is None and "hermes update exit code:" not in log.read_text():
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert not any(call.startswith("update --yes") for call in _calls(tmp_path)), _calls(tmp_path)
        assert "hermes update exit code: 3" in log.read_text()
        assert "delegate:" not in marker.read_text()
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        if script.poll() is None:
            os.killpg(script.pid, signal.SIGKILL)  # windows-footgun: ok — Linux process-group fixture
        script.wait(timeout=10)
