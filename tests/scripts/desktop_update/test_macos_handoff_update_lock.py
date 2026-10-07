"""The real macOS hand-off joins its custodian's lock and excludes another updater."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess

import pytest

from tests.installation_launcher_fixture import publish_fixture_launcher

pytestmark = pytest.mark.platforms("macos")
REPO = Path(__file__).resolve().parents[3]

# Only the update's payload is inert. The launcher, shell hand-off, custodian,
# marker publication, Python identity reader and checkout lock are production code.
UPDATE_PROBE = """
import json, os, subprocess, sys
from pathlib import Path

if '--help' in sys.argv:
    print('--keep-stash')
    sys.exit(0)
if sys.argv[1:2] != ['update']:
    sys.exit(1)

repo = Path(os.environ['HANDOFF_TEST_REPO'])
sys.path.insert(0, str(repo))
import hermes_cli
hermes_cli.__path__.insert(0, str(repo / 'hermes_cli'))
from hermes_cli.update_lock import UpdateLock, process_create_time

lock = UpdateLock(install_root=Path.cwd())
joined = lock.acquire()
peer = None
try:
    marker = Path(os.environ['HERMES_HOME'], '.hermes-update-in-progress')
    lines = marker.read_text().splitlines()
    if joined:
        # The peer has the same environment but no inherited checkout-lock fd.
        # Knowing the custodian's pid must never let an independent writer join.
        peer = subprocess.run([
            sys.executable, '-I', '-c',
            'import json, sys; from pathlib import Path; '
            'sys.path.insert(0, sys.argv[1]); '
            'from hermes_cli.update_lock import UpdateLock; '
            'lock = UpdateLock(install_root=Path.cwd()); '
            'joined = lock.acquire(); '
            'print(json.dumps({"joined": joined})); '
            'lock.release()', str(repo),
        ], capture_output=True, text=True, timeout=20, check=True)
    Path(os.environ['HANDOFF_LOCK_CAPTURE']).write_text(json.dumps({
        'joined': joined,
        'owner': int(lines[0]),
        'handoff': int(os.environ['HERMES_UPDATE_HANDOFF_PID']),
        'update_pid': os.getpid(),
        'update_ct': process_create_time(),
        'peer': None if peer is None else json.loads(peer.stdout),
        'holder': None if lock.holder is None else lock.holder.pid,
    }))
finally:
    lock.release()
sys.exit(0 if joined else 2)
"""


def test_macos_handoff_joins_real_lock_and_refuses_independent_update(tmp_path):
    home = tmp_path / "home"
    install = home / "hermes-agent"
    install.mkdir(parents=True)
    publish_fixture_launcher(install, UPDATE_PROBE)
    capture = tmp_path / "lock.json"
    env = {
        **os.environ,
        "HERMES_HOME": str(home),
        "HERMES_RUNTIME_DIR": str(tmp_path / "runtime"),
        "TMPDIR": str(tmp_path),
        "HANDOFF_TEST_REPO": str(REPO),
        "HANDOFF_LOCK_CAPTURE": str(capture),
        "HERMES_UPDATE_SHIM_GRACE_SECONDS": "0",
    }
    for key in ("PYTHONHOME", "PYTHONPATH", "HERMES_UPDATE_STARTED_AT", "HERMES_UPDATE_HANDOFF_PID"):
        env.pop(key, None)
    handoff = subprocess.Popen([
        "bash", str(REPO / "scripts/desktop-update/posix.sh"),
        "--daemonized", "--no-ui", "--install-root", str(install),
        "--relaunch-target", str(tmp_path / "missing.app"),
    ], env=env, cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True)
    try:
        stdout, stderr = handoff.communicate(timeout=120)
    finally:
        if handoff.poll() is None:
            os.killpg(handoff.pid, signal.SIGKILL)  # windows-footgun: ok -- owned macOS fixture group
            handoff.wait(timeout=10)
    assert capture.exists(), (stdout, stderr)
    probe = json.loads(capture.read_text())
    assert probe["handoff"] == probe["owner"], probe
    assert probe["owner"] != probe["update_pid"], probe
    assert probe["joined"] is True, probe
    assert probe["peer"] == {"joined": False}, probe
    assert handoff.returncode == 0, (stdout, stderr)
    assert not (home / ".hermes-update-in-progress").exists()
