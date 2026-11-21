"""The gateway is paused only by the update that owns the install (#149520).

An update that pauses the gateway and then refuses, or waits on another update's lock, leaves the
Desktop with no backend and nothing that says why ("Gateway starting" forever). The pause entry
point therefore checks ownership itself: while ANOTHER process tree holds the checkout kernel lock
it refuses (exit 2) before any gateway is asked to stop. A lock whose holder died is free: the
kernel released it with the holder, so the pause goes ahead. Real lock holders (child processes),
real kernel locks; the seam is the platform pause itself, recorded instead of stopping gateways.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import main as cli_main
from hermes_cli import update_cmd, update_lock

REPO_ROOT = Path(__file__).resolve().parents[2]

_HOLD = """
import sys, time
sys.path.insert(0, sys.argv[1])
from pathlib import Path
from hermes_cli.update_lock import UpdateLock
lock = UpdateLock(path=Path(sys.argv[3]), install_root=sys.argv[2])
assert lock.acquire()
print("held", flush=True)
time.sleep(120)
"""


@pytest.fixture
def install(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    paused: list[str] = []
    monkeypatch.setattr(cli_main, "_pause_windows_gateways_for_update", lambda: paused.append("windows") or None)
    monkeypatch.setattr(update_cmd._posix_pause, "arm_pause", lambda **_k: paused.append("posix") or None)
    return SimpleNamespace(root=root, paused=paused, home=tmp_path)


def _holder(install):
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLD, str(REPO_ROOT), str(install.root), str(install.home / "other-home" / "m")],
        stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True)
    assert proc.stdout.readline().strip() == "held"
    return proc


@pytest.mark.platforms("posix")
def test_no_gateway_is_paused_while_another_update_holds_the_lock(install, capsys):
    holder = _holder(install)
    try:
        with pytest.raises(SystemExit) as stopped:
            update_cmd._pause_gateways_for_update(SimpleNamespace(no_gateway_restart=False))
    finally:
        holder.kill()
        holder.wait()
    assert install.paused == [], "the gateway was paused by an update that does not own the install"
    assert stopped.value.code == update_lock.UPDATE_EXIT_CONCURRENT
    out = capsys.readouterr().out
    assert "Another Hermes update is already running" in out and "No gateway was paused" in out, out


@pytest.mark.platforms("posix")
def test_a_dead_holders_lock_does_not_stop_the_pause(install):
    holder = _holder(install)
    holder.kill()
    holder.wait()
    lock = update_lock.UpdateLock(path=install.home / "marker", install_root=install.root)
    assert lock.acquire(), "the kernel did not free a dead holder's lock"
    try:
        update_cmd._pause_gateways_for_update(SimpleNamespace(no_gateway_restart=False))
    finally:
        lock.release()
    assert install.paused == ["windows", "posix"]
