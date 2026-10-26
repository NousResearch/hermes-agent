"""The scratch prune's owner check survives a wall-clock step (NTP, VM resume)."""

import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from hermes_constants import get_scratch_dir, prune_scratch_dir
from tests.waiting import wait_until


@pytest.mark.platforms("linux")  # psutil's create_time adds the live /proc/stat btime only here
@pytest.mark.parametrize("step", [5.0, -5.0])
def test_prune_keeps_a_registered_gateways_shell_after_a_clock_step(tmp_path, monkeypatch, step):
    """A gateway registered in the spawn ledger runs a shell inside an idle scratch entry. The
    wall clock then steps by ``step`` seconds, which moves psutil's epoch ``create_time`` of every
    process on Linux. The gateway is still the same process: the prune must keep owning it, never
    kill its shell or delete the entry the shell works in."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    scratch = get_scratch_dir(tmp_path, prune=False)
    lane = scratch / "gateway-lane"
    lane.mkdir()
    shell_pidfile = tmp_path / "shell.pid"
    repo = str(Path(__file__).resolve().parents[1])
    gateway = subprocess.Popen(
        [sys.executable, "-c",
         f"import sys, subprocess, time; sys.path.insert(0, {repo!r}); "
         "from hermes_cli.process_identity import register_self; assert register_self('gateway'); "
         f"shell = subprocess.Popen(['sleep', '60'], cwd={str(lane)!r}); "
         f"open({str(shell_pidfile)!r}, 'w').write(str(shell.pid)); time.sleep(60)"],
        stdin=subprocess.DEVNULL, env={**os.environ, "HERMES_HOME": str(tmp_path)})
    try:
        shell = int(wait_until(lambda: shell_pidfile.exists() and shell_pidfile.read_text().strip(),
                               timeout=20, what="registered gateway started its shell"))
        real_btime = psutil.boot_time()
        monkeypatch.setattr(psutil._pslinux, "boot_time", lambda: real_btime + step)
        ancient = time.time() - 30 * 3600
        os.utime(lane, (ancient, ancient))

        prune_scratch_dir(scratch)

        assert psutil.Process(shell).status() != psutil.STATUS_ZOMBIE, "the gateway's shell was killed"
        assert lane.exists(), "the entry the gateway's shell works in was deleted"
    finally:
        for child in psutil.Process(gateway.pid).children(recursive=True):
            child.kill()
        gateway.kill()
        gateway.wait(timeout=10)
