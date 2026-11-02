"""The spawn ledger's boot-relative identity: a registered gateway's scratch survives a wall-clock
step (NTP, VM resume), and boot ticks never vouch for a process across a reboot."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from hermes_cli.data_cleanup_holders import _refuse_backend_writers
from hermes_cli.process_identity import (
    _ledger_path, boot_start, install_id, ledger_entries, reap_orphaned_mcp_helpers, register_self,
)
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


@pytest.mark.platforms("linux")  # boot ids and boot-relative ticks are recorded only here
def test_ledger_never_vouches_for_an_earlier_boot_or_an_exited_backend(tmp_path, monkeypatch):
    """The ledger survives reboots but boot ticks restart: an MCP-helper entry written in an
    earlier boot whose pid and start tick equal an unrelated process running now (early-boot
    services are deterministic) names a stranger, and its spawner is gone, so the orphan reaper
    must leave the stranger alone. Older epoch-only serve entries (written before boot ticks) keep
    main's epoch match: one whose backend has exited (a zombie its parent has not reaped yet) and
    one whose pid the stranger now holds no longer own HERMES_HOME. No row survives the next
    ledger write."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    stranger = subprocess.Popen(["sleep", "60"], stdin=subprocess.DEVNULL)
    backend = subprocess.Popen(["true"], stdin=subprocess.DEVNULL)
    spawner = subprocess.Popen(["true"], stdin=subprocess.DEVNULL)
    spawner.wait(timeout=10)
    try:
        wait_until(lambda: psutil.Process(backend.pid).status() == psutil.STATUS_ZOMBIE,
                   timeout=10, what="the backend exited")
        live = psutil.Process(stranger.pid)
        earlier_boot = time.time() - 3 * 86400
        _ledger_path().write_text(json.dumps([
            {"pid": stranger.pid, "create_time": live.create_time() - 3 * 86400, "purpose": "mcp-helper",
             "install": install_id(), "spawner_pid": spawner.pid, "spawner_create": earlier_boot,
             "registered_at": earlier_boot, "argv": "", "boot_start": boot_start(live), "spawner_boot": 1.0,
             "boot_id": "00000000-0000-4000-8000-000000000000"},
            {"pid": backend.pid, "create_time": psutil.Process(backend.pid).create_time(), "purpose": "serve",
             "install": install_id(), "spawner_pid": None, "spawner_create": None,
             "registered_at": time.time(), "argv": ""},
            {"pid": stranger.pid, "create_time": live.create_time() - 3 * 86400, "purpose": "serve",
             "install": install_id(), "spawner_pid": None, "spawner_create": None,
             "registered_at": earlier_boot, "argv": ""},
        ]), encoding="utf-8")

        assert stranger.pid not in [e["pid"] for e in ledger_entries()]
        assert reap_orphaned_mcp_helpers() == []
        assert stranger.poll() is None, "the orphan reaper killed a process from a later boot"
        _refuse_backend_writers(tmp_path)  # raises while a backend still owns the home
        assert register_self("gateway")
        rows = json.loads(_ledger_path().read_text(encoding="utf-8"))
        assert [row["pid"] for row in rows] == [os.getpid()]
    finally:
        stranger.kill()
        stranger.wait(timeout=10)
        backend.wait(timeout=10)
