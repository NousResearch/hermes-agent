"""The gateway pid scan follows a process's real home.

Regression: a web server started on a temp HERMES_HOME runs the orphan reaper at startup. Its
current-profile scan claimed every ``gateway run`` with no profile flag and no ``HERMES_HOME=`` on
argv, so it SIGTERMed the operator's launchd gateway, whose home arrives through the plist
environment and never appears on argv.

The test spawns only its own child and reads the process table. Nothing is signalled except that
child, and it exits on its own once its parent is gone.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import hermes_cli.gateway as gateway
from hermes_cli import dashboard_procs

# Named ``hermes`` so ``python <dir>/hermes gateway run`` satisfies the canonical gateway matcher.
_GATEWAY_STUB = """
import os, time
open(os.environ["STUB_READY"], "w").close()
parent, deadline = os.getppid(), time.monotonic() + 120
while os.getppid() == parent and time.monotonic() < deadline:
    time.sleep(0.1)
"""

def _gateway_argv(tmp_path: Path) -> list[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    script = bin_dir / "hermes"
    script.write_text(_GATEWAY_STUB, encoding="utf-8")
    return [sys.executable, str(script), "gateway", "run"]


def _wait_for(path: Path, proc: subprocess.Popen) -> None:
    deadline = time.monotonic() + 15.0
    while not path.exists():
        if proc.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError("gateway stub never came up")
        time.sleep(0.05)


@pytest.mark.spawns_gateway_lookalike
def test_scan_claims_a_bare_gateway_only_for_the_home_its_environment_names(tmp_path, monkeypatch):
    home_a, home_b = tmp_path / "home-a", tmp_path / "home-b"
    home_a.mkdir()
    home_b.mkdir()
    ready = tmp_path / "gateway.ready"
    monkeypatch.setattr(gateway, "_get_service_pids", lambda all_profiles=False: set())
    proc = subprocess.Popen(
        _gateway_argv(tmp_path),
        env={**os.environ, "HERMES_HOME": str(home_a), "STUB_READY": str(ready)},
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # Any other gateway on the host reads as an unreadable environment, so resolving it never
    # touches the operator's real home; only the stub's own environment is replayed.
    real_home_for_pid = dashboard_procs._hermes_home_for_pid
    monkeypatch.setattr(
        dashboard_procs, "_hermes_home_for_pid",
        lambda pid: real_home_for_pid(pid) if pid == proc.pid else None,
    )
    try:
        _wait_for(ready, proc)
        monkeypatch.setenv("HERMES_HOME", str(home_b))
        assert proc.pid not in gateway.find_gateway_pids()
        monkeypatch.setenv("HERMES_HOME", str(home_a))
        assert proc.pid in gateway.find_gateway_pids()
    finally:
        proc.terminate()
        proc.wait(timeout=10)

