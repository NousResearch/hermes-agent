"""The core-tool-deferral worker's host sandbox (evals/core_tool_deferral/sandbox.py).

A past live run let agents install real nightly cron jobs and write scripts into the
operator's real ~/.hermes. The contract: after isolate_host, a child shell's crontab and
HOME both resolve inside the run dir.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from evals.core_tool_deferral.sandbox import isolate_host


@pytest.mark.platforms("posix")  # the crontab shim is a POSIX sh script
def test_child_shell_crontab_and_home_stay_inside_the_run_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    run_root = tmp_path / "run"
    hermes_home = run_root / ".hermes"
    hermes_home.mkdir(parents=True)

    crontab_file = isolate_host(str(run_root), str(hermes_home))
    env = dict(os.environ)

    def sh(cmd: str) -> str:
        return subprocess.run(["sh", "-c", cmd], env=env, capture_output=True, text=True, check=True).stdout

    # Touch crontab only once the shim is proven first on PATH, so a regression can never
    # write the host's real crontab from the test itself.
    assert sh("command -v crontab").strip() == str(run_root / "bin" / "crontab")
    assert sh("echo '0 2 * * * ./backup.sh' | crontab - && crontab -l") == "0 2 * * * ./backup.sh\n"
    assert Path(crontab_file).read_text(encoding="utf-8") == "0 2 * * * ./backup.sh\n"
    assert sh('echo "$HOME"').strip() == str(hermes_home / "home")
    assert env["TERMINAL_HOME_MODE"] == "profile"
    assert "DBUS_SESSION_BUS_ADDRESS" not in env and "XDG_RUNTIME_DIR" not in env
