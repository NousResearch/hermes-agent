"""Unattended cron children own stdin, even after the gateway's terminal exits."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import textwrap

import pytest


# Run descriptor manipulation in a disposable interpreter, never in the test runner.
_RUNNER = r'''
import json, os, subprocess, sys
from cron.scheduler_script import _run_job_script_with_claim_heartbeat

home, lane, revoked = sys.argv[1:]
if revoked == "yes":
    import pty
    master, slave = pty.openpty()
    owner = subprocess.run(
        [sys.executable, "-c", "import os,fcntl,termios; os.setsid(); "
         "fcntl.ioctl(0,termios.TIOCSCTTY,0)"],
        stdin=slave, capture_output=True, text=True, timeout=10, check=False,
    )
    assert owner.returncode == 0, owner.stderr
    # Session-leader exit revokes this private PTY on macOS, just like the
    # observed gateway fd 0. A merely closed fd or hung-up PTY is different.
    os.dup2(slave, 0)
else:
    # A readable gateway stdin must also not leak data to an unattended job.
    fd = os.open(os.path.join(home, "input"), os.O_RDONLY)
    os.dup2(fd, 0)
    os.close(fd)

result = _run_job_script_with_claim_heartbeat({}, lane)
print(json.dumps(result))
'''


def _probe(tmp_path, lane, revoked):
    home = tmp_path / "home"
    scripts = home / "scripts"
    scripts.mkdir(parents=True)
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    (home / "input").write_text("gateway input must stay private\n", encoding="utf-8")
    probe = "import sys\nassert sys.stdin.read() == '', 'inherited gateway input'\nprint('stdio-ok')\n"
    (scripts / "probe.py").write_text(probe, encoding="utf-8")
    (scripts / "chat").write_text(probe, encoding="utf-8")
    (scripts / "probe.sh").write_text(
        f"exec {shlex.quote(sys.executable)} {shlex.quote(str(scripts / 'probe.py'))}\n",
        encoding="utf-8",
    )
    env = dict(os.environ, HERMES_HOME=str(home))
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_RUNNER), str(home), lane, revoked],
        cwd=Path(__file__).resolve().parents[2], env=env,
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr
    ok, output = json.loads(result.stdout)
    assert ok, output
    assert output == "stdio-ok"


@pytest.mark.parametrize("lane", ["probe.py", "probe.sh"])
def test_cron_children_receive_eof_instead_of_gateway_input(tmp_path, lane):
    _probe(tmp_path, lane, "no")


@pytest.mark.platforms("macos")
@pytest.mark.parametrize("lane", ["probe.py", "probe.sh"])
def test_cron_children_start_after_gateway_terminal_is_revoked(tmp_path, lane):
    _probe(tmp_path, lane, "yes")
