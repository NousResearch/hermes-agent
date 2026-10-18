"""Job-scoped ``secret_env`` (CORP-764 formal review): a process the script starts cannot keep
the values after the run. The script launches a detached child with its output on DEVNULL (so
nothing ties the run to it) and returns; whatever way the run ends, that child is gone once
``run_job`` returns. A child that keeps the script's pipes instead ends with the script, not at
the timeout.

Real ``run_job`` and real processes against a temp ``HERMES_HOME``; the only fake is the
``wincred`` backend. Windows only: secret_env fails closed elsewhere.
"""

from __future__ import annotations

import hashlib
import time

import psutil
import pytest

FAKE_VALUE = "FAKE-SECRET-VALUE-0123456789"
TARGET = "hermes/notice-report-slack"
NAME = "NOTICE_REPORT_SLACK_TOKEN"

_CHILD = f"""\
import os, pathlib, sys, time
beacon = pathlib.Path(sys.argv[1])
beacon.with_suffix(".pid").write_text(str(os.getpid()))
for _ in range(600):
    with beacon.open("a") as f:
        f.write(str(len(os.environ.get("{NAME}", ""))) + "\\n")
    time.sleep(0.05)
"""

_PARENT = f"""\
import os, pathlib, subprocess, sys, time
beacon = pathlib.Path(sys.argv[0]).with_name("beacon.txt")
mode = os.environ.get("CRON_CONTAINMENT_MODE", "ok")
# "inherit": the child keeps the script's stdout/stderr pipes open instead.
inherit = mode == "inherit"
subprocess.Popen(
    [sys.executable, "-c", {_CHILD!r}, str(beacon)], stdin=subprocess.DEVNULL,
    stdout=sys.stdout if inherit else subprocess.DEVNULL,
    stderr=sys.stderr if inherit else subprocess.DEVNULL,
    creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
deadline = time.monotonic() + 30
while not (beacon.exists() and beacon.stat().st_size) and time.monotonic() < deadline:
    time.sleep(0.05)
if mode == "hang":
    time.sleep(120)
print("parent done")
sys.exit(3 if mode == "fail" else 0)
"""


class FakeBackend:
    def resolve(self, target):
        return FAKE_VALUE


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "scripts").mkdir(parents=True)
    (home / "cron").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from cron import secret_env

    monkeypatch.setattr(secret_env, "_default_backends", lambda: {"wincred": FakeBackend()})
    return home


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("mode", ["ok", "fail", "hang", "inherit"])
def test_detached_child_does_not_outlive_the_run(hermes_env, monkeypatch, mode):
    import cron.scheduler
    from cron.jobs import create_job
    from cron.scheduler import run_job

    monkeypatch.setenv("CRON_CONTAINMENT_MODE", mode)
    monkeypatch.setattr(cron.scheduler, "_SCRIPT_TIMEOUT", 10)
    script = hermes_env / "scripts" / "spawner.py"
    script.write_text(_PARENT, encoding="utf-8")
    job = create_job(prompt=None, schedule="every 5m", script="spawner.py", no_agent=True,
                     deliver="local", secret_env={NAME: f"wincred://{TARGET}"},
                     secret_env_script_sha256=hashlib.sha256(script.read_bytes()).hexdigest())
    beacon = script.with_name("beacon.txt")

    success, _doc, final_response, _error = run_job(job)

    # "inherit" succeeds too: the run ends with the script, not at the timeout.
    assert success is (mode in ("ok", "inherit"))
    assert FAKE_VALUE not in final_response
    # The child really ran with the value, and is gone the moment the run returns.
    assert beacon.read_text().split()[0] == str(len(FAKE_VALUE))
    pid = int(beacon.with_suffix(".pid").read_text())
    assert not psutil.pid_exists(pid)
    seen = beacon.stat().st_size
    time.sleep(0.5)
    assert beacon.stat().st_size == seen
