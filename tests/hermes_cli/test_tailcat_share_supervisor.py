"""The supervised ``tailcat serve`` must not outlive the Hermes process that shares.

An orphaned tailcat keeps publishing the share after Hermes is gone, and the next
``hermes serve --share tailcat`` starts a second one on the same key.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

FAKE_TAILCAT = """#!/bin/sh
echo $$ > "$(dirname "$0")/tailcat.pid"
echo '{"listenAddr":"tcFAKEADDRESS"}'
exec sleep 120
"""

HOST = textwrap.dedent("""
    import sys, time
    from pathlib import Path
    from hermes_cli.tailcat_share import TailcatSupervisor

    sup = TailcatSupervisor(binary=Path(sys.argv[1]), key=Path(sys.argv[2]), port=1)
    sup.start()
    assert sup.wait_ready(20), sup.status.as_dict()
    print("ready", flush=True)
    time.sleep(120)
""")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers signal 0; it is dead for our purposes.
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


@pytest.mark.platforms("linux")
def test_tailcat_dies_when_the_sharing_process_is_killed(tmp_path):
    fake = tmp_path / "tailcat"
    fake.write_text(FAKE_TAILCAT)
    fake.chmod(0o755)
    host = subprocess.Popen(
        [sys.executable, "-c", HOST, str(fake), str(tmp_path / "server.key.json")],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert host.stdout is not None
        assert host.stdout.readline().strip() == "ready"
        tailcat_pid = int((tmp_path / "tailcat.pid").read_text())
        assert _alive(tailcat_pid)

        host.send_signal(signal.SIGKILL)  # no cleanup path runs
        host.wait(timeout=10)

        deadline = time.monotonic() + 10
        while _alive(tailcat_pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not _alive(tailcat_pid), "tailcat outlived the process sharing it"
    finally:
        if host.poll() is None:
            host.kill()
        pid_file = tmp_path / "tailcat.pid"
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
