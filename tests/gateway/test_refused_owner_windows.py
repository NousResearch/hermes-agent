"""#139845 / #140252 real Windows probes behind the named refusal states.

The classifier is unit-tested in ``test_gateway_refused_owner.py``; this proves the native probes
read real tokens and sessions: this process's own session/elevation, a same-token child (nothing to
name), and a session-0 system process seen from an interactive session (``gateway_other_session``,
the #140252 shape). Building an elevated owner seen from a filtered client needs a second logon
token, which a hosted runner cannot mint; that half stays a live retest.
"""
import csv
import io
import subprocess
import sys

import pytest

pytestmark = pytest.mark.platforms("windows")


def test_own_session_and_elevation_are_read_from_the_real_token():
    from gateway import runtime_bootstrap_windows as native
    from hermes_dir_acl import process_elevation, process_is_elevated
    import os

    assert isinstance(native._session_of(os.getpid()), int)
    assert process_elevation(os.getpid()) is process_elevation() is process_is_elevated()
    from hermes_cli.gateway_windows import _is_running_as_admin
    assert _is_running_as_admin() is process_is_elevated()


def test_a_same_token_owner_is_not_misnamed():
    from gateway import runtime_bootstrap_windows as native

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert native.probe_refused_owner(child.pid) is None
    finally:
        child.kill()
        child.wait(timeout=10)


def _session_zero_pid() -> int | None:
    rows = subprocess.run(["tasklist", "/fo", "csv", "/nh", "/fi", "SESSION eq 0"], capture_output=True,
                          text=True, check=False).stdout
    for row in csv.reader(io.StringIO(rows)):
        if len(row) > 1 and row[1].isdigit() and int(row[1]) > 4:
            return int(row[1])
    return None


def test_a_session_zero_owner_reads_as_another_logon_session():
    import os

    from gateway import runtime_bootstrap_windows as native

    if native._session_of(os.getpid()) == 0:
        pytest.skip("this runner itself runs in session 0")
    pid = _session_zero_pid()
    if pid is None:
        pytest.skip("no session-0 process visible")
    named = native.probe_refused_owner(pid)
    assert named is not None and named[0] == "gateway_other_session", named
    assert f"PID {pid}" in named[1] and "session 0" in named[1]
