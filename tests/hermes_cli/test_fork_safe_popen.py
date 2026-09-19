"""``fork_safe_popen``: a threaded macOS parent must never fork() to spawn (#97296).

CPython 3.11 takes posix_spawn only when cwd is None, close_fds and pass_fds are off, no
new session is requested and argv[0] names a directory; anything else fork()s. A threaded
parent holding Network.framework state (e.g. with a VPN network extension loaded) can then
crash the child in an atfork handler before exec: SIGSEGV, exit -11, empty output. The
requested cwd, session, fd set and environment must still reach the child unchanged, and a
missing program or cwd must still raise in the parent.

Observed through CPython's own audit events (``subprocess.Popen`` fires for every spawn,
``os.posix_spawn`` only on the no-fork path), in a fresh interpreter because audit hooks
cannot be removed.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

_HARNESS = textwrap.dedent("""
    import json, os, subprocess, sys, threading
    events = []
    sys.addaudithook(
        lambda e, a: events.append(e) if e in ("subprocess.Popen", "os.posix_spawn") else None)
    threading.Thread(target=threading.Event().wait, daemon=True).start()  # a threaded parent
    from hermes_cli._subprocess_compat import fork_safe_popen

    cwd = sys.argv[1]
    passed, leaked = 60, 61  # both inheritable in the parent; only `passed` is handed over
    os.dup2(os.pipe()[1], passed)
    os.dup2(os.pipe()[1], leaked)
    probe = ("import json, os; print(json.dumps({'cwd': os.getcwd(), 'pid': os.getpid(), "
             "'pgid': os.getpgrp(), 'sid': os.getsid(0), "
             "'fds': sorted(int(f) for f in os.listdir('/dev/fd'))}))")
    proc = fork_safe_popen(
        [os.path.basename(sys.executable), "-c", probe], cwd=cwd,
        env={**os.environ, "PATH": os.path.dirname(sys.executable)},
        start_new_session=True, pass_fds=(passed,), stdout=subprocess.PIPE, text=True)
    child = json.loads(proc.communicate(timeout=60)[0])
    spawn_events = list(events)
    # No locale in the env: Python's own startup would coerce LC_CTYPE into it (PEP 538).
    exact_env = fork_safe_popen(["env"], cwd=cwd, env={"PATH": "/usr/bin"}, stdout=subprocess.PIPE,
                                text=True).communicate(timeout=60)[0].splitlines()
    errors = {}
    for label, args, kwargs in (
            ("missing_program", ["hermes-no-such-program"], {}),
            ("missing_cwd", [sys.executable, "-c", "pass"], {"cwd": os.path.join(cwd, "gone")})):
        try:
            fork_safe_popen(args, **kwargs).wait()
        except OSError as exc:
            errors[label] = type(exc).__name__
    print(json.dumps({"events": spawn_events, "child": child, "parent_pgid": os.getpgrp(),
                      "passed": passed, "leaked": leaked, "env": exact_env, "errors": errors}))
""")


@pytest.mark.macos_only
@pytest.mark.live_system_guard_bypass
def test_threaded_parent_spawn_keeps_popen_semantics_without_fork(tmp_path):
    harness = subprocess.run(
        [sys.executable, "-c", _HARNESS, str(tmp_path)], cwd=_REPO_ROOT,
        capture_output=True, text=True, timeout=120, start_new_session=True)
    assert harness.returncode == 0, harness.stderr[-2000:]
    report = json.loads(harness.stdout.splitlines()[-1])
    assert report["events"] == ["subprocess.Popen", "os.posix_spawn"], "the spawn fork()ed"
    child = report["child"]
    assert child["cwd"] == os.path.realpath(tmp_path)
    assert child["pid"] == child["pgid"] == child["sid"] != report["parent_pgid"]
    assert report["passed"] in child["fds"] and report["leaked"] not in child["fds"]
    assert report["env"] == ["PATH=/usr/bin"]
    assert report["errors"] == {
        "missing_program": "FileNotFoundError", "missing_cwd": "FileNotFoundError"}
