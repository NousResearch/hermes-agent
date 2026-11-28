"""Behaviour contracts for bounded execution: caps, wall clock, exact cleanup."""

import sys

from isolated_tests.executor import Cleanup, run_bounded


def fake_docker(tmp_path, label, *, inspect_rc=0, rm_rc=0):
    """A stand-in docker CLI: inspect prints the label until rm deletes it."""
    state = tmp_path / "container"
    state.write_text(label)
    inspect = [sys.executable, "-c",
               "import os,sys\n"
               f"p={str(state)!r}\n"
               "if not os.path.exists(p): sys.stderr.write('Error: No such container: x'); sys.exit(1)\n"
               f"print(open(p).read()); sys.exit({inspect_rc})"]
    rm = [sys.executable, "-c", f"import os,sys; os.unlink({str(state)!r}) if {rm_rc}==0 else None; sys.exit({rm_rc})"]
    return Cleanup(inspect, rm, "run1")


def test_stdout_is_capped():
    out = run_bounded([sys.executable, "-c", "print('x' * 100000)"], {}, wall_seconds=30,
                      stdout_cap=100, stderr_cap=100, cleanup=None, cleanup_timeout=5)
    assert out.returncode == 0 and len(out.stdout) == 100 and out.stdout_truncated


def test_timeout_kills_and_removes_owned_container(tmp_path):
    out = run_bounded([sys.executable, "-c", "import time; time.sleep(60)"], {}, wall_seconds=2,
                      stdout_cap=10, stderr_cap=10, cleanup=fake_docker(tmp_path, "run1"),
                      cleanup_timeout=10)
    assert out.timed_out and out.returncode is None and out.cleanup_status == "removed"
