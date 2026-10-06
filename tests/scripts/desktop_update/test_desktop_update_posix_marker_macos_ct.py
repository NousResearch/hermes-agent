"""proc_ct must report the clock the Python reader compares against (update_lock's psutil).

The hand-off publishes the update child as the marker's delegate line (contract C1 rule 6), and
update_lock judges OUR OWN pid within ``_OWN_CREATE_TIME_EPSILON`` -- 5 ms. A reading floored to
whole seconds (what ``ps -o lstart=`` gives on macOS, which has no sub-second process clock) is
0.1-0.9 s off, so the delegate line read as a previous incarnation: the update found no partner
but the live custodian and refused its own hand-off with exit 2 ("Another Hermes update is
already running"), and no Desktop update could start on macOS.
"""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

pytestmark = pytest.mark.platforms("macos")

ROOT = Path(__file__).resolve().parents[3]
MARKER_SH = ROOT / "scripts" / "desktop-update" / "marker.sh"

PRELUDE = f"set -u\nlog() {{ :; }}\n. {shlex.quote(str(MARKER_SH))}\n"

JUDGE = """import pathlib, sys
sys.path.insert(0, sys.argv[2])
from hermes_cli.update_lock import judge_marker
print(judge_marker(pathlib.Path(sys.argv[1]).read_bytes())[0])
"""


def _sh(script: str) -> list[str]:
    proc = subprocess.run(["bash", "-c", PRELUDE + script], capture_output=True, text=True,
                          timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.splitlines()


@pytest.fixture
def custodian():
    """A live process with no lineage to the hand-off: what the marker's line 1 names."""
    proc = subprocess.Popen(["sleep", "300"])
    yield proc.pid
    proc.kill()
    proc.wait()


def test_proc_ct_reports_the_creation_time_the_python_reader_compares(custodian):
    psutil = pytest.importorskip("psutil")
    from hermes_cli.update_lock import _OWN_CREATE_TIME_EPSILON

    # our own pid is the hand-off's parked child: it is judged with the own-incarnation
    # epsilon -- never the 2 s tolerance a foreign pid gets.
    for pid in (os.getpid(), custodian):
        got = float(_sh(f"proc_ct {pid}")[0])
        want = psutil.Process(pid).create_time()
        assert abs(got - want) <= _OWN_CREATE_TIME_EPSILON, f"pid {pid}: proc_ct {got} vs psutil {want}"


def test_the_delegate_the_handoff_publishes_is_judged_ours(tmp_path, custodian):
    """The parked child's line 4, written from the shell's own pid, must read ``ours`` once that
    pid execs the update (the shell keeps its pid through the exec), so the update adopts the
    claim. A floored ct left the live custodian on line 1 as the only partner: it refused."""
    marker = tmp_path / "marker"
    judge = tmp_path / "judge.py"
    judge.write_text(JUDGE, encoding="utf-8")
    script = (
        f"printf '%s\\n%s\\nct:%s\\n' {custodian} \"$(marker_now)\" \"$(proc_ct {custodian})\""
        f" > {shlex.quote(str(marker))}\n"
        f"printf 'delegate:%s ct:%s\\n' \"$$\" \"$(proc_ct $$)\" >> {shlex.quote(str(marker))}\n"
        f"exec {shlex.quote(sys.executable)} {shlex.quote(str(judge))}"
        f" {shlex.quote(str(marker))} {shlex.quote(str(ROOT))}\n"
    )
    assert _sh(script) == ["ours"]
