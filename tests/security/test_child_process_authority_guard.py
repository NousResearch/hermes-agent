"""Behavior tests for the process-edge authority CI checker."""

from scripts.check_process_edge_authority import _find_escape_hatches_in_source


def test_guard_detects_missing_env_and_stdin_on_typed_consumer():
    findings = _find_escape_hatches_in_source(
        """
import subprocess
from tools.child_process_authority import build_child_process_env, probe_spec

def run_probe():
    spec = probe_spec(source="example")
    build_child_process_env(spec)
    subprocess.run(["probe"])
""",
        "agent/example.py",
    )

    assert any("missing explicit env policy" in finding for finding in findings)
    assert any("missing explicit stdin policy" in finding for finding in findings)


def test_guard_detects_explicit_ambient_env_on_typed_consumer():
    findings = _find_escape_hatches_in_source(
        """
import os
import subprocess as sp
from tools.child_process_authority import build_child_process_env, probe_spec

def run_probe():
    spec = probe_spec(source="example")
    build_child_process_env(spec)
    sp.run(["probe"], env=os.environ.copy(), stdin=sp.DEVNULL)
""",
        "agent/example.py",
    )

    assert any("explicitly requests ambient env" in finding for finding in findings)


def test_guard_accepts_explicit_brokered_spawn_policy():
    findings = _find_escape_hatches_in_source(
        """
from subprocess import DEVNULL, run
from tools.child_process_authority import build_child_process_env, probe_spec

spec = probe_spec(source="example")
run(["probe"], env=build_child_process_env(spec), stdin=DEVNULL)
""",
        "agent/example.py",
    )

    assert findings == []
