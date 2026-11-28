"""Every known-broken local memory backend fails the memory conformance suite.

Each mutant (``mutants.py``) runs in two subprocesses. First its own-agent control — the world
setup and the positive operation matrix, nothing adversarial — which must PASS: a mutant that
crashes in its own-agent setup is a broken harness, not a kill the suite takes credit for. Then
the whole suite, stopping at the first failure, which must fail: a mutant the suite passes is a
hole in the suite.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conformance.memory.mutants import MUTANTS

ROOT = Path(__file__).resolve().parents[3]
SUITE = ["tests/conformance/memory/test_local.py"]
#: The mutant's own-agent control: every operation the agent itself may make, and the world setup.
CONTROL = ["tests/conformance/memory/test_local.py", "-k", "test_every_write_succeeds_for_its_own_agent"]


def _spawn(name, kind, root, *pytest_args):
    """One mutant's subprocess on its own home: a pytest session reaps processes under its sandbox
    home at exit, and a shared one would let the first child to finish terminate its siblings."""
    env = {k: v for k, v in os.environ.items() if k != "HERMES_TEST_SANDBOX_HOME"}
    env.update(HERMES_MEMORY_MUTANT=name, HERMES_HOME=str(root / kind / "home"))
    # pytest creates ``--basetemp`` without parents, and nothing else makes root/<kind>/ first.
    (root / kind).mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp",
         str(root / kind / "tmp"), *pytest_args],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _collect(procs):
    return {(name, kind): (proc.communicate(timeout=240)[0], proc.returncode)[::-1]
            for (name, kind), proc in procs.items()}


@pytest.fixture(scope="module")
def outcomes(tmp_path_factory):
    """``{(mutant, "control"|"suite"): (returncode, output)}``: the control wave first (it must
    pass before any suite run is believed), then the suite wave, parallel within each wave."""
    controls = {}
    for name in MUTANTS:
        controls[(name, "control")] = _spawn(name, "control", tmp_path_factory.mktemp(name), *CONTROL)
    outcomes = _collect(controls)
    suites = {}
    for name in MUTANTS:
        suites[(name, "suite")] = _spawn(name, "suite", tmp_path_factory.mktemp(name + "-suite"),
                                         "-x", *SUITE)
    outcomes.update(_collect(suites))
    return outcomes


@pytest.mark.parametrize("mutant", sorted(MUTANTS))
def test_the_mutant_survives_its_own_agent_setup_and_the_suite_rejects_it(outcomes, mutant):
    code, output = outcomes[(mutant, "control")]
    assert code == 0, (
        f"harness error: mutant {mutant!r} does not survive its own-agent setup, so the suite "
        f"has not been shown to reject it for the right reason:\n{output[-2000:]}")
    code, output = outcomes[(mutant, "suite")]
    assert code == 1 and " failed" in output, f"the suite passed broken backend {mutant!r}:\n{output[-2000:]}"
