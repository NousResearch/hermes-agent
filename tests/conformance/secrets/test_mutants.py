"""Every known-broken secret store backend fails the conformance suite.

Each mutant (``mutants.py``) runs in two subprocesses. First its own-agent control — the world
setup and the positive operation and resolution matrix, nothing adversarial — which must PASS: a
mutant that crashes in its own-agent setup is a broken harness, not a kill the suite takes credit
for. Then the whole suite, which must fail — and each mutant declares the suite test that must
kill it (its node id in ``MUTANTS``), which the harness asserts is among the failures: a mutant
the suite rejects for some other reason alone has not been shown killed for the reason it exists.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conformance.secrets.mutants import MUTANTS

ROOT = Path(__file__).resolve().parents[3]
SUITE = ["tests/conformance/secrets/test_local.py", "tests/conformance/secrets/test_reference.py"]
#: The mutant's own-agent control: every operation the agent itself may make, and the resolution
#: the fixture's owners are entitled to.
CONTROL = [*SUITE, "-k", "succeeds or resolves"]


def _spawn(name, kind, root, *pytest_args):
    """One mutant's subprocess on its own home: a pytest session reaps processes under its sandbox
    home at exit, and a shared one would let the first child to finish terminate its siblings."""
    (root / kind / "home").mkdir(parents=True, exist_ok=True)  # pytest creates --basetemp itself
    (root / kind / "tmp").mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "HERMES_TEST_SANDBOX_HOME"}
    env.update(HERMES_SECRETS_MUTANT=name, HERMES_HOME=str(root / kind / "home"))
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
    pass before any suite run is believed), then the suite wave, parallel within each wave. The
    suite wave runs to the last failure (no early stop): the killing test the mutant declares
    must be among them, wherever the mutant breaks the suite first."""
    controls = {}
    for name in MUTANTS:
        controls[(name, "control")] = _spawn(name, "control", tmp_path_factory.mktemp(name), *CONTROL)
    outcomes = _collect(controls)
    suites = {}
    for name in MUTANTS:
        suites[(name, "suite")] = _spawn(name, "suite", tmp_path_factory.mktemp(name + "-suite"),
                                         *SUITE, "-rf")
    outcomes.update(_collect(suites))
    return outcomes


def _failed_ids(output):
    """The node ids of the tests a suite run failed, from pytest's short summary."""
    return {line[len("FAILED "):].split(" - ", 1)[0]
            for line in output.splitlines() if line.startswith("FAILED ")}


@pytest.mark.parametrize("mutant", sorted(MUTANTS))
def test_the_mutant_survives_its_own_agent_setup_and_the_suite_rejects_it(outcomes, mutant):
    code, output = outcomes[(mutant, "control")]
    assert code == 0, (
        f"harness error: mutant {mutant!r} does not survive its own-agent setup, so the suite "
        f"has not been shown to reject it for the right reason:\n{output[-2000:]}")
    code, output = outcomes[(mutant, "suite")]
    assert code == 1, f"the suite passed broken backend {mutant!r}:\n{output[-2000:]}"
    declared = MUTANTS[mutant][1]
    failed = _failed_ids(output)
    assert declared in failed, (
        f"the suite rejected mutant {mutant!r} without its declared killing test "
        f"{declared!r} failing (failures: {sorted(failed)}):\n{output[-2000:]}")
