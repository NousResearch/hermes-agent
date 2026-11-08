"""Every known-broken local backend fails the storage conformance suite.

Each mutant (``mutants.py``) runs the whole suite in a subprocess (stopping at the first failure);
a mutant the suite passes is a hole in the suite. The five adversarial-review mutants all passed
the suite before it was parametrized over every operation.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conformance.storage.mutants import MUTANTS

ROOT = Path(__file__).resolve().parents[3]
SUITE = ["tests/conformance/storage/test_local.py", "tests/conformance/storage/test_local_pinning.py"]


@pytest.fixture(scope="module")
def outcomes(tmp_path_factory):
    """Run the suite once per mutant, all in parallel: ``{mutant: (returncode, output)}``."""
    procs = {}
    for name in MUTANTS:
        root = tmp_path_factory.mktemp(name)
        # Its own home: a pytest session reaps processes under its sandbox home at exit, and a
        # shared one would let the first child to finish terminate its siblings.
        env = {k: v for k, v in os.environ.items() if k != "HERMES_TEST_SANDBOX_HOME"}
        env.update(HERMES_STORAGE_MUTANT=name, HERMES_HOME=str(root / "home"))
        procs[name] = subprocess.Popen(
            [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "--basetemp",
             str(root / "tmp"), *SUITE],
            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return {name: (proc.communicate(timeout=240)[0], proc.returncode)[::-1] for name, proc in procs.items()}


@pytest.mark.parametrize("mutant", sorted(MUTANTS))
def test_the_suite_rejects_the_mutant(outcomes, mutant):
    code, output = outcomes[mutant]
    assert code == 1 and " failed" in output, f"the suite passed broken backend {mutant!r}:\n{output[-2000:]}"
